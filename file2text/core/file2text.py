"""Главный класс File2Text для единого API."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from file2text.core.transcriber import Transcriber
from file2text.core.diarizer import Diarizer
from file2text.core.summarizer import Summarizer
from file2text.core.vectorizer import Vectorizer
from file2text.utils.audio_converter import AudioConverter
from file2text.utils.config import Config, load_config


@dataclass
class ProcessingResult:
    """Результат обработки аудио файла."""
    audio_path: str
    text: Optional[str] = None
    segments: List[Dict[str, Any]] = field(default_factory=list)
    speakers: Dict[str, str] = field(default_factory=dict)
    speaker_segments: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, str] = field(default_factory=dict)
    vectors: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результат в словарь."""
        result = {
            "audio_path": self.audio_path,
            "text": self.text,
            "speakers": self.speakers,
            "summary": self.summary,
            "metadata": self.metadata
        }
        if self.vectors is not None:
            result["vectors_shape"] = self.vectors.shape if hasattr(self.vectors, 'shape') else None
        return result


class File2Text:
    """Главный класс для обработки аудио в текст с суммаризацией и векторизацией."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        whisper_model: str = "medium",
        verbose: bool = False
    ):
        """
        Инициализация File2Text.
        
        Args:
            config: Объект конфигурации. Если None, загружается из переменных окружения
            whisper_model: Модель Whisper для транскрипции
            verbose: Выводить ли подробную информацию
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.verbose = verbose
        
        # Инициализируем компоненты
        self.transcriber = Transcriber(
            model=whisper_model or config.whisper_model,
            device=config.whisper_device,
            verbose=verbose
        )
        
        self.diarizer = Diarizer(
            auth_token=config.huggingface_token,
            verbose=verbose
        )
        
        self.summarizer = Summarizer(
            model=config.summarizer_model,
            device=config.whisper_device,
            verbose=verbose
        )
        
        self.vectorizer = Vectorizer(
            model=config.vectorizer_model,
            device=config.whisper_device,
            verbose=verbose
        )
        
        self.audio_converter = AudioConverter()
    
    def process(
        self,
        audio_path: str,
        transcribe: bool = True,
        diarize: bool = False,
        summarize: bool = False,
        vectorize: bool = False,
        **kwargs
    ) -> ProcessingResult:
        """
        Полный пайплайн обработки аудио или видео файла.
        
        Args:
            audio_path: Путь к аудио или видео файлу
            transcribe: Выполнить транскрипцию
            diarize: Выполнить диаризацию спикеров
            summarize: Выполнить суммаризацию
            vectorize: Выполнить векторизацию
            **kwargs: Дополнительные параметры для транскрипции
            
        Returns:
            ProcessingResult: Результат обработки
        """
        result = ProcessingResult(audio_path=audio_path)
        
        # Конвертируем медиа файл (аудио или видео) в WAV если нужно
        original_path = audio_path
        if not Path(audio_path).suffix.lower() == '.wav':
            if self.verbose:
                if self.audio_converter.is_video_file(audio_path):
                    print(f"Обнаружен видео файл, извлекаю аудио...")
                else:
                    print(f"Конвертирую аудио в WAV...")
            
            audio_path = self.audio_converter.convert_to_wav(audio_path)
            result.metadata['converted_audio_path'] = audio_path
            result.metadata['original_path'] = original_path
        
        # Транскрипция
        if transcribe:
            transcript_result = self.transcriber.transcribe(audio_path, **kwargs)
            result.text = transcript_result['text']
            result.segments = transcript_result.get('segments', [])
            result.metadata['language'] = transcript_result.get('language', 'ru')
        
        # Диаризация
        if diarize and result.segments:
            diarization = self.diarizer.diarize(audio_path)
            result.speaker_segments = self.diarizer.assign_speakers(result.segments, diarization)
            result.speakers = self.diarizer.get_speakers_text(result.speaker_segments)
        
        # Суммаризация
        if summarize:
            if result.text:
                result.summary['full'] = self.summarizer.summarize_full(result.text)
            
            if result.speakers:
                result.summary['by_speakers'] = self.summarizer.summarize_by_speakers(result.speakers)
        
        # Векторизация
        if vectorize:
            if result.text:
                result.vectors = self.vectorizer.vectorize(result.text)
        
        return result
    
    def transcribe(self, audio_path: str, **kwargs) -> str:
        """
        Только транскрипция аудио или видео файла.
        
        Args:
            audio_path: Путь к аудио или видео файлу
            **kwargs: Дополнительные параметры для транскрипции
            
        Returns:
            str: Транскрибированный текст
        """
        # Автоматически конвертируем если нужно
        if not Path(audio_path).suffix.lower() == '.wav':
            audio_path = self.audio_converter.convert_to_wav(audio_path)
        
        result = self.transcriber.transcribe(audio_path, **kwargs)
        return result['text']
    
    def transcribe_with_speakers(self, audio_path: str, **kwargs) -> ProcessingResult:
        """
        Транскрипция с диаризацией спикеров.
        
        Args:
            audio_path: Путь к аудио или видео файлу
            **kwargs: Дополнительные параметры для транскрипции
            
        Returns:
            ProcessingResult: Результат обработки
        """
        return self.process(audio_path, transcribe=True, diarize=True, **kwargs)
    
    def summarize(self, text: str, **kwargs) -> str:
        """Только суммаризация текста."""
        return self.summarizer.summarize(text, **kwargs)
    
    def vectorize(self, text: str) -> Any:
        """Только векторизация текста."""
        return self.vectorizer.vectorize(text)
