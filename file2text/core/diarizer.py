"""Модуль для диаризации спикеров в аудио."""

import os
from typing import Dict, List, Optional, Any
from pyannote.audio import Pipeline


class Diarizer:
    """Класс для диаризации спикеров в аудио файлах."""
    
    def __init__(
        self,
        auth_token: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Инициализация диаризатора.
        
        Args:
            auth_token: Hugging Face токен для доступа к модели диаризации.
                      Если None, берется из переменной окружения HUGGINGFACE_TOKEN
            verbose: Выводить ли подробную информацию
        """
        self.verbose = verbose
        
        # Получаем токен
        if not auth_token:
            auth_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if not auth_token:
            raise ValueError(
                "HUGGINGFACE_TOKEN не установлен. "
                "Установите переменную окружения HUGGINGFACE_TOKEN или "
                "передайте auth_token при инициализации."
            )
        
        if self.verbose:
            print("Загрузка модели диаризации...")
        
        # Загружаем модель диаризации
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=auth_token
        )
        
        if self.verbose:
            print("Модель диаризации загружена")
    
    def diarize(self, audio_path: str) -> Any:
        """
        Выполняет диаризацию спикеров в аудио файле.
        
        Args:
            audio_path: Путь к аудио файлу
            
        Returns:
            Результат диаризации от pyannote.audio
        """
        if self.verbose:
            print(f"Начинаю диаризацию: {audio_path}")
        
        diarization = self.pipeline(audio_path)
        
        if self.verbose:
            print("Диаризация завершена")
        
        return diarization
    
    def assign_speakers(
        self,
        transcript_segments: List[Dict[str, Any]],
        diarization: Any
    ) -> List[Dict[str, Any]]:
        """
        Сопоставляет спикеров с сегментами транскрипции.
        
        Args:
            transcript_segments: Список сегментов транскрипции с ключами:
                                start, end, text
            diarization: Результат диаризации от pyannote.audio
            
        Returns:
            List[Dict]: Список сегментов с информацией о спикере:
                       - speaker: Имя спикера
                       - text: Текст сегмента
                       - start: Время начала
                       - end: Время окончания
        """
        speaker_transcript = []
        
        for segment in transcript_segments:
            start = segment['start']
            end = segment['end']
            text = segment['text']
            
            # Найти спикера для данного сегмента
            speaker = None
            for turn, _, spk in diarization.itertracks(yield_label=True):
                if (turn.start <= start <= turn.end or 
                    turn.start <= end <= turn.end or 
                    (start <= turn.start and end >= turn.end)):
                    speaker = spk
                    break
            
            if speaker is None:
                speaker = "Unknown"
            
            speaker_transcript.append({
                'speaker': speaker,
                'text': text.strip(),
                'start': start,
                'end': end
            })
        
        return speaker_transcript
    
    def get_speakers_text(
        self,
        speaker_segments: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Группирует текст по спикерам.
        
        Args:
            speaker_segments: Список сегментов с информацией о спикерах
            
        Returns:
            Dict[str, str]: Словарь {speaker: text}
        """
        speakers_dict = {}
        
        for segment in speaker_segments:
            speaker = segment['speaker']
            text = segment['text']
            
            if speaker not in speakers_dict:
                speakers_dict[speaker] = []
            
            speakers_dict[speaker].append(text)
        
        # Объединяем тексты каждого спикера
        speakers_text = {}
        for speaker, texts in speakers_dict.items():
            speakers_text[speaker] = ' '.join(texts)
        
        return speakers_text
