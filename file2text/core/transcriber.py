"""Модуль для транскрипции аудио в текст с помощью Whisper."""

import whisper
import torch
from typing import Dict, List, Optional, Any
from pathlib import Path


class Transcriber:
    """Класс для транскрипции аудио файлов в текст."""
    
    def __init__(
        self,
        model: str = "medium",
        device: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Инициализация транскриптора.
        
        Args:
            model: Модель Whisper (tiny, base, small, medium, large-v2, large-v3)
            device: Устройство для обработки ("cuda" или "cpu"). Если None, определяется автоматически
            verbose: Выводить ли подробную информацию
        """
        self.model_name = model
        self.verbose = verbose
        
        # Определяем устройство
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if self.verbose:
            print(f"Используемое устройство: {self.device}")
            print(f"Загрузка модели Whisper: {model}...")
        
        # Загружаем модель
        try:
            self.model = whisper.load_model(model, device=self.device)
            if self.verbose:
                print(f"Модель {model} загружена успешно")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                error_msg = (
                    f"ОШИБКА: Не хватает памяти для модели {model}\n"
                    "Попробуйте использовать модель 'small' или 'base'\n"
                    "Или закройте другие приложения, использующие GPU"
                )
                if self.verbose:
                    print(error_msg)
                raise RuntimeError(error_msg) from e
            else:
                raise
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "ru",
        word_timestamps: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Транскрибирует аудио файл в текст.
        
        Args:
            audio_path: Путь к аудио файлу
            language: Язык аудио (по умолчанию "ru")
            word_timestamps: Включать ли временные метки слов
            **kwargs: Дополнительные параметры для whisper.transcribe()
            
        Returns:
            Dict с ключами:
                - text: Полный текст транскрипции
                - segments: Список сегментов с временными метками
                - language: Определенный язык
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
        
        # Параметры по умолчанию для лучшего качества
        default_params = {
            "task": "transcribe",
            "temperature": 0.0,
            "beam_size": 5,
            "best_of": 5,
            "patience": 1.0,
            "condition_on_previous_text": True,
            "initial_prompt": "Это разговор на русском языке. ",
            "verbose": self.verbose
        }
        
        # Объединяем параметры (kwargs имеют приоритет)
        params = {**default_params, **kwargs}
        params["language"] = language
        params["word_timestamps"] = word_timestamps
        
        if self.verbose:
            print(f"Начинаю транскрипцию: {audio_path}")
        
        result = self.model.transcribe(str(audio_path), **params)
        
        if self.verbose:
            print(f"Транскрипция завершена. Длина текста: {len(result['text'])} символов")
        
        return result
    
    def get_segments(self, audio_path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Получить сегменты транскрипции с временными метками.
        
        Args:
            audio_path: Путь к аудио файлу
            **kwargs: Дополнительные параметры для transcribe()
            
        Returns:
            List[Dict]: Список сегментов с ключами: start, end, text
        """
        result = self.transcribe(audio_path, **kwargs)
        return result.get("segments", [])
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Транскрибирует несколько аудио файлов.
        
        Args:
            audio_paths: Список путей к аудио файлам
            **kwargs: Дополнительные параметры для transcribe()
            
        Returns:
            List[Dict]: Список результатов транскрипции
        """
        results = []
        total = len(audio_paths)
        
        for i, audio_path in enumerate(audio_paths, 1):
            if self.verbose:
                print(f"Обработка {i}/{total}: {audio_path}")
            
            try:
                result = self.transcribe(audio_path, **kwargs)
                results.append({
                    "audio_path": audio_path,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                if self.verbose:
                    print(f"Ошибка при обработке {audio_path}: {e}")
                results.append({
                    "audio_path": audio_path,
                    "success": False,
                    "error": str(e)
                })
        
        return results
