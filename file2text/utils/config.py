"""Конфигурация для file2text."""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не установлен, используем только переменные окружения


@dataclass
class Config:
    """Конфигурация для file2text."""
    
    # Whisper настройки
    whisper_model: str = "medium"
    whisper_device: str = "cuda"  # "cuda" или "cpu"
    
    # Диаризация
    huggingface_token: Optional[str] = None
    
    # Суммаризация
    summarizer_model: str = "IlyaGusev/rut5_base_sum_gazeta"
    summary_max_length: int = 250
    summary_min_length: int = 50
    
    # Векторизация
    vectorizer_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_dimension: int = 384
    
    # Пути
    default_output_dir: str = "./output"
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        """Инициализация после создания объекта."""
        # Загружаем токен из переменных окружения если не указан
        if not self.huggingface_token:
            self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Устанавливаем cache_dir по умолчанию
        if not self.cache_dir:
            cache_home = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            self.cache_dir = os.path.join(cache_home, "file2text")
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Проверяем наличие токена для диаризации
        if not self.huggingface_token:
            raise ValueError(
                "HUGGINGFACE_TOKEN не установлен. "
                "Установите переменную окружения HUGGINGFACE_TOKEN или "
                "создайте файл .env с HUGGINGFACE_TOKEN=ваш_токен"
            )


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Загрузить конфигурацию из файла или переменных окружения.
    
    Args:
        config_path: Путь к файлу конфигурации (YAML/JSON). 
                    Если None, используется только .env и переменные окружения.
    
    Returns:
        Config: Объект конфигурации
    """
    # Пока используем только переменные окружения
    # В будущем можно добавить поддержку YAML/JSON
    
    return Config(
        whisper_model=os.getenv("WHISPER_MODEL", "medium"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cuda"),
        summarizer_model=os.getenv("SUMMARIZER_MODEL", "IlyaGusev/rut5_base_sum_gazeta"),
        vectorizer_model=os.getenv("VECTORIZER_MODEL", 
                                   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    )
