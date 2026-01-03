"""
file2text - Универсальная система для конвертации аудио в текст,
суммаризации и векторизации текста.
"""

__version__ = "1.0.0"

try:
    from file2text.core.file2text import File2Text
    from file2text.core.transcriber import Transcriber
    from file2text.core.diarizer import Diarizer
    from file2text.core.summarizer import Summarizer
    from file2text.core.vectorizer import Vectorizer
except ImportError:
    # Для случаев когда зависимости еще не установлены
    pass

__all__ = [
    "File2Text",
    "Transcriber",
    "Diarizer",
    "Summarizer",
    "Vectorizer",
]
