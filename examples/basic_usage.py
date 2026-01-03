"""Пример базового использования file2text."""

from file2text import File2Text

# Инициализация
processor = File2Text(verbose=True)

# Полный пайплайн обработки
result = processor.process(
    audio_path="audio.mp3",
    transcribe=True,
    diarize=True,
    summarize=True,
    vectorize=True
)

# Доступ к результатам
print(f"Текст: {result.text}")
print(f"Спикеры: {result.speakers}")
print(f"Суммаризация: {result.summary}")
print(f"Векторы: {result.vectors.shape if result.vectors is not None else None}")

# Только транскрипция
text = processor.transcribe("audio.mp3")

# Только суммаризация
summary = processor.summarize(text)

# Только векторизация
vectors = processor.vectorize(text)
