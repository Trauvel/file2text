# Модульная система file2text

## ✅ Реализовано

Создана модульная структура проекта с возможностью использования как библиотеки.

### Структура пакета

```
file2text/
├── __init__.py              # Главный экспорт
├── core/
│   ├── __init__.py
│   ├── transcriber.py      # Транскрипция (Whisper)
│   ├── diarizer.py         # Диаризация спикеров
│   ├── summarizer.py       # Суммаризация
│   ├── vectorizer.py       # Векторизация (с поддержкой векторных БД)
│   └── file2text.py        # Главный класс File2Text
├── utils/
│   ├── __init__.py
│   ├── audio_converter.py  # Конвертация аудио
│   ├── text_cleaner.py     # Очистка текста
│   └── config.py           # Конфигурация
├── cli/
│   ├── __init__.py
│   └── main.py             # CLI интерфейс (typer)
└── examples/
    └── basic_usage.py      # Пример использования
```

## 📦 Установка

```bash
# Установка в режиме разработки
pip install -e .

# Или с дополнительными зависимостями
pip install -e ".[cli,all]"
```

## 🚀 Использование

### Как библиотека

```python
from file2text import File2Text

# Инициализация
processor = File2Text(verbose=True)

# Полный пайплайн
result = processor.process(
    audio_path="audio.mp3",
    transcribe=True,
    diarize=True,
    summarize=True,
    vectorize=True
)

# Доступ к результатам
print(result.text)              # Полный текст
print(result.speakers)          # Текст по спикерам
print(result.summary)           # Суммаризация
print(result.vectors)           # Векторы
```

### Поэтапная обработка

```python
# Только транскрипция
text = processor.transcribe("audio.mp3")

# Транскрипция + диаризация
result = processor.transcribe_with_speakers("audio.mp3")

# Только суммаризация
summary = processor.summarize(text)

# Только векторизация
vectors = processor.vectorize(text)
```

### Использование отдельных модулей

```python
from file2text import Transcriber, Summarizer, Vectorizer

transcriber = Transcriber(model="medium")
text = transcriber.transcribe("audio.mp3")

summarizer = Summarizer()
summary = summarizer.summarize(text)

vectorizer = Vectorizer()
vectors = vectorizer.vectorize(text)
```

## 🖥️ CLI интерфейс

```bash
# Полный пайплайн
file2text process audio.mp3 --diarize --summarize --vectorize

# Только транскрипция
file2text transcribe audio.mp3 -o output.txt

# Только суммаризация
file2text summarize text.txt -o summary.txt

# Только векторизация
file2text vectorize text.txt -o vectors.npy
```

## ⚙️ Конфигурация

Создайте файл `.env`:

```env
HUGGINGFACE_TOKEN=ваш_токен
WHISPER_MODEL=medium
SUMMARIZER_MODEL=IlyaGusev/rut5_base_sum_gazeta
VECTORIZER_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## 🔮 Будущие возможности

- [ ] Интеграция с векторными БД (FAISS, Qdrant, Chroma)
- [ ] REST API для веб-приложений
- [ ] Кэширование результатов
- [ ] Пакетная обработка через CLI
- [ ] Поддержка YAML конфигурации

## 📝 Примечания

- Старые скрипты (`main.py`, `summarize.py`) сохранены для обратной совместимости
- Все модули можно использовать независимо
- Векторизатор готов для интеграции с векторными БД (метод `prepare_for_vector_db`)
