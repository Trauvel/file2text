# file2text

Универсальная система для конвертации аудио в текст, суммаризации и векторизации.

## 🚀 Возможности

- 🎤 **Транскрипция** - Конвертация аудио/видео в текст с помощью Whisper
- 👥 **Диаризация** - Разделение речи по спикерам
- 📝 **Суммаризация** - Суммаризация текста с удалением артефактов
- 🔢 **Векторизация** - Создание векторных представлений текста
- 🎯 **Модульная архитектура** - Использование как библиотека в других проектах

## 📦 Установка

```bash
# Клонируйте репозиторий
git clone https://github.com/yourusername/file2text.git
cd file2text

# Установите зависимости
pip install -e .

# Или с дополнительными возможностями
pip install -e ".[cli,all]"
```

## ⚙️ Настройка

Создайте файл `.env` в корне проекта:

```env
HUGGINGFACE_TOKEN=ваш_токен_здесь
WHISPER_MODEL=medium
SUMMARIZER_MODEL=IlyaGusev/rut5_base_sum_gazeta
VECTORIZER_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Получите токен Hugging Face на: https://huggingface.co/settings/tokens

## 💻 Использование

### Как библиотека

```python
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
# Полный пайплайн обработки
file2text process audio.mp3 --diarize --summarize --vectorize

# Только транскрипция
file2text transcribe audio.mp3 -o output.txt

# Только суммаризация
file2text summarize text.txt -o summary.txt

# Только векторизация
file2text vectorize text.txt -o vectors.npy
```

## 📁 Структура проекта

```
file2text/
├── file2text/          # Основной пакет
│   ├── core/           # Основные модули
│   ├── utils/          # Утилиты
│   └── cli/            # CLI интерфейс
├── examples/           # Примеры использования
│   └── legacy/         # Старые скрипты
├── docs/               # Документация
├── setup.py            # Установка пакета
└── README.md           # Этот файл
```

## 📚 Документация

Подробная документация находится в папке `docs/`:
- `ARCHITECTURE.md` - Архитектура системы
- `DEVELOPMENT_PLAN.md` - План разработки
- `MODULE_README.md` - Документация по модулям

## 🔧 Требования

- Python 3.8+
- CUDA (опционально, для GPU ускорения)
- FFmpeg (для конвертации аудио)

## 📝 Лицензия

MIT License

## 🤝 Вклад

Приветствуются pull requests и issues!

## 📧 Контакты

Ваш email или GitHub профиль
