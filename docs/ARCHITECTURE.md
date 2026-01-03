# Архитектура модульной системы file2text

## 🏗️ Общая архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    file2text (Main Package)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Transcriber  │  │  Diarizer    │  │ Summarizer   │     │
│  │  (Whisper)   │→ │ (pyannote)   │→ │ (RUT5)       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           │                                │
│                  ┌────────▼────────┐                       │
│                  │   Vectorizer    │                       │
│                  │ (sentence-      │                       │
│                  │  transformers)  │                       │
│                  └─────────────────┘                       │
│                                                             │
│  ┌──────────────────────────────────────────────┐         │
│  │         File2Text (Unified API)               │         │
│  │  - process() - полный пайплайн                │         │
│  │  - transcribe() - только транскрипция         │         │
│  │  - summarize() - только суммаризация          │         │
│  │  - vectorize() - только векторизация          │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Использование                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Как библиотека:                                         │
│     from file2text import File2Text                         │
│     processor = File2Text()                                 │
│     result = processor.process("audio.mp3")                │
│                                                              │
│  2. Через CLI:                                              │
│     file2text audio.mp3 --summarize --vectorize              │
│                                                              │
│  3. В других проектах:                                      │
│     import file2text                                        │
│     text = file2text.transcribe("audio.mp3")                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Структура модулей

### core/transcriber.py
```python
class Transcriber:
    def __init__(self, model="medium", device="cuda"):
        self.model = whisper.load_model(model, device=device)
    
    def transcribe(self, audio_path, **kwargs):
        """Транскрипция аудио в текст"""
        return self.model.transcribe(audio_path, **kwargs)
    
    def get_segments(self, audio_path):
        """Получить сегменты с временными метками"""
        result = self.transcribe(audio_path)
        return result['segments']
```

### core/diarizer.py
```python
class Diarizer:
    def __init__(self, auth_token=None):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=auth_token
        )
    
    def diarize(self, audio_path):
        """Диаризация спикеров"""
        return self.pipeline(audio_path)
    
    def assign_speakers(self, segments, diarization):
        """Сопоставить спикеров с сегментами"""
        # Логика сопоставления
```

### core/summarizer.py
```python
class Summarizer:
    def __init__(self, model="IlyaGusev/rut5_base_sum_gazeta"):
        self.pipeline = pipeline('summarization', model=model)
    
    def summarize(self, text, **kwargs):
        """Суммаризация текста"""
        # Логика суммаризации с очисткой
    
    def summarize_by_speakers(self, speaker_text):
        """Суммаризация по спикерам"""
```

### core/vectorizer.py (новый)
```python
class Vectorizer:
    def __init__(self, model="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model)
    
    def vectorize(self, text):
        """Векторизация текста"""
        return self.model.encode(text)
    
    def similarity(self, text1, text2):
        """Вычисление схожести"""
        vec1 = self.vectorize(text1)
        vec2 = self.vectorize(text2)
        return cosine_similarity([vec1], [vec2])[0][0]
    
    def search(self, query, texts, top_k=5):
        """Поиск похожих текстов"""
        query_vec = self.vectorize(query)
        text_vecs = self.vectorize_batch(texts)
        similarities = cosine_similarity([query_vec], text_vecs)[0]
        indices = similarities.argsort()[-top_k:][::-1]
        return [(texts[i], similarities[i]) for i in indices]
```

### Главный класс File2Text
```python
class File2Text:
    def __init__(self, config=None):
        self.transcriber = Transcriber(config.whisper_model)
        self.diarizer = Diarizer(config.hf_token)
        self.summarizer = Summarizer(config.summarizer_model)
        self.vectorizer = Vectorizer(config.vectorizer_model)
    
    def process(self, audio_path, **options):
        """Полный пайплайн обработки"""
        result = ProcessingResult()
        
        if options.get('transcribe', True):
            result.text = self.transcriber.transcribe(audio_path)
        
        if options.get('diarize', False):
            result.speakers = self.diarizer.assign_speakers(...)
        
        if options.get('summarize', False):
            result.summary = self.summarizer.summarize(result.text)
        
        if options.get('vectorize', False):
            result.vectors = self.vectorizer.vectorize(result.text)
        
        return result
```

## 🔄 Поток данных

```
Аудио файл
    │
    ▼
[Audio Converter] → WAV (16kHz, mono)
    │
    ▼
[Transcriber] → Текст + сегменты
    │
    ├─→ Полный текст
    │
    └─→ [Diarizer] → Текст по спикерам
            │
            ▼
        [Summarizer] → Суммаризация
            │
            ▼
        [Vectorizer] → Векторы (embeddings)
            │
            ▼
        Результат (JSON/объект)
```

## 💾 Формат результатов

```python
@dataclass
class ProcessingResult:
    audio_path: str
    text: str = None
    segments: List[Dict] = None
    speakers: Dict[str, str] = None
    summary: Dict[str, str] = None  # full, by_speakers, combined
    vectors: np.ndarray = None
    metadata: Dict = None
```

## 🔌 Интерфейсы использования

### 1. Простой импорт
```python
import file2text

processor = file2text.File2Text()
result = processor.process("audio.mp3")
```

### 2. Поэтапная обработка
```python
from file2text import Transcriber, Summarizer, Vectorizer

transcriber = Transcriber()
text = transcriber.transcribe("audio.mp3")

summarizer = Summarizer()
summary = summarizer.summarize(text)

vectorizer = Vectorizer()
vectors = vectorizer.vectorize(text)
```

### 3. CLI
```bash
file2text audio.mp3 --summarize --vectorize
```

### 4. Интеграция в другие проекты
```python
from file2text.core import Transcriber

class MyApp:
    def __init__(self):
        self.transcriber = Transcriber()
    
    def process_audio(self, path):
        return self.transcriber.transcribe(path)
```
