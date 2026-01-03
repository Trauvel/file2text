# Legacy скрипты

Эта папка содержит старые скрипты для обратной совместимости.

## Файлы

- `main.py` - Старый скрипт для транскрипции и диаризации
- `summarize.py` - Старый скрипт для суммаризации
- `logicCombrain.py` - Скрипт для кластеризации по темам (BERTopic)

## Использование

Эти скрипты сохранены для обратной совместимости. Рекомендуется использовать новый модульный API:

```python
from file2text import File2Text

processor = File2Text()
result = processor.process("audio.mp3", 
    transcribe=True,
    diarize=True,
    summarize=True
)
```

Или через CLI:

```bash
file2text process audio.mp3 --diarize --summarize
```
