import os
import whisper
from pyannote.audio import Pipeline
import subprocess
import torch

# Папки
FILES_DIR = 'files'
TEXT_DIR = 'text'

# Создаём папку для хранения текстов, если она не существует
os.makedirs(TEXT_DIR, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Выбор модели Whisper
# Доступные модели и их требования к VRAM:
# - tiny: ~1GB VRAM, быстро, низкое качество
# - base: ~1GB VRAM, быстро, среднее качество
# - small: ~2GB VRAM, хорошо, хороший баланс скорости/качества
# - medium: ~5GB VRAM, медленно, высокое качество (РЕКОМЕНДУЕТСЯ для RTX 3060 Ti 8GB)
# - large-v2: ~10GB VRAM, очень медленно, максимальное качество (может не хватить памяти)
# - large-v3: ~10GB VRAM, очень медленно, максимальное качество (может не хватить памяти)
WHISPER_MODEL = "medium"  # Измените на "small" если не хватает памяти, или "large-v2" для максимального качества

print(f"Загрузка модели Whisper: {WHISPER_MODEL}...")
try:
    model = whisper.load_model(WHISPER_MODEL, device=device)
    print(f"Модель {WHISPER_MODEL} загружена успешно")
except RuntimeError as e:
    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
        print(f"ОШИБКА: Не хватает памяти для модели {WHISPER_MODEL}")
        print("Попробуйте использовать модель 'small' или 'base'")
        print("Или закройте другие приложения, использующие GPU")
        raise
    else:
        raise

# Ваш Hugging Face токен (получите на https://huggingface.co/settings/tokens)
# Установите переменную окружения: export HUGGINGFACE_TOKEN="ваш_токен"
# Или создайте файл .env с HUGGINGFACE_TOKEN=ваш_токен
YOUR_AUTH_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
if not YOUR_AUTH_TOKEN:
    print("ВНИМАНИЕ: Hugging Face токен не найден!")
    print("Установите переменную окружения HUGGINGFACE_TOKEN или создайте файл .env")
    print("Получите токен на: https://huggingface.co/settings/tokens")
    raise ValueError("HUGGINGFACE_TOKEN не установлен")

# Инициализация пайплайна для диаризации с токеном
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=YOUR_AUTH_TOKEN)

# Функция для сопоставления спикеров с транскрипцией
def assign_speakers(transcript_segments, diarization_segments):
    speaker_transcript = []
    for segment in transcript_segments:
        start = segment['start']
        end = segment['end']
        text = segment['text']

        # Найти спикера для данного сегмента
        speaker = None
        for turn, _, spk in diarization_segments.itertracks(yield_label=True):
            if turn.start <= start <= turn.end or turn.start <= end <= turn.end or (start <= turn.start and end >= turn.end):
                speaker = spk
                break

        if speaker is None:
            speaker = "Unknown"

        speaker_transcript.append(f"Спикер {speaker}: {text.strip()}")

    return speaker_transcript

# Проход по файлам в папке
for filename in os.listdir(FILES_DIR):
    if filename.lower().endswith(('.mp3', '.mp4', '.wav')):
        file_path = os.path.abspath(os.path.join(FILES_DIR, filename))
        print(f"Обрабатываем файл: {file_path}")

        if not os.path.exists(file_path):
            print(f"Файл {file_path} не существует!")
            continue

        # Конвертация mp3 или mp4 в wav с оптимальными параметрами для Whisper
        if filename.lower().endswith(('.mp3', '.mp4')):
            wav_file_path = os.path.splitext(file_path)[0] + '.wav'
            print(f"Конвертирую {filename} в WAV...")
            # Оптимальные параметры для Whisper: моно, 16kHz, 16-bit PCM
            subprocess.run([
                'ffmpeg', '-i', file_path,
                '-ar', '16000',  # Частота дискретизации 16kHz (оптимально для Whisper)
                '-ac', '1',  # Моно канал
                '-sample_fmt', 's16',  # 16-bit PCM
                '-y',  # Перезаписать если файл существует
                wav_file_path
            ], check=True, capture_output=True)
            file_path = wav_file_path
            print("Конвертация завершена")

        # Шаг 1: Полная транскрипция всего аудио файла с улучшенными параметрами
        print("Начинаю транскрипцию с улучшенными параметрами...")
        result = model.transcribe(
            file_path,
            language='ru',
            word_timestamps=True,  # Включаем временные метки слов для лучшей точности
            task='transcribe',  # Явно указываем задачу транскрипции
            temperature=0.0,  # Детерминированный режим для стабильности
            beam_size=5,  # Увеличиваем beam size для лучшего качества
            best_of=5,  # Пробуем несколько вариантов и выбираем лучший
            patience=1.0,  # Терпение для beam search
            condition_on_previous_text=True,  # Используем контекст предыдущего текста
            initial_prompt="Это разговор на русском языке. ",  # Подсказка для модели
            verbose=False  # Убираем лишний вывод
        )
        full_text = result['text']
        segments = result['segments']
        print(f"Транскрипция завершена. Длина текста: {len(full_text)} символов")

        # Сохранение полного текста без разделения по спикерам
        full_text_filename = f"{os.path.splitext(filename)[0]}_full.txt"
        full_text_file_path = os.path.join(TEXT_DIR, full_text_filename)
        with open(full_text_file_path, "w", encoding="utf-8") as full_text_file:
            full_text_file.write(full_text)
        print(f"Полный текст сохранён в {full_text_file_path}")

        # Шаг 2: Диаризация (выделение спикеров)
        diarization = diarization_pipeline(file_path)

        # Шаг 3: Сопоставление спикеров с транскрипцией
        speaker_transcript = assign_speakers(segments, diarization)

        # Объединение всех сегментов в один текст
        speaker_text = "\n".join(speaker_transcript)

        # Шаг 4: Сохранение расшифрованного текста со спикерами
        text_filename = f"{os.path.splitext(filename)[0]}.txt"
        text_file_path = os.path.join(TEXT_DIR, text_filename)
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(speaker_text)
        print(f"Текст со спикерами сохранён в {text_file_path}")
        print(f"Для суммаризации запустите: python summarize.py")