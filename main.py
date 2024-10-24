import os
import whisper
from pyannote.audio import Pipeline
import torchaudio
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import subprocess

# Папки
FILES_DIR = 'files'
TEXT_DIR = 'text'
SUM_TEXT_DIR = 'sumText'

# Создаём папки для хранения, если они не существуют
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(SUM_TEXT_DIR, exist_ok=True)

# Инициализация модели Whisper
model = whisper.load_model("base")

# Ваш Hugging Face токен
YOUR_AUTH_TOKEN = "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"

# Инициализация пайплайна для диаризации с токеном
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=YOUR_AUTH_TOKEN)

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

# Функция для создания выжимки текста
def summarize_text(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("russian"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join([str(sentence) for sentence in summary])

# Проход по файлам в папке
for filename in os.listdir(FILES_DIR):
    if filename.lower().endswith(('.mp3', '.mp4', '.wav')):
        file_path = os.path.abspath(os.path.join(FILES_DIR, filename))
        print(f"Обрабатываем файл: {file_path}")

        if not os.path.exists(file_path):
            print(f"Файл {file_path} не существует!")
            continue

        # Конвертация mp3 или mp4 в wav
        if filename.lower().endswith(('.mp3', '.mp4')):
            wav_file_path = os.path.splitext(file_path)[0] + '.wav'
            subprocess.run(['ffmpeg', '-i', file_path, wav_file_path])
            file_path = wav_file_path

        # Шаг 1: Полная транскрипция всего аудио файла с временными метками
        result = model.transcribe(file_path, language='ru', word_timestamps=False)
        full_text = result['text']
        segments = result['segments']

        # Сохранение полного текста без разделения по спикерам
        full_text_filename = f"{os.path.splitext(filename)[0]}_full.txt"
        full_text_file_path = os.path.join(TEXT_DIR, full_text_filename)
        with open(full_text_file_path, "w", encoding="utf-8") as full_text_file:
            full_text_file.write(full_text)
        print(f"Полный текст сохранён в {full_text_file_path}")

        # Шаг 2: Диаризация (выделение спикеров)
        diarization = pipeline(file_path)

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

        # Шаг 5: Создание выжимки из полного текста
        summary_text = summarize_text(full_text)

        # Шаг 6: Сохранение выжимки
        summary_file_path = os.path.join(SUM_TEXT_DIR, text_filename)
        with open(summary_file_path, "w", encoding="utf-8") as summary_file:
            summary_file.write(summary_text)
        print(f"Суммаризированный текст сохранён в {summary_file_path}")
