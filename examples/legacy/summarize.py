import os
import torch
from transformers import pipeline
import re

# Папки
TEXT_DIR = 'text'
SUM_TEXT_DIR = 'sumText'

# Создаём папку для суммаризаций, если она не существует
os.makedirs(SUM_TEXT_DIR, exist_ok=True)

# Инициализация модели суммаризации
print("Загрузка модели суммаризации...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

summarizer = pipeline(
    'summarization',
    model='IlyaGusev/rut5_base_sum_gazeta',
    device=0 if torch.cuda.is_available() else -1,
    tokenizer='IlyaGusev/rut5_base_sum_gazeta'
)
print("Модель суммаризации загружена\n")

# Функция для очистки текста от артефактов
def clean_text(text):
    """Очищает текст от артефактов, повторений и незначимых реплик"""
    if not text:
        return text
    
    # Удаляем множественные пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Удаляем множественные повторения междометий (более 2 подряд)
    # Паттерн для междометий с пунктуацией
    text = re.sub(r'\b(Ага|Оке|Окей|Да|Нет|Угу|М-м|Хм)\s*[.!?,]?\s*(\1\s*[.!?,]?\s*){2,}', r'\1. ', text, flags=re.IGNORECASE)
    
    # Удаляем повторяющиеся слова (более 2 подряд)
    text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1 \1', text, flags=re.IGNORECASE)
    
    # Удаляем повторяющиеся фразы в предложениях (типа "симплоро, симплоро, симплоро")
    text = re.sub(r'(\b\w+\b)(\s*,\s*\1){2,}', r'\1', text, flags=re.IGNORECASE)
    
    # Разбиваем на предложения для удаления дубликатов
    sentences = re.split(r'[.!?]\s+', text)
    seen_sentences = {}
    cleaned_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Пропускаем очень короткие предложения (менее 5 символов)
        if len(sentence) < 5:
            continue
        
        # Нормализуем предложение для сравнения (убираем пунктуацию, лишние пробелы)
        normalized = re.sub(r'[^\w\s]', '', sentence.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Пропускаем точные дубликаты
        if normalized in seen_sentences:
            continue
        
        seen_sentences[normalized] = True
        cleaned_sentences.append(sentence)
    
    if not cleaned_sentences:
        return text
    
    text = '. '.join(cleaned_sentences)
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    
    # Удаляем повторяющиеся подстроки в предложениях (фразы длиной 10-80 символов)
    # Ищем паттерны типа "фраза, фраза, фраза" или "фраза. фраза. фраза"
    for length in range(80, 9, -10):
        pattern = rf'([^.!?]{{{length//2},{length}}})(\s*[,.]\s*\1){{2,}}'
        text = re.sub(pattern, r'\1', text)
    
    # Удаляем повторяющиеся части предложений (типа "а не для того, чтобы X, а не для того, чтобы X")
    # Ищем повторяющиеся фрагменты длиной 15-50 символов
    for length in range(50, 14, -5):
        pattern = rf'([^.!?]{{{length//2},{length}}})(\s*,\s*\1)+'
        text = re.sub(pattern, r'\1', text)
    
    # Финальная очистка пробелов и пунктуации
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s*([.!?])\s*\1+', r'\1', text)  # Убираем множественные знаки препинания
    
    return text

# Функция для постобработки суммаризации (удаление артефактов из результата)
def postprocess_summary(summary):
    """Постобработка суммаризации для удаления артефактов модели"""
    if not summary:
        return summary
    
    # Удаляем повторяющиеся фразы
    summary = clean_text(summary)
    
    # Удаляем повторяющиеся предложения
    sentences = re.split(r'[.!?]\s+', summary)
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Нормализуем для сравнения
        normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
        
        # Пропускаем точные дубликаты
        if normalized in seen:
            continue
        
        seen.add(normalized)
        unique_sentences.append(sentence)
    
    result = '. '.join(unique_sentences)
    if result and not result.endswith(('.', '!', '?')):
        result += '.'
    
    # Удаляем повторяющиеся слова в конце (артефакты модели)
    result = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', result, flags=re.IGNORECASE)
    
    return result.strip()

# Функция для разбиения длинного текста на чанки с перекрытием
def split_text_into_chunks(text, max_length=1000, overlap=200):
    """Разбивает текст на чанки с перекрытием для обработки длинных текстов"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    sentences = re.split(r'[.!?]\s+', text)
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_length and current_chunk:
            # Сохраняем текущий чанк
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
            
            # Начинаем новый чанк с перекрытием (последние несколько предложений)
            overlap_sentences = current_chunk[-3:] if len(current_chunk) >= 3 else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Добавляем последний чанк
    if current_chunk:
        chunk_text = '. '.join(current_chunk) + '.'
        chunks.append(chunk_text)
    
    return chunks

# Функция для суммаризации текста (с поддержкой длинных текстов)
def summarize_text(text, max_length=250, min_length=50):
    """Суммаризирует текст, разбивая длинные тексты на чанки"""
    if not text or len(text.strip()) < 50:
        return text
    
    # Очистка текста от артефактов перед суммаризацией
    text = clean_text(text)
    
    if not text or len(text.strip()) < 50:
        return text
    
    # Очистка текста от лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Если текст короткий, суммаризируем напрямую
    if len(text) <= 1000:
        try:
            result = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            summary = result[0]['summary_text'].strip()
            # Постобработка для удаления артефактов
            return postprocess_summary(summary)
        except Exception as e:
            print(f"Ошибка при суммаризации короткого текста: {e}")
            return text
    
    # Для длинных текстов разбиваем на чанки
    chunks = split_text_into_chunks(text, max_length=1000, overlap=200)
    summaries = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 50:
            continue
            
        try:
            result = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            chunk_summary = result[0]['summary_text'].strip()
            # Постобработка каждого чанка
            chunk_summary = postprocess_summary(chunk_summary)
            summaries.append(chunk_summary)
            print(f"  Обработан чанк {i+1}/{len(chunks)}")
        except Exception as e:
            print(f"Ошибка при суммаризации чанка {i+1}: {e}")
            summaries.append(chunk[:200] + "...")  # Fallback
    
    # Объединяем все суммаризации и очищаем от дубликатов
    combined_summary = ' '.join(summaries)
    combined_summary = postprocess_summary(combined_summary)
    
    # Если объединенная суммаризация все еще длинная, суммаризируем еще раз
    if len(combined_summary) > 1500:
        try:
            final_result = summarizer(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            final_summary = final_result[0]['summary_text'].strip()
            # Постобработка финальной суммаризации
            return postprocess_summary(final_summary)
        except Exception as e:
            print(f"Ошибка при финальной суммаризации: {e}")
            return combined_summary
    
    return combined_summary

# Функция для парсинга текста со спикерами
def parse_speakers_text(text):
    """Парсит текст со спикерами и группирует по спикерам"""
    speakers_dict = {}
    
    # Разбиваем текст на строки
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Ищем паттерн "Спикер SPEAKER_XX: текст"
        match = re.match(r'Спикер\s+(\S+):\s+(.+)', line)
        if match:
            speaker = match.group(1)
            text_part = match.group(2)
            
            if speaker not in speakers_dict:
                speakers_dict[speaker] = []
            
            speakers_dict[speaker].append(text_part)
    
    # Объединяем тексты каждого спикера и очищаем от артефактов
    speakers_text = {}
    for speaker, texts in speakers_dict.items():
        combined_text = ' '.join(texts)
        # Очищаем текст перед сохранением
        cleaned_text = clean_text(combined_text)
        if cleaned_text and len(cleaned_text.strip()) >= 50:
            speakers_text[speaker] = cleaned_text
    
    return speakers_text

# Получаем список всех файлов для обработки
full_text_files = []
speaker_text_files = []

if not os.path.exists(TEXT_DIR):
    print(f"Папка {TEXT_DIR} не существует! Сначала запустите main.py для расшифровки.")
    exit(1)

for filename in os.listdir(TEXT_DIR):
    if filename.endswith('_full.txt'):
        full_text_files.append(filename)
    elif filename.endswith('.txt') and not filename.endswith('_full.txt'):
        speaker_text_files.append(filename)

if not full_text_files and not speaker_text_files:
    print(f"В папке {TEXT_DIR} нет файлов для суммаризации!")
    exit(1)

print(f"Найдено файлов для суммаризации:")
print(f"  - Полных текстов: {len(full_text_files)}")
print(f"  - Текстов со спикерами: {len(speaker_text_files)}\n")

# Обрабатываем каждый файл
processed_files = set()

# Обработка полных текстов (суммаризация всего разговора)
for filename in full_text_files:
    base_name = filename.replace('_full.txt', '')
    processed_files.add(base_name)
    
    print(f"Обрабатываю файл: {filename}")
    
    # Читаем полный текст
    full_text_path = os.path.join(TEXT_DIR, filename)
    with open(full_text_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    if not full_text.strip():
        print(f"  Файл {filename} пуст, пропускаю...\n")
        continue
    
    # Суммаризация всего разговора
    print("  Начинаю суммаризацию всего разговора...")
    full_summary = summarize_text(full_text, max_length=300, min_length=100)
    
    summary_full_filename = f"{base_name}_summary_full.txt"
    summary_full_path = os.path.join(SUM_TEXT_DIR, summary_full_filename)
    with open(summary_full_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("=== СУММАРИЗАЦИЯ ВСЕГО РАЗГОВОРА ===\n\n")
        summary_file.write(full_summary)
    print(f"  Суммаризация всего разговора сохранена в {summary_full_filename}\n")

# Обработка текстов со спикерами (суммаризация по спикерам)
for filename in speaker_text_files:
    base_name = filename.replace('.txt', '')
    
    print(f"Обрабатываю файл: {filename}")
    
    # Читаем текст со спикерами
    speaker_text_path = os.path.join(TEXT_DIR, filename)
    with open(speaker_text_path, "r", encoding="utf-8") as f:
        speaker_text = f.read()
    
    if not speaker_text.strip():
        print(f"  Файл {filename} пуст, пропускаю...\n")
        continue
    
    # Парсим текст по спикерам
    speakers_text = parse_speakers_text(speaker_text)
    
    if not speakers_text:
        print(f"  Не удалось распарсить спикеров из файла {filename}, пропускаю...\n")
        continue
    
    # Суммаризация по спикерам
    print("  Начинаю суммаризацию по спикерам...")
    summary_by_speakers = []
    
    for speaker, text in speakers_text.items():
        if len(text.strip()) < 50:  # Пропускаем слишком короткие тексты
            continue
        
        print(f"    Суммаризирую текст спикера {speaker}...")
        speaker_summary = summarize_text(text, max_length=200, min_length=50)
        summary_by_speakers.append(f"=== СПИКЕР {speaker} ===\n\n{speaker_summary}\n")
    
    if not summary_by_speakers:
        print(f"  Нет данных для суммаризации по спикерам в файле {filename}\n")
        continue
    
    summary_speakers_filename = f"{base_name}_summary_speakers.txt"
    summary_speakers_path = os.path.join(SUM_TEXT_DIR, summary_speakers_filename)
    with open(summary_speakers_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("=== СУММАРИЗАЦИЯ ПО СПИКЕРАМ ===\n\n")
        summary_file.write("\n".join(summary_by_speakers))
    print(f"  Суммаризация по спикерам сохранена в {summary_speakers_filename}\n")
    
    # Создаем комбинированную суммаризацию, если есть полный текст
    if base_name in processed_files:
        # Читаем полную суммаризацию, если она была создана
        summary_full_filename = f"{base_name}_summary_full.txt"
        summary_full_path = os.path.join(SUM_TEXT_DIR, summary_full_filename)
        
        if os.path.exists(summary_full_path):
            with open(summary_full_path, "r", encoding="utf-8") as f:
                full_summary_content = f.read()
                # Извлекаем только текст суммаризации (без заголовка)
                full_summary = full_summary_content.split("===")[-1].strip()
            
            combined_summary_filename = f"{base_name}_summary_combined.txt"
            combined_summary_path = os.path.join(SUM_TEXT_DIR, combined_summary_filename)
            with open(combined_summary_path, "w", encoding="utf-8") as combined_file:
                combined_file.write("=== ОБЩАЯ СУММАРИЗАЦИЯ ===\n\n")
                combined_file.write(full_summary)
                combined_file.write("\n\n" + "="*50 + "\n\n")
                combined_file.write("=== СУММАРИЗАЦИЯ ПО СПИКЕРАМ ===\n\n")
                combined_file.write("\n".join(summary_by_speakers))
            print(f"  Комбинированная суммаризация сохранена в {combined_summary_filename}\n")

print("Суммаризация завершена!")

