"""Утилиты для очистки текста от артефактов."""

import re


def clean_text(text: str) -> str:
    """
    Очищает текст от артефактов, повторений и незначимых реплик.
    
    Args:
        text: Исходный текст
        
    Returns:
        str: Очищенный текст
    """
    if not text:
        return text
    
    # Удаляем множественные пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Удаляем множественные повторения междометий (более 2 подряд)
    text = re.sub(
        r'\b(Ага|Оке|Окей|Да|Нет|Угу|М-м|Хм)\s*[.!?,]?\s*(\1\s*[.!?,]?\s*){2,}',
        r'\1. ',
        text,
        flags=re.IGNORECASE
    )
    
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
    for length in range(80, 9, -10):
        pattern = rf'([^.!?]{{{length//2},{length}}})(\s*[,.]\s*\1){{2,}}'
        text = re.sub(pattern, r'\1', text)
    
    # Удаляем повторяющиеся части предложений
    for length in range(50, 14, -5):
        pattern = rf'([^.!?]{{{length//2},{length}}})(\s*,\s*\1)+'
        text = re.sub(pattern, r'\1', text)
    
    # Финальная очистка пробелов и пунктуации
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s*([.!?])\s*\1+', r'\1', text)  # Убираем множественные знаки препинания
    
    return text


def postprocess_summary(summary: str) -> str:
    """
    Постобработка суммаризации для удаления артефактов модели.
    
    Args:
        summary: Текст суммаризации
        
    Returns:
        str: Очищенная суммаризация
    """
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
