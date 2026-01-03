"""Модуль для суммаризации текста."""

import re
import torch
from transformers import pipeline
from typing import Dict, Optional
from file2text.utils.text_cleaner import clean_text, postprocess_summary


def _split_text_into_chunks(text: str, max_length: int = 1000, overlap: int = 200) -> list:
    """Разбивает текст на чанки с перекрытием для обработки длинных текстов."""
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
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
            overlap_sentences = current_chunk[-3:] if len(current_chunk) >= 3 else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunk_text = '. '.join(current_chunk) + '.'
        chunks.append(chunk_text)
    
    return chunks


class Summarizer:
    """Класс для суммаризации текста."""
    
    def __init__(
        self,
        model: str = "IlyaGusev/rut5_base_sum_gazeta",
        device: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Инициализация суммаризатора.
        
        Args:
            model: Модель для суммаризации
            device: Устройство для обработки. Если None, определяется автоматически
            verbose: Выводить ли подробную информацию
        """
        self.model_name = model
        self.verbose = verbose
        
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == "cuda" else -1
        
        if self.verbose:
            print("Загрузка модели суммаризации...")
        
        self.summarizer = pipeline(
            'summarization',
            model=model,
            device=self.device,
            tokenizer=model
        )
        
        if self.verbose:
            print("Модель суммаризации загружена")
    
    def summarize(
        self,
        text: str,
        max_length: int = 250,
        min_length: int = 50
    ) -> str:
        """
        Суммаризирует текст.
        
        Args:
            text: Исходный текст
            max_length: Максимальная длина суммаризации
            min_length: Минимальная длина суммаризации
            
        Returns:
            str: Суммаризированный текст
        """
        if not text or len(text.strip()) < 50:
            return text
        
        # Очистка текста от артефактов перед суммаризацией
        text = clean_text(text)
        
        if not text or len(text.strip()) < 50:
            return text
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Если текст короткий, суммаризируем напрямую
        if len(text) <= 1000:
            try:
                result = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )
                summary = result[0]['summary_text'].strip()
                return postprocess_summary(summary)
            except Exception as e:
                if self.verbose:
                    print(f"Ошибка при суммаризации короткого текста: {e}")
                return text
        
        # Для длинных текстов разбиваем на чанки
        chunks = _split_text_into_chunks(text, max_length=1000, overlap=200)
        summaries = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
                
            try:
                result = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )
                chunk_summary = result[0]['summary_text'].strip()
                chunk_summary = postprocess_summary(chunk_summary)
                summaries.append(chunk_summary)
                if self.verbose:
                    print(f"  Обработан чанк {i+1}/{len(chunks)}")
            except Exception as e:
                if self.verbose:
                    print(f"Ошибка при суммаризации чанка {i+1}: {e}")
                summaries.append(chunk[:200] + "...")
        
        combined_summary = ' '.join(summaries)
        combined_summary = postprocess_summary(combined_summary)
        
        # Если объединенная суммаризация все еще длинная, суммаризируем еще раз
        if len(combined_summary) > 1500:
            try:
                final_result = self.summarizer(
                    combined_summary,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )
                final_summary = final_result[0]['summary_text'].strip()
                return postprocess_summary(final_summary)
            except Exception as e:
                if self.verbose:
                    print(f"Ошибка при финальной суммаризации: {e}")
                return combined_summary
        
        return combined_summary
    
    def summarize_by_speakers(
        self,
        speakers_text: Dict[str, str],
        max_length: int = 200,
        min_length: int = 50
    ) -> Dict[str, str]:
        """
        Суммаризирует текст по каждому спикеру отдельно.
        
        Args:
            speakers_text: Словарь {speaker: text}
            max_length: Максимальная длина суммаризации
            min_length: Минимальная длина суммаризации
            
        Returns:
            Dict[str, str]: Словарь {speaker: summary}
        """
        summaries = {}
        
        for speaker, text in speakers_text.items():
            if len(text.strip()) < 50:
                continue
            
            if self.verbose:
                print(f"  Суммаризирую текст спикера {speaker}...")
            
            summary = self.summarize(text, max_length=max_length, min_length=min_length)
            summaries[speaker] = summary
        
        return summaries
    
    def summarize_full(
        self,
        text: str,
        max_length: int = 300,
        min_length: int = 100
    ) -> str:
        """
        Суммаризирует весь текст разговора.
        
        Args:
            text: Полный текст разговора
            max_length: Максимальная длина суммаризации
            min_length: Минимальная длина суммаризации
            
        Returns:
            str: Суммаризированный текст
        """
        return self.summarize(text, max_length=max_length, min_length=min_length)
