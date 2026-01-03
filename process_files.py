"""Скрипт для автоматической обработки файлов из папки files/."""

import os
import argparse
from pathlib import Path
from file2text import File2Text
from file2text.utils.config import load_config

# Папки
FILES_DIR = 'files'
TEXT_DIR = 'text'

# Создаём папку для текстов, если она не существует
os.makedirs(TEXT_DIR, exist_ok=True)


def process_audio_files(summarize: bool = False, vectorize: bool = False, model: str = None):
    """
    Обрабатывает все аудио файлы из папки files/.
    
    Args:
        summarize: Включить ли суммаризацию
        vectorize: Включить ли векторизацию
        model: Модель Whisper (если None, используется из конфигурации)
    """
    
    # Загружаем конфигурацию
    try:
        config = load_config()
    except ValueError as e:
        print(f"Ошибка конфигурации: {e}")
        print("Убедитесь, что установлена переменная окружения HUGGINGFACE_TOKEN")
        return
    
    # Инициализируем процессор
    processor = File2Text(
        config=config,
        whisper_model=model or config.whisper_model,
        verbose=True
    )
    
    # Получаем список файлов
    files_path = Path(FILES_DIR)
    if not files_path.exists():
        print(f"Папка {FILES_DIR} не существует!")
        return
    
    # Находим все медиа файлы (аудио и видео)
    from file2text.utils.audio_converter import AudioConverter
    
    media_files = [
        f for f in files_path.iterdir()
        if f.is_file() and AudioConverter.is_media_file(str(f))
    ]
    
    if not media_files:
        print(f"В папке {FILES_DIR} нет аудио или видео файлов для обработки.")
        return
    
    print(f"Найдено файлов для обработки: {len(media_files)}")
    print(f"  (аудио и видео файлы будут автоматически обработаны)\n")
    
    # Обрабатываем каждый файл
    for audio_file in media_files:
        try:
            print(f"\n{'='*60}")
            print(f"Обрабатываю файл: {audio_file.name}")
            print(f"{'='*60}")
            
            # Обрабатываем файл (транскрипция + диаризация)
            result = processor.process(
                audio_path=str(audio_file),
                transcribe=True,
                diarize=True,
                summarize=summarize,
                vectorize=vectorize
            )
            
            # Сохраняем полный текст
            base_name = audio_file.stem
            full_text_filename = f"{base_name}_full.txt"
            full_text_path = Path(TEXT_DIR) / full_text_filename
            
            with open(full_text_path, "w", encoding="utf-8") as f:
                f.write(result.text)
            print(f"✓ Полный текст сохранён: {full_text_path}")
            
            # Сохраняем текст со спикерами
            if result.speakers:
                text_filename = f"{base_name}.txt"
                text_path = Path(TEXT_DIR) / text_filename
                
                # Форматируем текст со спикерами
                speaker_lines = []
                for segment in result.speaker_segments:
                    speaker_lines.append(f"Спикер {segment['speaker']}: {segment['text']}")
                
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(speaker_lines))
                print(f"✓ Текст со спикерами сохранён: {text_path}")
            
            # Сохраняем суммаризацию если включена
            if summarize and result.summary:
                sum_text_dir = Path('sumText')
                sum_text_dir.mkdir(exist_ok=True)
                
                if 'full' in result.summary:
                    summary_full_path = sum_text_dir / f"{base_name}_summary_full.txt"
                    with open(summary_full_path, "w", encoding="utf-8") as f:
                        f.write("=== СУММАРИЗАЦИЯ ВСЕГО РАЗГОВОРА ===\n\n")
                        f.write(result.summary['full'])
                    print(f"✓ Суммаризация всего разговора сохранена: {summary_full_path}")
                
                if 'by_speakers' in result.summary:
                    summary_speakers_path = sum_text_dir / f"{base_name}_summary_speakers.txt"
                    with open(summary_speakers_path, "w", encoding="utf-8") as f:
                        f.write("=== СУММАРИЗАЦИЯ ПО СПИКЕРАМ ===\n\n")
                        for speaker, summary_text in result.summary['by_speakers'].items():
                            f.write(f"=== СПИКЕР {speaker} ===\n\n{summary_text}\n\n")
                    print(f"✓ Суммаризация по спикерам сохранена: {summary_speakers_path}")
            
            # Сохраняем векторы если включена векторизация
            if vectorize and result.vectors is not None:
                import numpy as np
                vectors_path = Path(TEXT_DIR) / f"{base_name}_vectors.npy"
                np.save(vectors_path, result.vectors)
                print(f"✓ Векторы сохранены: {vectors_path} (размерность: {result.vectors.shape})")
            
            # Удаляем обработанный файл
            try:
                audio_file.unlink()
                print(f"✓ Файл удалён: {audio_file.name}")
            except Exception as e:
                print(f"⚠ Ошибка при удалении файла {audio_file.name}: {e}")
            
            print(f"✓ Файл {audio_file.name} успешно обработан\n")
            
        except Exception as e:
            print(f"✗ Ошибка при обработке файла {audio_file.name}: {e}")
            print(f"  Файл не будет удалён из-за ошибки\n")
            continue
    
    print(f"\n{'='*60}")
    print("Обработка завершена!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обработка аудио файлов из папки files/ с использованием file2text"
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Включить суммаризацию текста"
    )
    parser.add_argument(
        "--vectorize",
        action="store_true",
        help="Включить векторизацию текста"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Модель Whisper (tiny, base, small, medium, large-v2, large-v3). "
             "По умолчанию используется из конфигурации."
    )
    
    args = parser.parse_args()
    
    process_audio_files(
        summarize=args.summarize,
        vectorize=args.vectorize,
        model=args.model
    )
