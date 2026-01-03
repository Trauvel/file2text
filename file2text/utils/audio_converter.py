"""Утилиты для конвертации аудио и видео файлов."""

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple


class AudioConverter:
    """Класс для конвертации аудио и видео файлов в формат, подходящий для Whisper."""
    
    @staticmethod
    def convert_to_wav(
        input_path: str,
        output_path: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_format: str = "s16"
    ) -> str:
        """
        Конвертирует аудио файл в WAV формат с оптимальными параметрами для Whisper.
        
        Args:
            input_path: Путь к входному файлу
            output_path: Путь к выходному файлу (если None, создается рядом с входным)
            sample_rate: Частота дискретизации (по умолчанию 16kHz для Whisper)
            channels: Количество каналов (1 = моно)
            sample_format: Формат сэмпла (s16 = 16-bit PCM)
            
        Returns:
            str: Путь к сконвертированному файлу
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_path}")
        
        # Определяем выходной путь
        if output_path is None:
            output_path = input_path.with_suffix('.wav')
        else:
            output_path = Path(output_path)
        
        # Проверяем, нужна ли конвертация
        if input_path.suffix.lower() == '.wav':
            # Можно проверить параметры существующего WAV файла
            # Пока просто возвращаем исходный путь
            return str(input_path)
        
        # Конвертируем через ffmpeg
        try:
            subprocess.run(
                [
                    'ffmpeg',
                    '-i', str(input_path),
                    '-ar', str(sample_rate),
                    '-ac', str(channels),
                    '-sample_fmt', sample_format,
                    '-y',  # Перезаписать если файл существует
                    str(output_path)
                ],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ошибка конвертации аудио: {e.stderr.decode()}")
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg не найден. Установите ffmpeg и добавьте его в PATH."
            )
        
        return str(output_path)
    
    @staticmethod
    def extract_audio_from_video(
        video_path: str,
        output_path: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_format: str = "s16"
    ) -> str:
        """
        Извлекает аудио из видео файла и конвертирует в WAV формат.
        
        Args:
            video_path: Путь к видео файлу
            output_path: Путь к выходному аудио файлу (если None, создается рядом с видео)
            sample_rate: Частота дискретизации (по умолчанию 16kHz для Whisper)
            channels: Количество каналов (1 = моно)
            sample_format: Формат сэмпла (s16 = 16-bit PCM)
            
        Returns:
            str: Путь к извлеченному аудио файлу
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Видео файл не найден: {video_path}")
        
        # Определяем выходной путь
        if output_path is None:
            output_path = video_path.with_suffix('.wav')
        else:
            output_path = Path(output_path)
        
        # Извлекаем аудио из видео через ffmpeg
        try:
            subprocess.run(
                [
                    'ffmpeg',
                    '-i', str(video_path),
                    '-vn',  # Без видео
                    '-acodec', 'pcm_s16le',  # Кодек для WAV
                    '-ar', str(sample_rate),
                    '-ac', str(channels),
                    '-sample_fmt', sample_format,
                    '-y',  # Перезаписать если файл существует
                    str(output_path)
                ],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Ошибка извлечения аудио из видео: {error_msg}")
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg не найден. Установите ffmpeg и добавьте его в PATH."
            )
        
        return str(output_path)
    
    @staticmethod
    def convert_to_wav(
        input_path: str,
        output_path: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_format: str = "s16",
        auto_detect: bool = True
    ) -> str:
        """
        Конвертирует аудио или видео файл в WAV формат с оптимальными параметрами для Whisper.
        Автоматически определяет тип файла и использует соответствующий метод конвертации.
        
        Args:
            input_path: Путь к входному файлу (аудио или видео)
            output_path: Путь к выходному файлу (если None, создается рядом с входным)
            sample_rate: Частота дискретизации (по умолчанию 16kHz для Whisper)
            channels: Количество каналов (1 = моно)
            sample_format: Формат сэмпла (s16 = 16-bit PCM)
            auto_detect: Автоматически определять тип файла (видео/аудио)
            
        Returns:
            str: Путь к сконвертированному файлу
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_path}")
        
        # Определяем выходной путь
        if output_path is None:
            output_path = input_path.with_suffix('.wav')
        else:
            output_path = Path(output_path)
        
        # Если уже WAV, проверяем параметры
        if input_path.suffix.lower() == '.wav':
            # Можно добавить проверку параметров существующего WAV
            # Пока просто возвращаем исходный путь
            return str(input_path)
        
        # Автоматически определяем тип файла
        if auto_detect:
            if AudioConverter.is_video_file(str(input_path)):
                # Это видео - извлекаем аудио
                return AudioConverter.extract_audio_from_video(
                    str(input_path),
                    str(output_path),
                    sample_rate,
                    channels,
                    sample_format
                )
            elif AudioConverter.is_audio_file(str(input_path)):
                # Это аудио - конвертируем
                pass  # Продолжаем с обычной конвертацией
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {input_path.suffix}")
        
        # Конвертируем аудио через ffmpeg
        try:
            subprocess.run(
                [
                    'ffmpeg',
                    '-i', str(input_path),
                    '-ar', str(sample_rate),
                    '-ac', str(channels),
                    '-sample_fmt', sample_format,
                    '-y',  # Перезаписать если файл существует
                    str(output_path)
                ],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Ошибка конвертации аудио: {error_msg}")
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg не найден. Установите ffmpeg и добавьте его в PATH."
            )
        
        return str(output_path)
    
    @staticmethod
    def is_audio_file(file_path: str) -> bool:
        """
        Проверяет, является ли файл аудио файлом.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: True если файл является аудио файлом
        """
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.opus'}
        return Path(file_path).suffix.lower() in audio_extensions
    
    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """
        Проверяет, является ли файл видео файлом.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: True если файл является видео файлом
        """
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
        return Path(file_path).suffix.lower() in video_extensions
    
    @staticmethod
    def is_media_file(file_path: str) -> bool:
        """
        Проверяет, является ли файл медиа файлом (аудио или видео).
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: True если файл является медиа файлом
        """
        return AudioConverter.is_audio_file(file_path) or AudioConverter.is_video_file(file_path)