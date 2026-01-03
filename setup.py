"""Setup файл для установки file2text как пакет."""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем README для длинного описания
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

setup(
    name="file2text",
    version="1.0.0",
    author="Your Name",
    description="Универсальная система для конвертации аудио в текст, суммаризации и векторизации",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/file2text",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai-whisper>=20231117",
        "pyannote.audio>=3.1.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "cli": [
            "typer>=0.9.0",
            "rich>=13.0.0",
        ],
        "vector-db": [
            # Для будущей интеграции с векторными БД
            # "faiss-cpu>=1.7.4",
            # "qdrant-client>=1.6.0",
        ],
        "all": [
            "typer>=0.9.0",
            "rich>=13.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "file2text=file2text.cli.main:app",
        ],
    },
)
