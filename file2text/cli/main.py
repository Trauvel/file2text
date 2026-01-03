"""CLI интерфейс для file2text."""

import typer
from pathlib import Path
from typing import Optional
import json

from file2text import File2Text
from file2text.utils.config import load_config

app = typer.Typer(help="file2text - Конвертация аудио в текст, суммаризация и векторизация")


@app.command()
def process(
    audio_path: str = typer.Argument(..., help="Путь к аудио файлу"),
    transcribe: bool = typer.Option(True, "--transcribe/--no-transcribe", help="Выполнить транскрипцию"),
    diarize: bool = typer.Option(False, "--diarize/--no-diarize", help="Выполнить диаризацию спикеров"),
    summarize: bool = typer.Option(False, "--summarize/--no-summarize", help="Выполнить суммаризацию"),
    vectorize: bool = typer.Option(False, "--vectorize/--no-vectorize", help="Выполнить векторизацию"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Путь для сохранения результатов (JSON)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Модель Whisper (tiny, base, small, medium, large-v2)"),
):
    """Полный пайплайн обработки аудио файла."""
    try:
        config = load_config()
        processor = File2Text(config=config, whisper_model=model or config.whisper_model, verbose=True)
        
        typer.echo(f"Обработка файла: {audio_path}")
        result = processor.process(
            audio_path=audio_path,
            transcribe=transcribe,
            diarize=diarize,
            summarize=summarize,
            vectorize=vectorize
        )
        
        # Выводим результаты
        if result.text:
            typer.echo(f"\nТранскрипция ({len(result.text)} символов):")
            typer.echo(result.text[:200] + "..." if len(result.text) > 200 else result.text)
        
        if result.speakers:
            typer.echo(f"\nСпикеры: {list(result.speakers.keys())}")
        
        if result.summary:
            if 'full' in result.summary:
                typer.echo(f"\nСуммаризация:")
                typer.echo(result.summary['full'])
        
        # Сохраняем в файл если указан
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            typer.echo(f"\nРезультаты сохранены в: {output}")
        
    except Exception as e:
        typer.echo(f"Ошибка: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def transcribe(
    audio_path: str = typer.Argument(..., help="Путь к аудио файлу"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Путь для сохранения текста"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Модель Whisper"),
):
    """Только транскрипция аудио в текст."""
    try:
        config = load_config()
        processor = File2Text(config=config, whisper_model=model or config.whisper_model, verbose=True)
        
        typer.echo(f"Транскрипция: {audio_path}")
        text = processor.transcribe(audio_path)
        
        if output:
            Path(output).write_text(text, encoding='utf-8')
            typer.echo(f"Текст сохранен в: {output}")
        else:
            typer.echo(text)
        
    except Exception as e:
        typer.echo(f"Ошибка: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def summarize(
    text_path: str = typer.Argument(..., help="Путь к текстовому файлу"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Путь для сохранения суммаризации"),
):
    """Суммаризация текста."""
    try:
        config = load_config()
        processor = File2Text(config=config, verbose=True)
        
        text = Path(text_path).read_text(encoding='utf-8')
        typer.echo(f"Суммаризация текста из: {text_path}")
        
        summary = processor.summarize(text)
        
        if output:
            Path(output).write_text(summary, encoding='utf-8')
            typer.echo(f"Суммаризация сохранена в: {output}")
        else:
            typer.echo(summary)
        
    except Exception as e:
        typer.echo(f"Ошибка: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def vectorize(
    text_path: str = typer.Argument(..., help="Путь к текстовому файлу"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Путь для сохранения векторов (.npy)"),
):
    """Векторизация текста."""
    try:
        import numpy as np
        
        config = load_config()
        processor = File2Text(config=config, verbose=True)
        
        text = Path(text_path).read_text(encoding='utf-8')
        typer.echo(f"Векторизация текста из: {text_path}")
        
        vectors = processor.vectorize(text)
        
        if output:
            np.save(output, vectors)
            typer.echo(f"Векторы сохранены в: {output}")
            typer.echo(f"Размерность: {vectors.shape}")
        else:
            typer.echo(f"Размерность вектора: {vectors.shape}")
        
    except Exception as e:
        typer.echo(f"Ошибка: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
