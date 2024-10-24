**Краткий гайд по установке и запуску кода**

1. **Установите Python 3.9+**

   - Скачайте и установите Python с официального сайта: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Убедитесь, что добавили Python в переменную PATH во время установки.

2. **Создайте виртуальное окружение (опционально)**

   - Создайте виртуальное окружение командой:
     ```sh
     python -m venv venv
     ```
   - Активируйте виртуальное окружение:
     - Windows: `venv\Scripts\activate`
     - MacOS/Linux: `source venv/bin/activate`

3. **Установите необходимые зависимости**

   - Убедитесь, что `pip` обновлен:
     ```sh
     pip install --upgrade pip
     ```
   - Установите зависимости, указанные ниже:
     ```sh
     pip install git+https://github.com/openai/whisper.git
     pip install pyannote.audio
     pip install sumy
     pip install sentence-transformers
     pip install torch torchaudio
     pip install transformers
     pip install nltk
     ```
   - Установите FFmpeg (для конвертации аудиофайлов):
     - Windows: скачайте и установите FFmpeg с [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) и добавьте его в переменную PATH.
     - Linux/MacOS: установите с помощью пакетного менеджера:
       ```sh
       sudo apt install ffmpeg  # Ubuntu/Debian
       brew install ffmpeg      # macOS
       ```

4. **Скачайте необходимые данные для NLTK**

   - Скачайте данные, включая токенизатор 'punkt':
     ```sh
     python -c "import nltk; nltk.download('punkt')"
     ```

5. **Создайте структуру папок**

   - В корне проекта создайте папки `files`, `text` и `sumText`. В папку `files` добавьте аудио или видео файлы для обработки.

6. **Запуск скрипта**

   - Запустите скрипт командой:
     ```sh
     python main.py
     ```
   - Скрипт автоматически обработает все файлы в папке `files`, создаст текстовые версии и суммаризации в папках `text` и `sumText`.

7. **Hugging Face Token**

   - Зарегистрируйтесь на [https://huggingface.co](https://huggingface.co) и получите API-токен.
   - Замените значение переменной `YOUR_AUTH_TOKEN` в коде на ваш токен.

