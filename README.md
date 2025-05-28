# Video Translation App

A fast and efficient video translation application that transcribes videos, enhances subtitles using AI, and translates them to Chinese.

## Features

- Video transcription using Whisper
- SRT enhancement using Qwen AI
- Translation to Chinese
- Optional subtitled video creation
- CPU-optimized for better performance

## Requirements

- Python 3.10+
- FFmpeg
- 4GB+ RAM
- 2GB+ free disk space

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd video-translation-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg:
- Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)
- Linux: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

## Usage

1. Start the backend:
```bash
cd app/backend
python -m uvicorn main:app --reload
```

2. Start the frontend:
```bash
cd app/frontend
python -m streamlit run app.py
```

3. Open your browser and go to `http://localhost:8501`

## Testing

The application includes automated tests to ensure functionality:

1. Run backend tests:
```bash
cd app/backend
python -m pytest test_api.py -v
```

2. Run frontend checks:
```bash
cd app/frontend
python -c "import streamlit; import requests; import dotenv"
```

## GitHub Actions

The repository includes GitHub Actions workflows that automatically run tests on:
- Push to main branch
- Pull requests to main branch

The workflow:
1. Sets up Python 3.10
2. Installs FFmpeg
3. Installs dependencies
4. Runs backend tests
5. Checks frontend imports

## Project Structure

```
video-translation-app/
├── app/
│   ├── backend/
│   │   ├── main.py
│   │   └── test_api.py
│   └── frontend/
│       └── app.py
├── .github/
│   └── workflows/
│       └── test.yml
├── requirements.txt
└── README.md
```

## Performance Optimizations

The app is optimized for CPU usage and speed:
- Uses Whisper tiny model
- Uses Qwen-1.8B model
- Optimized FFmpeg settings
- Efficient memory usage
- Minimal dependencies

## License

MIT License 