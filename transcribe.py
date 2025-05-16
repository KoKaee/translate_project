import os
import glob
import subprocess
from pathlib import Path
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from faster_whisper import WhisperModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
os.environ["USE_TORCH"] = "1"  # Force use of PyTorch
os.environ["USE_TF"] = "0"  # Disable TensorFlow

def get_video_files(directory="./videos/"):
    """Get all video files in the specified directory."""
    video_extensions = ["*.mp4", "*.mkv", "*.webm", "*.flv"]
    video_files = []

    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, ext)))

    return video_files

def transcribe_video(video_path):
    """Transcribe video using Whisper model."""
    print(f"Transcribing {video_path}...")

    # Using faster-whisper for better performance
    model_size = "medium"
    # Run on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    # Load the Whisper model
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Transcribe the audio
    segments, _ = model.transcribe(
        video_path, language="en", task="transcribe", vad_filter=True
    )

    # Format as SRT
    srt_content = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()

        srt_content.append(f"{i}\n{start} --> {end}\n{text}\n")

    return srt_content

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def main():
    # Load translation model
    print("Loading translation model...")
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer)

    video_files = get_video_files()

    if not video_files:
        print("No video files found in the current directory.")
        return

    print(f"Found {len(video_files)} video file(s).")

    for video_path in video_files:
        video_filename = Path(video_path).stem
        srt_path = f"{video_filename}.srt"

        # Transcribe video
        srt_content = transcribe_video(video_path)

if __name__ == "__main__":
    main()