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

def get_video_files(directory="./"):
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

def translate_text(text, translator):
    """Translate text from English to Chinese."""
    if not text.strip():
        return ""

    translated = translator(text, max_length=512)
    return translated[0]["translation_text"]


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

        # Translate each subtitle line
        print(f"Translating subtitles for {video_path}...")
        translated_srt = []
        for line in srt_content:
            parts = line.strip().split("\n")
            if len(parts) >= 3:  # Valid subtitle entry
                subtitle_index = parts[0]
                timestamp = parts[1]
                english_text = parts[2]

                # Translate to Chinese
                chinese_translation = translate_text(english_text, translator)

                # Combine English and Chinese
                combined_text = (
                    f"{english_text}\n{chinese_translation}"
                    if chinese_translation
                    else english_text
                )

                translated_srt.append(
                    f"{subtitle_index}\n{timestamp}\n{combined_text}\n\n"
                )

        # Write to SRT file
        with open(srt_path, "w", encoding="utf-8") as f:
            f.writelines(translated_srt)

        print(f"Subtitles with translation saved to {srt_path}")


if __name__ == "__main__":
    main()