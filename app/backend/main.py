import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
os.environ["USE_TORCH"] = "1"  # Force use of PyTorch
os.environ["USE_TF"] = "0"  # Disable TensorFlow

import shutil
import tempfile
import logging
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("backend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Video Translation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ────────────────────────────────────────────────────────────────────
whisper_model: Optional[WhisperModel] = None
translation_model = None
translation_tokenizer = None
translator = None

@app.on_event("startup")
async def load_models():
    global whisper_model, translation_model, translation_tokenizer, translator
    try:
        # Load Whisper model
        logger.info("Loading faster-whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        whisper_model = WhisperModel("base", device=device, compute_type=compute_type)
        logger.info("Whisper model loaded successfully")

        # Load translation model
        logger.info("Loading Helsinki-NLP translation model...")
        model_name = "Helsinki-NLP/opus-mt-en-zh"  # Changed to English to Chinese
        translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
        translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        translator = pipeline(
            "translation",
            model=translation_model,
            tokenizer=translation_tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("Translation model loaded successfully")

    except Exception as e:
        logger.error("Failed to load models", exc_info=e)
        raise

# ─── Helpers ───────────────────────────────────────────────────────────────────
def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def format_segments_to_srt(segments) -> str:
    srt = ""
    for i, seg in enumerate(segments, start=1):
        srt += f"{i}\n"
        srt += f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n"
        srt += f"{seg.text.strip()}\n\n"
    return srt

def translate_text(text: str) -> str:
    """Translate text using Helsinki-NLP model from English to Chinese"""
    if not text.strip():
        return ""
    
    try:
        # Translate text
        translated = translator(text, max_length=512)
        return translated[0]["translation_text"]
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/process_video")
async def process_video(
    file: UploadFile = File(...),
    create_subtitled_video: bool = Form(False)
) -> Dict[str, Any]:
    """
    Process video: transcribe, translate to Chinese, and optionally create subtitled video
    """
    if whisper_model is None or translator is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    start_time = datetime.now()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Save video file
        video_path = os.path.join(temp_dir, file.filename)
        with open(video_path, "wb") as f:
            f.write(await file.read())

        # 2. Transcribe video
        logger.info("Transcribing video...")
        segments, _ = whisper_model.transcribe(
            video_path, language="en", task="transcribe", vad_filter=True
        )
        segments = list(segments)
        original_srt = format_segments_to_srt(segments)
        
        # 3. Translate SRT
        logger.info("Translating SRT to Chinese...")
        translated_segments = []
        for segment in original_srt.strip().split("\n\n"):
            lines = segment.split("\n")
            if len(lines) >= 3:
                index = lines[0]
                timestamp = lines[1]
                english_text = lines[2]
                chinese_text = translate_text(english_text)
                
                # Combine English and Chinese text
                combined_text = f"{english_text}\n{chinese_text}" if chinese_text else english_text
                translated_segment = f"{index}\n{timestamp}\n{combined_text}\n"
                translated_segments.append(translated_segment)
        
        translated_srt = "\n\n".join(translated_segments)

        # 4. Create subtitled video if requested
        subtitled_video_path = None
        if create_subtitled_video:
            logger.info("Creating subtitled video...")
            srt_path = os.path.join(temp_dir, "subtitles.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(translated_srt)

            subtitled_video_path = os.path.join(temp_dir, "subtitled_video.mp4")
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"subtitles={srt_path}",
                subtitled_video_path
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            
            if proc.returncode != 0:
                logger.error("FFmpeg failed: %s", proc.stderr)
                raise HTTPException(status_code=500, detail=f"FFmpeg error:\n{proc.stderr}")

        # 5. Prepare response
        response = {
            "original_srt": original_srt,
            "translated_srt": translated_srt,
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "source_lang": "en",
            "target_lang": "zh"
        }

        # 6. Add subtitled video to response if created
        if create_subtitled_video and os.path.exists(subtitled_video_path):
            return FileResponse(
                path=subtitled_video_path,
                filename="subtitled_video.mp4",
                media_type="video/mp4",
                headers={
                    "X-Original-SRT": original_srt,
                    "X-Translated-SRT": translated_srt
                }
            )

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.debug(f"Removed temp dir {temp_dir}")

@app.get("/health")
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "whisper_model_loaded": whisper_model is not None,
        "translation_model_loaded": translator is not None
    } 