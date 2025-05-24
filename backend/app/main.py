import os
import shutil
import tempfile
import logging
import subprocess
from datetime import datetime
from typing import Optional

import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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
app = FastAPI(title="Video to Text API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Whisper Model ─────────────────────────────────────────────────────────────
whisper_model: Optional[WhisperModel] = None

@app.on_event("startup")
async def load_whisper_model():
    global whisper_model
    try:
        logger.info("Loading faster-whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        whisper_model = WhisperModel("base", device=device, compute_type=compute_type)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error("Failed to load Whisper model", exc_info=e)
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

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = datetime.now()
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(await file.read())
        temp_path = tf.name

    try:
        segments, _ = whisper_model.transcribe(
            temp_path, task="transcribe", vad_filter=True
        )
        segments = list(segments)
        srt_text = format_segments_to_srt(segments)
        full_text = " ".join(seg.text for seg in segments)
        proc_time = (datetime.now() - start_time).total_seconds()

        return {
            "text": full_text,
            "segments": segments,
            "srt": srt_text,
            "processing_time": proc_time
        }
    finally:
        os.unlink(temp_path)
        logger.debug(f"Deleted temp file {temp_path}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": whisper_model is not None}

@app.post("/comment_video")
async def comment_video(
    video_file: UploadFile = File(...),
    srt_file:   UploadFile = File(...)
):
    # 1) Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # 2) Write uploads
        base_name, _ = os.path.splitext(video_file.filename)
        vid_name = f"{base_name}.mp4"
        srt_name = f"{base_name}.srt"
        out_name = f"{base_name}_commented.mp4"

        vid_path = os.path.join(temp_dir, vid_name)
        srt_path = os.path.join(temp_dir, srt_name)
        out_path = os.path.join(temp_dir, out_name)

        # Save video
        with open(vid_path, "wb") as fv:
            fv.write(await video_file.read())
        # Save SRT
        with open(srt_path, "wb") as fs:
            fs.write(await srt_file.read())

        # 3) Run FFmpeg inside temp_dir, with relative filenames
        cmd = [
            "ffmpeg",
            "-i", vid_name,
            "-vf", f"subtitles={srt_name}",
            out_name
        ]
        logger.info("Running FFmpeg: %s", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True)

        if proc.returncode != 0:
            logger.error("FFmpeg failed: %s", proc.stderr)
            raise HTTPException(status_code=500, detail=f"FFmpeg error:\n{proc.stderr}")

        # 4) Verify output exists
        if not os.path.isfile(out_path):
            logger.error("Expected output not found, dir contents: %s", os.listdir(temp_dir))
            raise HTTPException(500, detail="FFmpeg did not produce output file")

        # 5) Return the burned-in video
        return FileResponse(
            path=out_path,
            filename=out_name,
            media_type="video/mp4"
        )

    finally:
        # Clean up entire temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.debug(f"Removed temp dir {temp_dir}")
