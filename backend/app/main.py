import os
import re
import shutil
import tempfile
import logging
import subprocess
from datetime import datetime
from typing import Optional

import torch
from faster_whisper import WhisperModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
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
app = FastAPI(title="Video to Text API with SRT Translation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Whisper Model ─────────────────────────────────────────────────────────────
whisper_model: Optional[WhisperModel] = None

# ─── Translation Model ─────────────────────────────────────────────────────────
tokenizer: Optional[M2M100Tokenizer] = None
translation_model: Optional[M2M100ForConditionalGeneration] = None
SRC_LANG = "en"
TGT_LANG = "fr"

@app.on_event("startup")
async def load_models():
    global whisper_model, tokenizer, translation_model
    # Load Whisper model
    try:
        logger.info("Loading faster-whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        whisper_model = WhisperModel("base", device=device, compute_type=compute_type)
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error("Failed to load Whisper model", exc_info=e)
        raise
    # Load translation model
    try:
        logger.info("Loading translation model facebook/m2m100_418M...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        logger.info("Translation model loaded successfully")
    except Exception as e:
        logger.error("Failed to load translation model", exc_info=e)
        raise

# ─── Helpers ───────────────────────────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_segments_to_srt(segments) -> str:
    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        srt_lines.append(f"{i}")
        srt_lines.append(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}")
        srt_lines.append(seg.text.strip() or "[ … ]")
        srt_lines.append("")  # blank line
    return "\n".join(srt_lines)


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:"*?<>|]+', '_', name)


def translate_text(text: str, src_lang: str = SRC_LANG, tgt_lang: str = TGT_LANG) -> str:
    if not text.strip():
        return ""
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    generated_tokens = translation_model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
        max_length=512
    )
    translated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    start_time = datetime.now()
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(await file.read())
        tmp_video_path = tf.name
    try:
        gen_segments, info = whisper_model.transcribe(
            tmp_video_path,
            language="en",
            task="transcribe",
            vad_filter=False
        )
        raw_segments = list(gen_segments)
        logger.info(f"Whisper returned {len(raw_segments)} segments over {info.duration:.1f}s of audio")
        if not raw_segments:
            raise HTTPException(status_code=422, detail="No speech detected in the video.")
        segments_dict = [{"start": seg.start, "end": seg.end, "text": seg.text.strip()} for seg in raw_segments]
        srt_text = format_segments_to_srt(raw_segments)
        full_text = " ".join(d["text"] for d in segments_dict)
        os.makedirs("srt_files", exist_ok=True)
        base = os.path.splitext(os.path.basename(file.filename))[0]
        disk_path = os.path.join("srt_files", f"{safe_filename(base)}.srt")
        with open(disk_path, "w", encoding="utf-8") as out:
            out.write(srt_text)
        return {
            "text": full_text,
            "segments": segments_dict,
            "srt": srt_text,
            "srt_file_path": disk_path,
            "processing_time": (datetime.now() - start_time).total_seconds()
        }
    finally:
        try:
            os.unlink(tmp_video_path)
            logger.debug(f"Deleted temp file {tmp_video_path}")
        except OSError:
            logger.warning(f"Could not delete {tmp_video_path}", exc_info=True)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": whisper_model is not None and translation_model is not None}

@app.post("/comment_video")
async def comment_video(
    video_file: UploadFile = File(...),
    srt_file: UploadFile = File(...)
):
    temp_dir = tempfile.mkdtemp()
    try:
        base_name, _ = os.path.splitext(video_file.filename)
        vid_name = f"{base_name}.mp4"
        srt_name = f"{base_name}.srt"
        out_name = f"{base_name}_commented.mp4"
        vid_path = os.path.join(temp_dir, vid_name)
        srt_path = os.path.join(temp_dir, srt_name)
        out_path = os.path.join(temp_dir, out_name)
        with open(vid_path, "wb") as fv:
            fv.write(await video_file.read())
        with open(srt_path, "wb") as fs:
            fs.write(await srt_file.read())
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
        if not os.path.isfile(out_path):
            logger.error("Expected output not found, dir contents: %s", os.listdir(temp_dir))
            raise HTTPException(500, detail="FFmpeg did not produce output file")
        return FileResponse(
            path=out_path,
            filename=out_name,
            media_type="video/mp4"
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.debug(f"Removed temp dir {temp_dir}")

@app.post("/translate_srt")
async def translate_srt(
    srt_file: UploadFile = File(...)
):
    if tokenizer is None or translation_model is None:
        raise HTTPException(status_code=503, detail="Translation model not loaded")
    # Read SRT content
    content = (await srt_file.read()).decode("utf-8")
    lines = content.splitlines()
    translated_lines = []
    buffer = []
    for line in lines:
        if line.strip() == "":
            if len(buffer) >= 3:
                index = buffer[0]
                timestamp = buffer[1]
                text_lines = buffer[2:]
                full_text = " ".join(text_lines).strip()
                translated_text = translate_text(full_text)
                translated_block = [index, timestamp] + [translated_text, ""]
                translated_lines.extend(translated_block)
            else:
                translated_lines.extend(buffer + [""])
            buffer = []
        else:
            buffer.append(line)
    # Handle last block if no trailing newline
    if buffer:
        if len(buffer) >= 3:
            index = buffer[0]
            timestamp = buffer[1]
            text_lines = buffer[2:]
            full_text = " ".join(text_lines).strip()
            translated_text = translate_text(full_text)
            translated_block = [index, timestamp] + [translated_text, ""]
            translated_lines.extend(translated_block)
        else:
            translated_lines.extend(buffer)
    # Write to temporary file
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix="_fr.srt", mode="w", encoding="utf-8")
    tmp_out.write("\n".join(translated_lines))
    tmp_out.flush()
    tmp_out.close()
    return FileResponse(
        path=tmp_out.name,
        filename=f"{safe_filename(srt_file.filename).rsplit('.', 1)[0]}_fr.srt",
        media_type="text/plain"
    )
