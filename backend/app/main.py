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
from fastapi import (
    FastAPI, UploadFile, File,
    HTTPException, Request, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# ─── Logging ───────────────────────────────────────────────────────────────────
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

# ─── CORS & Request Logging ────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    size = request.headers.get("content-length") or "?"
    logger.info(f"→ Incoming {request.method} {request.url.path}  size={size}")
    start = datetime.utcnow()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception("Unhandled error in request")
        raise
    process_time = (datetime.utcnow() - start).total_seconds()
    logger.info(f"← Completed {request.method} {request.url.path}  in={process_time:.2f}s status={response.status_code}")
    return response

# ─── Whisper & Translation Models ─────────────────────────────────────────────
whisper_model: Optional[WhisperModel] = None
tokenizer: Optional[M2M100Tokenizer] = None
translation_model: Optional[M2M100ForConditionalGeneration] = None
SRC_LANG = "en"
TGT_LANG = "fr"

@app.on_event("startup")
async def load_models():
    global whisper_model, tokenizer, translation_model
    # Whisper
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        whisper_model = WhisperModel("base", device=device, compute_type=compute_type)
        logger.info("✅ Whisper model loaded")
    except Exception:
        logger.exception("❌ Failed to load Whisper model")
        raise
    # M2M100
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        logger.info("✅ Translation model loaded")
    except Exception:
        logger.exception("❌ Failed to load translation model")
        raise

# ─── Helpers ───────────────────────────────────────────────────────────────────
def format_timestamp(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

def format_segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines += [
            str(i),
            f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}",
            seg.text.strip() or "[ … ]", ""
        ]
    return "\n".join(lines)

def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:"*?<>|]+', "_", name)

def translate_text(text: str, src_lang: str = SRC_LANG, tgt_lang: str = TGT_LANG) -> str:
    if not text.strip():
        return ""
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    tokens = translation_model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
        max_length=512
    )
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

# ─── Exception Handlers ───────────────────────────────────────────────────────
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        {"detail": exc.detail},
        status_code=exc.status_code
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        {"detail": "Internal server error"},
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    if whisper_model is None:
        raise HTTPException(503, "Model not loaded")
    # Basic upload validation
    if file.content_type.split("/")[0] != "video":
        raise HTTPException(400, "Invalid file type")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".mp4")
    try:
        content = await file.read()
        if not content:
            raise HTTPException(400, "Empty file")
        tmp.write(content)
        tmp.flush()
        segments_gen, info = whisper_model.transcribe(tmp.name, language="en", task="transcribe", vad_filter=False)
        segments = list(segments_gen)
        if not segments:
            raise HTTPException(422, "No speech detected")
        srt_text = format_segments_to_srt(segments)
        # return JSON payload
        return {
            "srt": srt_text,
            "processing_time": info.duration
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in /transcribe")
        raise HTTPException(500, "Transcription failed")
    finally:
        try:
            tmp.close()
            os.unlink(tmp.name)
        except:
            pass

@app.post("/translate_srt")
async def translate_srt_endpoint(srt_file: UploadFile = File(...)):
    if translation_model is None:
        raise HTTPException(503, "Translation model not loaded")
    if not srt_file.filename.lower().endswith(".srt"):
        raise HTTPException(400, "Please upload a .srt file")
    try:
        content = (await srt_file.read()).decode("utf-8", errors="ignore")
        if not content.strip():
            raise HTTPException(400, "Empty subtitle file")
        lines = content.splitlines()
        out_lines, buffer = [], []
        for line in lines + [""]:
            if not line.strip():
                if len(buffer) >= 3:
                    idx, ts, *txt = buffer
                    full = " ".join(txt).strip()
                    tr = translate_text(full)
                    out_lines += [idx, ts, tr, ""]
                else:
                    out_lines += buffer + [""]
                buffer = []
            else:
                buffer.append(line)
        # stream back as a temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_fr.srt", mode="w", encoding="utf-8")
        tmp.write("\n".join(out_lines))
        tmp.flush()
        tmp.close()
        return FileResponse(tmp.name, filename=f"{safe_filename(srt_file.filename)}_fr.srt")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in /translate_srt")
        raise HTTPException(500, "Translation failed")
    finally:
        # cleanup would happen after response is sent by FileResponse
        ...

# (comment_video stays unchanged, but you can wrap it similarly)
