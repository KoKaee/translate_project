import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
os.environ["USE_TORCH"] = "1"  # Force use of PyTorch
os.environ["USE_TF"] = "0"  # Disable TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage

import shutil
import tempfile
import logging
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, AutoModelForCausalLM

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
qwen_model = None
qwen_tokenizer = None

@app.on_event("startup")
async def load_models():
    global whisper_model, translation_model, translation_tokenizer, translator, qwen_model, qwen_tokenizer
    try:
        # Load Whisper model (tiny model for speed)
        logger.info("Loading faster-whisper model...")
        whisper_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=1
        )
        logger.info("Whisper model loaded successfully")

        # Load translation model (smaller model)
        logger.info("Loading Helsinki-NLP translation model...")
        model_name = "Helsinki-NLP/opus-mt-en-zh"
        translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
        translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )
        translator = pipeline(
            "translation",
            model=translation_model,
            tokenizer=translation_tokenizer,
            device=-1,  # Force CPU
            max_length=128  # Limit sequence length for speed
        )
        logger.info("Translation model loaded successfully")

        # Load Qwen model (smaller version)
        logger.info("Loading Qwen model...")
        qwen_model_name = "Qwen/Qwen-1_8B-Chat"  # Using smaller 1.8B model
        qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        ).eval()
        logger.info("Qwen model loaded successfully")

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
        # Translate text with shorter max length
        translated = translator(text, max_length=64)
        return translated[0]["translation_text"]
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

def enhance_srt_with_qwen(srt_content: str) -> str:
    """Enhance SRT content using Qwen model"""
    try:
        # Parse SRT content
        segments = []
        current_segment = {}
        
        for line in srt_content.strip().split('\n'):
            line = line.strip()
            if not line:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = {}
                continue
                
            if not current_segment:
                current_segment = {'index': line}
            elif 'timestamp' not in current_segment:
                current_segment['timestamp'] = line
            else:
                current_segment['text'] = line

        # Enhance each segment with shorter generation
        enhanced_segments = []
        for segment in segments:
            # Prepare prompt for Qwen
            prompt = f"""Enhance this subtitle (keep it short): {segment['text']}"""
            
            # Generate enhanced text with shorter parameters
            inputs = qwen_tokenizer(prompt, return_tensors="pt")
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=32,  # Shorter generation
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=False  # Faster generation
            )
            enhanced_text = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the enhanced text
            enhanced_text = enhanced_text.replace(prompt, "").strip()
            
            # Create enhanced segment
            enhanced_segment = f"{segment['index']}\n{segment['timestamp']}\n{enhanced_text}\n"
            enhanced_segments.append(enhanced_segment)
        
        return "\n\n".join(enhanced_segments)
    
    except Exception as e:
        logger.error(f"SRT enhancement error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SRT enhancement failed: {str(e)}")

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/process_video")
async def process_video(
    file: UploadFile = File(...),
    create_subtitled_video: bool = Form(False),
    enhance_srt: bool = Form(False)
) -> Dict[str, Any]:
    """
    Process video: transcribe, translate to Chinese, enhance SRT (optional), and optionally create subtitled video
    """
    if whisper_model is None or translator is None or (enhance_srt and qwen_model is None):
        raise HTTPException(status_code=503, detail="Required models not loaded")

    start_time = datetime.now()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Save video file
        video_path = os.path.join(temp_dir, file.filename)
        with open(video_path, "wb") as f:
            f.write(await file.read())

        # 2. Transcribe video with faster settings
        logger.info("Transcribing video...")
        segments, _ = whisper_model.transcribe(
            video_path,
            language="en",
            task="transcribe",
            vad_filter=True,
            beam_size=1,  # Faster beam search
            best_of=1,    # Fewer candidates
            temperature=0.0  # Deterministic output
        )
        segments = list(segments)
        original_srt = format_segments_to_srt(segments)
        
        # 3. Enhance SRT if requested
        if enhance_srt:
            logger.info("Enhancing SRT with Qwen...")
            original_srt = enhance_srt_with_qwen(original_srt)
        
        # 4. Translate SRT
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

        # 5. Create subtitled video if requested
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
                "-c:v", "libx264",  # Faster encoding
                "-preset", "ultrafast",  # Fastest preset
                "-crf", "28",  # Slightly lower quality for speed
                subtitled_video_path
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            
            if proc.returncode != 0:
                logger.error("FFmpeg failed: %s", proc.stderr)
                raise HTTPException(status_code=500, detail=f"FFmpeg error:\n{proc.stderr}")

        # 6. Prepare response
        response = {
            "original_srt": original_srt,
            "translated_srt": translated_srt,
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "source_lang": "en",
            "target_lang": "zh",
            "enhanced": enhance_srt
        }

        # 7. Add subtitled video to response if created
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
        "translation_model_loaded": translator is not None,
        "qwen_model_loaded": qwen_model is not None
    } 