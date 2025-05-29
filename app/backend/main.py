import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline
from faster_whisper import WhisperModel
import ffmpeg
from pathlib import Path
import logging
from typing import Optional
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

# Initialize models
whisper_model = None
translation_model = None
translation_tokenizer = None
text_enhancer = None

def load_models():
    """Load all required models"""
    global whisper_model, translation_model, translation_tokenizer, text_enhancer
    
    try:
        # Load Whisper model
        whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully")
        
        # Load translation model
        model_name = "facebook/m2m100_418M"
        translation_tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        translation_model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        logger.info("Translation model loaded successfully")
        
        # Load text enhancement model
        text_enhancer = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device="cpu"
        )
        logger.info("Text enhancement model loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def load_srt_blocks(srt_content):
    """Load SRT blocks from content"""
    blocks = []
    for block in srt_content.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) >= 2:
            index = lines[0]
            timestamp = lines[1]
            text = " ".join(lines[2:]).strip() if len(lines) > 2 else ""
            blocks.append((index, timestamp, text))
    return blocks

def enhance_text(text):
    """Enhance text using the text enhancement model"""
    if not text.strip():
        return text
        
    try:
        # Prepare prompt for text enhancement
        prompt = f"Fix grammar and punctuation in this text, keep the meaning the same: {text}"
        
        # Generate enhanced text
        enhanced = text_enhancer(prompt, max_length=512, do_sample=False)[0]['generated_text']
        
        # Clean up the enhanced text
        enhanced = enhanced.replace("Fix grammar and punctuation in this text, keep the meaning the same:", "").strip()
        
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing text: {str(e)}")
        return text

def enhance_srt(srt_content):
    """Enhance SRT content"""
    try:
        blocks = load_srt_blocks(srt_content)
        enhanced_blocks = []
        
        for index, timestamp, text in blocks:
            # Skip enhancement for short interjections
            if text.lower() in ["wow", "um", "bou"]:
                enhanced_blocks.append((index, timestamp, text))
            else:
                enhanced_text = enhance_text(text)
                enhanced_blocks.append((index, timestamp, enhanced_text))
        
        # Rebuild SRT
        output_lines = []
        for index, timestamp, text in enhanced_blocks:
            output_lines.extend([index, timestamp, text, ""])
        
        return "\n".join(output_lines)
    except Exception as e:
        logger.error(f"Error enhancing SRT: {str(e)}")
        return srt_content

def translate_text(text: str, src_lang: str = "en", tgt_lang: str = "zh") -> str:
    """Translate text using M2M100 model"""
    if not text.strip():
        return ""
    
    try:
        translation_tokenizer.src_lang = src_lang
        encoded = translation_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        generated_tokens = translation_model.generate(
            **encoded,
            forced_bos_token_id=translation_tokenizer.get_lang_id(tgt_lang),
            max_length=512
        )
        translated_text = translation_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text

def create_srt(segments, output_path):
    """Create SRT file from segments"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def format_timestamp(seconds):
    """Format seconds to SRT timestamp"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def create_subtitled_video(video_path: str, srt_path: str, output_path: str):
    """Create video with embedded subtitles"""
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, output_path, vf=f"subtitles={srt_path}")
        ffmpeg.run(stream, overwrite_output=True)
        return True
    except Exception as e:
        logger.error(f"Error creating subtitled video: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_model_loaded": whisper_model is not None,
        "translation_model_loaded": translation_model is not None
    }

@app.post("/enhance_srt")
async def enhance_srt_endpoint(srt_content: str = Form(...)):
    """Enhance SRT content"""
    try:
        enhanced_srt = enhance_srt(srt_content)
        return {"enhanced_srt": enhanced_srt}
    except Exception as e:
        logger.error(f"Error in enhance_srt endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error enhancing SRT: {str(e)}"}
        )

@app.post("/process_video")
async def process_video(
    file: UploadFile = File(...),
    create_subtitled_video: bool = Form(False),
    enhance_srt: bool = Form(False)
):
    """Process video and generate subtitles"""
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded video
            video_path = os.path.join(temp_dir, "input.mp4")
            with open(video_path, "wb") as f:
                f.write(await file.read())
            
            # Transcribe video
            segments, _ = whisper_model.transcribe(video_path)
            segments = list(segments)
            
            # Create original SRT
            original_srt_path = os.path.join(temp_dir, "original.srt")
            create_srt(segments, original_srt_path)
            
            # Read original SRT
            with open(original_srt_path, "r", encoding="utf-8") as f:
                original_srt = f.read()
            
            # Enhance SRT if requested
            if enhance_srt:
                original_srt = enhance_srt(original_srt)
            
            # Translate SRT
            translated_srt_path = os.path.join(temp_dir, "translated.srt")
            translated_segments = []
            
            for segment in segments:
                translated_text = translate_text(segment.text)
                segment.text = translated_text
                translated_segments.append(segment)
            
            create_srt(translated_segments, translated_srt_path)
            
            # Read translated SRT
            with open(translated_srt_path, "r", encoding="utf-8") as f:
                translated_srt = f.read()
            
            # Enhance translated SRT if requested
            if enhance_srt:
                translated_srt = enhance_srt(translated_srt)
            
            # Create subtitled video if requested
            if create_subtitled_video:
                output_video_path = os.path.join(temp_dir, "output.mp4")
                if create_subtitled_video(video_path, translated_srt_path, output_video_path):
                    return FileResponse(
                        output_video_path,
                        media_type="video/mp4",
                        headers={
                            "X-Original-SRT": original_srt,
                            "X-Translated-SRT": translated_srt
                        }
                    )
            
            # Return SRT files
            return {
                "original_srt": original_srt,
                "translated_srt": translated_srt,
                "enhanced": enhance_srt,
                "source_lang": "en",
                "target_lang": "zh"
            }
            
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing video: {str(e)}"}
        ) 