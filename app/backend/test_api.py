import os
import pytest
from fastapi.testclient import TestClient
from main import app
import tempfile
from pathlib import Path

client = TestClient(app)

def create_test_video():
    """Create a small test video file"""
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "test_video.mp4")
    
    # Create a 2-second test video using FFmpeg
    os.system(f'ffmpeg -f lavfi -i testsrc=duration=2:size=1280x720:rate=30 -c:v libx264 {video_path}')
    
    return video_path

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "whisper_model_loaded" in data
    assert "translation_model_loaded" in data
    assert "qwen_model_loaded" in data

def test_process_video():
    """Test video processing endpoint"""
    # Create test video
    video_path = create_test_video()
    
    # Test video processing
    with open(video_path, "rb") as f:
        files = {"file": ("test_video.mp4", f, "video/mp4")}
        data = {
            "create_subtitled_video": False,
            "enhance_srt": False
        }
        response = client.post("/process_video", files=files, data=data)
    
    # Clean up test video
    os.remove(video_path)
    os.rmdir(os.path.dirname(video_path))
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "original_srt" in data
    assert "translated_srt" in data
    assert "processing_time" in data
    assert "source_lang" in data
    assert "target_lang" in data
    assert data["source_lang"] == "en"
    assert data["target_lang"] == "zh"

def test_process_video_with_enhancement():
    """Test video processing with SRT enhancement"""
    # Create test video
    video_path = create_test_video()
    
    # Test video processing with enhancement
    with open(video_path, "rb") as f:
        files = {"file": ("test_video.mp4", f, "video/mp4")}
        data = {
            "create_subtitled_video": False,
            "enhance_srt": True
        }
        response = client.post("/process_video", files=files, data=data)
    
    # Clean up test video
    os.remove(video_path)
    os.rmdir(os.path.dirname(video_path))
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "original_srt" in data
    assert "translated_srt" in data
    assert "enhanced" in data
    assert data["enhanced"] == True

def test_process_video_with_subtitles():
    """Test video processing with subtitled video creation"""
    # Create test video
    video_path = create_test_video()
    
    # Test video processing with subtitles
    with open(video_path, "rb") as f:
        files = {"file": ("test_video.mp4", f, "video/mp4")}
        data = {
            "create_subtitled_video": True,
            "enhance_srt": False
        }
        response = client.post("/process_video", files=files, data=data)
    
    # Clean up test video
    os.remove(video_path)
    os.rmdir(os.path.dirname(video_path))
    
    # Check response
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert "X-Original-SRT" in response.headers
    assert "X-Translated-SRT" in response.headers

if __name__ == "__main__":
    pytest.main([__file__]) 