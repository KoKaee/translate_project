import os
import requests
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Get API base URL from environment variable
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/")

def upload_video(video_file, quality, language):
    """
    Upload video file to backend with chunked transfer
    """
    try:
        # Prepare the files and data for the request
        files = {
            'file': ('video.mp4', video_file, 'video/mp4')
        }
        data = {
            'quality': quality,
            'language': language
        }
        
        # Configure session with retry strategy
        session = requests.Session()
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Make the request with increased timeout and chunked transfer
        response = session.post(
            f"{API_BASE_URL}/video/upload",
            files=files,
            data=data,
            timeout=(30, 600),  # (connect timeout, read timeout)
            stream=True  # Enable chunked transfer
        )
        
        # Check if the request was successful
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Error uploading video: {str(e)}"}

def get_processing_status(video_id):
    """
    Mock function to simulate processing status
    """
    return {
        "stage": "completed",
        "progress": 100,
        "message": "Processing completed"
    }

def get_srt_content(video_id):
    """
    Get SRT content for a video with retry logic
    """
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"{API_BASE_URL}/srt/{video_id}",
                timeout=(30, 60)  # (connect timeout, read timeout)
            )
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                return {"error": f"Error getting SRT content: {str(e)}"}
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff 