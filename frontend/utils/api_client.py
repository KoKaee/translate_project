import os
import requests
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Get API base URL from environment variable
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/")

def upload_video(video_file, quality, language):
    try:
        files = {
            'file': ('video.mp4', video_file, 'video/mp4')
        }
        data = {
            'quality': quality,
            'language': language
        }

        session = requests.Session()
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        response = session.post(
            f"{API_BASE_URL}/video/upload",
            files=files,
            data=data,
            timeout=(30, 600),
            stream=True
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Error uploading video: {str(e)}"}

def get_processing_status(video_id):
    return {
        "stage": "completed",
        "progress": 100,
        "message": "Processing completed"
    }

def get_srt_content(video_id):
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"{API_BASE_URL}/srt/{video_id}",
                timeout=(30, 60)
            )
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return {"error": f"Error getting SRT content: {str(e)}"}
            time.sleep(retry_delay * (attempt + 1))

# ✅ New function to call /translate_srt
def translate_srt(srt_file, target_language="fr"):
    try:
        files = {
            'file': ('subtitle.srt', srt_file, 'application/x-subrip')
        }
        data = {
            'target_language': target_language
        }
        response = requests.post(
            f"{API_BASE_URL}/translate_srt",
            files=files,
            data=data,
            timeout=(30, 300),
            stream=True
        )
        response.raise_for_status()
        return response.text  # or .content if binary
    except requests.exceptions.RequestException as e:
        return {"error": f"Error translating SRT: {str(e)}"}
