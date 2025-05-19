def upload_video(file, quality, language):
    """
    Mock function to simulate video upload
    """
    return {"success": True, "video_id": "mock_video_123"}

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
    Mock function to simulate SRT content
    """
    return """1
00:00:01,000 --> 00:00:04,000
This is a sample subtitle

2
00:00:05,000 --> 00:00:08,000
It's just for demonstration purposes""" 