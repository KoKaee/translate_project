# Streamlit Frontend Implementation Guide

## Project Structure
```
frontend/
├── app.py                 # Main Streamlit application
├── components/            # Reusable UI components
│   ├── __init__.py
│   ├── video_upload.py    # Video upload component
│   ├── progress_bar.py    # Progress tracking component
│   ├── metadata_display.py # Video metadata display
│   └── result_preview.py  # SRT preview component
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── api_client.py      # API communication
│   ├── validators.py      # Input validation
│   └── formatters.py      # Data formatting
├── config/               # Configuration files
│   ├── __init__.py
│   └── settings.py       # App settings and constants
└── requirements.txt      # Frontend dependencies
```

## Core Dependencies
```txt
streamlit>=1.24.0
requests>=2.28.0
python-dotenv>=0.19.0
streamlit-option-menu>=0.3.2
```

## Implementation Steps

### 1. Basic App Structure (app.py)
```python
import streamlit as st
from components.video_upload import render_upload_section
from components.progress_bar import render_progress
from components.metadata_display import render_metadata
from components.result_preview import render_preview

def main():
    st.set_page_config(
        page_title="YouTube Subtitle Generator",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("YouTube Subtitle Generator")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        # Add settings controls here
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Upload", "Processing", "Results"])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_progress()
    
    with tab3:
        render_preview()

if __name__ == "__main__":
    main()
```

### 2. Video Upload Component (components/video_upload.py)
```python
import streamlit as st
from utils.validators import validate_youtube_url
from utils.api_client import upload_video

def render_upload_section():
    st.header("Upload YouTube Video")
    
    # URL input
    url = st.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    # Quality selection
    quality = st.selectbox(
        "Select Video Quality",
        ["720p", "1080p", "480p", "360p"]
    )
    
    # Language selection
    language = st.selectbox(
        "Select Target Language",
        ["English", "Chinese", "French"]
    )
    
    if st.button("Process Video"):
        if validate_youtube_url(url):
            with st.spinner("Processing..."):
                # Call API to start processing
                response = upload_video(url, quality, language)
                if response["success"]:
                    st.success("Video processing started!")
                    # Store video_id in session state
                    st.session_state.video_id = response["video_id"]
                else:
                    st.error("Failed to process video")
        else:
            st.error("Invalid YouTube URL")
```

### 3. Progress Tracking (components/progress_bar.py)
```python
import streamlit as st
import time
from utils.api_client import get_processing_status

def render_progress():
    st.header("Processing Status")
    
    if "video_id" not in st.session_state:
        st.info("Please upload a video first")
        return
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Poll for status updates
    while True:
        status = get_processing_status(st.session_state.video_id)
        
        if status["stage"] == "completed":
            progress_bar.progress(100)
            status_text.text("Processing completed!")
            break
        elif status["stage"] == "failed":
            st.error("Processing failed")
            break
        
        # Update progress
        progress = status["progress"]
        progress_bar.progress(progress)
        status_text.text(f"Processing: {status['message']}")
        
        time.sleep(2)  # Poll every 2 seconds
```

### 4. Result Preview (components/result_preview.py)
```python
import streamlit as st
from utils.api_client import get_srt_content

def render_preview():
    st.header("Generated Subtitles")
    
    if "video_id" not in st.session_state:
        st.info("Please upload and process a video first")
        return
    
    # Get SRT content
    srt_content = get_srt_content(st.session_state.video_id)
    
    if srt_content:
        # Display SRT content
        st.text_area("SRT Content", srt_content, height=400)
        
        # Download button
        st.download_button(
            "Download SRT",
            srt_content,
            file_name="subtitles.srt",
            mime="text/plain"
        )
    else:
        st.info("No subtitles available yet")
```

### 5. API Client (utils/api_client.py)
```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def upload_video(url, quality, language):
    response = requests.post(
        f"{API_BASE_URL}/api/video/upload",
        json={
            "url": url,
            "quality": quality,
            "language": language
        }
    )
    return response.json()

def get_processing_status(video_id):
    response = requests.get(
        f"{API_BASE_URL}/api/video/status/{video_id}"
    )
    return response.json()

def get_srt_content(video_id):
    response = requests.get(
        f"{API_BASE_URL}/api/srt/{video_id}"
    )
    return response.json().get("srt_content")
```

## UI/UX Guidelines

### 1. Layout
- Use wide layout for better space utilization
- Implement tabs for different stages
- Keep sidebar for settings and controls
- Use consistent spacing and padding

### 2. User Feedback
- Show loading spinners during API calls
- Display clear success/error messages
- Use progress bars for long operations
- Implement tooltips for complex features

### 3. Error Handling
- Validate inputs before submission
- Show user-friendly error messages
- Implement retry mechanisms
- Handle API failures gracefully

### 4. State Management
- Use Streamlit session state for persistence
- Store video_id and processing status
- Maintain user preferences
- Handle page refreshes

## Testing Strategy

### 1. Unit Tests
- Test individual components
- Validate input handling
- Check API client functions
- Test utility functions

### 2. Integration Tests
- Test component interactions
- Validate API communication
- Check state management
- Test error scenarios

### 3. UI Tests
- Test responsive design
- Validate user interactions
- Check accessibility
- Test different screen sizes

## Deployment Checklist

1. **Environment Setup**
   - Set up virtual environment
   - Install dependencies
   - Configure environment variables
   - Set up API endpoints

2. **Security**
   - Implement API key management
   - Add input sanitization
   - Set up CORS policies
   - Configure rate limiting

3. **Performance**
   - Optimize API calls
   - Implement caching
   - Minimize page refreshes
   - Optimize large file handling

4. **Monitoring**
   - Add error logging
   - Implement usage tracking
   - Set up performance monitoring
   - Configure alerts

## Next Steps

1. Set up the project structure
2. Implement basic components
3. Add API integration
4. Implement error handling
5. Add testing
6. Deploy and monitor

## Notes
- Keep the UI simple and intuitive
- Focus on user experience
- Implement proper error handling
- Document all components and functions
- Regular testing and updates 