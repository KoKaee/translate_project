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
            st.error(f"Processing failed: {status.get('error', 'Unknown error')}")
            break
        
        # Update progress
        progress = status.get("progress", 0)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {status.get('message', 'In progress...')}")
        
        time.sleep(2)  # Poll every 2 seconds 