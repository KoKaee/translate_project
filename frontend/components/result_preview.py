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