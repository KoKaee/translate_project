import streamlit as st
import requests
import tempfile
import os
from pathlib import Path

# Configure the page
st.set_page_config(
    page_title="Video Translation App",
    page_icon="🎥",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000"

def download_file(url, filename):
    """Download a file from a URL"""
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        return True
    return False

def enhance_srt(srt_content):
    """Enhance SRT content using the backend API"""
    try:
        response = requests.post(
            f"{API_URL}/enhance_srt",
            data={"srt_content": srt_content}
        )
        if response.status_code == 200:
            return response.json()["enhanced_srt"]
        return srt_content
    except Exception as e:
        st.error(f"Error enhancing SRT: {str(e)}")
        return srt_content

def main():
    st.title("🎥 Video Translation App")
    st.markdown("""
    This app helps you:
    1. Upload a video
    2. Extract subtitles
    3. Enhance subtitles (optional)
    4. Translate subtitles to Chinese
    5. Create a subtitled video (optional)
    """)

    # File upload
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        # Options
        col1, col2 = st.columns(2)
        with col1:
            create_subtitled_video = st.checkbox("Create subtitled video", value=False)
        with col2:
            enhance_srt_option = st.checkbox("Enhance subtitles", value=False, 
                help="Improve subtitle quality by fixing broken sentences and adding proper punctuation")

        # Process button
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                try:
                    # Prepare files
                    files = {"file": (uploaded_file.name, uploaded_file, "video/mp4")}
                    data = {
                        "create_subtitled_video": create_subtitled_video,
                        "enhance_srt": enhance_srt_option
                    }

                    # Send request
                    response = requests.post(f"{API_URL}/process_video", files=files, data=data)

                    if response.status_code == 200:
                        # Handle video response
                        if create_subtitled_video and response.headers.get("content-type") == "video/mp4":
                            # Save video
                            video_path = "output_video.mp4"
                            with open(video_path, "wb") as f:
                                f.write(response.content)
                            
                            # Display video
                            st.video(video_path)
                            
                            # Display SRT files
                            original_srt = response.headers.get("X-Original-SRT", "")
                            translated_srt = response.headers.get("X-Translated-SRT", "")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Original Subtitles")
                                st.text_area("", original_srt, height=300)
                            with col2:
                                st.subheader("Translated Subtitles")
                                st.text_area("", translated_srt, height=300)
                            
                            # Download buttons
                            st.download_button(
                                "Download Original SRT",
                                original_srt,
                                file_name="original.srt",
                                mime="text/plain"
                            )
                            st.download_button(
                                "Download Translated SRT",
                                translated_srt,
                                file_name="translated.srt",
                                mime="text/plain"
                            )
                            st.download_button(
                                "Download Subtitled Video",
                                open(video_path, "rb").read(),
                                file_name="output_video.mp4",
                                mime="video/mp4"
                            )
                        else:
                            # Handle SRT response
                            result = response.json()
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Original Subtitles")
                                st.text_area("", result["original_srt"], height=300)
                            with col2:
                                st.subheader("Translated Subtitles")
                                st.text_area("", result["translated_srt"], height=300)
                            
                            # Download buttons
                            st.download_button(
                                "Download Original SRT",
                                result["original_srt"],
                                file_name="original.srt",
                                mime="text/plain"
                            )
                            st.download_button(
                                "Download Translated SRT",
                                result["translated_srt"],
                                file_name="translated.srt",
                                mime="text/plain"
                            )
                    else:
                        st.error(f"Error: {response.json().get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main() 