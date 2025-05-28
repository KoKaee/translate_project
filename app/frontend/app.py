import streamlit as st
import os
import tempfile
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API URL from environment variable
API_URL = os.getenv("API_URL", "http://localhost:8000")

def main():
    st.set_page_config(
        page_title="Video Translation App",
        page_icon="🎥",
        layout="wide"
    )

    st.title("🎥 Video Translation App")
    st.markdown("""
    Upload a video to transcribe, enhance, and translate it to Chinese.
    The app will generate both English and Chinese subtitles.
    """)

    # Language selection
    st.subheader("Target Language")
    target_lang = st.selectbox(
        "Select target language",
        ["Chinese"],
        index=0
    )

    # File upload
    st.subheader("Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file:
        # Display file size
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Convert to MB
        st.info(f"File size: {file_size:.2f} MB")

        # Processing options
        st.subheader("Processing Options")
        col1, col2 = st.columns(2)
        
        with col1:
            enhance_srt = st.checkbox(
                "Enhance SRT with AI",
                help="Use Qwen AI to improve subtitle quality and readability"
            )
        
        with col2:
            create_subtitled_video = st.checkbox(
                "Create subtitled video",
                help="Generate a video with embedded subtitles"
            )

        # Process button
        if st.button("Process Video"):
            with st.spinner("Processing video... This may take a few minutes."):
                try:
                    # Prepare files for upload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    data = {
                        "create_subtitled_video": create_subtitled_video,
                        "enhance_srt": enhance_srt
                    }

                    # Make API request
                    response = requests.post(
                        f"{API_URL}/process_video",
                        files=files,
                        data=data
                    )

                    if response.status_code == 200:
                        st.success("Video processed successfully!")

                        # Handle subtitled video response
                        if create_subtitled_video:
                            # Save the video file
                            video_data = response.content
                            video_filename = f"subtitled_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                            
                            with open(video_filename, "wb") as f:
                                f.write(video_data)
                            
                            # Get SRT content from headers
                            original_srt = response.headers.get("X-Original-SRT", "")
                            translated_srt = response.headers.get("X-Translated-SRT", "")

                            # Download buttons for video and SRTs
                            st.download_button(
                                "Download Subtitled Video",
                                video_data,
                                file_name=video_filename,
                                mime="video/mp4"
                            )
                        else:
                            # Handle JSON response
                            result = response.json()
                            original_srt = result["original_srt"]
                            translated_srt = result["translated_srt"]

                        # Display processing time
                        if "processing_time" in response.json():
                            st.info(f"Processing time: {response.json()['processing_time']:.2f} seconds")

                        # Display and download SRTs
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Original SRT")
                            st.text_area("", original_srt, height=300)
                            st.download_button(
                                "Download Original SRT",
                                original_srt,
                                file_name="original.srt",
                                mime="text/plain"
                            )

                        with col2:
                            st.subheader("Translated SRT")
                            st.text_area("", translated_srt, height=300)
                            st.download_button(
                                "Download Translated SRT",
                                translated_srt,
                                file_name="translated.srt",
                                mime="text/plain"
                            )

                    else:
                        st.error(f"Error: {response.text}")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 