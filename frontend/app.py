import streamlit as st
import os
import tempfile
import subprocess
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frontend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def process_video(video_file, quality="medium"):
    """Process video file using FFmpeg"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.mp4")
            with open(input_path, "wb") as f:
                f.write(video_file.getbuffer())
            output_path = os.path.join(temp_dir, "output.mp4")
            if quality == "high":
                ffmpeg_cmd = ['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-crf', '18', '-c:a', 'aac', '-b:a', '192k', output_path]
            elif quality == "medium":
                ffmpeg_cmd = ['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', '-b:a', '128k', output_path]
            else:
                ffmpeg_cmd = ['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-crf', '28', '-c:a', 'aac', '-b:a', '96k', output_path]
            process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {process.stderr}")
                return None, f"Error processing video: {process.stderr}"
            with open(output_path, "rb") as f:
                processed_video = f.read()
            return processed_video, None
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}")
        return None, f"Error processing video: {str(e)}"


def transcribe_video(video_file):
    """Send video to backend for transcription"""
    try:
        files = {"file": (video_file.name, video_file, "video/mp4")}
        response = requests.post(f"{API_URL}/transcribe", files=files)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in transcribe_video: {str(e)}")
        return None, f"Error transcribing video: {str(e)}"


def comment_video(video_file, srt_file):
    """Send video and SRT to backend for commenting."""
    try:
        files = {
            "video_file": (video_file.name, video_file, "video/mp4"),
            "srt_file": (srt_file.name, srt_file, "text/plain"),
        }
        response = requests.post(f"{API_URL}/comment_video", files=files, stream=True)
        response.raise_for_status()
        from io import BytesIO
        video_data = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            video_data.write(chunk)
        video_data.seek(0)
        return video_data.getvalue(), None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in comment_video: {str(e)}")
        return None, f"Error commenting video: {str(e)}"


def translate_srt(srt_file):
    """Send SRT file to backend for translation"""
    try:
        files = {"srt_file": (srt_file.name, srt_file, "text/plain")}
        response = requests.post(f"{API_URL}/translate_srt", files=files, stream=True)
        response.raise_for_status()
        return response.content, None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in translate_srt: {str(e)}")
        return None, f"Error translating SRT: {str(e)}"


def main():
    st.set_page_config(
        page_title="Video Processor",
        page_icon="🎥",
        layout="wide"
    )
    st.title("🎥 Video Processor")
    if not check_ffmpeg():
        st.error("FFmpeg is not installed. Please install FFmpeg to use this application.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Get SRT", "Comment Video", "Translate SRT"])

    with tab1:
        st.subheader("🎙️ Transcribe Video to SRT")
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"], key="srt_upload")
        if uploaded_file:
            st.info(f"File size: {uploaded_file.size / (1024 * 1024):.2f} MB")
            if st.button("Transcribe Video"):
                with st.spinner("Transcribing video..."):
                    start = datetime.now()
                    result, error = transcribe_video(uploaded_file)
                    if error:
                        st.error(error)
                    elif result:
                        st.success(f"Transcribed in {(datetime.now()-start).total_seconds():.2f}s")
                        srt_content = result.get("srt")
                        if srt_content:
                            filename = f"{os.path.splitext(uploaded_file.name)[0]}.srt"
                            st.download_button("Download SRT file", srt_content, file_name=filename, mime="text/plain")
                        else:
                            st.warning("No SRT content received.")
        else:
            st.info("Upload a video to enable transcription.")

    with tab2:
        st.subheader("💬 Overlay Subtitles on Video")
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"], key="comment_video")
        uploaded_srt = st.file_uploader("Upload an SRT file", type=["srt"], key="comment_srt")
        if st.button("Create Commented Video", disabled=not uploaded_video or not uploaded_srt):
            if uploaded_video and uploaded_srt:
                with st.spinner("Creating commented video..."):
                    start = datetime.now()
                    video_data, error = comment_video(uploaded_video, uploaded_srt)
                    if error:
                        st.error(error)
                    elif video_data:
                        st.success(f"Created in {(datetime.now()-start).total_seconds():.2f}s")
                        st.download_button("Download Commented Video",
                                           video_data,
                                           file_name=f"{os.path.splitext(uploaded_video.name)[0]}_commented.mp4",
                                           mime="video/mp4")
            else:
                st.warning("Please upload both a video and an SRT file.")

    with tab3:
        st.subheader("🌍 Translate SRT to French")
        uploaded_srt_tr = st.file_uploader("Upload an English .srt file", type=["srt"], key="translate_srt")
        if st.button("Translate SRT", disabled=not uploaded_srt_tr):
            if uploaded_srt_tr:
                with st.spinner("Translating SRT..."):
                    translated_data, error = translate_srt(uploaded_srt_tr)
                    if error:
                        st.error(error)
                    elif translated_data:
                        out_name = f"{os.path.splitext(uploaded_srt_tr.name)[0]}_fr.srt"
                        st.success("Translation complete!")
                        st.download_button("Download Translated SRT",
                                           translated_data,
                                           file_name=out_name,
                                           mime="text/plain")
            else:
                st.warning("Please upload an SRT file to translate.")

if __name__ == "__main__":
    main()
