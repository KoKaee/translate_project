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
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            input_path = os.path.join(temp_dir, "input.mp4")
            with open(input_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            # Process video based on quality
            output_path = os.path.join(temp_dir, "output.mp4")
            
            # FFmpeg command based on quality
            if quality == "high":
                ffmpeg_cmd = [
                    'ffmpeg', '-i', input_path,
                    '-c:v', 'libx264', '-crf', '18',  # High quality
                    '-c:a', 'aac', '-b:a', '192k',    # High quality audio
                    output_path
                ]
            elif quality == "medium":
                ffmpeg_cmd = [
                    'ffmpeg', '-i', input_path,
                    '-c:v', 'libx264', '-crf', '23',  # Medium quality
                    '-c:a', 'aac', '-b:a', '128k',    # Medium quality audio
                    output_path
                ]
            else:  # low
                ffmpeg_cmd = [
                    'ffmpeg', '-i', input_path,
                    '-c:v', 'libx264', '-crf', '28',  # Lower quality
                    '-c:a', 'aac', '-b:a', '96k',     # Lower quality audio
                    output_path
                ]
            
            # Run FFmpeg
            process = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {process.stderr}")
                return None, f"Error processing video: {process.stderr}"
            
            # Read the processed file
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

        # Handle the video file response
        # Create a BytesIO object to hold the video data
        from io import BytesIO
        video_data = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            video_data.write(chunk)
        video_data.seek(0) # Go back to the start of the BytesIO object

        return video_data.getvalue(), None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error in comment_video: {str(e)}")
        return None, f"Error commenting video: {str(e)}"

def main():
    st.set_page_config(
        page_title="Video Processor",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Video Processor")
    
    # Check FFmpeg installation
    if not check_ffmpeg():
        st.error("FFmpeg is not installed. Please install FFmpeg to use this application.")
        st.stop()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to process"
    )
    
    if uploaded_file is not None:
        # Show video info
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        st.info(f"File size: {file_size:.2f} MB")
        
        # Create tabs for different operations
        tab1, tab2 = st.tabs(["Get SRT", "Comment Video"])
        
        with tab1:
            # Transcribe button
            if st.button("Transcribe Video"):
                with st.spinner("Transcribing video..."):
                    start_time = datetime.now()
                    
                    # Transcribe the video
                    result, error = transcribe_video(uploaded_file)
                    
                    if error:
                        st.error(error)
                    elif result:
                        # Calculate processing time
                        process_time = (datetime.now() - start_time).total_seconds()
                        
                        # Show success message
                        st.success(f"Video transcribed successfully in {process_time:.2f} seconds!")
                        
                        # Display transcription
                        st.subheader("Transcription")
                        st.write(result.get("text", "No text available"))
                        
                        # Display segments with timestamps
                        segments = result.get("segments")
                        if segments:
                            st.subheader("Segments")
                            for segment in segments:
                                st.write(f"[{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}")

                        # Add download button for SRT
                        srt_content = result.get("srt")
                        if srt_content:
                            st.download_button(
                                label="Download SRT file",
                                data=srt_content,
                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}.srt",
                                mime="text/plain"
                            )

        with tab2:
            st.subheader("Overlay Subtitles on Video")
            
            # SRT file upload
            uploaded_srt_file = st.file_uploader(
                "Choose an SRT file",
                type=['srt'],
                help="Upload an SRT file to overlay on the video"
            )

            # Comment Video button
            if st.button("Create Commented Video", disabled=uploaded_srt_file is None):
                 if uploaded_file is not None and uploaded_srt_file is not None:
                    with st.spinner("Creating commented video..."):
                        start_time = datetime.now()

                        # Call backend endpoint to comment video
                        commented_video_data, error = comment_video(uploaded_file, uploaded_srt_file)

                        if error:
                            st.error(error)
                        elif commented_video_data:
                            process_time = (datetime.now() - start_time).total_seconds()
                            st.success(f"Commented video created successfully in {process_time:.2f} seconds!")

                            # Add download button for commented video
                            st.download_button(
                                label="Download Commented Video",
                                data=commented_video_data,
                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_commented.mp4",
                                mime="video/mp4"
                            )

if __name__ == "__main__":
    main() 