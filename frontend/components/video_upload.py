import streamlit as st
import tempfile
import os
import ffmpeg
import subprocess
from utils.api_client import upload_video

def check_ffmpeg():
    """
    Check if ffmpeg is properly installed and accessible
    """
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        return False
    except FileNotFoundError:
        return False

def has_audio_stream(video_file):
    """
    Check if the video file has an audio stream
    """
    try:
        probe = ffmpeg.probe(video_file)
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        return len(audio_streams) > 0
    except ffmpeg.Error as e:
        st.error(f"Error checking audio stream: {e.stderr.decode() if e.stderr else str(e)}")
        return False

def format_file_size(size_bytes):
    """
    Format file size in human readable format
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def extract_audio(video_file):
    """
    Extract audio from video file using ffmpeg
    Returns path to temporary audio file
    """
    if not check_ffmpeg():
        st.error("FFmpeg is not properly installed or not in system PATH")
        return None

    # Check if video has audio
    if not has_audio_stream(video_file):
        st.error("This video file does not contain any audio. Please upload a video with audio.")
        return None

    # Create temporary file for audio
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_audio.close()
    
    try:
        # Extract audio using ffmpeg with detailed error logging
        stream = ffmpeg.input(video_file)
        stream = ffmpeg.output(
            stream,
            temp_audio.name,
            acodec='pcm_s16le',  # 16-bit PCM
            ac=1,                # Mono audio
            ar='16k',           # 16kHz sample rate
            loglevel='info'     # More detailed logging
        )
        
        # Run ffmpeg with error capture
        try:
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            st.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            return None
            
        return temp_audio.name
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return None

def render_upload_section():
    """
    Render video upload section with simplified interface
    """
    st.header("Upload Video")
    
    # Check FFmpeg installation
    if not check_ffmpeg():
        st.error("FFmpeg is not properly installed or not in system PATH. Please ensure FFmpeg is installed and accessible.")
        return
    
    # File upload with increased size limit (2GB)
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV. Make sure your video contains audio. Maximum file size: 2GB"
    )
    
    if uploaded_file:
        # Display file size
        file_size = len(uploaded_file.getvalue())
        st.info(f"File size: {format_file_size(file_size)}")
    
    # Quality selection
    quality = st.selectbox(
        "Select Quality",
        ["high", "medium", "low"],
        index=1,
        help="Higher quality means longer processing time"
    )
    
    # Language selection
    language = st.selectbox(
        "Select Language",
        ["en", "fr", "es", "de", "it"],
        format_func=lambda x: {
            "en": "English",
            "fr": "French",
            "es": "Spanish",
            "de": "German",
            "it": "Italian"
        }[x],
        help="Select the language for transcription"
    )
    
    # Process button
    if uploaded_file and st.button("Process Video", type="primary"):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Save uploaded file
            status_text.text("Saving video file...")
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            temp_video.write(uploaded_file.getvalue())
            temp_video.close()
            progress_bar.progress(20)
            
            # Step 2: Extract audio
            status_text.text("Extracting audio...")
            audio_path = extract_audio(temp_video.name)
            if not audio_path:
                return
            progress_bar.progress(40)
            
            # Step 3: Upload audio
            status_text.text("Uploading audio...")
            with open(audio_path, 'rb') as audio_file:
                result = upload_video(audio_file, quality, language)
            progress_bar.progress(80)
            
            # Step 4: Cleanup and finish
            status_text.text("Cleaning up...")
            os.unlink(temp_video.name)
            os.unlink(audio_path)
            progress_bar.progress(100)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Audio uploaded successfully!")
                # Store video_id in session state for result tab
                st.session_state.video_id = result.get("video_id")
                # Switch to results tab
                st.session_state.active_tab = "Results"
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Cleanup in case of error
            if 'temp_video' in locals():
                os.unlink(temp_video.name)
            if 'audio_path' in locals() and audio_path:
                os.unlink(audio_path) 