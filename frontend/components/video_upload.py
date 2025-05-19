import streamlit as st
from utils.api_client import upload_video

def render_upload_section():
    st.header("Upload Video")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
        # Display video preview
        st.video(uploaded_file)
        
        # Quality selection
        quality = st.selectbox(
            "Select Video Quality",
            ["Original", "720p", "480p", "360p"]
        )
        
        # Language selection
        language = st.selectbox(
            "Select Target Language",
            ["English", "Chinese", "French"]
        )
        
        if st.button("Process Video"):
            with st.spinner("Processing..."):
                # Call API to start processing
                response = upload_video(uploaded_file, quality, language)
                if response["success"]:
                    st.success("Video processing started!")
                    # Store video_id in session state
                    st.session_state.video_id = response["video_id"]
                else:
                    st.error("Failed to process video") 