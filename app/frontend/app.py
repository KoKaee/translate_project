import streamlit as st
import os
import tempfile
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

def main():
    st.set_page_config(
        page_title="Video Translation App",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Video Translation App")
    
    # Language selection
    target_language = st.selectbox(
        "Select target language",
        ["fr", "es", "de", "it", "pt", "ru", "ja", "ko", "zh-cn"],
        format_func=lambda x: {
            "fr": "French",
            "es": "Spanish",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh-cn": "Chinese (Simplified)"
        }[x]
    )

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to process"
    )

    if uploaded_file is None:
        st.info("Please upload a video file to get started.")
        return

    # Show video info
    file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
    st.info(f"File size: {file_size:.2f} MB")
    
    # Process video button
    if st.button("Process Video"):
        with st.spinner("Processing video..."):
            start_time = datetime.now()
            
            # Prepare the request
            files = {'file': uploaded_file}
            data = {
                'target_lang': target_language,
                'create_subtitled_video': False  # We'll handle this separately
            }
            
            try:
                # Send request to backend
                response = requests.post(f"{API_URL}/process_video", files=files, data=data)
                response.raise_for_status()
                result = response.json()
                
                # Calculate processing time
                process_time = (datetime.now() - start_time).total_seconds()
                st.success(f"Video processed successfully in {process_time:.2f} seconds!")
                
                # Display original and translated SRT
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original SRT")
                    st.text_area("Original subtitles", result['original_srt'], height=300)
                    st.download_button(
                        label="Download Original SRT",
                        data=result['original_srt'],
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_original.srt",
                        mime="text/plain"
                    )
                
                with col2:
                    st.subheader("Translated SRT")
                    st.text_area("Translated subtitles", result['translated_srt'], height=300)
                    st.download_button(
                        label="Download Translated SRT",
                        data=result['translated_srt'],
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_{target_language}.srt",
                        mime="text/plain"
                    )
                
                # Option to create subtitled video
                if st.button("Create Subtitled Video"):
                    with st.spinner("Creating subtitled video..."):
                        # Request subtitled video
                        data['create_subtitled_video'] = True
                        response = requests.post(f"{API_URL}/process_video", files=files, data=data)
                        response.raise_for_status()
                        
                        # Download button for the subtitled video
                        st.download_button(
                            label="Download Subtitled Video",
                            data=response.content,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_subtitled.mp4",
                            mime="video/mp4"
                        )
                
            except requests.exceptions.RequestException as e:
                st.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main() 