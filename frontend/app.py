import streamlit as st
from components.video_upload import render_upload_section
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
    tab1, tab2 = st.tabs(["Upload", "Results"])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_preview()

if __name__ == "__main__":
    main() 