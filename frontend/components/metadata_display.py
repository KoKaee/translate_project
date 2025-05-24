import streamlit as st

def render_metadata(video_info):
    """
    Render video metadata section
    
    Args:
        video_info: Dictionary containing video metadata
    """
    st.subheader("Video Information")
    
    if video_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Duration:**", video_info.get("duration", "N/A"))
            st.write("**Resolution:**", video_info.get("resolution", "N/A"))
            st.write("**Format:**", video_info.get("format", "N/A"))
        
        with col2:
            st.write("**Size:**", video_info.get("size", "N/A"))
            st.write("**FPS:**", video_info.get("fps", "N/A"))
            st.write("**Codec:**", video_info.get("codec", "N/A"))
    else:
        st.info("No video information available") 