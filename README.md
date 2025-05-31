# Project Title
## Overview
This project is designed to handle video transcription and translation, converting video files into SRT subtitle files and translating them from English to French. The application is divided into backend and frontend components, each serving specific roles in the overall functionality.

## Project Structure
```
project-root/
├── backend/
│   └── app/
│       ├── main.py
│       └── translator.py
├── frontend/
│   ├── app.py
│   ├── components/
│   │   ├── metadata_display.py
│   │   ├── progress_bar.py
│   │   ├── result_preview.py
│   │   └── video_upload.py
│   └── utils/
│       └── api_client.py
├── srt_files/
├── enhanced_srt_files/
├── translated_srt_files/
└── translator/
    ├── model/
    │   └── mistral-7b-instruct-v0.2.Q4_K_M.
    ├── subtitle_enhancer_ai.py
    ├── subtitle_splitter.py
    └── translator.py
```
## Backend
### main.py
- Purpose : Handles API requests for transcription and translation.
- Key Functions :
  - load_models() : Loads the Whisper and translation models.
  - /transcribe : Endpoint for transcribing video files into SRT format.
  - /translate_srt : Endpoint for translating SRT files from English to French.
### translator.py
- Purpose : Contains functions for translating text using the M2M100 model.
- Key Functions :
  - translate_text() : Translates given text from source to target language.
  - translate_srt_file() : Translates an entire SRT file.
## Frontend
### app.py
- Purpose : Provides a user interface for uploading videos and SRT files, and downloading the translated SRT files.
- Key Components :
  - video_upload.py : Handles video file uploads.
  - result_preview.py : Displays the results of transcription and translation.
### utils/api_client.py
- Purpose : Manages API requests to the backend.
## Usage
1. Setup : Ensure all dependencies are installed as per requirements.txt in both backend and frontend directories.
2. Running the Backend :
   - Navigate to the backend directory.
   - Run the FastAPI server using uvicorn app.main:app --reload .
3. Running the Frontend :
   - Navigate to the frontend directory.
   - Start the Streamlit app using streamlit run app.py .
4. Using the Application :
   - Upload a video file to transcribe and translate.
   - Download the translated SRT file.
## Notes
- Ensure the models are correctly loaded and paths are set for file storage.
- Check logs for any errors during processing.


### Dear Teacher...
____
 
***We tried our best to completed the project as per the requirements. However, we encountered some challenges during the integration of the subtitle splitting and enhancement functionalities into the Streamlit application.***

***The files located in the translator folder, including subtitle_enhancer_ai.py and subtitle_splitter.py , are functioning correctly when executed locally. However, while integrating these functionalities into the Streamlit application, some challenges were encountered, particularly with the SRT splitting and enhancement processes. Despite these issues in the Streamlit environment, the scripts perform as expected in a local setup.***

** Anyways, we had fun doing this project. Thanks ! **
