# Project Setup Guide

## Prerequisites

1. **Python Installation**
   - Install Python 3.8 or higher
   - Verify installation:
     ```bash
     python --version
     pip --version
     ```

2. **FFmpeg Installation**
   - Windows:
     ```bash
     # Using chocolatey
     choco install ffmpeg
     
     # Or download from https://ffmpeg.org/download.html
     ```
   - Linux:
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```
   - macOS:
     ```bash
     brew install ffmpeg
     ```

## Environment Setup

1. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   # Upgrade pip
   python -m pip install --upgrade pip

   # Install requirements
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   Create a `.env` file in the project root:
   ```env
   # API Configuration
   API_BASE_URL=http://localhost:8000
   
   # Security
   SECRET_KEY=your-secret-key-here
   
   # Optional: API Keys
   OPENAI_API_KEY=your-openai-key
   QWEN_API_KEY=your-qwen-key
   ```

## Development Setup

1. **Code Formatting**
   ```bash
   # Install pre-commit hooks
   pre-commit install

   # Format code
   black .
   isort .
   ```

2. **Running the Application**
   ```bash
   # Start the Streamlit app
   streamlit run app.py

   # Start the API server (in a separate terminal)
   uvicorn api.main:app --reload
   ```

3. **Testing**
   ```bash
   # Run tests
   pytest

   # Run tests with coverage
   pytest --cov=.
   ```

## Project Structure Setup

1. **Create Directories**
   ```bash
   mkdir -p frontend/components
   mkdir -p frontend/utils
   mkdir -p frontend/config
   mkdir -p tests
   ```

2. **Initialize Python Packages**
   ```bash
   touch frontend/components/__init__.py
   touch frontend/utils/__init__.py
   touch frontend/config/__init__.py
   ```

## Verification Steps

1. **Check Installation**
   ```bash
   # Verify Streamlit
   streamlit --version

   # Verify FFmpeg
   ffmpeg -version

   # Verify Python packages
   pip list
   ```

2. **Test Basic Functionality**
   ```bash
   # Run the Streamlit app
   streamlit run app.py
   ```
   - Open http://localhost:8501 in your browser
   - Verify the interface loads correctly

## Troubleshooting

1. **Common Issues**
   - If FFmpeg is not found:
     - Ensure FFmpeg is in your system PATH
     - Restart your terminal after installation
   
   - If packages fail to install:
     - Try updating pip: `python -m pip install --upgrade pip`
     - Check Python version compatibility
     - Try installing packages one by one

2. **Virtual Environment Issues**
   - If activation fails:
     - Windows: Try using `.\venv\Scripts\activate.bat`
     - Linux/macOS: Ensure execute permissions: `chmod +x venv/bin/activate`

3. **Port Conflicts**
   - If port 8501 is in use:
     - Change Streamlit port: `streamlit run app.py --server.port 8502`
     - Or kill the process using the port

## Next Steps

1. Review the project structure
2. Set up your development environment
3. Start implementing the components
4. Run tests to verify functionality

## Notes
- Keep your virtual environment activated while working
- Regularly update dependencies
- Follow the coding standards
- Document any issues or solutions 