# AI-Powered Mock Investor Pitching Platform

## Summary
A sophisticated platform that helps founders practice investor pitches with AI-powered simulated investors. The system provides realistic investor interactions, comprehensive pitch analysis, and actionable feedback through advanced prompt engineering and LangGraph workflows.

## Structure
- **app/**: Core application package with database models and services
- **data/**: Runtime data storage for uploads, responses, and sessions
- **static/**: Static web assets including JavaScript files
- **templates/**: HTML templates for web interface
- **Dockerfile**: Multi-stage container configuration
- **main.py**: Application entry point with FastAPI and Socket.IO setup

## Language & Runtime
**Language**: Python
**Version**: 3.11
**Framework**: FastAPI v0.115.12
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- **AI & ML**: google-generativeai v0.8.5, langchain v0.3.15, langgraph v0.2.65, openai-whisper v20240930
- **Web Framework**: fastapi v0.115.12, uvicorn v0.34.3, python-socketio v5.13.0
- **Audio Processing**: librosa v0.10.2, pydub v0.25.1, noisereduce v3.0.3, webrtcvad v2.0.10
- **Video Analysis**: opencv-python v4.10.0.84, mediapipe v0.10.20, fer v22.5.1, tensorflow v2.19.0
- **Database**: pymongo v4.13.2, motor v3.7.1

## Build & Installation
```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables in .env file
# Required: GEMINI_API_KEY, OPENAI_API_KEY, MONGODB_URL

# Start the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000  # Development
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4  # Production
```

## Docker
**Dockerfile**: Multi-stage build with Python 3.11
**Base Image**: python:3.11-slim
**Exposed Port**: 8080
**Run Command**:
```bash
docker build -t ai-mock-investor-pitch .
docker run -p 8080:8080 -e GEMINI_API_KEY=your_key -e OPENAI_API_KEY=your_key ai-mock-investor-pitch
```

## Main Components
**API Server**: FastAPI with Socket.IO integration for real-time communication
**AI Engine**: LangGraph workflow with Google Gemini model for investor simulation
**Audio Processing**: Whisper transcription and Google Cloud TTS for voice synthesis
**Video Analysis**: MediaPipe, FER, and CVZone for posture and expression analysis
**Database**: MongoDB for session storage and analytics

## Testing
**Framework**: Python's built-in unittest/asyncio
**Test Files**: test_session_report.py
**Run Command**:
```bash
python test_session_report.py
```

## Key Features
- Three distinct AI investor personas with different questioning styles
- 9-stage structured pitch evaluation process
- Real-time audio processing with noise reduction
- Video analysis for posture and expression feedback
- Comprehensive 14-category evaluation system
- MongoDB integration for session storage and analytics