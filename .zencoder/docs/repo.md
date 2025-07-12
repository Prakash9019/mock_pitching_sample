# AI-Powered Mock Investor Pitching Platform

## Summary
A sophisticated platform that helps founders practice investor pitches with AI-powered simulated investors. The system provides realistic investor interactions, comprehensive pitch analysis, and actionable feedback through advanced prompt engineering and LangGraph workflows.

## Structure
- **app/**: Core application package with database models and services
  - **services/**: Backend services for audio, transcription, and AI workflows
  - **models/**: Database models and schemas
- **data/**: Runtime data storage for uploads, responses, and sessions
- **static/**: Static web assets including JavaScript files
- **templates/**: HTML templates for web interface
- **main.py**: Application entry point with FastAPI and Socket.IO setup

## Language & Runtime
**Language**: Python
**Version**: 3.11
**Framework**: FastAPI v0.115.12
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- **AI & ML**: google-generativeai v0.8.5, langchain v0.3.15, langgraph v0.2.65
- **Web Framework**: fastapi v0.115.12, uvicorn v0.34.3, python-socketio v5.13.0
- **Audio Processing**: librosa v0.10.2, pydub v0.25.1, webrtcvad v2.0.10
- **Cloud Storage**: google-cloud-storage for audio conversation storage
- **Database**: pymongo v4.13.2, motor v3.7.1

## Build & Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables in .env file
# Required: GOOGLE_APPLICATION_CREDENTIALS, MONGODB_URL

# Start the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Audio Conversation System
The application features a sophisticated audio conversation system that:
- Captures user audio through WebSockets using Socket.IO
- Uses Voice Activity Detection (VAD) to detect speech
- Transcribes speech to text
- Generates AI responses through LangGraph workflows
- Converts AI responses to speech using TTS
- Stores both user and AI audio for later analysis
- Combines audio into a complete conversation with proper alternating pattern
- Uploads conversations to Google Cloud Storage
- Returns audio URL in the session analysis response

## Key Components

### Audio Storage Service
The `AudioConversationStorage` class in `app/services/audio_conversation_storage.py`:
- Manages recording sessions with start/stop functionality
- Stores user and AI audio segments with timestamps
- Combines segments into a complete conversation
- Uploads to Google Cloud Storage with signed URLs

### WebSocket Handler
The `AudioWebSocketHandler` in `app/services/audio_websocket_handler.py`:
- Manages real-time audio streaming via Socket.IO
- Processes audio through VAD for speech detection
- Triggers transcription when speech is detected
- Generates TTS for AI responses
- Maintains conversation state and turn-taking

### API Endpoints
Key endpoints include:
- `/api/pitch/start`: Start a new pitch session
- `/api/pitch/message/{session_id}`: Send a text message
- `/api/pitch/end/{session_id}`: End session, generate analysis, and return audio URL

## Frontend Integration
The application provides comprehensive frontend integration examples for React, Vue.js, and Angular with:
- WebSocket connection for real-time audio streaming
- Audio recording and processing
- Speech-to-text and text-to-speech handling
- Session management and analysis display

## Audio Conversation Flow
1. User audio is captured and streamed to server via WebSockets
2. Server processes audio using VAD to detect speech
3. When speech is detected, audio is saved and transcribed
4. AI generates response through LangGraph workflow
5. Response is converted to speech and sent back to client
6. Both user and AI audio are stored with proper timestamps
7. When session ends, audio is combined and uploaded to cloud storage
8. Audio URL is included in the final analysis response