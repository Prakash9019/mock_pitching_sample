# 🚀 AI-Powered Mock Investor Pitching Platform

An interactive platform that helps founders practice their investor pitches with AI-powered simulated investors. The application processes audio input, transcribes it, generates realistic investor responses using AI, and converts those responses to natural-sounding speech.

## 🌟 Key Features

- **Real-time Audio Processing**: Supports both WAV and MP3 audio inputs
- **Advanced Speech Recognition**: Powered by OpenAI's Whisper for accurate transcription
- **AI-Powered Investor Simulation**: Utilizes Google's Gemini for generating realistic investment questions and responses
- **Natural-Sounding Voice**: Multiple TTS options including ElevenLabs and Google Cloud TTS
- **Interactive Sessions**: Maintains conversation context throughout the pitch practice
- **Multiple Investor Personas**: Choose from different investor types (skeptical, technical, friendly)
- **Session Management**: Saves conversation history for review and improvement
- **WebSocket Support**: Real-time bidirectional communication for seamless interaction

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)
- API Keys for:
  - Google Gemini (required)
  - OpenAI (for Whisper transcription)
  - ElevenLabs or Google Cloud TTS (for voice output)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mock-pitching.git
   cd mock-pitching
   ```

2. **Set up a virtual environment**:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   # Required
   GEMINI_API_KEY=your_gemini_api_key
   
   # For speech-to-text (optional)
   OPENAI_API_KEY=your_openai_api_key
   
   # For text-to-speech (choose one)
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   # OR
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/google-credentials.json
   ```

### Running the Application

1. **Start the server**:
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Access the web interface**:
   Open your browser and navigate to `http://localhost:8000`

3. **For development with auto-reload**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

## 🎯 Features in Detail

### Investor Personas

- **Skeptical Investor**: Challenges assumptions and probes for weaknesses
- **Technical Investor**: Dives deep into technical details and implementation
- **Friendly Investor**: Provides constructive feedback and encouragement

### Conversation Flow
1. Record or upload your pitch audio
2. The system transcribes your pitch
3. AI generates context-aware investor questions
4. Listen to the investor's response
5. Continue the conversation naturally

### Session Management
- All conversations are saved with unique session IDs
- Review past sessions with full transcripts
- Download audio responses for later review

## 💻 API Documentation

### REST API Endpoints

#### `POST /pitch`
Submit an audio pitch for processing and receive an AI investor response.

**Request (multipart/form-data):**
- `audio_file` (required): Audio file in WAV or MP3 format
- `investor_persona` (optional, default="skeptical"): Type of investor (skeptical, technical, friendly)

**Success Response (200 OK):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-06-14T06:30:00.000Z",
  "founder_transcript": "Our startup is building an AI-powered...",
  "investor_response": "That's interesting. Could you tell me more about...",
  "response_audio_url": "/download/550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid input or missing required fields
- `413 Payload Too Large`: Audio file exceeds size limit
- `500 Internal Server Error`: Server-side processing error

#### `GET /download/{session_id}`
Download the audio response for a specific session.

**Path Parameters:**
- `session_id` (required): The unique identifier for the session

**Responses:**
- `200 OK`: Returns the audio file (MP3 format)
- `404 Not Found`: Session not found
- `410 Gone`: Session audio has expired

#### `GET /sessions/{session_id}`
Retrieve the details of a specific pitch session.

**Path Parameters:**
- `session_id` (required): The unique identifier for the session

**Success Response (200 OK):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-06-14T06:30:00.000Z",
  "investor_persona": "skeptical",
  "conversation": [
    {
      "role": "founder",
      "content": "Our startup is building...",
      "timestamp": "2025-06-14T06:30:05.123Z"
    },
    {
      "role": "investor",
      "content": "That's interesting. Could you...",
      "timestamp": "2025-06-14T06:30:10.456Z"
    }
  ]
}
```

### WebSocket Interface

Connect to `ws://localhost:8000/ws` for real-time bidirectional communication.

**Events:**
- `connect`: Connection established
- `audio_chunk`: Send/receive audio chunks (base64 encoded)
- `transcript`: Receive real-time transcription
- `response`: Receive AI-generated responses
- `error`: Error notifications
- `disconnect`: Connection terminated

**Example Usage:**
```javascript
const socket = new WebSocket('ws://localhost:8000/ws');

socket.onopen = () => {
  console.log('Connected to WebSocket');
  // Start sending audio chunks
};

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  switch (data.type) {
    case 'transcript':
      console.log('Transcript:', data.text);
      break;
    case 'response':
      console.log('AI Response:', data.text);
      playAudio(data.audio);
      break;
    case 'error':
      console.error('Error:', data.message);
      break;
  }
};
```

## 🧪 Testing

### Automated Tests

Run the test suite with:

```bash
pytest tests/
```

### Manual API Testing

1. **Using cURL**:
   ```bash
   # Submit a pitch
   curl -X POST http://localhost:8000/pitch \
     -F "audio_file=@path/to/your/pitch.mp3" \
     -F "investor_persona=technical"
   
   # Get session details
   curl http://localhost:8000/sessions/YOUR_SESSION_ID
   
   # Download response audio
   curl -O http://localhost:8000/download/YOUR_SESSION_ID
   ```

2. **Using Python Script**:
   ```python
   import requests
   
   # Submit a pitch
   with open('pitch.mp3', 'rb') as f:
       response = requests.post(
           'http://localhost:8000/pitch',
           files={'audio_file': f},
           data={'investor_persona': 'friendly'}
       )
   print(response.json())
   ```

## 🏗️ Project Structure

```
mock-pitching/
├── app/
│   ├── __init__.py
│   ├── main.py                # Main FastAPI application and WebSocket handlers
│   ├── services/              # Core service modules
│   │   ├── __init__.py
│   │   ├── ai_response.py     # AI response generation using Google Gemini
│   │   ├── text_to_speech.py  # TTS integration (ElevenLabs/Google TTS)
│   │   └── transcription.py   # Speech-to-text with Whisper
│   └── data/                  # Runtime data (not versioned)
│       ├── uploads/           # User-uploaded audio files
│       ├── responses/         # Generated audio responses
│       └── sessions/          # Session data and conversation history
├── tests/                     # Test files
│   ├── __init__.py
│   ├── test_api.py            # API test cases
│   └── conftest.py            # Test fixtures
├── .env.example               # Example environment variables
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                # Git ignore file
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key for AI responses |
| `OPENAI_API_KEY` | ❌ | OpenAI API key for Whisper STT |
| `ELEVENLABS_API_KEY` | ❌ | ElevenLabs API key for TTS (or use Google TTS) |
| `GOOGLE_APPLICATION_CREDENTIALS` | ❌ | Path to Google Cloud credentials JSON |
| `MAX_AUDIO_SIZE_MB` | ❌ | Max upload size in MB (default: 10) |
| `LOG_LEVEL` | ❌ | Logging level (default: INFO) |

## 🚀 Deployment

### Production

1. Set up a production WSGI server (Gunicorn with Uvicorn workers recommended):
   ```bash
   pip install gunicorn uvloop httptools
   gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 app.main:app
   ```

2. Set up a reverse proxy (Nginx example):
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;

       location / {
           proxy_pass http://localhost:8000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
       }
   }
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the awesome web framework
- [Google Gemini](https://ai.google.dev/) for AI conversation capabilities
- [OpenAI Whisper](https://openai.com/research/whisper) for speech recognition
- [ElevenLabs](https://elevenlabs.io/) for high-quality text-to-speech

## 📈 Future Enhancements

- [ ] User authentication and profiles
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Custom investor persona creation
- [ ] Integration with pitch deck analysis
- [ ] Real-time feedback during pitches
- [ ] Mobile application
- [ ] Video call simulation