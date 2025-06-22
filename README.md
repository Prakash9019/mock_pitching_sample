# 🚀 AI-Powered Mock Investor Pitching Platform

A sophisticated, professional-grade platform that helps founders practice their investor pitches with AI-powered simulated investors. Built with advanced prompt engineering and LangGraph workflows, this platform provides realistic investor interactions, comprehensive pitch analysis, and actionable feedback to prepare founders for real investor meetings.

## 🌟 Key Features

### 🎯 **Advanced AI Investor Simulation**
- **Sophisticated Investor Personas**: Three distinct investor types with detailed psychological profiles and decision frameworks
- **LangGraph Workflow Engine**: Structured 9-stage pitch evaluation process
- **Context-Aware Conversations**: Maintains conversation history and builds progressive understanding
- **Professional-Grade Analysis**: Investor-quality evaluation with detailed scoring and feedback

### 🎙️ **Audio & Communication**
- **Real-time Audio Processing**: Supports WAV and MP3 audio inputs with noise reduction
- **Advanced Speech Recognition**: Powered by OpenAI's Whisper for accurate transcription
- **Natural Voice Synthesis**: High-quality TTS with persona-specific voice characteristics
- **WebSocket Integration**: Real-time bidirectional communication for seamless interaction

### 📊 **Comprehensive Analytics**
- **14-Category Evaluation System**: 10 content categories + 4 communication metrics
- **Detailed Scoring Framework**: 100-point scale with investor-grade assessment criteria
- **Performance Tracking**: Session analytics, progress monitoring, and improvement insights
- **Exportable Reports**: Professional pitch analysis reports for review and sharing

### 🗄️ **Enterprise Features**
- **MongoDB Integration**: Persistent session storage and analytics
- **Session Management**: Complete conversation history and replay capabilities
- **Multi-User Support**: Scalable architecture for multiple concurrent users
- **Health Monitoring**: Comprehensive health checks and system monitoring

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (3.9+ recommended)
- **MongoDB** (local installation or MongoDB Atlas)
- **FFmpeg** (for audio processing)
- **API Keys**:
  - Google Gemini API (required)
  - OpenAI API (for Whisper transcription)
  - Google Cloud TTS or ElevenLabs (for voice synthesis)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ai-mock-investor-pitch.git
   cd ai-mock-investor-pitch
   ```

2. **Set up virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory:
   ```env
   # Required - AI Services
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Required - Database
   MONGODB_URL=mongodb://localhost:27017
   DATABASE_NAME=pitch_platform
   
   # Optional - Text-to-Speech (choose one)
   GOOGLE_APPLICATION_CREDENTIALS=path/to/google-credentials.json
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   
   # Optional - Configuration
   MAX_AUDIO_SIZE_MB=25
   LOG_LEVEL=INFO
   ENVIRONMENT=development
   ```

5. **Set up MongoDB**:
   ```bash
   # Local MongoDB (if not using Atlas)
   # Windows: Download and install from mongodb.com
   # macOS: brew install mongodb-community
   # Ubuntu: sudo apt install mongodb
   
   # Start MongoDB service
   # Windows: net start MongoDB
   # macOS/Linux: sudo systemctl start mongod
   ```

### Running the Application

1. **Start the server**:
   ```bash
   # Development mode with auto-reload
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   
   # Production mode
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Access the platform**:
   - **Web Interface**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health

3. **Verify installation**:
   ```bash
   # Test API endpoint
   curl http://localhost:8000/health
   
   # Expected response:
   # {"status": "healthy", "database": "healthy", ...}
   ```

## 🎯 Platform Architecture

### 🤖 AI Investor Personas

#### **Sarah Martinez - The Skeptical Investor**
- **Background**: Senior Partner at Venture Capital, Former McKinsey Consultant
- **Approach**: Data-driven, evidence-focused, systematic evaluation
- **Questioning Style**: Challenges assumptions, demands proof points, focuses on metrics
- **Decision Framework**: Bottom-up analysis, risk-adjusted returns, concrete validation

#### **Dr. Alex Chen - The Technical Investor**
- **Background**: CTO-turned-Investor, Former Google Principal Engineer
- **Approach**: Architecture-first, innovation-focused, technical depth
- **Questioning Style**: Deep technical dives, scalability assessment, engineering excellence
- **Decision Framework**: Systems thinking, technical risk evaluation, innovation potential

#### **Michael Thompson - The Friendly Investor**
- **Background**: Angel Investor & Former Entrepreneur
- **Approach**: People-first, story-driven, empathetic evaluation
- **Questioning Style**: Vision-focused, team dynamics, founder journey
- **Decision Framework**: Founder-market fit, passion assessment, mission alignment

### 📋 9-Stage Pitch Evaluation Process

1. **Greeting & Introduction** - Rapport building and context gathering
2. **Problem & Solution** - Problem validation and solution effectiveness
3. **Target Market** - Customer segmentation and market opportunity
4. **Business Model** - Revenue strategy and unit economics
5. **Competition** - Competitive landscape and differentiation
6. **Traction** - Growth metrics and market validation
7. **Team** - Team capability and execution assessment
8. **Funding Needs** - Investment requirements and use of funds
9. **Future Plans** - Strategic vision and scaling roadmap

### 🔄 Conversation Flow

1. **Session Initialization**: Choose investor persona and start conversation
2. **Audio Input**: Record or upload pitch audio (WAV/MP3)
3. **Transcription**: Advanced speech-to-text with noise reduction
4. **AI Processing**: Context-aware response generation using LangGraph
5. **Voice Synthesis**: Natural-sounding investor responses
6. **Progressive Evaluation**: Stage-by-stage assessment and feedback
7. **Comprehensive Analysis**: Detailed scoring and improvement recommendations

## 💻 API Documentation

### 🔗 REST API Endpoints

#### **Core Pitch Processing**

##### `POST /pitch`
Submit an audio pitch for processing and receive AI investor response.

**Request (multipart/form-data):**
```
audio_file: File (WAV/MP3, max 25MB)
investor_persona: String (optional, default="skeptical")
```

**Response (200 OK):**
```json
{
  "session_id": "uuid-string",
  "timestamp": "2025-01-14T10:30:00.000Z",
  "founder_transcript": "Our startup is building...",
  "investor_response": "That's interesting. Could you tell me more about...",
  "response_audio_url": "/download/session_id"
}
```

##### `GET /download/{filename}`
Download audio response files.

**Parameters:**
- `filename`: Session ID or full filename

**Response:** MP3 audio file

#### **Session Management**

##### `GET /sessions/{session_id}`
Retrieve complete session details and conversation history.

**Response:**
```json
{
  "session_id": "uuid",
  "created_at": "timestamp",
  "investor_persona": "skeptical",
  "conversation_history": [...],
  "analytics": {...}
}
```

##### `GET /pitch-analysis/{session_id}`
Get comprehensive pitch analysis and scoring.

**Response:**
```json
{
  "overall_score": 85,
  "overall_rating": "Good",
  "category_scores": {
    "hooks_story": {"score": 78, "rating": "Good", "description": "..."},
    "problem_urgency": {"score": 82, "rating": "Good", "description": "..."}
  },
  "strengths": [...],
  "weaknesses": [...],
  "recommendations": [...]
}
```

#### **System Endpoints**

##### `GET /health`
System health check with database status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-14T10:30:00.000Z",
  "database": "healthy",
  "version": "1.0.0"
}
```

### 🔌 WebSocket Interface

Real-time communication for interactive pitch sessions.

**Connection:** `ws://localhost:8000/socket.io/`

#### **Socket.IO Events**

##### **Client → Server Events**

```javascript
// Start a new pitch session
socket.emit('start_session', {
  persona: 'skeptical',  // 'skeptical', 'technical', 'friendly'
  system: 'improved'     // Use enhanced AI system
});

// Send text message
socket.emit('message', {
  session_id: 'uuid',
  message: 'Hello, I\'m John from TechCorp...',
  generate_audio: true
});

// Send audio message
socket.emit('audio_message', {
  session_id: 'uuid',
  audio_data: 'base64-encoded-audio',
  persona: 'skeptical',
  generate_audio: true
});
```

##### **Server → Client Events**

```javascript
// Session started
socket.on('session_started', (data) => {
  console.log('Session ID:', data.session_id);
  console.log('Investor:', data.investor_name);
});

// Investor response received
socket.on('response', (data) => {
  console.log('Response:', data.response);
  console.log('Audio URL:', data.audio_url);
  console.log('Session complete:', data.session_complete);
});

// Error handling
socket.on('error', (error) => {
  console.error('Error:', error.message);
});
```

#### **Example Implementation**

```javascript
const socket = io('http://localhost:8000');

// Start session
socket.emit('start_session', {
  persona: 'skeptical',
  system: 'improved'
});

// Handle responses
socket.on('response', (data) => {
  displayMessage(data.response, 'investor');
  if (data.audio_url) {
    playAudio(data.audio_url);
  }
});

// Send message
function sendMessage(text) {
  socket.emit('message', {
    session_id: currentSessionId,
    message: text,
    generate_audio: true
  });
}
```

## 🧪 Testing & Development

### 🔍 Manual Testing

#### **API Testing with cURL**

```bash
# Health check
curl http://localhost:8000/health

# Submit a pitch
curl -X POST http://localhost:8000/pitch \
  -F "audio_file=@sample_pitch.mp3" \
  -F "investor_persona=technical"

# Get session details
curl http://localhost:8000/sessions/YOUR_SESSION_ID

# Download response audio
curl -O http://localhost:8000/download/YOUR_SESSION_ID
```

#### **Python Testing Script**

```python
import requests
import json

# Test pitch submission
def test_pitch_submission():
    with open('sample_pitch.mp3', 'rb') as f:
        response = requests.post(
            'http://localhost:8000/pitch',
            files={'audio_file': f},
            data={'investor_persona': 'skeptical'}
        )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Session ID: {data['session_id']}")
        print(f"Transcript: {data['founder_transcript']}")
        print(f"Response: {data['investor_response']}")
        return data['session_id']
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Test session retrieval
def test_session_retrieval(session_id):
    response = requests.get(f'http://localhost:8000/sessions/{session_id}')
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))

# Run tests
session_id = test_pitch_submission()
if session_id:
    test_session_retrieval(session_id)
```

### 🐛 Debugging & Logs

#### **Enable Debug Logging**

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or in .env file
LOG_LEVEL=DEBUG
```

#### **Monitor Logs**

```bash
# View application logs
tail -f logs/app.log

# Monitor MongoDB operations
tail -f logs/database.log

# Check system health
curl http://localhost:8000/health | jq
```

## 🏗️ Project Structure

```
ai-mock-investor-pitch/
├── 📁 app/                           # Core application package
│   ├── database.py                   # MongoDB connection and management
│   ├── models.py                     # Data models and schemas
│   └── 📁 services/                  # Business logic services
│       ├── intelligent_ai_agent_improved.py  # Enhanced AI agent with personas
│       ├── langgraph_workflow.py     # LangGraph workflow engine
│       ├── transcription.py          # Speech-to-text processing
│       ├── enhanced_text_to_speech.py # Advanced TTS with personas
│       └── database_service.py       # Database operations
│
├── 📁 templates/                     # HTML templates
│   ├── index.html                    # Main web interface
│   └── pitch_analysis.html          # Analysis results page
│
├── 📁 static/                        # Static web assets
│   └── 📁 js/
│       └── main.js                   # Frontend JavaScript
│
├── 📁 data/                          # Runtime data (not versioned)
│   ├── uploads/                      # User-uploaded audio files
│   ├── responses/                    # Generated audio responses
│   ├── sessions/                     # Session data (file-based backup)
│   └── sessions_workflow/            # LangGraph session states
│
├── 📄 main.py                        # FastAPI application entry point
├── 📄 requirements.txt               # Python dependencies
├── 📄 Dockerfile                     # Container configuration
├── 📄 cloudbuild.yaml               # Google Cloud Build configuration
├── 📄 .env.example                  # Environment variables template
├── 📄 README.md                     # This documentation
│
└── 📁 Documentation/                 # Additional documentation
    ├── PROMPT_IMPROVEMENTS_SUMMARY.md
    ├── DATABASE_INTEGRATION.md
    ├── FRONTEND_INTEGRATION_SUMMARY.md
    └── AI_IMPROVEMENTS_GUIDE.md
```

### 🔧 Key Components

#### **Core Services**
- **`intelligent_ai_agent_improved.py`**: Advanced AI agent with sophisticated investor personas
- **`langgraph_workflow.py`**: Structured 9-stage pitch evaluation workflow
- **`transcription.py`**: Audio processing with noise reduction and Whisper STT
- **`enhanced_text_to_speech.py`**: High-quality TTS with persona-specific voices

#### **Data Layer**
- **`database.py`**: MongoDB connection management and health monitoring
- **`models.py`**: Pydantic models for data validation and serialization
- **`database_service.py`**: Database operations and session persistence

#### **Web Interface**
- **`main.py`**: FastAPI application with REST API and WebSocket endpoints
- **`templates/`**: Jinja2 templates for web interface
- **`static/`**: Frontend assets with real-time communication

## ⚙️ Configuration

### 🔐 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| **AI Services** |
| `GEMINI_API_KEY` | ✅ | - | Google Gemini API key for AI responses |
| `OPENAI_API_KEY` | ✅ | - | OpenAI API key for Whisper transcription |
| **Database** |
| `MONGODB_URL` | ✅ | `mongodb://localhost:27017` | MongoDB connection string |
| `DATABASE_NAME` | ✅ | `pitch_platform` | Database name |
| **Text-to-Speech** |
| `GOOGLE_APPLICATION_CREDENTIALS` | ❌ | - | Path to Google Cloud credentials JSON |
| `ELEVENLABS_API_KEY` | ❌ | - | ElevenLabs API key for premium TTS |
| **Application Settings** |
| `MAX_AUDIO_SIZE_MB` | ❌ | `25` | Maximum audio upload size |
| `LOG_LEVEL` | ❌ | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `ENVIRONMENT` | ❌ | `development` | Environment mode |
| `HOST` | ❌ | `0.0.0.0` | Server host |
| `PORT` | ❌ | `8000` | Server port |

### 📋 Configuration Examples

#### **Development Configuration (.env)**
```env
# AI Services
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here

# Database
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=pitch_platform_dev

# Optional Services
GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-cloud-key.json

# Development Settings
LOG_LEVEL=DEBUG
ENVIRONMENT=development
MAX_AUDIO_SIZE_MB=25
```

#### **Production Configuration**
```env
# AI Services
GEMINI_API_KEY=prod_gemini_key
OPENAI_API_KEY=prod_openai_key

# Database (MongoDB Atlas)
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/
DATABASE_NAME=pitch_platform_prod

# Production Settings
LOG_LEVEL=INFO
ENVIRONMENT=production
MAX_AUDIO_SIZE_MB=50
HOST=0.0.0.0
PORT=8000
```

## 🚀 Deployment

### 🐳 Docker Deployment

#### **Build and Run with Docker**

```bash
# Build the image
docker build -t ai-pitch-platform .

# Run with environment variables
docker run -d \
  --name pitch-platform \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  -e MONGODB_URL=mongodb://host.docker.internal:27017 \
  ai-pitch-platform

# Check logs
docker logs pitch-platform
```

#### **Docker Compose (Recommended)**

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MONGODB_URL=mongodb://mongodb:27017
      - DATABASE_NAME=pitch_platform
    depends_on:
      - mongodb
    volumes:
      - ./data:/app/data

  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    environment:
      - MONGO_INITDB_DATABASE=pitch_platform

volumes:
  mongodb_data:
```

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f app
```

### ☁️ Cloud Deployment

#### **Google Cloud Platform**

```bash
# Deploy using Cloud Build
gcloud builds submit --config cloudbuild.yaml

# Deploy to Cloud Run
gcloud run deploy pitch-platform \
  --image gcr.io/PROJECT_ID/pitch-platform \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key,OPENAI_API_KEY=your_key
```

#### **Production Server Setup**

```bash
# Install production dependencies
pip install gunicorn uvloop httptools

# Run with Gunicorn
gunicorn -k uvicorn.workers.UvicornWorker \
  -w 4 \
  -b 0.0.0.0:8000 \
  --timeout 300 \
  --keep-alive 2 \
  main:app

# Or use systemd service
sudo systemctl enable pitch-platform
sudo systemctl start pitch-platform
```

#### **Nginx Configuration**

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    client_max_body_size 50M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # WebSocket support
    location /socket.io/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## 📊 Performance & Monitoring

### 🔍 Health Monitoring

```bash
# System health check
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health | jq '.database'

# Monitor application metrics
curl http://localhost:8000/metrics  # If metrics endpoint is enabled
```

### 📈 Performance Optimization

#### **Database Optimization**
- Index frequently queried fields
- Use MongoDB aggregation pipelines for analytics
- Implement connection pooling
- Regular database maintenance

#### **Audio Processing Optimization**
- Implement audio compression
- Use background tasks for TTS generation
- Cache frequently used audio responses
- Optimize file storage and cleanup

#### **API Performance**
- Enable response compression
- Implement request rate limiting
- Use async/await for I/O operations
- Monitor response times and errors

## 🛠️ Development

### 🔧 Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Set up pre-commit hooks
pre-commit install

# Run development server with auto-reload
uvicorn main:app --reload --log-level debug
```

### 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest -m "unit"
pytest -m "integration"
```

### 📝 Code Quality

```bash
# Format code
black .

# Lint code
flake8 app/

# Type checking
mypy app/
```

## 🤝 Contributing

### 🔄 Development Workflow

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**:
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation as needed
4. **Test your changes**:
   ```bash
   pytest
   black .
   flake8 app/
   ```
5. **Commit and push**:
   ```bash
   git commit -m 'Add amazing feature'
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### 📋 Contribution Guidelines

- **Code Style**: Follow PEP 8 and use Black for formatting
- **Testing**: Maintain test coverage above 80%
- **Documentation**: Update README and docstrings
- **Commit Messages**: Use conventional commit format
- **Pull Requests**: Include description and testing instructions

## 🙏 Acknowledgments

### 🛠️ Core Technologies
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast web framework
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - Workflow orchestration
- **[Google Gemini](https://ai.google.dev/)** - Advanced AI conversation capabilities
- **[OpenAI Whisper](https://openai.com/research/whisper)** - State-of-the-art speech recognition
- **[MongoDB](https://www.mongodb.com/)** - Flexible document database
- **[Socket.IO](https://socket.io/)** - Real-time bidirectional communication

### 🎯 AI & ML Libraries
- **[LangChain](https://langchain.com/)** - LLM application framework
- **[Pydantic](https://pydantic.dev/)** - Data validation and settings management
- **[NumPy](https://numpy.org/)** & **[Librosa](https://librosa.org/)** - Audio processing

## 🚀 Future Roadmap

### 🎯 Short-term Goals (Next 3 months)
- [ ] **User Authentication**: Secure user accounts and session management
- [ ] **Advanced Analytics**: Detailed performance metrics and progress tracking
- [ ] **Mobile Optimization**: Responsive design and mobile-specific features
- [ ] **API Rate Limiting**: Implement usage quotas and throttling

### 🌟 Medium-term Goals (3-6 months)
- [ ] **Custom Investor Personas**: User-defined investor profiles and characteristics
- [ ] **Multi-language Support**: International language support for global users
- [ ] **Integration APIs**: Connect with CRM systems and pitch deck tools
- [ ] **Advanced Audio Features**: Real-time noise cancellation and audio enhancement

### 🔮 Long-term Vision (6+ months)
- [ ] **Video Call Simulation**: Full video conferencing experience with AI investors
- [ ] **Pitch Deck Analysis**: AI-powered slide deck evaluation and feedback
- [ ] **Industry-Specific Training**: Specialized investor personas for different sectors
- [ ] **Machine Learning Insights**: Predictive analytics for pitch success probability

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for founders preparing to change the world** 🌍