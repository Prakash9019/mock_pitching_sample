# AI-Powered Mock Investor Pitching

This FastAPI application allows founders to practice their investor pitches with an AI-simulated investor. The system accepts audio input, transcribes it, generates a realistic investor response, and converts that response to speech.

## Features

- Audio input processing (.wav or .mp3)
- Speech-to-text transcription using OpenAI Whisper
- AI-generated investor responses using Google's Gemini
- Text-to-speech conversion using ElevenLabs or Google Cloud TTS
- Session history storage
- Customizable investor personas

## Setup

### Prerequisites

- Python 3.8+
- API keys for:
  - OpenAI (for Whisper)
  - Google Gemini
  - ElevenLabs or Google Cloud TTS

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd moke-pitch
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit the `.env` file and add your API keys.

### Running the Application

Start the FastAPI server:
```bash
cd app
uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

## API Endpoints

### POST /pitch

Submit a founder's pitch audio for processing.

**Request:**
- `audio_file`: Audio file (.wav or .mp3)
- `investor_persona`: Type of investor (skeptical, technical, friendly)

**Response:**
```json
{
  "session_id": "uuid",
  "timestamp": "iso-datetime",
  "founder_transcript": "transcribed text",
  "investor_response": "AI-generated response",
  "response_audio_url": "/download/uuid"
}
```

### GET /download/{session_id}

Download the investor's audio response.

### GET /sessions/{session_id}

Get details about a specific pitch session.

## Testing

You can test the API using the provided test script:

```bash
python test_api.py path/to/audio_file.mp3
```

Or using curl:

```bash
curl -X POST http://localhost:8000/pitch \
  -F "audio_file=@path/to/audio_file.mp3" \
  -F "investor_persona=skeptical"
```

## Investor Personas

- **Skeptical**: Focuses on business metrics, market validation, and financial projections
- **Technical**: Focuses on technology stack, scalability, and technical differentiation
- **Friendly**: Supportive but thorough, focuses on vision, team, and growth strategy

## Project Structure

```
moke-pitch/
├── app/
│   ├── main.py                # FastAPI application
│   ├── services/
│   │   ├── __init__.py
│   │   ├── transcription.py   # Audio transcription service
│   │   ├── ai_response.py     # AI response generation
│   │   └── text_to_speech.py  # Text-to-speech conversion
│   └── data/                  # Created at runtime
│       ├── uploads/           # Uploaded audio files
│       ├── responses/         # Generated audio responses
│       └── sessions/          # Session data
├── requirements.txt
├── .env.example
├── test_api.py
└── README.md
```

## Future Enhancements

- Database integration for persistent storage
- User authentication and profiles
- More customizable investor personas
- Real-time feedback during pitches
- Analytics dashboard for pitch performance