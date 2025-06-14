from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse
import os
import json
import uuid
import tempfile
from datetime import datetime
import shutil
from typing import Optional, Union, BinaryIO
import logging

# Socket.IO integration
import socketio
from fastapi.middleware.cors import CORSMiddleware

# Import service modules
from services.transcription import transcribe_audio
from services.ai_response import generate_investor_response
from services.text_to_speech import convert_text_to_speech

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
fastapi_app = FastAPI(
    title="AI Mock Investor Pitch",
    description="API for simulating investor pitch sessions with AI-generated responses",
    version="1.0.0"
)

fastapi_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
app = socketio.ASGIApp(sio, fastapi_app)

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')
RESPONSE_DIR = os.path.join(DATA_DIR, 'responses')
SESSION_DIR = os.path.join(DATA_DIR, 'sessions')

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESPONSE_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
logger.info(f"Data directories created at: {DATA_DIR}")
logger.info(f"Upload directory: {UPLOAD_DIR}")
logger.info(f"Response directory: {RESPONSE_DIR}")
logger.info(f"Session directory: {SESSION_DIR}")

@fastapi_app.get("/")
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>🎙️ AI Investor Pitch</title>
    </head>
    <body>
      <h1>🎤 Talk to the AI Investor</h1>
      <button id="record">🎙️ Record Pitch</button>
      <audio id="player" controls></audio>

      <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
      <script>
        const socket = io("http://localhost:8000");
        const recordButton = document.getElementById("record");
        const audioPlayer = document.getElementById("player");

        recordButton.onclick = async () => {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          const mediaRecorder = new MediaRecorder(stream);
          const chunks = [];

          mediaRecorder.ondataavailable = e => chunks.push(e.data);
          mediaRecorder.onstop = async () => {
            const blob = new Blob(chunks, { type: 'audio/wav' });
            const buffer = await blob.arrayBuffer();
            socket.emit("audio_chunk", buffer);
          };

          mediaRecorder.start();
          setTimeout(() => mediaRecorder.stop(), 4000);  // Record 4 seconds
        };

        socket.on("ai_response", (data) => {
          const blob = new Blob([data], { type: "audio/mpeg" });
          const url = URL.createObjectURL(blob);
          audioPlayer.src = url;
          audioPlayer.play();
        });
      </script>
    </body>
    </html>
    """)

@fastapi_app.post("/pitch")
async def process_pitch(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    investor_persona: Optional[str] = "skeptical"
):
    if not audio_file.filename.endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 files are supported")

    try:
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        upload_path = os.path.join(UPLOAD_DIR, f"{session_id}_{audio_file.filename}")
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        transcript = transcribe_audio(upload_path)
        investor_response = generate_investor_response(transcript, investor_persona)
        response_audio_path = os.path.join(RESPONSE_DIR, f"{session_id}_response.mp3")
        convert_text_to_speech(investor_response, response_audio_path)

        session_data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "founder_transcript": transcript,
            "investor_response": investor_response,
            "investor_persona": investor_persona,
            "original_audio": upload_path,
            "response_audio": response_audio_path
        }

        session_file = os.path.join(SESSION_DIR, f"{session_id}.json")
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)

        return {
            "session_id": session_id,
            "timestamp": timestamp,
            "founder_transcript": transcript,
            "investor_response": investor_response,
            "response_audio_url": f"/download/{session_id}"
        }

    except Exception as e:
        logger.error(f"Error processing pitch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing pitch: {str(e)}")

@fastapi_app.get("/download/{session_id}")
async def download_response(session_id: str):
    response_path = os.path.join(RESPONSE_DIR, f"{session_id}_response.mp3")
    if not os.path.exists(response_path):
        raise HTTPException(status_code=404, detail="Response audio not found")
    return FileResponse(
        path=response_path,
        filename=f"investor_response_{session_id}.mp3",
        media_type="audio/mpeg"
    )

@fastapi_app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session_path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")
    with open(session_path, "r") as f:
        session_data = json.load(f)
    return session_data

@sio.event
async def connect(sid, environ):
    logger.info(f"WebSocket connected: {sid}")

@sio.event
async def audio_chunk(sid, data):
    try:
        logger.info(f"Received audio chunk from {sid}")
        
        # Save the incoming audio to a temporary file for transcription
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio.write(data)
            temp_audio_path = temp_audio.name
        
        try:
            logger.info("Starting transcription...")
            transcript = transcribe_audio(temp_audio_path)
            logger.info(f"Transcription complete: {transcript[:100]}...")
            
            logger.info("Generating investor response...")
            investor_reply = generate_investor_response(transcript, persona="skeptical")
            
            logger.info("Converting response to speech...")
            # Get audio data directly without saving to file
            audio_data = convert_text_to_speech(investor_reply)
            logger.info(f"Generated {len(audio_data)} bytes of audio data")
            
            # Send the audio data directly to the client
            await sio.emit("ai_response", audio_data, to=sid)
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_audio_path}: {e}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await sio.emit("error", {"message": str(e)}, to=sid)


@sio.event
async def disconnect(sid):
    logger.info(f"WebSocket disconnected: {sid}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)