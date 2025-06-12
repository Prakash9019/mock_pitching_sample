from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import json
import uuid
from datetime import datetime
import shutil
from typing import Optional
import logging

# Import service modules
from services.transcription import transcribe_audio
from services.ai_response import generate_investor_response
from services.text_to_speech import convert_text_to_speech

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Mock Investor Pitch",
    description="API for simulating investor pitch sessions with AI-generated responses",
    version="1.0.0"
)

# Create necessary directories
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/responses", exist_ok=True)
os.makedirs("data/sessions", exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "AI Mock Investor Pitch API is running"}

@app.post("/pitch")
async def process_pitch(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    investor_persona: Optional[str] = "skeptical"
):
    """
    Process a founder's pitch audio and generate an investor response.
    
    Args:
        audio_file: The audio file containing the founder's pitch (.wav or .mp3)
        investor_persona: The type of investor persona to simulate (default: skeptical)
        
    Returns:
        A JSON response with session details and a link to download the investor's audio response
    """
    # Validate file type
    if not audio_file.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 files are supported")
    
    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Save the uploaded audio file
        upload_path = f"data/uploads/{session_id}_{audio_file.filename}"
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Transcribe the audio
        logger.info(f"Transcribing audio for session {session_id}")
        transcript = transcribe_audio(upload_path)
        
        # Generate investor response
        logger.info(f"Generating investor response for session {session_id}")
        investor_response = generate_investor_response(transcript, investor_persona)
        
        # Convert response to speech
        logger.info(f"Converting response to speech for session {session_id}")
        response_audio_path = f"data/responses/{session_id}_response.mp3"
        convert_text_to_speech(investor_response, response_audio_path)
        
        # Save session data
        session_data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "founder_transcript": transcript,
            "investor_response": investor_response,
            "investor_persona": investor_persona,
            "original_audio": upload_path,
            "response_audio": response_audio_path
        }
        
        with open(f"data/sessions/{session_id}.json", "w") as f:
            json.dump(session_data, f, indent=2)
        
        # Schedule cleanup in background (optional)
        # background_tasks.add_task(cleanup_files, session_id)
        
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

@app.get("/download/{session_id}")
async def download_response(session_id: str):
    """
    Download the investor's audio response for a specific session.
    
    Args:
        session_id: The unique session identifier
        
    Returns:
        The audio file as a downloadable response
    """
    response_path = f"data/responses/{session_id}_response.mp3"
    
    if not os.path.exists(response_path):
        raise HTTPException(status_code=404, detail="Response audio not found")
    
    return FileResponse(
        path=response_path,
        filename=f"investor_response_{session_id}.mp3",
        media_type="audio/mpeg"
    )

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Retrieve the details of a specific pitch session.
    
    Args:
        session_id: The unique session identifier
        
    Returns:
        The session data as JSON
    """
    session_path = f"data/sessions/{session_id}.json"
    
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")
    
    with open(session_path, "r") as f:
        session_data = json.load(f)
    
    return session_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)