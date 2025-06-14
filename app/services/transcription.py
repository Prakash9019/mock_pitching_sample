# transcription.py
import os
import logging
import subprocess
import sys
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

# Check if FFmpeg is installed
def check_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error(
            "FFmpeg is not installed. Please install FFmpeg and add it to your PATH. "
            "See https://ffmpeg.org/download.html for installation instructions."
        )
        return False

# Check FFmpeg installation before proceeding
if not check_ffmpeg():
    sys.exit(1)

try:
    
    from whisper import load_model
    WHISPER_AVAILABLE = True
except ImportError:
    logger.error(
        "Whisper is not installed. Install it with: "
        "pip install openai-whisper"
    )
    WHISPER_AVAILABLE = False
    
# Initialize OpenAI client (if using OpenAI API)
# openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_audio(audio_file_path: str, model_size: str = "base") -> str:
    """
    Transcribe an audio file using OpenAI's Whisper model.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        model_size: Size of the Whisper model to use (tiny, base, small, medium, large)
        
    Returns:
        The transcribed text
        
    Raises:
        RuntimeError: If Whisper is not available or if there's an error during transcription
    """
    if not WHISPER_AVAILABLE:
        raise RuntimeError(
            "Whisper is not available. Please install it with: "
            "pip install openai-whisper"
        )
    
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    logger.info(f"Transcribing audio file: {audio_file_path}")
    
    try:
        # Load the Whisper model
        model = load_model(model_size)
        
        # Transcribe the audio file
        result = model.transcribe(
            audio_file_path,
            fp16=False  # Disable mixed precision for better compatibility
        )
        
        transcript = result["text"].strip()
        
        if not transcript:
            logger.warning("Transcription returned empty result")
            return "[No speech detected]"
            
        logger.info(f"Transcription successful. First 100 chars: {transcript[:100]}")
        return transcript
        
    except Exception as e:
        error_msg = f"Error transcribing audio: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError("Failed to transcribe audio. Please check the logs for details.")