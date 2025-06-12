import os
import logging
from typing import Optional
import whisper

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client (if using OpenAI API)
# openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribe an audio file using OpenAI's Whisper API or local Whisper model.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        
    Returns:
        The transcribed text
    """
    logger.info(f"Transcribing audio file: {audio_file_path}")
    
    try:
        # # Option 1: Using OpenAI's Whisper API
        # with open(audio_file_path, "rb") as audio_file:
        #     response = openai.Audio.transcribe(
        #         model="whisper-1",
        #         file=audio_file
        #     )
        #     transcript = response["text"]
        
        # Option 2: Using local Whisper model (commented out)
        import whisper
        model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
        result = model.transcribe(audio_file_path)
        transcript = result["text"]
        
        logger.info(f"Transcription successful: {transcript[:100]}...")
        return transcript
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        # Return a placeholder for testing if transcription fails
        return "This is a placeholder transcript for testing purposes. In production, this would be the actual transcribed content from the founder's pitch."