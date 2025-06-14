# transcription.py
import os
import logging
import subprocess
import sys
import tempfile
import numpy as np
from typing import Dict, Optional, Tuple, Union
import soundfile as sf
import librosa
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import split_on_silence

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
    import whisper
    from whisper import load_model
    WHISPER_AVAILABLE = True
except ImportError:
    logger.error(
        "Whisper is not installed. Install it with: "
        "pip install openai-whisper"
    )
    WHISPER_AVAILABLE = False

# Default model size (can be overridden by environment variable)
DEFAULT_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large"]

def preprocess_audio(audio_path: str) -> str:
    """
    Preprocess audio file to improve transcription quality.
    - Converts to 16kHz mono WAV
    - Normalizes volume
    - Reduces background noise
    - Removes long silences
    """
    try:
        # Load audio using pydub
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono and set sample rate to 16kHz
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Normalize volume
        audio = audio.normalize()
        
        # Split on silence and keep only segments longer than 200ms
        chunks = split_on_silence(
            audio,
            min_silence_len=200,
            silence_thresh=audio.dBFS-14,
            keep_silence=100
        )
        
        # Combine non-silent chunks
        processed_audio = AudioSegment.empty()
        for chunk in chunks:
            processed_audio += chunk
        
        # Reduce background noise
        samples = np.array(processed_audio.get_array_of_samples())
        sample_rate = processed_audio.frame_rate
        
        # Convert to float32 for noise reduction
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        
        # Perform noise reduction
        reduced_noise = nr.reduce_noise(
            y=samples,
            sr=sample_rate,
            stationary=True,
            n_std_thresh_stationary=1.5
        )
        
        # Save processed audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            processed_path = tmp_file.name
            sf.write(processed_path, reduced_noise, sample_rate)
            
        return processed_path
        
    except Exception as e:
        logger.warning(f"Audio preprocessing failed: {str(e)}. Using original audio.")
        return audio_path

def transcribe_audio(
    audio_file_path: str,
    model_size: str = None,
    language: str = "en",
    temperature: float = 0.2,
    beam_size: int = 5,
    best_of: int = 5,
    patience: float = 1.0
) -> Dict[str, Union[str, float]]:
    """
    Transcribe an audio file using OpenAI's Whisper model with enhanced accuracy.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        model_size: Size of the Whisper model (tiny, base, small, medium, large)
        language: Language code (e.g., 'en', 'es', 'fr'). If None, auto-detects.
        temperature: Sampling temperature (0.0 to 1.0, lower is more deterministic)
        beam_size: Number of beams in beam search (only used when temperature is zero)
        best_of: Number of candidates to consider (only used when temperature > 0)
        patience: Patience for beam search (only used when temperature is zero)
        
    Returns:
        Dictionary containing:
        - 'text': The transcribed text
        - 'language': Detected language
        - 'confidence': Confidence score (0-1)
        - 'segments': List of segments with timing and confidence
        
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
    
    model_size = model_size or DEFAULT_MODEL_SIZE
    if model_size not in SUPPORTED_MODELS:
        logger.warning(f"Model size {model_size} not supported. Using {DEFAULT_MODEL_SIZE}.")
        model_size = DEFAULT_MODEL_SIZE
    
    logger.info(f"Transcribing audio file: {audio_file_path} with model: {model_size}")
    
    try:
        # Preprocess audio
        processed_audio_path = preprocess_audio(audio_file_path)
        
        # Load the Whisper model
        model = load_model(model_size)
        
        # Transcribe with enhanced parameters
        result = model.transcribe(
            processed_audio_path,
            language=language,
            temperature=temperature,
            beam_size=beam_size if temperature == 0 else None,
            best_of=best_of if temperature > 0 else None,
            patience=patience if temperature == 0 else None,
            fp16=False,  # Disable mixed precision for better compatibility
            verbose=False
        )
        
        # Clean up processed audio file
        if processed_audio_path != audio_file_path and os.path.exists(processed_audio_path):
            try:
                os.unlink(processed_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {processed_audio_path}: {e}")
        
        # Calculate average confidence
        segments = result.get('segments', [])
        avg_confidence = np.mean([seg.get('confidence', 0.8) for seg in segments]) if segments else 0.8
        
        return {
            'text': result['text'].strip() or "[No speech detected]",
            'language': result.get('language', 'en'),
            'confidence': float(avg_confidence),
            'segments': segments
        }
        
    except Exception as e:
        error_msg = f"Error transcribing audio: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError("Failed to transcribe audio. Please check the logs for details.")