# transcription.py
import os
import logging
import subprocess
import sys
import tempfile
import time
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

# Transcription cache to avoid duplicate transcriptions
_TRANSCRIPTION_CACHE = {}
_CACHE_EXPIRY = 60 * 60  # Cache expiry in seconds (1 hour)
_ACTIVE_TRANSCRIPTIONS = set()  # Track currently processing transcriptions

def get_file_hash(file_path: str) -> str:
    """Generate a hash for a file to use as cache key"""
    try:
        # Get file size and modification time for quick comparison
        file_stat = os.stat(file_path)
        file_size = file_stat.st_size
        mod_time = file_stat.st_mtime
        
        # Combine with file path for a unique identifier
        hash_input = f"{file_path}:{file_size}:{mod_time}"
        return hash_input
    except Exception as e:
        logger.warning(f"Error generating file hash: {e}")
        # Fallback to just the file path
        return file_path

def preprocess_audio(audio_path: str) -> str:
    """
    Preprocess audio file to improve transcription quality with optimized performance.
    - Converts to 16kHz mono WAV
    - Normalizes volume if needed
    - Skips unnecessary processing for small files
    """
    try:
        # Check file size first - skip processing for very small files
        file_size = os.path.getsize(audio_path)
        if file_size < 5000:  # Less than 5KB
            logger.info(f"Skipping preprocessing for small file ({file_size} bytes)")
            return audio_path
            
        # Load audio using pydub
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono and set sample rate to 16kHz
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Only normalize if volume is very low
        if audio.dBFS < -35:  # Only normalize if volume is very low
            audio = audio.normalize(headroom=1.0)
        
        # Skip silence removal for short audio (less than 3 seconds)
        if len(audio) < 3000:
            processed_audio = audio
        else:
            # Only remove extended silences (over 1 second)
            chunks = split_on_silence(
                audio,
                min_silence_len=1000,  # 1 second of silence
                silence_thresh=audio.dBFS-16,  # More permissive threshold
                keep_silence=300  # Keep more silence for natural speech rhythm
            )
            
            # If no chunks were found (no long silences), use the original audio
            if not chunks:
                processed_audio = audio
            else:
                # Combine non-silent chunks
                processed_audio = AudioSegment.empty()
                for chunk in chunks:
                    processed_audio += chunk
        
        # Save processed audio to a temporary file - skip noise reduction for performance
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            processed_path = tmp_file.name
            
            # Export directly without additional processing
            processed_audio.export(
                processed_path, 
                format="wav",
                parameters=["-acodec", "pcm_s16le"]  # Use standard PCM format
            )
            
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
    patience: float = 1.0,
    use_cache: bool = True
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
        use_cache: Whether to use the transcription cache
        
    Returns:
        Dictionary containing:
        - 'text': The transcribed text
        - 'language': Detected language
        - 'confidence': Confidence score (0-1)
        - 'segments': List of segments with timing and confidence
        
    Raises:
        RuntimeError: If Whisper is not available or if there's an error during transcription
    """
    global _ACTIVE_TRANSCRIPTIONS
    
    if not WHISPER_AVAILABLE:
        raise RuntimeError(
            "Whisper is not available. Please install it with: "
            "pip install openai-whisper"
        )
    
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    # Generate a hash for the file
    file_hash = get_file_hash(audio_file_path)
    
    # Check if this file is already being transcribed
    if file_hash in _ACTIVE_TRANSCRIPTIONS:
        logger.info(f"Transcription already in progress for: {audio_file_path}")
        return {'text': '', 'language': 'en', 'confidence': 0.0, 'segments': [], 'status': 'in_progress'}
    
    # Check cache first if enabled
    if use_cache:
        if file_hash in _TRANSCRIPTION_CACHE:
            cache_entry = _TRANSCRIPTION_CACHE[file_hash]
            cache_time = cache_entry.get('timestamp', 0)
            current_time = time.time()
            
            # Check if cache is still valid
            if current_time - cache_time < _CACHE_EXPIRY:
                logger.info(f"Using cached transcription for: {audio_file_path}")
                return cache_entry['result']
            else:
                # Cache expired, remove it
                del _TRANSCRIPTION_CACHE[file_hash]
    
    # Mark this file as being transcribed
    _ACTIVE_TRANSCRIPTIONS.add(file_hash)
    
    try:
        # Check file size - skip very small files that likely don't contain speech
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:  # Less than 1KB
            logger.warning(f"Audio file too small ({file_size} bytes), likely no speech: {audio_file_path}")
            return {'text': '[No speech detected]', 'language': 'en', 'confidence': 0.0, 'segments': []}
        
        model_size = model_size or DEFAULT_MODEL_SIZE
        if model_size not in SUPPORTED_MODELS:
            logger.warning(f"Model size {model_size} not supported. Using {DEFAULT_MODEL_SIZE}.")
            model_size = DEFAULT_MODEL_SIZE
        
        logger.info(f"Transcribing audio file: {audio_file_path} with model: {model_size}")
        
        # Use smaller model for very short audio clips
        if file_size < 10000:  # Less than 10KB
            logger.info(f"Using 'tiny' model for small audio file ({file_size} bytes)")
            model_size = "tiny"
        
        # Preprocess audio - optimize for speed
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
        
        transcription_result = {
            'text': result['text'].strip() or "[No speech detected]",
            'language': result.get('language', 'en'),
            'confidence': float(avg_confidence),
            'segments': segments
        }
        
        # Cache the result if caching is enabled
        if use_cache:
            _TRANSCRIPTION_CACHE[file_hash] = {
                'result': transcription_result,
                'timestamp': time.time()
            }
            
            # Limit cache size to prevent memory issues
            if len(_TRANSCRIPTION_CACHE) > 100:  # Keep only 100 most recent entries
                oldest_key = min(_TRANSCRIPTION_CACHE.keys(), 
                                key=lambda k: _TRANSCRIPTION_CACHE[k]['timestamp'])
                del _TRANSCRIPTION_CACHE[oldest_key]
        
        return transcription_result
        
    except Exception as e:
        error_msg = f"Error transcribing audio: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {'text': '[Transcription error]', 'language': 'en', 'confidence': 0.0, 'segments': []}
    finally:
        # Remove from active transcriptions
        if file_hash in _ACTIVE_TRANSCRIPTIONS:
            _ACTIVE_TRANSCRIPTIONS.remove(file_hash)