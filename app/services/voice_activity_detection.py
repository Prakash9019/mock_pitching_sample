#!/usr/bin/env python3
"""
Voice Activity Detection (VAD) Service
Handles real-time audio streaming and automatic speech detection
"""

import asyncio
import logging
import time
import io
import wave
import tempfile
import os
from typing import Optional, Callable, Dict, Any
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

# For VAD - we'll use webrtcvad which is excellent for voice activity detection
try:
    import webrtcvad
    VAD_AVAILABLE = True
    logger.info("WebRTC VAD available")
except ImportError:
    VAD_AVAILABLE = False
    logger.warning("webrtcvad not available. Using volume-based VAD fallback.")

# For audio processing
try:
    import librosa
    LIBROSA_AVAILABLE = True
    logger.info("librosa available for advanced audio processing")
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available. Using basic audio processing.")

class VoiceActivityDetector:
    """
    Real-time Voice Activity Detection system
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 30,
                 silence_threshold_seconds: float = 3.0,  # Reduced to 3 seconds
                 min_speech_duration_seconds: float = 0.5,  # Reduced to 0.5 seconds
                 vad_aggressiveness: int = 1):  # Less aggressive for better sensitivity
        """
        Initialize VAD system
        
        Args:
            sample_rate: Audio sample rate (16kHz recommended for VAD)
            frame_duration_ms: Frame duration in milliseconds (10, 20, or 30)
            silence_threshold_seconds: Seconds of silence before considering speech ended
            min_speech_duration_seconds: Minimum speech duration to consider valid
            vad_aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.silence_threshold_seconds = silence_threshold_seconds
        self.min_speech_duration_seconds = min_speech_duration_seconds
        
        # Calculate frame size
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Initialize VAD
        if VAD_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(vad_aggressiveness)
                logger.info(f"WebRTC VAD initialized with aggressiveness {vad_aggressiveness}")
            except Exception as e:
                logger.error(f"Failed to initialize WebRTC VAD: {e}")
                self.vad = None
        else:
            self.vad = None
            logger.info("Using volume-based VAD (WebRTC VAD not available)")
        
        # State tracking
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.audio_buffer = deque(maxlen=int(sample_rate * 30))  # 30 seconds max buffer
        self.speech_frames = []
        
        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable[[bytes], None]] = None
        self.on_silence_detected: Optional[Callable] = None
        
    def set_callbacks(self, 
                     on_speech_start: Optional[Callable] = None,
                     on_speech_end: Optional[Callable[[bytes], None]] = None,
                     on_silence_detected: Optional[Callable] = None):
        """Set callback functions for VAD events"""
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_silence_detected = on_silence_detected
    
    def _is_speech_webrtc(self, audio_frame: bytes) -> bool:
        """Use WebRTC VAD to detect speech"""
        if not self.vad:
            return False
        
        try:
            # WebRTC VAD expects specific frame sizes
            if len(audio_frame) != self.frame_size * 2:  # 2 bytes per sample (16-bit)
                return False
            
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except Exception as e:
            logger.error(f"WebRTC VAD error: {e}")
            return False
    
    def _is_speech_volume(self, audio_frame: bytes) -> bool:
        """Fallback volume-based speech detection"""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_frame, dtype=np.int16)
            
            if len(audio_data) == 0:
                return False
            
            # Calculate RMS (Root Mean Square) for volume
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            
            # Dynamic threshold adjustment based on recent audio levels
            if not hasattr(self, '_volume_history'):
                self._volume_history = deque(maxlen=100)  # Keep last 100 measurements
                self._base_threshold = 50  # Lower base threshold for better sensitivity
            
            self._volume_history.append(rms)
            
            # Calculate adaptive threshold
            if len(self._volume_history) > 10:
                avg_volume = np.mean(list(self._volume_history))
                # Threshold is base + 1.5x average background noise (more sensitive)
                adaptive_threshold = max(self._base_threshold, avg_volume * 1.5)
            else:
                adaptive_threshold = self._base_threshold
            
            is_speech = rms > adaptive_threshold
            
            # Debug logging
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
            
            # Log every 50 frames for debugging, and always log when speech is detected
            if self._debug_counter % 50 == 0 or is_speech:
                logger.info(f"Volume VAD: RMS={rms:.1f}, Threshold={adaptive_threshold:.1f}, Speech={is_speech}")
            
            return is_speech
            
        except Exception as e:
            logger.error(f"Volume-based VAD error: {e}")
            return False
    
    def process_audio_chunk(self, audio_chunk: bytes) -> Dict[str, Any]:
        """
        Process incoming audio chunk and detect voice activity
        
        Args:
            audio_chunk: Raw audio data (16-bit PCM)
            
        Returns:
            Dict with VAD results and actions
        """
        current_time = time.time()
        
        # Add to buffer
        try:
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            self.audio_buffer.extend(audio_data)
            
            # Debug logging for audio data
            if hasattr(self, '_chunk_counter'):
                self._chunk_counter += 1
            else:
                self._chunk_counter = 0
            
            if self._chunk_counter % 20 == 0:  # Log every 20 chunks
                logger.info(f"Audio chunk: {len(audio_chunk)} bytes, {len(audio_data)} samples, max={np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0}")
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {"error": f"Audio processing error: {e}"}
        
        # Detect speech in this chunk
        if self.vad:
            is_speech = self._is_speech_webrtc(audio_chunk)
        else:
            is_speech = self._is_speech_volume(audio_chunk)
        
        result = {
            "is_speech": is_speech,
            "is_speaking": self.is_speaking,
            "action": None,
            "audio_data": None,
            "speech_duration": 0
        }
        
        if is_speech:
            # Speech detected
            if not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start_time = current_time
                self.speech_frames = []
                result["action"] = "speech_started"
                
                if self.on_speech_start:
                    try:
                        self.on_speech_start()
                    except Exception as e:
                        logger.error(f"Error in speech_start callback: {e}")
            
            # Add to speech buffer
            self.speech_frames.append(audio_chunk)
            self.last_speech_time = current_time
            
        else:
            # No speech detected
            if self.is_speaking:
                # Check if silence duration exceeds threshold
                silence_duration = current_time - self.last_speech_time
                
                if silence_duration >= self.silence_threshold_seconds:
                    # Speech ended
                    speech_duration = current_time - self.speech_start_time
                    
                    if speech_duration >= self.min_speech_duration_seconds:
                        # Valid speech detected
                        audio_data = self._combine_speech_frames()
                        result["action"] = "speech_ended"
                        result["audio_data"] = audio_data
                        result["speech_duration"] = speech_duration
                        
                        if self.on_speech_end:
                            try:
                                self.on_speech_end(audio_data)
                            except Exception as e:
                                logger.error(f"Error in speech_end callback: {e}")
                    else:
                        # Speech too short, ignore
                        result["action"] = "speech_too_short"
                        
                        if self.on_silence_detected:
                            try:
                                self.on_silence_detected()
                            except Exception as e:
                                logger.error(f"Error in silence_detected callback: {e}")
                    
                    # Reset state
                    self.is_speaking = False
                    self.speech_start_time = None
                    self.last_speech_time = None
                    self.speech_frames = []
        
        return result
    
    def _combine_speech_frames(self) -> bytes:
        """Combine all speech frames into a single audio file"""
        if not self.speech_frames:
            return b""
        
        # Combine all frames
        combined_audio = b"".join(self.speech_frames)
        
        # Create WAV file
        return self._create_wav_file(combined_audio)
    
    def _create_wav_file(self, audio_data: bytes) -> bytes:
        """Create a WAV file from raw audio data"""
        try:
            # Create in-memory WAV file
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
            
            wav_buffer.seek(0)
            return wav_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating WAV file: {e}")
            return b""
    
    def reset(self):
        """Reset VAD state"""
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.speech_frames = []
        self.audio_buffer.clear()
        logger.info("VAD state reset")

class AudioStreamProcessor:
    """
    Handles real-time audio streaming and processing
    """
    
    def __init__(self, vad_config: Optional[Dict] = None):
        """Initialize audio stream processor"""
        self.vad_config = vad_config or {}
        self.vad = VoiceActivityDetector(**self.vad_config)
        self.is_active = False
        self.session_id = None
        
        # Callbacks for integration with pitch system
        self.on_transcription_ready: Optional[Callable[[str, bytes], None]] = None
        
    def set_transcription_callback(self, callback: Callable[[str, bytes], None]):
        """Set callback for when transcription is ready"""
        self.on_transcription_ready = callback
    
    def start_session(self, session_id: str):
        """Start a new audio processing session"""
        self.session_id = session_id
        self.is_active = True
        self.vad.reset()
        
        # Set VAD callbacks
        self.vad.set_callbacks(
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
            on_silence_detected=self._on_silence_detected
        )
        
        logger.info(f"Audio stream session started: {session_id}")
    
    def stop_session(self):
        """Stop the current audio processing session"""
        self.is_active = False
        self.vad.reset()
        logger.info(f"Audio stream session stopped: {self.session_id}")
        self.session_id = None
    
    def process_audio_stream(self, audio_chunk: bytes) -> Dict[str, Any]:
        """Process incoming audio stream chunk"""
        if not self.is_active:
            return {"error": "Session not active"}
        
        return self.vad.process_audio_chunk(audio_chunk)
    
    def _on_speech_start(self):
        """Called when speech starts"""
        logger.info(f"Speech started in session {self.session_id}")
    
    def _on_speech_end(self, audio_data: bytes):
        """Called when speech ends - trigger transcription"""
        logger.info(f"Speech ended in session {self.session_id}, triggering transcription")
        
        if self.on_transcription_ready and audio_data:
            try:
                # Save audio to temporary file for transcription
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name
                
                # Trigger transcription callback
                self.on_transcription_ready(self.session_id, temp_file_path)
                
            except Exception as e:
                logger.error(f"Error processing speech end: {e}")
    
    def _on_silence_detected(self):
        """Called when silence is detected"""
        logger.debug(f"Silence detected in session {self.session_id}")

# Global audio processor instance
audio_processor = AudioStreamProcessor()

def get_audio_processor() -> AudioStreamProcessor:
    """Get the global audio processor instance"""
    return audio_processor

def initialize_vad_system(config: Optional[Dict] = None) -> AudioStreamProcessor:
    """Initialize the VAD system with custom configuration"""
    global audio_processor
    
    if config:
        audio_processor = AudioStreamProcessor(config)
    
    logger.info("VAD system initialized")
    return audio_processor