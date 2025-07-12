# audio_conversation_storage.py
import os
import logging
import tempfile
import time
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from google.oauth2 import service_account
from google.cloud import storage
import wave
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
import io

# Import additional libraries for audio processing
try:
    from scipy import signal
    from pydub.effects import high_pass_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available - some audio enhancements will be disabled")

# Configure logging
logger = logging.getLogger(__name__)

class AudioConversationStorage:
    """
    Service for storing and managing audio conversations using Google Cloud Storage
    Combines user and AI audio into single conversation files
    """
    
    def __init__(self, bucket_name: str = None):
        self.client = None
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME', 'ai-pitch-conversations')
        self.bucket = None
        self._initialize_client()
        
        # Audio settings
        self.sample_rate = 16000  # Standard sample rate
        self.channels = 2  # Stereo: left=user, right=AI
        self.format = "wav"  # Default format (can switch to mp3 for compression)
        
        # Session audio buffers
        self.session_audio_buffers: Dict[str, Dict] = {}
        
    def _initialize_client(self):
        """Initialize Google Cloud Storage client with service account credentials"""
        try:
            # Get credentials from environment variables (same as TTS)
            credentials_info = {
                'type': os.getenv('TYPE'),
                'project_id': os.getenv('PROJECT_ID'),
                'private_key_id': os.getenv('PRIVATE_KEY_ID'),
                'private_key': os.getenv('PRIVATE_KEY').replace('\\n', '\n') if os.getenv('PRIVATE_KEY') else None,
                'client_email': os.getenv('CLIENT_EMAIL'),
                'client_id': os.getenv('CLIENT_ID'),
                'auth_uri': os.getenv('AUTH_URI'),
                'token_uri': os.getenv('TOKEN_URI'),
                'auth_provider_x509_cert_url': os.getenv('AUTH_PROVIDER_X509_CERT_URL'),
                'client_x509_cert_url': os.getenv('CLIENT_X509_CERT_URL'),
                'universe_domain': os.getenv('UNIVERSE_DOMAIN', 'googleapis.com')
            }
            
            # Check if all required credentials are present
            if not all([credentials_info['type'], credentials_info['project_id'], 
                       credentials_info['private_key'], credentials_info['client_email']]):
                logger.error("Missing required Google Cloud credentials for Storage")
                return
            
            # Create credentials object
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Initialize the client
            self.client = storage.Client(credentials=credentials)
            
            # Get or create bucket
            try:
                self.bucket = self.client.bucket(self.bucket_name)
                # Check if bucket exists
                self.bucket.reload()
                logger.info(f"Connected to existing GCS bucket: {self.bucket_name}")
            except Exception as e:
                logger.warning(f"Bucket {self.bucket_name} not accessible: {e}")
                # Try to create bucket (this might fail due to permissions)
                try:
                    self.bucket = self.client.create_bucket(self.bucket_name)
                    logger.info(f"Created new GCS bucket: {self.bucket_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create bucket {self.bucket_name}: {create_error}")
                    self.bucket = None
                    return
            
            logger.info("Google Cloud Storage client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Google Cloud Storage client: {str(e)}")
            self.client = None
            self.bucket = None
    
    def start_session_recording(self, session_id: str, persona: str = "friendly") -> bool:
        """Start recording audio for a session"""
        try:
            if not self.client or not self.bucket:
                logger.error("Google Cloud Storage not initialized")
                return False
            
            # Initialize session audio buffer
            self.session_audio_buffers[session_id] = {
                'persona': persona,
                'user_audio_segments': [],
                'ai_audio_segments': [],
                'timestamps': [],
                'start_time': time.time(),
                'total_duration': 0.0
            }
            
            logger.info(f"Started audio recording for session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting session recording: {e}")
            return False
    
    def add_user_audio(self, session_id: str, audio_data: bytes, timestamp: float = None) -> bool:
        """Add user audio segment to session buffer"""
        try:
            if session_id not in self.session_audio_buffers:
                logger.error(f"❌ Session {session_id} not found for user audio - available sessions: {list(self.session_audio_buffers.keys())}")
                return False
            
            if not audio_data or len(audio_data) == 0:
                logger.error(f"❌ Empty audio data provided for session {session_id}")
                return False
            
            if timestamp is None:
                timestamp = time.time()
            
            logger.info(f"🎤 Processing user audio for session {session_id}: {len(audio_data)} bytes")
            
            # Convert audio data to AudioSegment for processing
            audio_segment = self._bytes_to_audio_segment(audio_data)
            if audio_segment is None:
                logger.error(f"❌ Failed to convert user audio data for session {session_id}")
                return False
            
            # Validate audio segment
            if len(audio_segment) < 100:  # Less than 100ms
                logger.warning(f"⚠️ User audio segment too short ({len(audio_segment)}ms) for session {session_id}")
                return False
            
            # Store audio segment with metadata
            self.session_audio_buffers[session_id]['user_audio_segments'].append({
                'audio': audio_segment,
                'timestamp': timestamp,
                'duration': len(audio_segment) / 1000.0,  # Duration in seconds
                'type': 'user'
            })
            
            user_count = len(self.session_audio_buffers[session_id]['user_audio_segments'])
            logger.info(f"✅ Added user audio segment to session {session_id}: {len(audio_segment)}ms (total user segments: {user_count})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding user audio for session {session_id}: {e}")
            return False
    
    def add_ai_audio(self, session_id: str, audio_data: bytes, timestamp: float = None) -> bool:
        """Add AI audio segment to session buffer"""
        try:
            if session_id not in self.session_audio_buffers:
                logger.error(f"❌ Session {session_id} not found for AI audio - available sessions: {list(self.session_audio_buffers.keys())}")
                return False
            
            if not audio_data or len(audio_data) == 0:
                logger.error(f"❌ Empty AI audio data provided for session {session_id}")
                return False
            
            if timestamp is None:
                timestamp = time.time()
            
            logger.info(f"🤖 Processing AI audio for session {session_id}: {len(audio_data)} bytes")
            
            # Convert audio data to AudioSegment for processing
            audio_segment = self._bytes_to_audio_segment(audio_data)
            if audio_segment is None:
                logger.error(f"❌ Failed to convert AI audio data for session {session_id}")
                return False
            
            # Validate audio segment
            if len(audio_segment) < 100:  # Less than 100ms
                logger.warning(f"⚠️ AI audio segment too short ({len(audio_segment)}ms) for session {session_id}")
                return False
            
            # Store audio segment with metadata
            self.session_audio_buffers[session_id]['ai_audio_segments'].append({
                'audio': audio_segment,
                'timestamp': timestamp,
                'duration': len(audio_segment) / 1000.0,  # Duration in seconds
                'type': 'ai'
            })
            
            ai_count = len(self.session_audio_buffers[session_id]['ai_audio_segments'])
            logger.info(f"✅ Added AI audio segment to session {session_id}: {len(audio_segment)}ms (total AI segments: {ai_count})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding AI audio for session {session_id}: {e}")
            return False
    
    def finalize_session_recording(self, session_id: str, use_mp3: bool = True) -> Optional[str]:
        """
        Finalize session recording and upload to Google Cloud Storage
        Returns the public URL of the uploaded audio file
        
        This is now an asynchronous operation - it returns a placeholder URL immediately
        and the actual upload happens in the background
        """
        try:
            if session_id not in self.session_audio_buffers:
                logger.error(f"Session {session_id} not found")
                return None
            
            if not self.client or not self.bucket:
                logger.error("Google Cloud Storage not initialized")
                return None
            
            # Generate placeholder URL that will be updated later
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_format = "mp3"  # Always use MP3 for better compression
            file_extension = f".{file_format}"
            filename = f"conversations/{session_id}_{timestamp}{file_extension}"
            
            # Create a placeholder URL
            placeholder_url = f"https://storage.googleapis.com/{self.bucket_name}/{filename}?processing=true"
            
            # Start background processing
            import threading
            thread = threading.Thread(
                target=self._process_and_upload_audio,
                args=(session_id, filename, file_format)
            )
            thread.daemon = True
            thread.start()
            
            logger.info(f"Started background processing for session {session_id}")
            
            # Return placeholder URL immediately
            return placeholder_url
            
        except Exception as e:
            logger.error(f"Error starting session recording finalization: {e}")
            return None
    
    def _process_and_upload_audio(self, session_id: str, filename: str, file_format: str):
        """Background process to combine, process and upload audio"""
        try:
            if session_id not in self.session_audio_buffers:
                logger.error(f"Session {session_id} not found for background processing")
                return
            
            session_data = self.session_audio_buffers[session_id]
            
            # Check if we have any audio segments to process
            user_segments = session_data.get('user_audio_segments', [])
            ai_segments = session_data.get('ai_audio_segments', [])
            
            if not user_segments and not ai_segments:
                logger.warning(f"No audio segments found for session {session_id}")
                return
                
            # Generate a placeholder URL immediately to return to client
            from datetime import timedelta
            blob = self.bucket.blob(filename)
            url = blob.generate_signed_url(
                expiration=datetime.now() + timedelta(days=7),  # 7 days
                method='GET'
            )
            
            # Update session data with file info before processing
            session_data['uploaded_file'] = {
                'filename': filename,
                'url': url,
                'format': file_format,
                'processing': True,
                'upload_time': datetime.now().isoformat()
            }
            
            # Combine all audio segments into a single conversation - most time-consuming part
            logger.info(f"Combining audio segments for session {session_id}")
            combined_audio = self._combine_audio_segments(session_data)
            if combined_audio is None:
                logger.error("Failed to combine audio segments")
                return
            
            # Optimize audio quality with reduced processing
            # Only normalize if volume is too low
            if combined_audio.dBFS < -25:
                logger.info(f"Normalizing audio volume for session {session_id}")
                combined_audio = combined_audio.normalize(headroom=1.0)
            
            # Convert to desired format with optimized settings
            logger.info(f"Converting audio to {file_format} for session {session_id}")
            audio_bytes = self._audio_segment_to_bytes(combined_audio, file_format)
            if audio_bytes is None:
                logger.error("Failed to convert audio to bytes")
                return
            
            # Upload to Google Cloud Storage with optimized chunk size
            logger.info(f"Uploading audio to GCS for session {session_id}")
            
            # Set a smaller chunk size for faster uploads of small files
            blob.chunk_size = 262144  # 256 KB chunks (default is 1 MB)
            
            # Upload with retry logic
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    blob.upload_from_string(
                        audio_bytes,
                        content_type=f"audio/{file_format}"
                    )
                    break
                except Exception as upload_error:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise upload_error
                    logger.warning(f"Upload attempt {retry_count} failed, retrying: {upload_error}")
                    time.sleep(1)  # Wait before retry
            
            # Update session data with complete file info
            session_data['uploaded_file'].update({
                'size_bytes': len(audio_bytes),
                'duration_seconds': len(combined_audio) / 1000.0,
                'processing': False
            })
            
            logger.info(f"Successfully uploaded conversation audio for session {session_id}: {filename}")
            logger.info(f"Audio URL: {url}")
            
        except Exception as e:
            logger.error(f"Error in background audio processing: {e}")
        finally:
            # Clean up session buffer
            if session_id in self.session_audio_buffers:
                del self.session_audio_buffers[session_id]
    
    def _bytes_to_audio_segment(self, audio_data: bytes) -> Optional[AudioSegment]:
        """Convert audio bytes to AudioSegment with improved quality"""
        try:
            logger.debug(f"Converting audio data: {len(audio_data)} bytes")
            
            # Try different formats
            formats_to_try = ['wav', 'mp3', 'raw']  # Try WAV first as it's most common
            
            for fmt in formats_to_try:
                try:
                    if fmt == 'raw':
                        # Assume raw PCM data at 16kHz, 16-bit
                        logger.debug(f"Trying raw PCM format")
                        audio_segment = AudioSegment(
                            data=audio_data,
                            sample_width=2,  # 16-bit = 2 bytes
                            frame_rate=self.sample_rate,
                            channels=1
                        )
                    else:
                        logger.debug(f"Trying {fmt} format")
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=fmt)
                    
                    # Normalize to our standard format with higher quality
                    audio_segment = audio_segment.set_frame_rate(44100)  # Higher sample rate for better quality
                    audio_segment = audio_segment.set_channels(1)  # Mono for individual segments
                    
                    # Apply basic noise reduction and clarity enhancement
                    # Normalize to a good volume level
                    if audio_segment.dBFS < -25:
                        audio_segment = audio_segment.normalize(headroom=0.5)
                    
                    # Apply a high-pass filter to remove low-frequency noise
                    try:
                        from pydub.effects import high_pass_filter
                        audio_segment = high_pass_filter(audio_segment, 80)  # Remove very low frequencies
                    except Exception as e:
                        logger.debug(f"Could not apply high-pass filter: {e}")
                    
                    logger.debug(f"✅ Successfully converted audio as {fmt}: {len(audio_segment)}ms")
                    return audio_segment
                    
                except Exception as e:
                    logger.debug(f"Failed to load audio as {fmt}: {e}")
                    continue
            
            logger.error(f"❌ Failed to convert audio data to AudioSegment - tried all formats")
            logger.error(f"Audio data info: {len(audio_data)} bytes, first 20 bytes: {audio_data[:20].hex() if len(audio_data) >= 20 else audio_data.hex()}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error converting bytes to audio segment: {e}")
            return None
    
    def _combine_audio_segments(self, session_data: Dict) -> Optional[AudioSegment]:
        """Combine user and AI audio segments into a single conversation with alternating speakers"""
        try:
            # Get all segments and sort by timestamp
            all_segments = []
            
            # Add user segments
            for segment in session_data['user_audio_segments']:
                all_segments.append(segment)
            
            # Add AI segments  
            for segment in session_data['ai_audio_segments']:
                all_segments.append(segment)
            
            if not all_segments:
                logger.warning("No audio segments to combine")
                return None
            
            # Sort by timestamp
            all_segments.sort(key=lambda x: x['timestamp'])
            
            logger.info(f"Combining {len(all_segments)} segments:")
            for i, seg in enumerate(all_segments):
                logger.info(f"  {i+1}. {seg['type']} - {seg['duration']:.2f}s at {seg['timestamp']:.2f}")
            
            # Create conversation by concatenating segments with minimal gaps
            conversation = AudioSegment.empty()
            gap_duration_ms = 500  # 0.5 second gap between speakers (for better clarity)
            
            # Ensure alternating pattern (AI then user then AI...)
            organized_segments = []
            current_type = None
            last_timestamp = 0
            
            # First pass: organize segments by speaker and remove duplicates
            for segment in all_segments:
                # Skip segments that are too close to the previous segment of the same type
                # This helps eliminate duplicates or echo recordings
                if current_type == segment['type'] and segment['timestamp'] - last_timestamp < 1.0:
                    # Segments less than 1 second apart from same speaker are likely duplicates
                    # Only keep the longer one
                    if organized_segments and segment['duration'] > organized_segments[-1]['duration']:
                        # This segment is longer, replace the previous one
                        logger.info(f"Replacing shorter {segment['type']} segment with longer one")
                        organized_segments[-1] = segment
                    else:
                        logger.info(f"Skipping likely duplicate {segment['type']} segment")
                        continue
                
                # New speaker or sufficient time gap - add to organized segments
                elif current_type is None or segment['type'] != current_type or segment['timestamp'] - last_timestamp >= 1.0:
                    organized_segments.append(segment)
                    current_type = segment['type']
                    last_timestamp = segment['timestamp']
                # Same speaker with small time gap - combine with previous segment
                elif segment['type'] == current_type:
                    if organized_segments:
                        last_segment = organized_segments[-1]
                        # Add a smaller gap between utterances from same speaker
                        small_gap = AudioSegment.silent(duration=200, frame_rate=self.sample_rate)
                        last_segment['audio'] = last_segment['audio'] + small_gap + segment['audio']
                        last_segment['duration'] += segment['duration'] + 0.2  # Add gap duration
                        last_timestamp = segment['timestamp']
            
            # Create an empty conversation to start with
            conversation = AudioSegment.empty()
            
            # Now build the conversation from organized segments
            for i, segment in enumerate(organized_segments):
                # Skip segments with no audio
                if 'audio' not in segment or segment['audio'] is None:
                    logger.warning(f"Skipping segment {i+1} with missing audio")
                    continue
                    
                # Skip segments with zero-length audio
                if len(segment['audio']) == 0:
                    logger.warning(f"Skipping segment {i+1} with zero-length audio")
                    continue
                    
                # Ensure audio is mono and correct sample rate
                audio = segment['audio'].set_channels(1).set_frame_rate(self.sample_rate)
                
                # For user audio, apply enhanced processing to improve clarity
                if segment['type'] == 'user':
                    # Apply more aggressive processing for user audio to improve clarity
                    
                    # Step 1: Normalize audio to a higher level for better clarity
                    audio = audio.normalize(headroom=0.5)  # More aggressive normalization
                    
                    # Step 2: Trim silence more aggressively
                    audio = self._trim_silence(audio, silence_threshold=-30, min_silence_len=200, keep_silence_ms=100)
                    
                    # Step 3: Apply a high-pass filter to remove low-frequency noise
                    try:
                        # Use the imported high_pass_filter function
                        audio = high_pass_filter(audio, 100)  # Higher cutoff for clearer speech
                    except Exception as e:
                        logger.warning(f"Could not apply high-pass filter: {e}")
                    
                    # Step 4: Apply a slight compression to even out volume levels
                    try:
                        # Simulate compression by boosting quieter parts
                        samples = np.array(audio.get_array_of_samples())
                        sample_rate = audio.frame_rate
                        
                        # Convert to float32
                        if samples.dtype == np.int16:
                            samples = samples.astype(np.float32) / 32768.0
                        
                        # Apply soft compression
                        threshold = 0.15
                        ratio = 0.7
                        makeup_gain = 1.2
                        
                        # Simple compression algorithm
                        mask = np.abs(samples) > threshold
                        samples[mask] = threshold + (np.abs(samples[mask]) - threshold) * ratio * np.sign(samples[mask])
                        samples = samples * makeup_gain
                        
                        # Clip to prevent distortion
                        samples = np.clip(samples, -0.95, 0.95)
                        
                        # Convert back to int16
                        samples = (samples * 32768.0).astype(np.int16)
                        
                        # Create new audio segment
                        # Use the already imported AudioSegment from the module level
                        audio = AudioSegment(
                            samples.tobytes(),
                            frame_rate=sample_rate,
                            sample_width=2,
                            channels=1
                        )
                    except Exception as e:
                        logger.warning(f"Could not apply compression: {e}")
                    
                    # Step 5: Final normalization to ensure consistent volume
                    audio = audio.normalize(headroom=0.5)
                    
                else:
                    # For AI audio, just trim silence and normalize
                    audio = self._trim_silence(audio)
                    audio = audio.normalize(headroom=0.7)
                
                # Skip empty segments
                if len(audio) < 100:  # Less than 100ms
                    logger.info(f"Skipping empty segment {i+1}")
                    continue
                
                # Add small gap before segment (except first)
                if len(conversation) > 0:
                    gap = AudioSegment.silent(duration=gap_duration_ms, frame_rate=self.sample_rate)
                    conversation += gap
                
                # Add the audio segment
                conversation += audio
                logger.info(f"Added {segment['type']} segment: {len(audio)}ms")
            
            if len(conversation) == 0:
                logger.warning("No valid audio segments found")
                return None
            
            # Final processing on the complete conversation
            try:
                # Step 1: Normalize to ensure consistent volume
                conversation = conversation.normalize(headroom=0.3)  # More aggressive normalization
                
                # Step 2: Apply a slight EQ to enhance speech clarity
                try:
                    import numpy as np
                    
                    # Convert to numpy array for processing
                    samples = np.array(conversation.get_array_of_samples())
                    sample_rate = conversation.frame_rate
                    
                    # Convert to float32
                    if samples.dtype == np.int16:
                        samples = samples.astype(np.float32) / 32768.0
                    
                    # Apply a simple speech enhancement filter
                    # Boost frequencies in the speech range (1kHz-3kHz)
                    if SCIPY_AVAILABLE:
                        # Design a bandpass filter to enhance speech frequencies
                        try:
                            b, a = signal.butter(2, [1000/(sample_rate/2), 3000/(sample_rate/2)], btype='bandpass')
                            filtered_samples = signal.lfilter(b, a, samples)
                            
                            # Mix original with filtered for a subtle enhancement
                            enhanced_samples = samples * 0.7 + filtered_samples * 0.3
                        except Exception as e:
                            logger.warning(f"Error in speech enhancement filter: {e}")
                            enhanced_samples = samples  # Use original if filter fails
                    else:
                        # If scipy not available, just use the original samples
                        enhanced_samples = samples
                    
                    try:
                        # Convert back to int16
                        enhanced_samples = np.clip(enhanced_samples, -0.95, 0.95)
                        enhanced_samples = (enhanced_samples * 32768.0).astype(np.int16)
                        
                        # Create new audio segment
                        # Use the already imported AudioSegment from the module level
                        conversation = AudioSegment(
                            enhanced_samples.tobytes(),
                            frame_rate=sample_rate,
                            sample_width=2,
                            channels=1
                        )
                        logger.info("Applied speech enhancement filter to final conversation")
                    except Exception as e:
                        logger.warning(f"Could not apply speech enhancement: {e}")
                except Exception as e:
                    logger.warning(f"Could not enhance audio: {e}")
                
                # Step 3: Final normalization
                conversation = conversation.normalize(headroom=0.3)
                
            except Exception as e:
                logger.warning(f"Error in final audio processing, using basic normalization: {e}")
                # Fallback to basic normalization
                conversation = conversation.normalize(headroom=0.5)
            
            logger.info(f"Final conversation: {len(conversation)}ms ({len(conversation)/1000:.1f}s)")
            return conversation
            
        except Exception as e:
            logger.error(f"Error combining audio segments: {e}")
            return None
    
    def _trim_silence(self, audio: AudioSegment, silence_threshold: int = -40, min_silence_len: int = 100, keep_silence_ms: int = 100) -> AudioSegment:
        """
        Remove silence from beginning and end of audio segment
        
        Args:
            audio: The audio segment to trim
            silence_threshold: The threshold (in dB) below which is considered silence
            min_silence_len: Minimum length of silence (in ms) to detect
            keep_silence_ms: How much silence to keep at the beginning and end (in ms)
        """
        try:
            # Find first and last non-silent parts
            start_trim = 0
            end_trim = len(audio)
            
            # Find start of speech (first non-silent chunk)
            chunk_size = min_silence_len  # Use the provided min_silence_len
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if chunk.dBFS > silence_threshold:
                    # Include some silence before speech for natural sound
                    start_trim = max(0, i - keep_silence_ms)
                    break
            
            # Find end of speech (last non-silent chunk)
            for i in range(len(audio) - chunk_size, 0, -chunk_size):
                chunk = audio[i:i+chunk_size]
                if chunk.dBFS > silence_threshold:
                    # Include some silence after speech for natural sound
                    end_trim = min(len(audio), i + chunk_size + keep_silence_ms)
                    break
            
            # Return trimmed audio
            if start_trim < end_trim:
                trimmed = audio[start_trim:end_trim]
                if len(trimmed) > 50:  # At least 50ms
                    return trimmed
            
            # If trimming failed, return original
            return audio
            
        except Exception as e:
            logger.error(f"Error trimming silence: {e}")
            return audio
    
    def _audio_segment_to_bytes(self, audio_segment: AudioSegment, format: str = "wav") -> Optional[bytes]:
        """Convert AudioSegment to bytes in specified format with optimized settings"""
        try:
            buffer = io.BytesIO()
            
            if format.lower() == "mp3":
                # Use a higher bitrate for better audio quality
                # 128k provides better clarity for speech
                audio_segment.export(
                    buffer, 
                    format="mp3", 
                    bitrate="128k",
                    codec="libmp3lame",
                    parameters=["-q:a", "2"]  # Higher quality setting (0-9, lower is better)
                )
            else:
                # For WAV, use higher quality settings
                audio_segment.export(
                    buffer, 
                    format="wav",
                    parameters=["-acodec", "pcm_s16le", "-ar", "44100"]  # Higher sample rate
                )
            
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            logger.error(f"Error converting audio segment to bytes: {e}")
            return None
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a session's audio recording"""
        if session_id not in self.session_audio_buffers:
            return None
        
        session_data = self.session_audio_buffers[session_id]
        return {
            'session_id': session_id,
            'persona': session_data['persona'],
            'user_segments': len(session_data['user_audio_segments']),
            'ai_segments': len(session_data['ai_audio_segments']),
            'total_user_duration': sum(seg['duration'] for seg in session_data['user_audio_segments']),
            'total_ai_duration': sum(seg['duration'] for seg in session_data['ai_audio_segments']),
            'recording_duration': time.time() - session_data['start_time']
        }
    
    def list_conversation_files(self, limit: int = 50) -> List[Dict]:
        """List uploaded conversation files"""
        try:
            if not self.bucket:
                return []
            
            blobs = self.bucket.list_blobs(prefix="conversations/", max_results=limit)
            files = []
            
            for blob in blobs:
                files.append({
                    'filename': blob.name,
                    'size_bytes': blob.size,
                    'created': blob.time_created.isoformat() if blob.time_created else None,
                    'updated': blob.updated.isoformat() if blob.updated else None,
                    'content_type': blob.content_type,
                    'public_url': blob.public_url if blob.public_url else None
                })
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing conversation files: {e}")
            return []
    
    def delete_conversation_file(self, filename: str) -> bool:
        """Delete a conversation file from storage"""
        try:
            if not self.bucket:
                return False
            
            blob = self.bucket.blob(filename)
            blob.delete()
            
            logger.info(f"Deleted conversation file: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting conversation file {filename}: {e}")
            return False

# Global storage service instance
audio_storage_service = AudioConversationStorage()

# Convenience functions for external use
def start_session_recording(session_id: str, persona: str = "friendly") -> bool:
    """Start recording audio for a session"""
    return audio_storage_service.start_session_recording(session_id, persona)

def add_user_audio(session_id: str, audio_data: bytes, timestamp: float = None) -> bool:
    """Add user audio segment to session"""
    return audio_storage_service.add_user_audio(session_id, audio_data, timestamp)

def add_ai_audio(session_id: str, audio_data: bytes, timestamp: float = None) -> bool:
    """Add AI audio segment to session"""
    return audio_storage_service.add_ai_audio(session_id, audio_data, timestamp)

def finalize_session_recording(session_id: str, use_mp3: bool = False) -> Optional[str]:
    """Finalize session recording and get URL"""
    return audio_storage_service.finalize_session_recording(session_id, use_mp3)

def get_session_audio_info(session_id: str) -> Optional[Dict]:
    """Get session audio information"""
    return audio_storage_service.get_session_info(session_id)

def get_storage_stats() -> Dict:
    """Get storage statistics for monitoring"""
    try:
        active_sessions = len(audio_storage_service.session_audio_buffers)
        total_user_segments = sum(
            len(session_data.get('user_audio_segments', []))
            for session_data in audio_storage_service.session_audio_buffers.values()
        )
        total_ai_segments = sum(
            len(session_data.get('ai_audio_segments', []))
            for session_data in audio_storage_service.session_audio_buffers.values()
        )
        
        return {
            "active_sessions": active_sessions,
            "total_user_segments": total_user_segments,
            "total_ai_segments": total_ai_segments,
            "gcs_available": audio_storage_service.bucket is not None,
            "bucket_name": audio_storage_service.bucket_name
        }
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        return {
            "active_sessions": 0,
            "error": str(e)
        }

def list_conversation_files(limit: int = 50) -> List[Dict]:
    """List all conversation files"""
    return audio_storage_service.list_conversation_files(limit)

def delete_conversation_file(filename: str) -> bool:
    """Delete a conversation file"""
    return audio_storage_service.delete_conversation_file(filename)