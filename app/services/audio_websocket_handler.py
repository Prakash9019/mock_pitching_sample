#!/usr/bin/env python3
"""
WebSocket handler for real-time audio streaming and VAD
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Dict, Any, Optional
from datetime import datetime
import socketio
import numpy as np

from .voice_activity_detection import get_audio_processor
from .transcription import transcribe_audio
from .langgraph_workflow import get_pitch_workflow
from .audio_conversation_storage import (
    start_session_recording, add_user_audio, add_ai_audio, 
    finalize_session_recording, get_session_audio_info
)

logger = logging.getLogger(__name__)

class AudioWebSocketHandler:
    """
    Handles WebSocket connections for real-time audio streaming
    """
    
    def __init__(self, sio: socketio.AsyncServer):
        self.sio = sio
        self.audio_processor = get_audio_processor()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Set up transcription callback
        self.audio_processor.set_transcription_callback(self._on_transcription_ready)
        
        # Register WebSocket event handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register WebSocket event handlers"""
        
        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection"""
            logger.info(f"Audio WebSocket client connected: {sid}")
            await self.sio.emit('audio_connection_status', {
                'status': 'connected',
                'message': 'Audio streaming ready'
            }, room=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            logger.info(f"Audio WebSocket client disconnected: {sid}")
            
            # Clean up any active sessions for this client
            sessions_to_remove = []
            for session_id, session_data in self.active_sessions.items():
                if session_data.get('socket_id') == sid:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                await self._cleanup_session(session_id)
        
        @self.sio.event
        async def start_audio_session(sid, data):
            """Start a new audio streaming session"""
            try:
                session_id = data.get('session_id')
                persona = data.get('persona', 'friendly')
                
                if not session_id:
                    await self.sio.emit('audio_error', {
                        'error': 'session_id required'
                    }, room=sid)
                    return
                
                # Initialize pitch workflow
                workflow = get_pitch_workflow()
                result = workflow.start_session(session_id, persona)
                
                # Start audio processing
                self.audio_processor.start_session(session_id)
                
                # Start audio conversation recording
                audio_recording_started = start_session_recording(session_id, persona)
                if not audio_recording_started:
                    logger.warning(f"Failed to start audio recording for session {session_id}")
                
                # Track session
                self.active_sessions[session_id] = {
                    'socket_id': sid,
                    'persona': persona,
                    'status': 'active',
                    'workflow': workflow,
                    'audio_recording_enabled': audio_recording_started,
                    'start_time': time.time(),
                    'last_audio_time': time.time(),
                    'audio_buffer': b'',  # Buffer for continuous audio capture
                    'buffer_duration': 0,  # Track buffer duration in seconds
                    'ai_speaking': False,  # Track if AI is currently speaking
                    'user_can_speak': True,  # Track if user should be recorded
                    'last_ai_end_time': 0,  # When AI finished speaking
                    'conversation_turn': 'user'  # Current conversation turn: 'user' or 'ai'
                }
                
                # Generate TTS for the initial greeting message
                greeting_message = result.get('message', '')
                if greeting_message:
                    logger.info(f"Generating TTS for initial greeting: '{greeting_message[:50]}...'")
                    
                    # CRITICAL FIX: Allow user to speak during initial greeting
                    # Don't block user recording for the initial greeting - user might want to interrupt
                    await self._generate_tts_for_response_non_blocking(session_id, greeting_message, sid)
                    
                    # Set a fallback timer to ensure user recording is enabled
                    import asyncio
                    asyncio.create_task(self._fallback_enable_user_recording(session_id, 5.0))  # 5 second fallback
                else:
                    await self.sio.emit('audio_session_started', {
                        'session_id': session_id,
                        'message': '',
                        'status': 'ready_for_audio'
                    }, room=sid)
                
                logger.info(f"Audio session started: {session_id} for client {sid}")
                
            except Exception as e:
                logger.error(f"Error starting audio session: {e}")
                await self.sio.emit('audio_error', {
                    'error': f'Failed to start session: {str(e)}'
                }, room=sid)
        
        @self.sio.event
        async def audio_stream(sid, data):
            """Handle incoming audio stream data"""
            try:
                session_id = data.get('session_id')
                audio_data = data.get('audio_data')  # Base64 encoded audio
                
                logger.info(f"Received audio stream for session {session_id}, data length: {len(audio_data) if audio_data else 0}")
                
                if not session_id or session_id not in self.active_sessions:
                    logger.warning(f"Invalid session: {session_id}")
                    await self.sio.emit('audio_error', {
                        'error': 'Invalid session'
                    }, room=sid)
                    return
                
                if not audio_data:
                    logger.warning("No audio data received")
                    return
                
                # Get session data
                session_data = self.active_sessions.get(session_id, {})
                
                # Track last processed chunk to avoid duplicates
                if not hasattr(session_data, 'last_chunk_hash'):
                    session_data['last_chunk_hash'] = None
                
                # Decode audio data
                import base64
                import numpy as np
                import hashlib
                
                try:
                    # Decode base64 audio data
                    audio_bytes = base64.b64decode(audio_data)
                    logger.info(f"Decoded audio: {len(audio_bytes)} bytes")
                    
                    # Generate a hash of the first part of the audio to detect duplicates
                    chunk_hash = hashlib.md5(audio_bytes[:500]).hexdigest()
                    
                    # Skip if this is a duplicate of the last chunk
                    if chunk_hash == session_data.get('last_chunk_hash'):
                        logger.info(f"Skipping duplicate audio chunk for session {session_id}")
                        return
                        
                    # Update last chunk hash
                    session_data['last_chunk_hash'] = chunk_hash
                    
                    # Convert from Int16 to proper PCM format for VAD
                    # The browser sends Int16 data, convert to bytes
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    # Check audio quality - skip processing for very low volume audio
                    audio_float = audio_array.astype(np.float32) / 32768.0
                    rms_volume = np.sqrt(np.mean(np.square(audio_float)))
                    
                    # Skip very quiet audio (likely background noise)
                    if rms_volume < 0.01:  # Very low volume threshold
                        logger.info(f"Skipping very low volume audio: RMS={rms_volume:.4f}")
                        return
                    
                    # Get sample rate from frontend or detect it
                    original_sample_rate = data.get('sample_rate', 44100)  # Use frontend-provided rate
                    if not original_sample_rate or original_sample_rate <= 0:
                        # Fallback to detection if not provided
                        original_sample_rate = self._detect_sample_rate(len(audio_array), data.get('duration', 0.1))
                    
                    target_sample_rate = 16000  # VAD processing requires 16kHz
                    
                    if original_sample_rate != target_sample_rate:
                        # Resample audio to target rate
                        audio_array = self._resample_audio(audio_array, original_sample_rate, target_sample_rate)
                        logger.info(f"Resampled audio: {original_sample_rate}Hz → {target_sample_rate}Hz")
                    
                    pcm_audio = audio_array.tobytes()
                    
                    logger.info(f"Processed audio: {len(pcm_audio)} bytes, {len(audio_array)} samples at {target_sample_rate}Hz")
                    
                except Exception as e:
                    logger.error(f"Error decoding/converting audio data: {e}")
                    return
                
                # Check if AI is currently speaking - if so, be more selective about processing
                if session_data.get('ai_speaking', False):
                    # Only process audio if it's likely to be user speech (higher volume threshold)
                    if rms_volume < 0.03:  # Higher threshold during AI speech
                        logger.info(f"AI is speaking - skipping low volume audio: RMS={rms_volume:.4f}")
                        return
                    else:
                        logger.info(f"Processing user audio during AI speech (high volume): RMS={rms_volume:.4f}")
                
                # Process audio through VAD if available
                if self.audio_processor:
                    vad_result = self.audio_processor.process_audio_stream(pcm_audio)
                    
                    # Log VAD activity
                    if vad_result.get('is_speech') or vad_result.get('action'):
                        logger.info(f"🎙️ VAD ACTIVITY for {session_id}: {vad_result}")
                    
                    # Send VAD status to client
                    if vad_result.get('action'):
                        logger.info(f"🔊 VAD ACTION for {session_id}: {vad_result['action']}")
                        
                        await self.sio.emit('vad_status', {
                            'session_id': session_id,
                            'action': vad_result['action'],
                            'is_speaking': vad_result['is_speaking'],
                            'speech_duration': vad_result.get('speech_duration', 0)
                        }, room=sid)
                        
                        # If speech ended, process the accumulated audio
                        if vad_result['action'] == 'speech_ended' and vad_result.get('audio_data'):
                            logger.info(f"🎤 VAD DETECTED USER SPEECH ENDED: {len(vad_result['audio_data'])} bytes for session {session_id}")
                            
                            # Process the speech segment
                            await self._process_speech_segment(session_id, vad_result['audio_data'], sid)
                        elif vad_result['action'] == 'speech_ended':
                            logger.warning(f"⚠️ VAD detected speech ended but NO AUDIO DATA for session {session_id}")
                    
                    # Only store raw audio if VAD detects speech and AI is not speaking
                    # This helps prevent duplicate recordings and feedback loops
                    if (vad_result.get('is_speech', False) and 
                        session_data.get('audio_recording_enabled', False) and 
                        len(pcm_audio) > 1600 and
                        not session_data.get('ai_speaking', False)):
                        
                        # Store raw audio directly (bypassing VAD)
                        audio_stored = add_user_audio(session_id, pcm_audio, time.time())
                        if audio_stored:
                            logger.info(f"✅ Raw user audio chunk stored for session {session_id}: {len(pcm_audio)} bytes")
                
                else:
                    # No VAD processor available - process audio directly
                    logger.warning(f"⚠️ NO VAD PROCESSOR AVAILABLE for session {session_id}")
                    logger.info(f"🔄 Processing raw audio directly: {len(pcm_audio)} bytes")
                    
                    # Store raw audio as user speech only if AI is not speaking
                    if len(pcm_audio) > 1600 and not session_data.get('ai_speaking', False):  # At least 100ms of audio at 16kHz
                        await self._process_speech_segment(session_id, pcm_audio, sid)
                
            except Exception as e:
                logger.error(f"Error processing audio stream: {e}")
                await self.sio.emit('audio_error', {
                    'error': f'Audio processing error: {str(e)}'
                }, room=sid)
        
        @self.sio.event
        async def stop_audio_session(sid, data):
            """Stop audio streaming session"""
            try:
                session_id = data.get('session_id')
                
                if session_id and session_id in self.active_sessions:
                    await self._cleanup_session(session_id)
                    
                    await self.sio.emit('audio_session_stopped', {
                        'session_id': session_id,
                        'status': 'stopped'
                    }, room=sid)
                
            except Exception as e:
                logger.error(f"Error stopping audio session: {e}")
                await self.sio.emit('audio_error', {
                    'error': f'Failed to stop session: {str(e)}'
                }, room=sid)
        
        @self.sio.event
        async def ai_audio_finished(sid, data):
            """Handle notification that AI audio finished playing"""
            try:
                session_id = data.get('session_id')
                
                if not session_id or session_id not in self.active_sessions:
                    await self.sio.emit('audio_error', {
                        'error': 'Invalid session'
                    }, room=sid)
                    return
                
                # AI finished speaking - resume user recording
                self.set_ai_speaking_state(session_id, False)
                logger.info(f"AI audio finished playing for session {session_id}")
                
            except Exception as e:
                logger.error(f"Error handling AI audio finished: {e}")
                await self.sio.emit('audio_error', {
                    'error': f'AI audio finish error: {str(e)}'
                }, room=sid)
        
        @self.sio.event
        async def manual_transcription(sid, data):
            """Handle manual transcription request (fallback)"""
            try:
                session_id = data.get('session_id')
                text = data.get('text')
                
                if not session_id or session_id not in self.active_sessions:
                    await self.sio.emit('audio_error', {
                        'error': 'Invalid session'
                    }, room=sid)
                    return
                
                if not text:
                    await self.sio.emit('audio_error', {
                        'error': 'Text required'
                    }, room=sid)
                    return
                
                # Process the text directly
                await self._process_transcription(session_id, text)
                
            except Exception as e:
                logger.error(f"Error processing manual transcription: {e}")
                await self.sio.emit('audio_error', {
                    'error': f'Transcription error: {str(e)}'
                }, room=sid)
        
        @self.sio.event
        async def test_tts(sid, data):
            """Handle test TTS request"""
            try:
                text = data.get('text', 'This is a test of the text-to-speech system.')
                persona = data.get('persona', 'friendly')
                
                logger.info(f"Test TTS request: {text} with persona: {persona}")
                
                # Generate TTS audio
                import tempfile
                import base64
                import os
                from .enhanced_text_to_speech import convert_text_to_speech_with_persona
                
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    temp_audio_path = temp_file.name
                
                try:
                    # Generate TTS audio using enhanced_text_to_speech
                    logger.info(f"Attempting to generate TTS for: '{text}' with persona: {persona}")
                    audio_data = convert_text_to_speech_with_persona(text, persona, temp_audio_path)
                    
                    if audio_data is None:
                        logger.error("TTS generation returned None - Google Cloud TTS failed")
                        await self.sio.emit('audio_error', {
                            'error': 'Google Cloud TTS failed - check credentials'
                        }, room=sid)
                        return
                    
                    # Check if file was created
                    if not os.path.exists(temp_audio_path):
                        logger.error(f"TTS audio file not created at: {temp_audio_path}")
                        await self.sio.emit('audio_error', {
                            'error': 'TTS audio file not generated'
                        }, room=sid)
                        return
                    
                    # Read audio file and send to client
                    with open(temp_audio_path, 'rb') as audio_file:
                        audio_file_data = audio_file.read()
                    
                    if len(audio_file_data) == 0:
                        logger.error("Generated audio file is empty")
                        await self.sio.emit('audio_error', {
                            'error': 'Generated audio file is empty'
                        }, room=sid)
                        return
                    
                    audio_base64 = base64.b64encode(audio_file_data).decode('utf-8')
                    
                    await self.sio.emit('test_tts_response', {
                        'text': text,
                        'persona': persona,
                        'audio_data': audio_base64
                    }, room=sid)
                    
                    logger.info(f"Test TTS response sent successfully - audio size: {len(audio_file_data)} bytes")
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"Error processing test TTS: {e}")
                await self.sio.emit('audio_error', {
                    'error': f'Test TTS error: {str(e)}'
                }, room=sid)
        
        @self.sio.event
        async def debug_ping(sid, data):
            """Handle debug ping for WebSocket testing"""
            logger.info(f"Debug ping received from {sid}: {data}")
            await self.sio.emit('debug_pong', {
                'timestamp': data.get('timestamp'),
                'server_time': __import__('time').time() * 1000,
                'message': 'WebSocket connection is working!'
            }, room=sid)

        @self.sio.event
        async def force_enable_user_recording(sid, data):
            """Force enable user recording if it gets stuck"""
            try:
                session_id = data.get('session_id')
                if not session_id:
                    await self.sio.emit('audio_error', {
                        'error': 'session_id required'
                    }, room=sid)
                    return
                
                if session_id in self.active_sessions:
                    session_data = self.active_sessions[session_id]
                    session_data['user_can_speak'] = True
                    session_data['ai_speaking'] = False
                    session_data['conversation_turn'] = 'user'
                    
                    logger.info(f"🔧 Force-enabled user recording for session {session_id}")
                    await self.sio.emit('user_recording_status', {
                        'session_id': session_id,
                        'user_can_speak': True,
                        'ai_speaking': False,
                        'message': 'User recording force-enabled'
                    }, room=sid)
                else:
                    await self.sio.emit('audio_error', {
                        'error': f'Session {session_id} not found'
                    }, room=sid)
                    
            except Exception as e:
                logger.error(f"Error force-enabling user recording: {e}")
                await self.sio.emit('audio_error', {
                    'error': f'Force enable error: {str(e)}'
                }, room=sid)

        @self.sio.event
        async def test_user_audio_storage(sid, data):
            """Test user audio storage with fake data"""
            try:
                session_id = data.get('session_id')
                if not session_id:
                    await self.sio.emit('audio_error', {
                        'error': 'session_id required'
                    }, room=sid)
                    return
                
                # Create fake user audio data
                fake_audio = b"fake_user_audio_data_for_testing" * 100  # Make it bigger
                
                logger.info(f"🧪 TESTING user audio storage for session {session_id}")
                logger.info(f"   Fake audio size: {len(fake_audio)} bytes")
                
                # Force process as user speech
                await self._process_speech_segment(session_id, fake_audio, sid)
                
                await self.sio.emit('test_result', {
                    'session_id': session_id,
                    'message': 'Test user audio storage completed',
                    'audio_size': len(fake_audio)
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error testing user audio storage: {e}")
                await self.sio.emit('audio_error', {
                    'error': f'Test error: {str(e)}'
                }, room=sid)
    
    async def _on_transcription_ready(self, session_id: str, audio_file_path: str):
        """Called when audio is ready for transcription"""
        try:
            logger.info(f"Transcribing audio for session {session_id}")
            
            # Transcribe the audio
            transcript_result = transcribe_audio(audio_file_path)
            
            # Clean up temporary file immediately
            try:
                os.unlink(audio_file_path)
            except:
                pass
            
            # Check if transcription is in progress (another thread is handling it)
            if transcript_result and transcript_result.get('status') == 'in_progress':
                logger.info(f"Transcription already in progress for session {session_id}, skipping duplicate")
                return
                
            # Check if we have valid text
            transcript_text = transcript_result.get('text', '') if transcript_result else ''
            
            if transcript_text and transcript_text not in ['[No speech detected]', '[Transcription error]']:
                # Process valid transcription
                await self._process_transcription(session_id, transcript_text)
            elif transcript_text == '[No speech detected]':
                # Just log this, don't notify client for no speech
                logger.info(f"No speech detected in audio for session {session_id}")
            else:
                # Notify client of transcription failure
                session_data = self.active_sessions.get(session_id)
                if session_data:
                    await self.sio.emit('transcription_error', {
                        'session_id': session_id,
                        'error': 'Failed to transcribe audio'
                    }, room=session_data['socket_id'])
            
        except Exception as e:
            logger.error(f"Error in transcription callback: {e}")
    
    async def _process_transcription(self, session_id: str, transcript_text: str):
        """Process transcribed text through the pitch workflow"""
        try:
            session_data = self.active_sessions.get(session_id)
            if not session_data:
                logger.error(f"Session not found: {session_id}")
                return
            
            # Skip empty or very short transcripts
            if not transcript_text or len(transcript_text.strip()) < 2:
                logger.info(f"Skipping empty or very short transcript for session {session_id}")
                return
                
            logger.info(f"Processing transcript for session {session_id}: {transcript_text[:50]}...")
            
            # Send transcript to client
            await self.sio.emit('transcription_result', {
                'session_id': session_id,
                'transcript': transcript_text
            }, room=session_data['socket_id'])
            
            # Process through pitch workflow
            workflow = session_data['workflow']
            response = workflow.process_message(session_id, transcript_text)
            
            if 'error' in response:
                await self.sio.emit('ai_error', {
                    'session_id': session_id,
                    'error': response['error']
                }, room=session_data['socket_id'])
                return
            
            # Send AI response to client
            await self.sio.emit('ai_response', {
                'session_id': session_id,
                'message': response['message'],
                'stage': response['stage'],
                'analysis': response.get('analysis', {}),
                'ready_for_audio': True  # Ready for next audio input
            }, room=session_data['socket_id'])
            
            logger.info(f"AI response sent for session {session_id}")
            
            # Generate TTS audio for the AI response (use blocking version for conversation responses)
            await self._generate_tts_for_response(session_id, response['message'], session_data['socket_id'])
            
        except Exception as e:
            logger.error(f"Error processing transcription: {e}")
            session_data = self.active_sessions.get(session_id)
            if session_data:
                await self.sio.emit('ai_error', {
                    'session_id': session_id,
                    'error': f'Processing error: {str(e)}'
                }, room=session_data['socket_id'])
    
    async def _cleanup_session(self, session_id: str):
        """Clean up a session"""
        try:
            if session_id in self.active_sessions:
                session_data = self.active_sessions[session_id]
                
                # FLUSH ANY REMAINING AUDIO BUFFER FIRST
                if session_data.get('audio_buffer') and len(session_data['audio_buffer']) > 0:
                    logger.info(f"Flushing remaining audio buffer for session {session_id}")
                    await self._flush_audio_buffer(session_id)
                
                # Always try to finalize audio recording
                # Even if audio_recording_enabled is False, we might have added audio manually
                logger.info(f"Finalizing audio recording for session {session_id}")
                
                # Always use MP3 for better compatibility and smaller file size
                use_mp3 = True
                
                # Get audio info before finalizing
                audio_info = get_session_audio_info(session_id)
                
                # If no audio info, try to start recording as a fallback
                if not audio_info:
                    from app.services.audio_conversation_storage import start_session_recording
                    logger.warning(f"No audio info found for session {session_id}, attempting fallback recording")
                    start_session_recording(session_id, session_data.get('persona', 'friendly'))
                    audio_info = get_session_audio_info(session_id)
                
                # Finalize and upload audio conversation
                audio_url = finalize_session_recording(session_id, use_mp3=use_mp3)
                if audio_url:
                    logger.info(f"✅ Audio conversation uploaded successfully: {audio_url}")
                    
                    # Save audio conversation data to database
                    await self._save_audio_conversation_to_db(session_id, audio_url, audio_info, use_mp3)
                    
                    # Store the URL in a global cache for the API endpoint to access
                    if not hasattr(self, 'audio_url_cache'):
                        self.audio_url_cache = {}
                    self.audio_url_cache[session_id] = audio_url
                    logger.info(f"✅ Cached audio URL for session {session_id}: {audio_url}")
                else:
                    logger.error(f"❌ Failed to upload audio conversation for session {session_id}")
                
                # Stop audio processing
                self.audio_processor.stop_session()
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                
                logger.info(f"Session cleaned up: {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
    
    async def _save_audio_conversation_to_db(self, session_id: str, audio_url: str, audio_info: Dict, use_mp3: bool):
        """Save audio conversation data to database"""
        try:
            from app.database import get_database_service
            
            db_service = get_database_service()
            if not db_service:
                logger.error("Database service not available")
                return
            
            # Prepare audio conversation data
            audio_data = {
                'session_id': session_id,
                'audio_file_url': audio_url,
                'audio_filename': audio_url.split('/')[-1].split('?')[0] if audio_url else None,
                'audio_format': 'mp3' if use_mp3 else 'wav',
                'total_duration_seconds': audio_info.get('recording_duration', 0),
                'user_speaking_duration': audio_info.get('total_user_duration', 0),
                'ai_speaking_duration': audio_info.get('total_ai_duration', 0),
                'user_audio_segments': audio_info.get('user_segments', 0),
                'ai_audio_segments': audio_info.get('ai_segments', 0),
                'total_segments': audio_info.get('user_segments', 0) + audio_info.get('ai_segments', 0),
                'storage_provider': 'google_cloud_storage',
                'bucket_name': os.getenv('GCS_BUCKET_NAME', 'ai-pitch-conversations'),
                'upload_timestamp': datetime.now(),
                'included_in_analysis': True,
                'analysis_notes': f"Audio conversation recorded with {audio_info.get('persona', 'unknown')} persona"
            }
            
            # Save to database
            audio_conversation_id = await db_service.save_audio_conversation(audio_data)
            logger.info(f"Audio conversation saved to database: {audio_conversation_id}")
            
        except Exception as e:
            logger.error(f"Error saving audio conversation to database: {e}")
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session"""
        return self.active_sessions.get(session_id)
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions"""
        return self.active_sessions.copy()
    
    async def _flush_audio_buffer(self, session_id: str):
        """Flush accumulated audio buffer to storage"""
        try:
            session_data = self.active_sessions.get(session_id, {})
            if not session_data or not session_data.get('audio_buffer'):
                return
            
            buffer_data = session_data['audio_buffer']
            buffer_duration = session_data['buffer_duration']
            
            logger.info(f"Flushing audio buffer for {session_id}: {len(buffer_data)} bytes, {buffer_duration:.1f}s")
            
            # Store the buffered audio
            from app.services.audio_conversation_storage import add_user_audio
            audio_stored = add_user_audio(session_id, buffer_data, time.time())
            
            if audio_stored:
                logger.info(f"✅ Buffered user audio stored for session {session_id}")
                # Clear the buffer
                session_data['audio_buffer'] = b''
                session_data['buffer_duration'] = 0
            else:
                logger.warning(f"❌ Failed to store buffered audio for session {session_id}")
                
        except Exception as e:
            logger.error(f"Error flushing audio buffer: {e}")
    
    def _detect_sample_rate(self, num_samples: int, duration_seconds: float) -> int:
        """Detect the sample rate based on number of samples and duration"""
        if duration_seconds <= 0:
            # Default assumption for browser audio
            return 44100
        
        calculated_rate = int(num_samples / duration_seconds)
        
        # Round to common sample rates
        common_rates = [8000, 16000, 22050, 44100, 48000]
        closest_rate = min(common_rates, key=lambda x: abs(x - calculated_rate))
        
        logger.info(f"Detected sample rate: {calculated_rate}Hz → {closest_rate}Hz")
        return closest_rate
    
    def _resample_audio(self, audio_array: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio array from one sample rate to another"""
        try:
            if from_rate == to_rate:
                return audio_array
            
            # Try to use scipy for better quality resampling
            try:
                from scipy import signal
                # Use scipy's high-quality resampling
                original_length = len(audio_array)
                target_length = int(original_length * to_rate / from_rate)
                resampled = signal.resample(audio_array.astype(np.float32), target_length)
                resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
                logger.info(f"Resampled with scipy: {original_length} → {target_length} samples")
                return resampled
            except ImportError:
                logger.info("scipy not available, using linear interpolation")
            
            # Fallback to linear interpolation
            original_length = len(audio_array)
            target_length = int(original_length * to_rate / from_rate)
            
            # Create indices for interpolation
            original_indices = np.linspace(0, original_length - 1, original_length)
            target_indices = np.linspace(0, original_length - 1, target_length)
            
            # Interpolate
            resampled = np.interp(target_indices, original_indices, audio_array.astype(np.float32))
            
            # Convert back to int16
            resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
            
            logger.info(f"Resampled with interpolation: {original_length} → {target_length} samples")
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_array
    
    def set_ai_speaking_state(self, session_id: str, is_speaking: bool):
        """Set whether AI is currently speaking (to pause user recording)"""
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            session_data['ai_speaking'] = is_speaking
            
            if is_speaking:
                # AI started speaking - pause user recording
                session_data['user_can_speak'] = False
                session_data['conversation_turn'] = 'ai'
                logger.info(f"🤖 AI speaking started - pausing user recording for {session_id}")
            else:
                # AI finished speaking - resume user recording after short delay
                session_data['last_ai_end_time'] = time.time()
                session_data['conversation_turn'] = 'user'
                logger.info(f"👤 AI speaking ended - resuming user recording for {session_id}")
                
                # Small delay to avoid capturing AI audio tail
                import asyncio
                asyncio.create_task(self._resume_user_recording_delayed(session_id, 0.5))
    
    async def _resume_user_recording_delayed(self, session_id: str, delay_seconds: float):
        """Resume user recording after a delay"""
        await asyncio.sleep(delay_seconds)
        if session_id in self.active_sessions:
            # Double-check that AI is not still speaking before resuming
            session_data = self.active_sessions[session_id]
            if not session_data.get('ai_speaking', False):
                session_data['user_can_speak'] = True
                session_data['conversation_turn'] = 'user'
                logger.info(f"✅ User recording resumed for {session_id}")
            else:
                logger.info(f"⏸️ Delaying user recording resume - AI still speaking for {session_id}")
                # Try again in 1 second
                asyncio.create_task(self._resume_user_recording_delayed(session_id, 1.0))

    async def _fallback_enable_user_recording(self, session_id: str, delay_seconds: float):
        """Fallback to enable user recording if frontend doesn't call ai_audio_finished"""
        await asyncio.sleep(delay_seconds)
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            if session_data.get('ai_speaking', False):
                logger.warning(f"🔧 FALLBACK: Force-enabling user recording for {session_id} (AI speaking state stuck)")
                session_data['ai_speaking'] = False
                session_data['user_can_speak'] = True
                session_data['conversation_turn'] = 'user'
                
                # Notify frontend
                await self.sio.emit('user_recording_status', {
                    'session_id': session_id,
                    'user_can_speak': True,
                    'ai_speaking': False,
                    'message': 'User recording enabled (fallback)'
                }, room=session_data.get('socket_id'))

    async def _generate_tts_for_response_non_blocking(self, session_id: str, ai_response: str, socket_id: str):
        """Generate TTS audio for an AI response without blocking user recording"""
        try:
            # Get session data
            session_data = self.active_sessions.get(session_id, {})
            persona = session_data.get('persona', 'friendly')
            
            # Import TTS service
            from app.services.enhanced_text_to_speech import convert_text_to_speech_with_persona
            
            # Convert to speech and send audio
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_audio_path = temp_file.name
            
            try:
                # DON'T set AI speaking state for initial greeting - allow user to interrupt
                logger.info(f"Generating NON-BLOCKING TTS for response: '{ai_response[:50]}...' with persona: {persona}")
                audio_result = convert_text_to_speech_with_persona(ai_response, persona, temp_audio_path)
                
                if audio_result is None:
                    logger.error("Google Cloud TTS failed for non-blocking response - no audio generated")
                    return
                
                # Check if file was created and has content
                if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                    logger.error(f"TTS audio file not created or empty: {temp_audio_path}")
                    return
                
                # Read audio file and send to client
                with open(temp_audio_path, 'rb') as audio_file:
                    audio_file_data = audio_file.read()
                
                # Store AI audio in conversation storage
                if session_data.get('audio_recording_enabled', False):
                    logger.info(f"🤖 Storing NON-BLOCKING AI audio for session {session_id}: {len(audio_file_data)} bytes")
                    ai_audio_stored = add_ai_audio(session_id, audio_file_data, time.time())
                    if ai_audio_stored:
                        logger.info(f"✅ Non-blocking AI audio stored successfully for session {session_id}")
                    else:
                        logger.error(f"❌ Failed to store non-blocking AI audio for session {session_id}")
                else:
                    logger.warning(f"⚠️ Audio recording not enabled for non-blocking AI audio in session {session_id}")
                
                import base64
                audio_base64 = base64.b64encode(audio_file_data).decode('utf-8')
                
                logger.info(f"Sending non-blocking ai_response event to room: {socket_id}")
                await self.sio.emit('ai_response', {
                    'session_id': session_id,
                    'audio_data': audio_base64,
                    'message': ai_response,
                    'non_blocking': True  # Flag to indicate this doesn't block user recording
                }, room=socket_id)
                
                logger.info(f"Non-blocking TTS audio response sent successfully - size: {len(audio_file_data)} bytes")
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error generating non-blocking AI response: {e}")
            await self.sio.emit('audio_error', {
                'error': f'Non-blocking AI response error: {str(e)}'
            }, room=socket_id)
    
    # Removed force_flush_user_audio - no longer needed with VAD-only approach
    
    async def _process_speech_segment(self, session_id: str, audio_data: bytes, socket_id: str):
        """Process a complete speech segment - store audio and trigger transcription"""
        try:
            # Use a more efficient logging approach
            logger.info(f"Processing speech segment for session {session_id}: {len(audio_data)} bytes")
            
            # Get session data
            session_data = self.active_sessions.get(session_id, {})
            if not session_data:
                logger.error(f"Session {session_id} not found")
                return
                
            # Check if we should process this segment
            # Only log state if debug logging is enabled
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🔍 State for {session_id}: AI speaking={session_data.get('ai_speaking', False)}, "
                           f"User can speak={session_data.get('user_can_speak', True)}, "
                           f"Turn={session_data.get('conversation_turn', 'unknown')}")
            
            # Track the last processed audio hash to prevent duplicates
            if not hasattr(session_data, 'last_audio_hash'):
                session_data['last_audio_hash'] = None
                
            # Generate a simple hash of the audio data to detect duplicates
            import hashlib
            current_audio_hash = hashlib.md5(audio_data[:1000]).hexdigest()  # Hash first 1000 bytes
            
            # Check if this is a duplicate of the last processed audio
            if session_data.get('last_audio_hash') == current_audio_hash:
                logger.info(f"⚠️ Skipping duplicate audio segment for session {session_id}")
                return
                
            # Update the last processed audio hash
            session_data['last_audio_hash'] = current_audio_hash
            
            # Store user audio in conversation storage FIRST
            # This is always done regardless of state to ensure complete conversation recording
            if session_data.get('audio_recording_enabled', False):
                # Check if AI is currently speaking - if so, we might want to skip recording
                # to avoid capturing AI audio feedback
                if session_data.get('ai_speaking', False):
                    logger.info(f"⚠️ AI is speaking - checking audio quality before storing")
                    
                    # Check audio quality - only store if it's likely to be user speech
                    # Convert to numpy array for analysis
                    import numpy as np
                    try:
                        # Convert from Int16 to float32
                        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Calculate RMS volume
                        rms = np.sqrt(np.mean(np.square(audio_array)))
                        
                        # Only store if volume is significant (likely user speech, not feedback)
                        if rms > 0.05:  # Threshold for significant volume
                            audio_stored = add_user_audio(session_id, audio_data, time.time())
                            if audio_stored:
                                logger.info(f"✅ User audio stored for session {session_id} (volume: {rms:.4f})")
                            else:
                                logger.warning(f"❌ Failed to store user audio for session {session_id}")
                        else:
                            logger.info(f"⚠️ Skipping low-volume audio ({rms:.4f}) - likely not user speech")
                    except Exception as e:
                        logger.warning(f"Error analyzing audio quality: {e}")
                        # Fall back to storing the audio anyway
                        add_user_audio(session_id, audio_data, time.time())
                else:
                    # AI is not speaking, store audio normally
                    audio_stored = add_user_audio(session_id, audio_data, time.time())
                    if audio_stored:
                        logger.info(f"✅ User audio stored for session {session_id}")
                    else:
                        logger.warning(f"❌ Failed to store user audio for session {session_id}")
            else:
                # Try to enable recording if not already enabled
                from app.services.audio_conversation_storage import start_session_recording
                recording_started = start_session_recording(session_id, session_data.get('persona', 'friendly'))
                if recording_started:
                    logger.info(f"✅ Started audio recording for session {session_id}")
                    session_data['audio_recording_enabled'] = True
                    # Try storing again
                    add_user_audio(session_id, audio_data, time.time())
                else:
                    logger.warning(f"⚠️ Audio recording not enabled for session {session_id}")
            
            # Check if we should process transcription
            # Only process if user is allowed to speak or if AI is not speaking
            should_transcribe = (
                session_data.get('user_can_speak', True) or 
                not session_data.get('ai_speaking', False)
            )
            
            if not should_transcribe:
                logger.info(f"Skipping transcription for session {session_id} (AI is speaking)")
                return
                
            # Save audio to temporary file for transcription
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Process transcription
            logger.info(f"Processing transcription for: {temp_file_path}")
            await self._on_transcription_ready(session_id, temp_file_path)
            
        except Exception as e:
            logger.error(f"Error processing speech segment: {e}")
    
    async def _process_speech_segment_with_transcript(self, session_id: str, transcript_text: str, socket_id: str):
        """Process speech segment with already transcribed text"""
        try:
            logger.info(f"Processing transcribed speech for session {session_id}: {transcript_text[:50]}...")
            
            # Send transcription to client
            await self.sio.emit('transcription_result', {
                'session_id': session_id,
                'transcript': {'text': transcript_text},
                'text': transcript_text
            }, room=socket_id)
            
            # Generate AI response
            if transcript_text.strip():
                await self._generate_ai_response(session_id, transcript_text, socket_id)
                    
        except Exception as e:
            logger.error(f"Error processing transcribed speech: {e}")
            await self.sio.emit('audio_error', {
                'error': f'Speech processing error: {str(e)}'
            }, room=socket_id)
    
    async def _generate_ai_response(self, session_id: str, transcript: str, socket_id: str):
        """Generate AI response and convert to speech"""
        try:
            # Get session data
            session_data = self.active_sessions.get(session_id, {})
            persona = session_data.get('persona', 'friendly')
            
            # Import AI services
            from app.services.langgraph_workflow import process_pitch_message_async
            from app.services.enhanced_text_to_speech import convert_text_to_speech_with_persona
            
            # Generate AI response using LangGraph workflow
            ai_response = await process_pitch_message_async(session_id, transcript)
            logger.info(f"AI response generated: {ai_response}")
            
            # Send AI response to client
            await self.sio.emit('ai_response', {
                'session_id': session_id,
                'message': ai_response
            }, room=socket_id)
            
            # Convert to speech and send audio
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_audio_path = temp_file.name
            
            try:
                # Generate TTS audio using enhanced_text_to_speech
                logger.info(f"Generating TTS for AI response: '{ai_response[:50]}...' with persona: {persona}")
                audio_result = convert_text_to_speech_with_persona(ai_response, persona, temp_audio_path)
                
                if audio_result is None:
                    logger.error("Google Cloud TTS failed - no audio generated")
                    # Don't send audio response, let frontend use fallback
                    return
                
                # Check if file was created and has content
                if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                    logger.error(f"TTS audio file not created or empty: {temp_audio_path}")
                    return
                
                # Read audio file and send to client
                with open(temp_audio_path, 'rb') as audio_file:
                    audio_file_data = audio_file.read()
                
                import base64
                audio_base64 = base64.b64encode(audio_file_data).decode('utf-8')
                
                logger.info(f"Sending ai_response event to room: {socket_id}")
                await self.sio.emit('ai_response', {
                    'session_id': session_id,
                    'audio_data': audio_base64,
                    'message': ai_response
                }, room=socket_id)
                
                logger.info(f"AI audio response sent successfully - size: {len(audio_file_data)} bytes, base64 size: {len(audio_base64)} chars")
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            await self.sio.emit('audio_error', {
                'error': f'AI response error: {str(e)}'
            }, room=socket_id)
    
    async def _generate_tts_for_response(self, session_id: str, ai_response: str, socket_id: str):
        """Generate TTS audio for an AI response"""
        try:
            # Get session data
            session_data = self.active_sessions.get(session_id, {})
            persona = session_data.get('persona', 'friendly')
            
            # Import TTS service
            from app.services.enhanced_text_to_speech import convert_text_to_speech_with_persona
            
            # Convert to speech and send audio
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_audio_path = temp_file.name
            
            try:
                # Set AI speaking state to pause user recording
                self.set_ai_speaking_state(session_id, True)
                
                # Generate TTS audio using enhanced_text_to_speech
                logger.info(f"Generating TTS for response: '{ai_response[:50]}...' with persona: {persona}")
                audio_result = convert_text_to_speech_with_persona(ai_response, persona, temp_audio_path)
                
                if audio_result is None:
                    logger.error("Google Cloud TTS failed for response - no audio generated")
                    return
                
                # Check if file was created and has content
                if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                    logger.error(f"TTS audio file not created or empty: {temp_audio_path}")
                    return
                
                # Read audio file and send to client
                with open(temp_audio_path, 'rb') as audio_file:
                    audio_file_data = audio_file.read()
                
                # ALWAYS store AI audio in conversation storage regardless of session settings
                # This ensures we have complete AI audio for the conversation
                logger.info(f"🤖 Storing AI audio for session {session_id}: {len(audio_file_data)} bytes")
                
                # Start recording if not already enabled
                if not session_data.get('audio_recording_enabled', False):
                    from app.services.audio_conversation_storage import start_session_recording
                    recording_started = start_session_recording(session_id, session_data.get('persona', 'friendly'))
                    if recording_started:
                        logger.info(f"✅ Started audio recording for session {session_id}")
                        session_data['audio_recording_enabled'] = True
                    else:
                        logger.error(f"❌ Failed to start audio recording for session {session_id}")
                
                # Store the AI audio
                ai_audio_stored = add_ai_audio(session_id, audio_file_data, time.time())
                if ai_audio_stored:
                    logger.info(f"✅ AI audio stored successfully for session {session_id}")
                else:
                    logger.error(f"❌ Failed to store AI audio for session {session_id}")
                
                import base64
                audio_base64 = base64.b64encode(audio_file_data).decode('utf-8')
                
                logger.info(f"Sending ai_response event to room: {socket_id}")
                await self.sio.emit('ai_response', {
                    'session_id': session_id,
                    'audio_data': audio_base64,
                    'message': ai_response
                }, room=socket_id)
                
                logger.info(f"TTS audio response sent successfully - size: {len(audio_file_data)} bytes, base64 size: {len(audio_base64)} chars")
                
                # AI will finish speaking when audio ends - we'll handle this in frontend
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error generating TTS for response: {e}")
            await self.sio.emit('audio_error', {
                'error': f'TTS generation error: {str(e)}'
            }, room=socket_id)

# Global handler instance
audio_websocket_handler: Optional[AudioWebSocketHandler] = None

def initialize_audio_websocket_handler(sio: socketio.AsyncServer) -> AudioWebSocketHandler:
    """Initialize the audio WebSocket handler"""
    global audio_websocket_handler
    audio_websocket_handler = AudioWebSocketHandler(sio)
    logger.info("Audio WebSocket handler initialized")
    return audio_websocket_handler

def get_audio_websocket_handler() -> Optional[AudioWebSocketHandler]:
    """Get the global audio WebSocket handler"""
    return audio_websocket_handler

def get_session_audio_url(session_id: str) -> Optional[str]:
    """Get the audio URL for a session from the handler's cache"""
    handler = get_audio_websocket_handler()
    if handler and hasattr(handler, 'audio_url_cache'):
        return handler.audio_url_cache.get(session_id)
    return None