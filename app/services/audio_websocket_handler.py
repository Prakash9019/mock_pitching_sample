#!/usr/bin/env python3
"""
WebSocket handler for real-time audio streaming and VAD
"""

import asyncio
import json
import logging
import os
import tempfile
from typing import Dict, Any, Optional
import socketio

from .voice_activity_detection import get_audio_processor
from .transcription import transcribe_audio
from .langgraph_workflow import get_pitch_workflow

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
                
                # Track session
                self.active_sessions[session_id] = {
                    'socket_id': sid,
                    'persona': persona,
                    'status': 'active',
                    'workflow': workflow
                }
                
                await self.sio.emit('audio_session_started', {
                    'session_id': session_id,
                    'message': result.get('message', ''),
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
                
                # Decode audio data
                import base64
                import numpy as np
                try:
                    # Decode base64 audio data
                    audio_bytes = base64.b64decode(audio_data)
                    logger.info(f"Decoded audio: {len(audio_bytes)} bytes")
                    
                    # Convert from Int16 to proper PCM format for VAD
                    # The browser sends Int16 data, convert to bytes
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    # Convert to 16kHz if needed (browser usually sends 44.1kHz or 48kHz)
                    # For now, assume it's already 16kHz from browser settings
                    pcm_audio = audio_array.tobytes()
                    
                    logger.info(f"Converted to PCM: {len(pcm_audio)} bytes, {len(audio_array)} samples")
                    
                except Exception as e:
                    logger.error(f"Error decoding/converting audio data: {e}")
                    return
                
                # Process audio through VAD
                if self.audio_processor:
                    vad_result = self.audio_processor.process_audio_stream(pcm_audio)
                    logger.info(f"VAD result: {vad_result}")
                    
                    # Send VAD status to client
                    if vad_result.get('action'):
                        await self.sio.emit('vad_status', {
                            'session_id': session_id,
                            'action': vad_result['action'],
                            'is_speaking': vad_result['is_speaking'],
                            'speech_duration': vad_result.get('speech_duration', 0)
                        }, room=sid)
                        
                        # If speech ended, process the accumulated audio
                        if vad_result['action'] == 'speech_ended' and vad_result.get('audio_data'):
                            await self._process_speech_segment(session_id, vad_result['audio_data'], sid)
                else:
                    logger.error("Audio processor not available")
                
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
    
    async def _on_transcription_ready(self, session_id: str, audio_file_path: str):
        """Called when audio is ready for transcription"""
        try:
            logger.info(f"Transcribing audio for session {session_id}")
            
            # Transcribe the audio
            transcript = transcribe_audio(audio_file_path)
            
            # Clean up temporary file
            try:
                os.unlink(audio_file_path)
            except:
                pass
            
            if transcript:
                await self._process_transcription(session_id, transcript)
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
    
    async def _process_transcription(self, session_id: str, transcript: str):
        """Process transcribed text through the pitch workflow"""
        try:
            session_data = self.active_sessions.get(session_id)
            if not session_data:
                logger.error(f"Session not found: {session_id}")
                return
            
            logger.info(f"Processing transcript for session {session_id}: {transcript[:50]}...")
            
            # Send transcript to client
            await self.sio.emit('transcription_result', {
                'session_id': session_id,
                'transcript': transcript
            }, room=session_data['socket_id'])
            
            # Process through pitch workflow
            workflow = session_data['workflow']
            response = workflow.process_message(session_id, transcript)
            
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
            
            # Generate TTS audio for the AI response
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
                # Stop audio processing
                self.audio_processor.stop_session()
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                
                logger.info(f"Session cleaned up: {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session"""
        return self.active_sessions.get(session_id)
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions"""
        return self.active_sessions.copy()
    
    async def _process_speech_segment(self, session_id: str, audio_data: bytes, socket_id: str):
        """Process a complete speech segment - transcribe and generate AI response"""
        try:
            logger.info(f"Processing speech segment for session {session_id}")
            
            # Save audio to temporary file for transcription
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Import transcription service
                from app.services.transcription import transcribe_audio
                
                # Transcribe the audio
                transcript = transcribe_audio(temp_file_path)
                logger.info(f"Transcription result: {transcript}")
                
                # Send transcription to client
                await self.sio.emit('transcription_result', {
                    'session_id': session_id,
                    'transcript': transcript
                }, room=socket_id)
                
                # Generate AI response
                if transcript.strip():
                    await self._generate_ai_response(session_id, transcript, socket_id)
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error processing speech segment: {e}")
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
                
                logger.info(f"Sending ai_audio_response event to room: {socket_id}")
                await self.sio.emit('ai_audio_response', {
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
                
                import base64
                audio_base64 = base64.b64encode(audio_file_data).decode('utf-8')
                
                logger.info(f"Sending ai_audio_response event to room: {socket_id}")
                await self.sio.emit('ai_audio_response', {
                    'session_id': session_id,
                    'audio_data': audio_base64,
                    'message': ai_response
                }, room=socket_id)
                
                logger.info(f"TTS audio response sent successfully - size: {len(audio_file_data)} bytes, base64 size: {len(audio_base64)} chars")
                
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