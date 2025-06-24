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
                
                if not session_id or session_id not in self.active_sessions:
                    await self.sio.emit('audio_error', {
                        'error': 'Invalid session'
                    }, room=sid)
                    return
                
                if not audio_data:
                    return
                
                # Decode audio data
                import base64
                try:
                    audio_bytes = base64.b64decode(audio_data)
                except Exception as e:
                    logger.error(f"Error decoding audio data: {e}")
                    return
                
                # Process audio through VAD
                vad_result = self.audio_processor.process_audio_stream(audio_bytes)
                
                # Send VAD status to client
                if vad_result.get('action'):
                    await self.sio.emit('vad_status', {
                        'session_id': session_id,
                        'action': vad_result['action'],
                        'is_speaking': vad_result['is_speaking'],
                        'speech_duration': vad_result.get('speech_duration', 0)
                    }, room=sid)
                
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