#!/usr/bin/env python3
"""
Video WebSocket handler for real-time video analysis
Integrates with existing audio VAD system
"""

import asyncio
import json
import logging
import base64
import cv2
import numpy as np
from typing import Dict, Any, Optional
import socketio

from .video_analysis import get_video_analyzer
from .enhanced_video_analysis import get_enhanced_video_analyzer
from .langgraph_workflow import get_pitch_workflow

logger = logging.getLogger(__name__)

class VideoWebSocketHandler:
    """
    Handles WebSocket connections for real-time video analysis
    Works alongside the existing audio WebSocket handler
    """
    
    def __init__(self, sio: socketio.AsyncServer):
        self.sio = sio
        self.video_analyzer = get_video_analyzer()
        self.enhanced_video_analyzer = get_enhanced_video_analyzer()  # Add enhanced analyzer
        self.pitch_workflow = get_pitch_workflow()  # Add workflow integration
        self.pitch_workflow = get_pitch_workflow()  # Add workflow integration
        self.active_video_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Analysis settings
        self.analysis_interval = 0.5  # Analyze every 0.5 seconds for better responsiveness
        self.last_analysis_time = {}
        
        # Set up video analysis callbacks
        if self.video_analyzer:
            self.video_analyzer.set_analysis_callback(self._on_video_analysis_update)
        
        if self.enhanced_video_analyzer:
            self.enhanced_video_analyzer.set_analysis_callback(self._on_enhanced_video_analysis_update)
        
        # Register WebSocket event handlers
        self._register_video_handlers()
    
    def _register_video_handlers(self):
        """Register video-specific WebSocket event handlers"""
        
        @self.sio.event
        async def start_video_analysis(sid, data):
            """Start video analysis for a session"""
            try:
                session_id = data.get('session_id')
                
                if not session_id:
                    await self.sio.emit('video_error', {
                        'error': 'session_id required'
                    }, room=sid)
                    return
                
                if not self.video_analyzer and not self.enhanced_video_analyzer:
                    await self.sio.emit('video_error', {
                        'error': 'Video analyzer not available'
                    }, room=sid)
                    return
                
                # Start video analysis (prefer enhanced analyzer)
                if self.enhanced_video_analyzer:
                    self.enhanced_video_analyzer.start_analysis(session_id)
                    analyzer_type = "enhanced"
                    logger.info(f"Started enhanced video analysis for session {session_id}")
                else:
                    self.video_analyzer.start_analysis(session_id)
                    analyzer_type = "basic"
                    logger.info(f"Started basic video analysis for session {session_id}")
                
                # Initialize video analysis in workflow state
                try:
                    if self.pitch_workflow:
                        config = {"configurable": {"thread_id": session_id}}
                        current_state = self.pitch_workflow.workflow.get_state(config)
                        
                        if current_state.values:
                            state_update = {"video_analysis_enabled": True}
                            self.pitch_workflow.workflow.update_state(config, state_update)
                            logger.info(f"Enabled video analysis in workflow state for session {session_id}")
                        else:
                            logger.warning(f"No workflow state found for session {session_id}")
                except Exception as e:
                    logger.error(f"Error enabling video analysis in workflow: {e}")
                
                # Track video session
                self.active_video_sessions[session_id] = {
                    'socket_id': sid,
                    'status': 'active',
                    'start_time': asyncio.get_event_loop().time()
                }
                
                self.last_analysis_time[session_id] = 0
                
                await self.sio.emit('video_analysis_started', {
                    'session_id': session_id,
                    'status': 'ready_for_video',
                    'analyzer_type': analyzer_type,
                    'message': f'{analyzer_type.title()} video analysis started successfully'
                }, room=sid)
                
                logger.info(f"Video analysis started for session: {session_id}")
                
            except Exception as e:
                logger.error(f"Error starting video analysis: {e}")
                await self.sio.emit('video_error', {
                    'error': f'Failed to start video analysis: {str(e)}'
                }, room=sid)
        
        @self.sio.event
        async def video_frame(sid, data):
            """Handle incoming video frame data"""
            try:
                session_id = data.get('session_id')
                frame_data = data.get('frame_data')  # Base64 encoded image
                
                if not session_id or session_id not in self.active_video_sessions:
                    await self.sio.emit('video_error', {
                        'error': 'Invalid video session'
                    }, room=sid)
                    return
                
                if not frame_data:
                    return
                
                # Check if we should analyze this frame (throttling)
                current_time = asyncio.get_event_loop().time()
                last_time = self.last_analysis_time.get(session_id, 0)
                
                if current_time - last_time < self.analysis_interval:
                    return  # Skip this frame
                
                self.last_analysis_time[session_id] = current_time
                
                # Decode frame data
                try:
                    # Remove data URL prefix if present
                    if frame_data.startswith('data:image'):
                        frame_data = frame_data.split(',')[1]
                    
                    # Decode base64 image
                    image_bytes = base64.b64decode(frame_data)
                    
                    # Convert to OpenCV format
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        logger.warning("Failed to decode video frame")
                        return
                    
                except Exception as e:
                    logger.error(f"Error decoding video frame: {e}")
                    return
                
                # Analyze frame with enhanced analyzer (preferred) or fallback to basic
                if self.enhanced_video_analyzer:
                    logger.debug(f"Analyzing video frame with enhanced analyzer for session {session_id}")
                    analysis_result = self.enhanced_video_analyzer.analyze_frame(frame)
                    
                    # Log analysis results for debugging
                    if analysis_result and analysis_result.get("status") != "throttled":
                        logger.info(f"Enhanced video analysis completed for session {session_id}: {analysis_result.get('status', 'unknown')}")
                        
                        # Log specific analysis components
                        if analysis_result.get('hand_analysis'):
                            hands = analysis_result['hand_analysis'].get('hands_detected', 0)
                            logger.info(f"  - Hands detected: {hands}")
                        
                        if analysis_result.get('emotion_analysis'):
                            emotion = analysis_result['emotion_analysis'].get('dominant_emotion', 'none')
                            logger.info(f"  - Dominant emotion: {emotion}")
                        
                        if analysis_result.get('pose_analysis'):
                            pose = analysis_result['pose_analysis'].get('pose_detected', False)
                            logger.info(f"  - Pose detected: {pose}")
                    
                elif self.video_analyzer:
                    logger.debug(f"Analyzing video frame with basic analyzer for session {session_id}")
                    analysis_result = self.video_analyzer.analyze_frame(frame)
                    logger.info(f"Basic video analysis result: {analysis_result}")
                else:
                    logger.error("No video analyzer available")
                
            except Exception as e:
                logger.error(f"Error processing video frame: {e}")
                await self.sio.emit('video_error', {
                    'error': f'Video processing error: {str(e)}'
                }, room=sid)
        
        @self.sio.event
        async def stop_video_analysis(sid, data):
            """Stop video analysis for a session"""
            try:
                session_id = data.get('session_id')
                
                if session_id and session_id in self.active_video_sessions:
                    await self._cleanup_video_session(session_id)
                    
                    await self.sio.emit('video_analysis_stopped', {
                        'session_id': session_id,
                        'status': 'stopped'
                    }, room=sid)
                
            except Exception as e:
                logger.error(f"Error stopping video analysis: {e}")
                await self.sio.emit('video_error', {
                    'error': f'Failed to stop video analysis: {str(e)}'
                }, room=sid)
        
        @self.sio.event
        async def enable_video_analysis(sid, data):
            """Explicitly enable video analysis for a session"""
            try:
                session_id = data.get('session_id')
                enabled = data.get('enabled', True)
                
                if not session_id:
                    await self.sio.emit('video_error', {
                        'error': 'session_id required'
                    }, room=sid)
                    return
                
                # Force enable video analysis in workflow state
                if self.pitch_workflow:
                    config = {"configurable": {"thread_id": session_id}}
                    try:
                        current_state = self.pitch_workflow.workflow.get_state(config)
                        if current_state.values:
                            state_update = {"video_analysis_enabled": enabled}
                            self.pitch_workflow.workflow.update_state(config, state_update)
                            logger.info(f"Explicitly set video_analysis_enabled={enabled} for session {session_id}")
                            
                            await self.sio.emit('video_status', {
                                'session_id': session_id,
                                'status': 'video_analysis_enabled' if enabled else 'video_analysis_disabled'
                            }, room=sid)
                    except Exception as e:
                        logger.error(f"Error enabling video analysis in workflow: {e}")
                        await self.sio.emit('video_error', {
                            'error': f'Failed to enable video analysis: {str(e)}'
                        }, room=sid)
            except Exception as e:
                logger.error(f"Error in enable_video_analysis: {e}")
                await self.sio.emit('video_error', {
                    'error': f'Failed to process request: {str(e)}'
                }, room=sid)
        
        @self.sio.event
        async def get_video_summary(sid, data):
            """Get video analysis summary for a session"""
            try:
                session_id = data.get('session_id')
                
                if not session_id:
                    await self.sio.emit('video_error', {
                        'error': 'session_id required'
                    }, room=sid)
                    return
                
                if not self.video_analyzer and not self.enhanced_video_analyzer:
                    await self.sio.emit('video_error', {
                        'error': 'Video analyzer not available'
                    }, room=sid)
                    return
                
                # Get session summary (prefer enhanced analyzer)
                if self.enhanced_video_analyzer:
                    summary = self.enhanced_video_analyzer.get_enhanced_session_summary()
                    summary['analyzer_type'] = 'enhanced'
                else:
                    summary = self.video_analyzer.get_session_summary()
                    summary['analyzer_type'] = 'basic'
                
                await self.sio.emit('video_summary', {
                    'session_id': session_id,
                    'summary': summary
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error getting video summary: {e}")
                await self.sio.emit('video_error', {
                    'error': f'Failed to get video summary: {str(e)}'
                }, room=sid)
    
    async def _on_video_analysis_update(self, analysis_result: Dict[str, Any]):
        """Called when basic video analysis produces results"""
        try:
            logger.debug(f"Basic video analysis callback triggered")
            await self._send_video_analysis_to_client(analysis_result, "basic")
        except Exception as e:
            logger.error(f"Error in basic video analysis callback: {e}")
    
    async def _on_enhanced_video_analysis_update(self, analysis_result: Dict[str, Any]):
        """Called when enhanced video analysis produces results"""
        try:
            # Skip throttled results
            if analysis_result.get("status") == "throttled":
                return
                
            logger.debug(f"Enhanced video analysis callback triggered")
            await self._send_video_analysis_to_client(analysis_result, "enhanced")
        except Exception as e:
            logger.error(f"Error in enhanced video analysis callback: {e}")
    
    async def _send_video_analysis_to_client(self, analysis_result: Dict[str, Any], analyzer_type: str):
        """Send video analysis results to client"""
        try:
            session_id = analysis_result.get('session_id')
            if not session_id or session_id not in self.active_video_sessions:
                logger.warning(f"Invalid session for video analysis: {session_id}")
                return
            
            session_data = self.active_video_sessions[session_id]
            socket_id = session_data['socket_id']
            
            # Prepare enhanced analysis data for client
            if analyzer_type == "enhanced":
                client_data = {
                    'session_id': session_id,
                    'timestamp': analysis_result.get('timestamp'),
                    'analyzer_type': 'enhanced',
                    'analysis': {
                        'hand_analysis': analysis_result.get('hand_analysis', {}),
                        'emotion_analysis': analysis_result.get('emotion_analysis', {}),
                        'pose_analysis': analysis_result.get('pose_analysis', {}),
                        'overall_scores': analysis_result.get('overall_scores', {}),
                        'recommendations': analysis_result.get('recommendations', [])
                    }
                }
            else:
                # Basic analysis format
                client_data = {
                    'session_id': session_id,
                    'timestamp': analysis_result.get('timestamp'),
                    'analyzer_type': 'basic',
                    'analysis': {
                        'hands': analysis_result.get('hands', {}),
                        'face': analysis_result.get('face', {}),
                        'pose': analysis_result.get('pose', {}),
                        'overall_assessment': analysis_result.get('overall_assessment', {})
                    }
                }
            
            # Send real-time analysis to client
            await self.sio.emit('video_analysis_update', client_data, room=socket_id)
            
            # Process insights for workflow integration
            await self._process_video_insights(session_id, analysis_result)
            
            # Integrate with LangGraph workflow
            await self._integrate_with_workflow(session_id, analysis_result)
            
            # Integrate with LangGraph workflow
            await self._integrate_with_workflow(session_id, analysis_result)
            
        except Exception as e:
            logger.error(f"Error sending video analysis to client: {e}")
    
    async def _process_video_insights(self, session_id: str, analysis_result: Dict[str, Any]):
        """Process video insights and integrate with pitch workflow"""
        try:
            # Extract key insights
            insights = []
            
            # Hand gesture insights
            hands = analysis_result.get('hands', {})
            if hands.get('hands_detected', 0) > 0:
                gestures = hands.get('gestures', [])
                for gesture in gestures:
                    if gesture.get('confidence', 0) > 0.7:
                        insights.append(f"Strong {gesture.get('type', 'unknown')} gesture detected")
            
            # Facial expression insights
            face = analysis_result.get('face', {})
            if face.get('faces_detected', 0) > 0:
                eye_contact_score = face.get('eye_contact_score', 0)
                if eye_contact_score > 0.8:
                    insights.append("Excellent eye contact maintained")
                elif eye_contact_score < 0.4:
                    insights.append("Poor eye contact detected")
            
            # Posture insights
            pose = analysis_result.get('pose', {})
            if pose.get('pose_detected', False):
                engagement = pose.get('engagement_level', 'neutral')
                if engagement == 'highly_engaged':
                    insights.append("Highly engaged body language")
                elif engagement == 'disengaged':
                    insights.append("Disengaged posture detected")
            
            # Send insights to workflow if significant
            if insights and session_id in self.active_video_sessions:
                session_data = self.active_video_sessions[session_id]
                socket_id = session_data['socket_id']
                
                await self.sio.emit('video_insights', {
                    'session_id': session_id,
                    'insights': insights,
                    'timestamp': analysis_result.get('timestamp')
                }, room=socket_id)
            
        except Exception as e:
            logger.error(f"Error processing video insights: {e}")
    
    async def _integrate_with_workflow(self, session_id: str, analysis_result: Dict[str, Any]):
        """Integrate video analysis results with LangGraph workflow"""
        try:
            if not self.pitch_workflow:
                return
            
            # Extract video insights for workflow integration
            video_insights = []
            gesture_feedback = []
            posture_feedback = []
            expression_feedback = []
            
            # Process enhanced analysis results
            if analysis_result.get('hand_analysis'):
                hand_analysis = analysis_result['hand_analysis']
                if hand_analysis.get('hands_detected', 0) > 0:
                    gestures = hand_analysis.get('gestures', [])
                    for gesture in gestures:
                        if gesture.get('confidence', 0) > 0.7:
                            gesture_feedback.append(f"Strong {gesture.get('type', 'unknown')} gesture detected")
                            video_insights.append(f"Effective hand gesture: {gesture.get('type', 'unknown')}")
            
            if analysis_result.get('pose_analysis'):
                pose_analysis = analysis_result['pose_analysis']
                if pose_analysis.get('pose_detected'):
                    engagement = pose_analysis.get('engagement_level', 'neutral')
                    if engagement == 'highly_engaged':
                        posture_feedback.append("Excellent engaged posture")
                        video_insights.append("Highly engaged body language")
                    elif engagement == 'disengaged':
                        posture_feedback.append("Posture suggests disengagement")
                        video_insights.append("Body language indicates low engagement")
            
            if analysis_result.get('emotion_analysis'):
                emotion_analysis = analysis_result['emotion_analysis']
                dominant_emotion = emotion_analysis.get('dominant_emotion')
                confidence = emotion_analysis.get('emotion_confidence', 0)
                
                if dominant_emotion and confidence > 0.6:
                    if dominant_emotion in ['happy', 'confident']:
                        expression_feedback.append(f"Positive facial expression: {dominant_emotion}")
                        video_insights.append(f"Confident facial expression detected")
                    elif dominant_emotion in ['nervous', 'fear']:
                        expression_feedback.append(f"Nervous expression detected: {dominant_emotion}")
                        video_insights.append(f"Facial expression suggests nervousness")
            
            # Update workflow state with video analysis data
            if video_insights or gesture_feedback or posture_feedback or expression_feedback:
                config = {"configurable": {"thread_id": session_id}}
                
                try:
                    current_state = self.pitch_workflow.workflow.get_state(config)
                    if current_state.values:
                        # Update the state with video analysis data
                        state_update = {}
                        
                        if video_insights:
                            existing_insights = current_state.values.get('video_insights', [])
                            state_update['video_insights'] = existing_insights + video_insights[-3:]  # Keep last 3 insights
                        
                        if gesture_feedback:
                            existing_gestures = current_state.values.get('gesture_feedback', [])
                            state_update['gesture_feedback'] = existing_gestures + gesture_feedback[-3:]
                        
                        if posture_feedback:
                            existing_posture = current_state.values.get('posture_feedback', [])
                            state_update['posture_feedback'] = existing_posture + posture_feedback[-3:]
                        
                        if expression_feedback:
                            existing_expression = current_state.values.get('expression_feedback', [])
                            state_update['expression_feedback'] = existing_expression + expression_feedback[-3:]
                        
                        # Enable video analysis flag
                        state_update['video_analysis_enabled'] = True
                        
                        # Update the workflow state
                        if state_update:
                            self.pitch_workflow.workflow.update_state(config, state_update)
                            logger.info(f"Updated workflow state with video analysis data for session {session_id}")
                
                except Exception as e:
                    logger.error(f"Error updating workflow state with video data: {e}")
            
        except Exception as e:
            logger.error(f"Error integrating video analysis with workflow: {e}")
    
    async def _integrate_with_workflow(self, session_id: str, analysis_result: Dict[str, Any]):
        """Integrate video analysis results with LangGraph workflow"""
        try:
            if not self.pitch_workflow:
                return
            
            # Extract video insights for workflow integration
            video_insights = []
            gesture_feedback = []
            posture_feedback = []
            expression_feedback = []
            
            # Process enhanced analysis results
            if analysis_result.get('hand_analysis'):
                hand_analysis = analysis_result['hand_analysis']
                if hand_analysis.get('hands_detected', 0) > 0:
                    gestures = hand_analysis.get('gestures', [])
                    for gesture in gestures:
                        if gesture.get('confidence', 0) > 0.7:
                            gesture_feedback.append(f"Strong {gesture.get('type', 'unknown')} gesture detected")
                            video_insights.append(f"Effective hand gesture: {gesture.get('type', 'unknown')}")
            
            if analysis_result.get('pose_analysis'):
                pose_analysis = analysis_result['pose_analysis']
                if pose_analysis.get('pose_detected'):
                    engagement = pose_analysis.get('engagement_level', 'neutral')
                    if engagement == 'highly_engaged':
                        posture_feedback.append("Excellent engaged posture")
                        video_insights.append("Highly engaged body language")
                    elif engagement == 'disengaged':
                        posture_feedback.append("Posture suggests disengagement")
                        video_insights.append("Body language indicates low engagement")
            
            if analysis_result.get('emotion_analysis'):
                emotion_analysis = analysis_result['emotion_analysis']
                dominant_emotion = emotion_analysis.get('dominant_emotion')
                confidence = emotion_analysis.get('emotion_confidence', 0)
                
                if dominant_emotion and confidence > 0.6:
                    if dominant_emotion in ['happy', 'confident']:
                        expression_feedback.append(f"Positive facial expression: {dominant_emotion}")
                        video_insights.append(f"Confident facial expression detected")
                    elif dominant_emotion in ['nervous', 'fear']:
                        expression_feedback.append(f"Nervous expression detected: {dominant_emotion}")
                        video_insights.append(f"Facial expression suggests nervousness")
            
            # Update workflow state with video analysis data
            if video_insights or gesture_feedback or posture_feedback or expression_feedback:
                config = {"configurable": {"thread_id": session_id}}
                
                try:
                    current_state = self.pitch_workflow.workflow.get_state(config)
                    if current_state.values:
                        # Update the state with video analysis data
                        state_update = {}
                        
                        if video_insights:
                            existing_insights = current_state.values.get('video_insights', [])
                            state_update['video_insights'] = existing_insights + video_insights[-3:]  # Keep last 3 insights
                        
                        if gesture_feedback:
                            existing_gestures = current_state.values.get('gesture_feedback', [])
                            state_update['gesture_feedback'] = existing_gestures + gesture_feedback[-3:]
                        
                        if posture_feedback:
                            existing_posture = current_state.values.get('posture_feedback', [])
                            state_update['posture_feedback'] = existing_posture + posture_feedback[-3:]
                        
                        if expression_feedback:
                            existing_expression = current_state.values.get('expression_feedback', [])
                            state_update['expression_feedback'] = existing_expression + expression_feedback[-3:]
                        
                        # Enable video analysis flag
                        state_update['video_analysis_enabled'] = True
                        
                        # Update the workflow state
                        if state_update:
                            self.pitch_workflow.workflow.update_state(config, state_update)
                            logger.info(f"Updated workflow state with video analysis data for session {session_id}")
                
                except Exception as e:
                    logger.error(f"Error updating workflow state with video data: {e}")
            
        except Exception as e:
            logger.error(f"Error integrating video analysis with workflow: {e}")
    
    async def _cleanup_video_session(self, session_id: str):
        """Clean up a video session"""
        try:
            if session_id in self.active_video_sessions:
                # Stop video analysis
                if self.enhanced_video_analyzer:
                    self.enhanced_video_analyzer.stop_analysis()
                elif self.video_analyzer:
                    self.video_analyzer.stop_analysis()
                
                # Remove from active sessions
                del self.active_video_sessions[session_id]
                
                # Clean up timing data
                if session_id in self.last_analysis_time:
                    del self.last_analysis_time[session_id]
                
                logger.info(f"Video session cleaned up: {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up video session {session_id}: {e}")
    
    def get_video_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific video session"""
        return self.active_video_sessions.get(session_id)
    
    def get_active_video_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active video sessions"""
        return self.active_video_sessions.copy()

# Global video handler instance
video_websocket_handler: Optional[VideoWebSocketHandler] = None

def initialize_video_websocket_handler(sio: socketio.AsyncServer) -> VideoWebSocketHandler:
    """Initialize the video WebSocket handler"""
    global video_websocket_handler
    video_websocket_handler = VideoWebSocketHandler(sio)
    logger.info("Video WebSocket handler initialized")
    return video_websocket_handler

def get_video_websocket_handler() -> Optional[VideoWebSocketHandler]:
    """Get the global video WebSocket handler"""
    return video_websocket_handler