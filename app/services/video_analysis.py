#!/usr/bin/env python3
"""
Video Analysis Service for Pitch Practice
Monitors hand gestures, facial expressions, and body language
"""

import cv2
import numpy as np
import logging
import time
import asyncio
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import deque
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import MediaPipe for advanced analysis
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe available for advanced video analysis")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available. Using basic OpenCV analysis.")

class VideoAnalyzer:
    """
    Real-time video analysis for pitch practice
    Analyzes hand gestures, facial expressions, and body language
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize video analyzer"""
        self.config = config or {}
        
        # Analysis settings
        self.fps = self.config.get('fps', 30)
        self.analysis_interval = self.config.get('analysis_interval', 1.0)  # seconds
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        # MediaPipe components
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.mp_face = mp.solutions.face_detection
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Initialize MediaPipe models
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.face_detection = self.mp_face.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # Analysis state
        self.is_analyzing = False
        self.current_session_id = None
        self.analysis_history = deque(maxlen=1000)  # Keep last 1000 analyses
        
        # Metrics tracking
        self.gesture_counts = {
            'pointing': 0,
            'open_palm': 0,
            'closed_fist': 0,
            'thumbs_up': 0,
            'peace_sign': 0,
            'no_hands_visible': 0
        }
        
        self.expression_counts = {
            'confident': 0,
            'nervous': 0,
            'engaged': 0,
            'neutral': 0,
            'smiling': 0
        }
        
        self.posture_metrics = {
            'upright_posture': 0,
            'leaning_forward': 0,
            'leaning_back': 0,
            'slouching': 0,
            'good_eye_contact': 0
        }
        
        # Callbacks
        self.on_analysis_update: Optional[Callable[[Dict], None]] = None
        
        logger.info("Video analyzer initialized")
    
    def set_analysis_callback(self, callback: Callable[[Dict], None]):
        """Set callback for analysis updates"""
        self.on_analysis_update = callback
    
    def start_analysis(self, session_id: str):
        """Start video analysis for a session"""
        self.current_session_id = session_id
        self.is_analyzing = True
        
        # Reset metrics
        self.gesture_counts = {k: 0 for k in self.gesture_counts}
        self.expression_counts = {k: 0 for k in self.expression_counts}
        self.posture_metrics = {k: 0 for k in self.posture_metrics}
        
        logger.info(f"Video analysis started for session: {session_id}")
    
    def stop_analysis(self):
        """Stop video analysis"""
        self.is_analyzing = False
        logger.info(f"Video analysis stopped for session: {self.current_session_id}")
        self.current_session_id = None
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a single video frame
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            Analysis results dictionary
        """
        if not self.is_analyzing:
            return {"error": "Analysis not active"}
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            analysis_result = {
                "timestamp": time.time(),
                "session_id": self.current_session_id,
                "hands": {},
                "face": {},
                "pose": {},
                "overall_assessment": {}
            }
            
            if MEDIAPIPE_AVAILABLE:
                # Analyze hands
                analysis_result["hands"] = self._analyze_hands(rgb_frame)
                
                # Analyze face
                analysis_result["face"] = self._analyze_face(rgb_frame)
                
                # Analyze pose
                analysis_result["pose"] = self._analyze_pose(rgb_frame)
            else:
                # Basic OpenCV analysis
                analysis_result = self._basic_opencv_analysis(frame)
            
            # Generate overall assessment
            analysis_result["overall_assessment"] = self._generate_overall_assessment(analysis_result)
            
            # Update metrics
            self._update_metrics(analysis_result)
            
            # Store in history
            self.analysis_history.append(analysis_result)
            
            # Trigger callback
            if self.on_analysis_update:
                try:
                    self.on_analysis_update(analysis_result)
                except Exception as e:
                    logger.error(f"Error in analysis callback: {e}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return {"error": str(e)}
    
    def _analyze_hands(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """Analyze hand gestures using MediaPipe"""
        try:
            results = self.hands.process(rgb_frame)
            
            hand_analysis = {
                "hands_detected": 0,
                "gestures": [],
                "hand_positions": [],
                "gesture_confidence": 0.0
            }
            
            if results.multi_hand_landmarks:
                hand_analysis["hands_detected"] = len(results.multi_hand_landmarks)
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract hand landmarks
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    
                    # Analyze gesture
                    gesture = self._classify_hand_gesture(landmarks)
                    hand_analysis["gestures"].append(gesture)
                    
                    # Get hand position (center of palm)
                    palm_center = self._get_palm_center(landmarks)
                    hand_analysis["hand_positions"].append(palm_center)
                
                # Calculate average confidence
                if hand_analysis["gestures"]:
                    confidences = [g.get("confidence", 0) for g in hand_analysis["gestures"]]
                    hand_analysis["gesture_confidence"] = np.mean(confidences)
            
            return hand_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing hands: {e}")
            return {"error": str(e)}
    
    def _analyze_face(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """Analyze facial expressions using MediaPipe"""
        try:
            results = self.face_detection.process(rgb_frame)
            
            face_analysis = {
                "faces_detected": 0,
                "expressions": [],
                "eye_contact_score": 0.0,
                "confidence_indicators": []
            }
            
            if results.detections:
                face_analysis["faces_detected"] = len(results.detections)
                
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Analyze expression (simplified)
                    expression = self._analyze_facial_expression(rgb_frame, bbox)
                    face_analysis["expressions"].append(expression)
                    
                    # Estimate eye contact (based on face orientation)
                    eye_contact = self._estimate_eye_contact(bbox)
                    face_analysis["eye_contact_score"] = eye_contact
                    
                    # Confidence indicators
                    confidence_indicators = self._analyze_confidence_indicators(expression)
                    face_analysis["confidence_indicators"].extend(confidence_indicators)
            
            return face_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing face: {e}")
            return {"error": str(e)}
    
    def _analyze_pose(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """Analyze body posture using MediaPipe"""
        try:
            results = self.pose.process(rgb_frame)
            
            pose_analysis = {
                "pose_detected": False,
                "posture_score": 0.0,
                "body_language": [],
                "engagement_level": "neutral"
            }
            
            if results.pose_landmarks:
                pose_analysis["pose_detected"] = True
                
                # Extract key landmarks
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                # Analyze posture
                posture_score = self._analyze_posture(landmarks)
                pose_analysis["posture_score"] = posture_score
                
                # Analyze body language
                body_language = self._analyze_body_language(landmarks)
                pose_analysis["body_language"] = body_language
                
                # Determine engagement level
                engagement = self._determine_engagement_level(landmarks, posture_score)
                pose_analysis["engagement_level"] = engagement
            
            return pose_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing pose: {e}")
            return {"error": str(e)}
    
    def _classify_hand_gesture(self, landmarks: List[List[float]]) -> Dict[str, Any]:
        """Classify hand gesture from landmarks"""
        try:
            # Simplified gesture classification
            # In a real implementation, you'd use more sophisticated algorithms
            
            # Get finger tip and base positions
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            # Simple gesture detection logic
            gesture = {
                "type": "unknown",
                "confidence": 0.5,
                "description": ""
            }
            
            # Check for pointing gesture (index finger extended)
            if self._is_finger_extended(landmarks, 8):  # Index finger
                gesture = {
                    "type": "pointing",
                    "confidence": 0.8,
                    "description": "Pointing gesture detected"
                }
            
            # Check for open palm (all fingers extended)
            elif all(self._is_finger_extended(landmarks, tip) for tip in [8, 12, 16, 20]):
                gesture = {
                    "type": "open_palm",
                    "confidence": 0.9,
                    "description": "Open palm gesture"
                }
            
            # Check for closed fist (no fingers extended)
            elif not any(self._is_finger_extended(landmarks, tip) for tip in [8, 12, 16, 20]):
                gesture = {
                    "type": "closed_fist",
                    "confidence": 0.7,
                    "description": "Closed fist"
                }
            
            return gesture
            
        except Exception as e:
            logger.error(f"Error classifying gesture: {e}")
            return {"type": "error", "confidence": 0.0, "description": str(e)}
    
    def _is_finger_extended(self, landmarks: List[List[float]], tip_index: int) -> bool:
        """Check if a finger is extended based on landmarks"""
        try:
            # Get tip and base positions
            tip = landmarks[tip_index]
            
            # Simple heuristic: compare tip y-coordinate with base
            if tip_index == 8:  # Index finger
                base = landmarks[6]
            elif tip_index == 12:  # Middle finger
                base = landmarks[10]
            elif tip_index == 16:  # Ring finger
                base = landmarks[14]
            elif tip_index == 20:  # Pinky
                base = landmarks[18]
            else:
                return False
            
            # Finger is extended if tip is above base (lower y value)
            return tip[1] < base[1]
            
        except Exception:
            return False
    
    def _get_palm_center(self, landmarks: List[List[float]]) -> Tuple[float, float]:
        """Get center of palm from landmarks"""
        try:
            # Use wrist and middle finger base as reference
            wrist = landmarks[0]
            middle_base = landmarks[9]
            
            center_x = (wrist[0] + middle_base[0]) / 2
            center_y = (wrist[1] + middle_base[1]) / 2
            
            return (center_x, center_y)
            
        except Exception:
            return (0.5, 0.5)  # Default center
    
    def _analyze_facial_expression(self, frame: np.ndarray, bbox) -> Dict[str, Any]:
        """Analyze facial expression from face region"""
        try:
            # Simplified expression analysis
            # In practice, you'd use more sophisticated emotion detection
            
            expression = {
                "type": "neutral",
                "confidence": 0.6,
                "emotions": {
                    "confident": 0.3,
                    "nervous": 0.2,
                    "engaged": 0.4,
                    "neutral": 0.1
                }
            }
            
            # Determine primary expression
            primary_emotion = max(expression["emotions"], key=expression["emotions"].get)
            expression["type"] = primary_emotion
            expression["confidence"] = expression["emotions"][primary_emotion]
            
            return expression
            
        except Exception as e:
            logger.error(f"Error analyzing expression: {e}")
            return {"type": "error", "confidence": 0.0}
    
    def _estimate_eye_contact(self, bbox) -> float:
        """Estimate eye contact score based on face orientation"""
        try:
            # Simplified eye contact estimation
            # In practice, you'd analyze eye gaze direction
            
            # For now, assume good eye contact if face is centered and upright
            center_x = bbox.xmin + bbox.width / 2
            center_y = bbox.ymin + bbox.height / 2
            
            # Score based on how centered the face is
            center_score = 1.0 - abs(center_x - 0.5) * 2
            vertical_score = 1.0 - abs(center_y - 0.4) * 2  # Slightly above center is good
            
            eye_contact_score = (center_score + vertical_score) / 2
            return max(0.0, min(1.0, eye_contact_score))
            
        except Exception:
            return 0.5  # Default neutral score
    
    def _analyze_confidence_indicators(self, expression: Dict) -> List[str]:
        """Analyze confidence indicators from facial expression"""
        indicators = []
        
        try:
            if expression.get("type") == "confident":
                indicators.append("confident_expression")
            
            if expression.get("confidence", 0) > 0.7:
                indicators.append("strong_expression")
            
            # Add more indicators based on expression analysis
            emotions = expression.get("emotions", {})
            if emotions.get("engaged", 0) > 0.6:
                indicators.append("engaged_appearance")
            
            if emotions.get("nervous", 0) > 0.7:
                indicators.append("signs_of_nervousness")
            
        except Exception as e:
            logger.error(f"Error analyzing confidence indicators: {e}")
        
        return indicators
    
    def _analyze_posture(self, landmarks: List[List[float]]) -> float:
        """Analyze posture quality from pose landmarks"""
        try:
            # Get key landmarks for posture analysis
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            nose = landmarks[0]
            
            # Calculate shoulder alignment
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
            shoulder_score = max(0, 1.0 - shoulder_diff * 10)
            
            # Calculate spine alignment (simplified)
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            spine_alignment = abs(shoulder_center_y - hip_center_y)
            spine_score = max(0, 1.0 - spine_alignment * 5)
            
            # Overall posture score
            posture_score = (shoulder_score + spine_score) / 2
            return posture_score
            
        except Exception as e:
            logger.error(f"Error analyzing posture: {e}")
            return 0.5  # Default neutral score
    
    def _analyze_body_language(self, landmarks: List[List[float]]) -> List[str]:
        """Analyze body language indicators"""
        body_language = []
        
        try:
            # Get key landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            
            # Check for open vs closed posture
            elbow_distance = abs(left_elbow[0] - right_elbow[0])
            if elbow_distance > 0.3:
                body_language.append("open_posture")
            else:
                body_language.append("closed_posture")
            
            # Check for leaning
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            if shoulder_center_x < 0.45:
                body_language.append("leaning_left")
            elif shoulder_center_x > 0.55:
                body_language.append("leaning_right")
            else:
                body_language.append("centered_posture")
            
        except Exception as e:
            logger.error(f"Error analyzing body language: {e}")
        
        return body_language
    
    def _determine_engagement_level(self, landmarks: List[List[float]], posture_score: float) -> str:
        """Determine overall engagement level"""
        try:
            # Simple engagement scoring
            if posture_score > 0.8:
                return "highly_engaged"
            elif posture_score > 0.6:
                return "engaged"
            elif posture_score > 0.4:
                return "neutral"
            else:
                return "disengaged"
                
        except Exception:
            return "neutral"
    
    def _basic_opencv_analysis(self, frame: np.ndarray) -> Dict[str, Any]:
        """Basic analysis using only OpenCV (fallback)"""
        try:
            # Convert to grayscale for basic analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Basic face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            analysis = {
                "timestamp": time.time(),
                "session_id": self.current_session_id,
                "faces_detected": len(faces),
                "basic_analysis": True,
                "message": "Using basic OpenCV analysis (MediaPipe not available)"
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in basic analysis: {e}")
            return {"error": str(e)}
    
    def _generate_overall_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment from analysis results"""
        try:
            assessment = {
                "overall_score": 0.0,
                "strengths": [],
                "areas_for_improvement": [],
                "recommendations": []
            }
            
            scores = []
            
            # Hand gesture assessment
            hands = analysis.get("hands", {})
            if hands.get("hands_detected", 0) > 0:
                gesture_score = hands.get("gesture_confidence", 0.5)
                scores.append(gesture_score)
                
                if gesture_score > 0.7:
                    assessment["strengths"].append("Good hand gesture usage")
                else:
                    assessment["areas_for_improvement"].append("Hand gestures could be more expressive")
            
            # Face analysis assessment
            face = analysis.get("face", {})
            if face.get("faces_detected", 0) > 0:
                eye_contact_score = face.get("eye_contact_score", 0.5)
                scores.append(eye_contact_score)
                
                if eye_contact_score > 0.7:
                    assessment["strengths"].append("Good eye contact")
                else:
                    assessment["areas_for_improvement"].append("Maintain better eye contact")
            
            # Pose assessment
            pose = analysis.get("pose", {})
            if pose.get("pose_detected", False):
                posture_score = pose.get("posture_score", 0.5)
                scores.append(posture_score)
                
                if posture_score > 0.7:
                    assessment["strengths"].append("Good posture")
                else:
                    assessment["areas_for_improvement"].append("Improve posture and body alignment")
            
            # Calculate overall score
            if scores:
                assessment["overall_score"] = np.mean(scores)
            
            # Generate recommendations
            if assessment["overall_score"] < 0.6:
                assessment["recommendations"].append("Practice in front of a mirror to improve body language")
                assessment["recommendations"].append("Work on maintaining eye contact with the camera")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error generating assessment: {e}")
            return {"error": str(e)}
    
    def _update_metrics(self, analysis: Dict[str, Any]):
        """Update running metrics from analysis"""
        try:
            # Update gesture counts
            hands = analysis.get("hands", {})
            for gesture in hands.get("gestures", []):
                gesture_type = gesture.get("type", "unknown")
                if gesture_type in self.gesture_counts:
                    self.gesture_counts[gesture_type] += 1
            
            # Update expression counts
            face = analysis.get("face", {})
            for expression in face.get("expressions", []):
                expr_type = expression.get("type", "neutral")
                if expr_type in self.expression_counts:
                    self.expression_counts[expr_type] += 1
            
            # Update posture metrics
            pose = analysis.get("pose", {})
            engagement = pose.get("engagement_level", "neutral")
            if engagement == "highly_engaged":
                self.posture_metrics["upright_posture"] += 1
            elif engagement == "engaged":
                self.posture_metrics["good_eye_contact"] += 1
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session analysis"""
        try:
            total_analyses = len(self.analysis_history)
            
            if total_analyses == 0:
                return {"message": "No analysis data available"}
            
            # Calculate averages and totals
            summary = {
                "session_id": self.current_session_id,
                "total_frames_analyzed": total_analyses,
                "analysis_duration": self._get_analysis_duration(),
                "gesture_summary": self.gesture_counts.copy(),
                "expression_summary": self.expression_counts.copy(),
                "posture_summary": self.posture_metrics.copy(),
                "overall_scores": self._calculate_overall_scores(),
                "recommendations": self._generate_session_recommendations()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return {"error": str(e)}
    
    def _get_analysis_duration(self) -> float:
        """Calculate total analysis duration"""
        if len(self.analysis_history) < 2:
            return 0.0
        
        start_time = self.analysis_history[0]["timestamp"]
        end_time = self.analysis_history[-1]["timestamp"]
        return end_time - start_time
    
    def _calculate_overall_scores(self) -> Dict[str, float]:
        """Calculate overall scores from analysis history"""
        try:
            scores = {
                "gesture_score": 0.0,
                "expression_score": 0.0,
                "posture_score": 0.0,
                "overall_score": 0.0
            }
            
            if not self.analysis_history:
                return scores
            
            # Extract scores from history
            gesture_scores = []
            expression_scores = []
            posture_scores = []
            
            for analysis in self.analysis_history:
                # Gesture scores
                hands = analysis.get("hands", {})
                if hands.get("gesture_confidence"):
                    gesture_scores.append(hands["gesture_confidence"])
                
                # Expression scores (simplified)
                face = analysis.get("face", {})
                if face.get("eye_contact_score"):
                    expression_scores.append(face["eye_contact_score"])
                
                # Posture scores
                pose = analysis.get("pose", {})
                if pose.get("posture_score"):
                    posture_scores.append(pose["posture_score"])
            
            # Calculate averages
            if gesture_scores:
                scores["gesture_score"] = np.mean(gesture_scores)
            if expression_scores:
                scores["expression_score"] = np.mean(expression_scores)
            if posture_scores:
                scores["posture_score"] = np.mean(posture_scores)
            
            # Overall score
            all_scores = [s for s in [scores["gesture_score"], scores["expression_score"], scores["posture_score"]] if s > 0]
            if all_scores:
                scores["overall_score"] = np.mean(all_scores)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating scores: {e}")
            return {"gesture_score": 0.0, "expression_score": 0.0, "posture_score": 0.0, "overall_score": 0.0}
    
    def _generate_session_recommendations(self) -> List[str]:
        """Generate recommendations based on session analysis"""
        recommendations = []
        
        try:
            scores = self._calculate_overall_scores()
            
            # Gesture recommendations
            if scores["gesture_score"] < 0.6:
                recommendations.append("Use more expressive hand gestures to emphasize key points")
                recommendations.append("Practice natural hand movements that complement your speech")
            
            # Expression recommendations
            if scores["expression_score"] < 0.6:
                recommendations.append("Maintain better eye contact with your audience")
                recommendations.append("Work on facial expressions to show confidence and engagement")
            
            # Posture recommendations
            if scores["posture_score"] < 0.6:
                recommendations.append("Improve your posture - stand or sit up straight")
                recommendations.append("Avoid slouching or leaning too much to one side")
            
            # Overall recommendations
            if scores["overall_score"] < 0.5:
                recommendations.append("Practice your pitch in front of a mirror to improve body language")
                recommendations.append("Record yourself to review and improve your presentation style")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations

# Global video analyzer instance
video_analyzer: Optional[VideoAnalyzer] = None

def get_video_analyzer() -> Optional[VideoAnalyzer]:
    """Get the global video analyzer instance"""
    return video_analyzer

def initialize_video_analyzer(config: Optional[Dict] = None) -> VideoAnalyzer:
    """Initialize the video analyzer"""
    global video_analyzer
    video_analyzer = VideoAnalyzer(config)
    logger.info("Video analyzer initialized")
    return video_analyzer