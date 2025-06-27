#!/usr/bin/env python3
"""
Enhanced Video Analysis Service using Professional Libraries
- CVZone for advanced hand gesture recognition
- FER for facial emotion recognition
- MediaPipe for pose and body language analysis
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

# Import professional libraries
try:
    import cvzone
    from cvzone.HandTrackingModule import HandDetector
    CVZONE_AVAILABLE = True
    logger.info("CVZone available for advanced hand tracking")
except ImportError:
    CVZONE_AVAILABLE = False
    logger.warning("CVZone not available")

try:
    from fer import FER
    FER_AVAILABLE = True
    logger.info("FER available for emotion recognition")
except ImportError:
    FER_AVAILABLE = False
    logger.warning("FER not available")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe available for pose analysis")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available")

class EnhancedVideoAnalyzer:
    """
    Professional video analysis using state-of-the-art libraries
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize enhanced video analyzer"""
        self.config = config or {}
        
        # Analysis settings
        self.fps = self.config.get('fps', 30)
        self.analysis_interval = self.config.get('analysis_interval', 0.5)  # Analyze every 0.5 seconds
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # Initialize CVZone Hand Detector
        if CVZONE_AVAILABLE:
            self.hand_detector = HandDetector(
                staticMode=False,
                maxHands=2,
                modelComplexity=1,
                detectionCon=0.7,
                minTrackCon=0.5
            )
            logger.info("CVZone Hand Detector initialized")
        else:
            self.hand_detector = None
        
        # Initialize FER Emotion Detector
        if FER_AVAILABLE:
            try:
                # Try with MTCNN first, fallback to default if it fails
                try:
                    self.emotion_detector = FER(mtcnn=True)
                    logger.info("FER Emotion Detector initialized with MTCNN")
                except Exception:
                    self.emotion_detector = FER(mtcnn=False)
                    logger.info("FER Emotion Detector initialized with default face detection")
            except Exception as e:
                logger.warning(f"FER initialization failed: {e}")
                self.emotion_detector = None
        else:
            self.emotion_detector = None
        
        # Initialize MediaPipe Pose
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            logger.info("MediaPipe Pose initialized")
        else:
            self.pose = None
        
        # Analysis state
        self.is_analyzing = False
        self.current_session_id = None
        self.analysis_history = deque(maxlen=2000)  # Keep more history for better analysis
        self.last_analysis_time = 0
        
        # Enhanced metrics tracking
        self.gesture_metrics = {
            'total_gestures': 0,
            'effective_gestures': 0,
            'gesture_variety': set(),
            'gesture_timing': [],
            'hand_visibility_score': 0.0
        }
        
        self.emotion_metrics = {
            'dominant_emotions': {},
            'confidence_levels': [],
            'emotion_stability': 0.0,
            'positive_emotion_ratio': 0.0
        }
        
        self.posture_metrics = {
            'posture_scores': [],
            'engagement_levels': [],
            'body_language_indicators': [],
            'movement_patterns': []
        }
        
        # Callbacks
        self.on_analysis_update: Optional[Callable[[Dict], None]] = None
        
        logger.info("Enhanced Video Analyzer initialized successfully")
    
    def set_analysis_callback(self, callback: Callable[[Dict], None]):
        """Set callback for analysis updates"""
        self.on_analysis_update = callback
    
    def start_analysis(self, session_id: str):
        """Start enhanced video analysis"""
        self.current_session_id = session_id
        self.is_analyzing = True
        self.last_analysis_time = 0
        
        # Reset all metrics
        self.gesture_metrics = {
            'total_gestures': 0,
            'effective_gestures': 0,
            'gesture_variety': set(),
            'gesture_timing': [],
            'hand_visibility_score': 0.0
        }
        
        self.emotion_metrics = {
            'dominant_emotions': {},
            'confidence_levels': [],
            'emotion_stability': 0.0,
            'positive_emotion_ratio': 0.0
        }
        
        self.posture_metrics = {
            'posture_scores': [],
            'engagement_levels': [],
            'body_language_indicators': [],
            'movement_patterns': []
        }
        
        logger.info(f"Enhanced video analysis started for session: {session_id}")
    
    def stop_analysis(self):
        """Stop video analysis"""
        self.is_analyzing = False
        logger.info(f"Enhanced video analysis stopped for session: {self.current_session_id}")
        self.current_session_id = None
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze video frame using professional libraries
        """
        if not self.is_analyzing:
            return {"error": "Analysis not active"}
        
        current_time = time.time()
        
        # Throttle analysis to avoid overwhelming the system
        if current_time - self.last_analysis_time < self.analysis_interval:
            return {"status": "throttled"}
        
        self.last_analysis_time = current_time
        
        try:
            analysis_result = {
                "timestamp": current_time,
                "session_id": self.current_session_id,
                "hand_analysis": {},
                "emotion_analysis": {},
                "pose_analysis": {},
                "overall_scores": {},
                "recommendations": []
            }
            
            # Analyze hands using CVZone
            if self.hand_detector:
                analysis_result["hand_analysis"] = self._analyze_hands_cvzone(frame)
            
            # Analyze emotions using FER
            if self.emotion_detector:
                analysis_result["emotion_analysis"] = self._analyze_emotions_fer(frame)
            
            # Analyze pose using MediaPipe
            if self.pose:
                analysis_result["pose_analysis"] = self._analyze_pose_mediapipe(frame)
            
            # Calculate overall scores
            analysis_result["overall_scores"] = self._calculate_enhanced_scores(analysis_result)
            
            # Generate intelligent recommendations
            analysis_result["recommendations"] = self._generate_intelligent_recommendations(analysis_result)
            
            # Update metrics
            self._update_enhanced_metrics(analysis_result)
            
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
            logger.error(f"Error in enhanced frame analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_hands_cvzone(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze hands using CVZone HandDetector"""
        try:
            hands, img = self.hand_detector.findHands(frame, draw=False)
            
            hand_analysis = {
                "hands_detected": len(hands),
                "gestures": [],
                "hand_positions": [],
                "gesture_effectiveness": 0.0,
                "hand_visibility": 0.0
            }
            
            if hands:
                for hand in hands:
                    # Get hand landmarks
                    lmList = hand["lmList"]
                    handType = hand["type"]  # "Left" or "Right"
                    
                    # Classify gesture using CVZone's finger detection
                    fingers = self.hand_detector.fingersUp(hand)
                    gesture = self._classify_cvzone_gesture(fingers, lmList, handType)
                    
                    hand_analysis["gestures"].append(gesture)
                    
                    # Get hand center position
                    center = hand["center"]
                    hand_analysis["hand_positions"].append({
                        "hand_type": handType,
                        "center": center,
                        "bbox": hand["bbox"]
                    })
                
                # Calculate overall hand metrics
                hand_analysis["gesture_effectiveness"] = self._calculate_gesture_effectiveness(hand_analysis["gestures"])
                hand_analysis["hand_visibility"] = self._calculate_hand_visibility(hands, frame.shape)
            
            return hand_analysis
            
        except Exception as e:
            logger.error(f"Error in CVZone hand analysis: {e}")
            return {"error": str(e)}
    
    def _classify_cvzone_gesture(self, fingers: List[int], landmarks: List, hand_type: str) -> Dict[str, Any]:
        """Classify gesture using CVZone finger detection"""
        try:
            total_fingers = sum(fingers)
            
            # Enhanced gesture classification
            gesture_map = {
                (0, 0, 0, 0, 0): {
                    "name": "closed_fist",
                    "description": "Closed fist - strong emphasis",
                    "effectiveness": 0.8,
                    "pitch_impact": "high"
                },
                (1, 1, 1, 1, 1): {
                    "name": "open_palm",
                    "description": "Open palm - welcoming, honest",
                    "effectiveness": 0.95,
                    "pitch_impact": "excellent"
                },
                (0, 1, 0, 0, 0): {
                    "name": "pointing",
                    "description": "Index finger pointing - directing attention",
                    "effectiveness": 0.7,
                    "pitch_impact": "good"
                },
                (1, 0, 0, 0, 0): {
                    "name": "thumbs_up",
                    "description": "Thumbs up - positive reinforcement",
                    "effectiveness": 0.9,
                    "pitch_impact": "excellent"
                },
                (0, 1, 1, 0, 0): {
                    "name": "peace_victory",
                    "description": "Peace/Victory sign",
                    "effectiveness": 0.6,
                    "pitch_impact": "moderate"
                },
                (1, 1, 0, 0, 0): {
                    "name": "gun_gesture",
                    "description": "Finger gun - casual pointing",
                    "effectiveness": 0.5,
                    "pitch_impact": "low"
                }
            }
            
            finger_tuple = tuple(fingers)
            
            if finger_tuple in gesture_map:
                gesture_info = gesture_map[finger_tuple].copy()
            else:
                # Handle other combinations
                if total_fingers == 2:
                    gesture_info = {
                        "name": "two_fingers",
                        "description": f"Two fingers up - counting gesture",
                        "effectiveness": 0.6,
                        "pitch_impact": "moderate"
                    }
                elif total_fingers == 3:
                    gesture_info = {
                        "name": "three_fingers",
                        "description": f"Three fingers up - counting/emphasis",
                        "effectiveness": 0.7,
                        "pitch_impact": "good"
                    }
                else:
                    gesture_info = {
                        "name": "custom_gesture",
                        "description": f"Custom gesture ({total_fingers} fingers)",
                        "effectiveness": 0.4,
                        "pitch_impact": "neutral"
                    }
            
            # Add additional context
            gesture_info.update({
                "hand_type": hand_type,
                "fingers_up": fingers,
                "total_fingers": total_fingers,
                "confidence": 0.85,  # CVZone is generally reliable
                "timestamp": time.time()
            })
            
            return gesture_info
            
        except Exception as e:
            logger.error(f"Error classifying CVZone gesture: {e}")
            return {
                "name": "error",
                "description": "Gesture classification error",
                "effectiveness": 0.0,
                "pitch_impact": "none"
            }
    
    def _analyze_emotions_fer(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze facial emotions using FER"""
        try:
            # FER expects RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect emotions
            emotions = self.emotion_detector.detect_emotions(rgb_frame)
            
            emotion_analysis = {
                "faces_detected": len(emotions),
                "emotions": [],
                "dominant_emotion": None,
                "confidence_score": 0.0,
                "pitch_suitability": "neutral"
            }
            
            if emotions:
                for face_emotions in emotions:
                    emotion_scores = face_emotions['emotions']
                    face_box = face_emotions['box']
                    
                    # Find dominant emotion
                    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                    confidence = emotion_scores[dominant_emotion]
                    
                    # Analyze pitch suitability
                    pitch_suitability = self._analyze_emotion_for_pitch(emotion_scores)
                    
                    emotion_info = {
                        "dominant_emotion": dominant_emotion,
                        "confidence": confidence,
                        "all_emotions": emotion_scores,
                        "face_box": face_box,
                        "pitch_suitability": pitch_suitability,
                        "recommendations": self._get_emotion_recommendations(dominant_emotion, confidence)
                    }
                    
                    emotion_analysis["emotions"].append(emotion_info)
                
                # Overall analysis
                if emotion_analysis["emotions"]:
                    primary_emotion = emotion_analysis["emotions"][0]
                    emotion_analysis["dominant_emotion"] = primary_emotion["dominant_emotion"]
                    emotion_analysis["confidence_score"] = primary_emotion["confidence"]
                    emotion_analysis["pitch_suitability"] = primary_emotion["pitch_suitability"]
            
            return emotion_analysis
            
        except Exception as e:
            logger.error(f"Error in FER emotion analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_pose_mediapipe(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze body pose using MediaPipe"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            pose_analysis = {
                "pose_detected": False,
                "posture_score": 0.0,
                "engagement_level": "neutral",
                "body_language": [],
                "posture_recommendations": []
            }
            
            if results.pose_landmarks:
                pose_analysis["pose_detected"] = True
                landmarks = results.pose_landmarks.landmark
                
                # Convert landmarks to list format
                landmark_list = []
                for landmark in landmarks:
                    landmark_list.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                # Analyze posture
                posture_score = self._analyze_enhanced_posture(landmark_list)
                pose_analysis["posture_score"] = posture_score
                
                # Determine engagement level
                engagement = self._determine_enhanced_engagement(landmark_list, posture_score)
                pose_analysis["engagement_level"] = engagement
                
                # Analyze body language
                body_language = self._analyze_enhanced_body_language(landmark_list)
                pose_analysis["body_language"] = body_language
                
                # Generate posture recommendations
                pose_analysis["posture_recommendations"] = self._get_posture_recommendations(posture_score, body_language)
            
            return pose_analysis
            
        except Exception as e:
            logger.error(f"Error in MediaPipe pose analysis: {e}")
            return {"error": str(e)}
    
    def _calculate_gesture_effectiveness(self, gestures: List[Dict]) -> float:
        """Calculate overall gesture effectiveness"""
        if not gestures:
            return 0.0
        
        effectiveness_scores = [g.get("effectiveness", 0.0) for g in gestures]
        return np.mean(effectiveness_scores)
    
    def _calculate_hand_visibility(self, hands: List[Dict], frame_shape: Tuple) -> float:
        """Calculate hand visibility score"""
        if not hands:
            return 0.0
        
        total_visibility = 0.0
        for hand in hands:
            bbox = hand["bbox"]
            # Calculate what percentage of the frame the hands occupy
            hand_area = bbox[2] * bbox[3]  # width * height
            frame_area = frame_shape[0] * frame_shape[1]
            visibility = min(1.0, hand_area / (frame_area * 0.1))  # Normalize
            total_visibility += visibility
        
        return total_visibility / len(hands)
    
    def _analyze_emotion_for_pitch(self, emotion_scores: Dict[str, float]) -> str:
        """Analyze emotion suitability for pitching"""
        # Define emotion weights for pitching effectiveness
        pitch_weights = {
            'happy': 0.9,
            'neutral': 0.7,
            'surprise': 0.6,
            'angry': 0.3,
            'sad': 0.2,
            'disgust': 0.1,
            'fear': 0.2
        }
        
        weighted_score = sum(emotion_scores[emotion] * pitch_weights.get(emotion, 0.5) 
                           for emotion in emotion_scores)
        
        if weighted_score > 0.7:
            return "excellent"
        elif weighted_score > 0.5:
            return "good"
        elif weighted_score > 0.3:
            return "moderate"
        else:
            return "needs_improvement"
    
    def _get_emotion_recommendations(self, dominant_emotion: str, confidence: float) -> List[str]:
        """Get recommendations based on detected emotion"""
        recommendations = []
        
        emotion_advice = {
            'happy': ["Great! Your positive energy is engaging", "Maintain this confident expression"],
            'neutral': ["Try to show more enthusiasm", "Smile more to appear more approachable"],
            'sad': ["Try to project more confidence", "Practice positive facial expressions"],
            'angry': ["Soften your expression", "Take a deep breath and relax your face"],
            'fear': ["Project more confidence", "Practice relaxation techniques before presenting"],
            'surprise': ["Good engagement, but try to appear more composed", "Balance surprise with confidence"],
            'disgust': ["Check your facial expression", "Ensure you appear approachable and positive"]
        }
        
        if dominant_emotion in emotion_advice:
            recommendations.extend(emotion_advice[dominant_emotion])
        
        if confidence < 0.6:
            recommendations.append("Work on maintaining consistent facial expressions")
        
        return recommendations
    
    def _analyze_enhanced_posture(self, landmarks: List[List[float]]) -> float:
        """Enhanced posture analysis"""
        try:
            # Key landmarks for posture analysis
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            nose = landmarks[0]
            left_ear = landmarks[7]
            right_ear = landmarks[8]
            
            scores = []
            
            # Shoulder alignment (should be level)
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
            shoulder_score = max(0, 1.0 - shoulder_diff * 20)
            scores.append(shoulder_score)
            
            # Head position (should be upright)
            head_tilt = abs(left_ear[1] - right_ear[1])
            head_score = max(0, 1.0 - head_tilt * 15)
            scores.append(head_score)
            
            # Spine alignment
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            spine_alignment = abs(shoulder_center_x - hip_center_x)
            spine_score = max(0, 1.0 - spine_alignment * 10)
            scores.append(spine_score)
            
            # Forward head posture check
            ear_center_x = (left_ear[0] + right_ear[0]) / 2
            forward_head = abs(ear_center_x - shoulder_center_x)
            forward_head_score = max(0, 1.0 - forward_head * 8)
            scores.append(forward_head_score)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"Error in enhanced posture analysis: {e}")
            return 0.5
    
    def _determine_enhanced_engagement(self, landmarks: List[List[float]], posture_score: float) -> str:
        """Determine engagement level with enhanced criteria"""
        try:
            # Factor in multiple indicators
            engagement_score = posture_score
            
            # Check for forward lean (indicates engagement)
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            
            # Slight forward lean is good for engagement
            lean_factor = shoulder_center_y - hip_center_y
            if 0.02 < lean_factor < 0.08:  # Optimal forward lean
                engagement_score += 0.1
            
            # Determine engagement level
            if engagement_score > 0.8:
                return "highly_engaged"
            elif engagement_score > 0.6:
                return "engaged"
            elif engagement_score > 0.4:
                return "moderately_engaged"
            else:
                return "disengaged"
                
        except Exception:
            return "neutral"
    
    def _analyze_enhanced_body_language(self, landmarks: List[List[float]]) -> List[str]:
        """Enhanced body language analysis"""
        body_language = []
        
        try:
            # Analyze various body language indicators
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # Open vs closed posture
            elbow_distance = abs(left_elbow[0] - right_elbow[0])
            if elbow_distance > 0.25:
                body_language.append("open_posture")
            else:
                body_language.append("closed_posture")
            
            # Hand position analysis
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            wrist_avg_y = (left_wrist[1] + right_wrist[1]) / 2
            
            if wrist_avg_y < shoulder_center_y:
                body_language.append("hands_raised")
            elif wrist_avg_y > shoulder_center_y + 0.2:
                body_language.append("hands_lowered")
            else:
                body_language.append("hands_neutral")
            
            # Shoulder position
            if left_shoulder[1] < right_shoulder[1] - 0.02:
                body_language.append("left_shoulder_raised")
            elif right_shoulder[1] < left_shoulder[1] - 0.02:
                body_language.append("right_shoulder_raised")
            else:
                body_language.append("shoulders_level")
            
        except Exception as e:
            logger.error(f"Error in body language analysis: {e}")
        
        return body_language
    
    def _get_posture_recommendations(self, posture_score: float, body_language: List[str]) -> List[str]:
        """Get specific posture recommendations"""
        recommendations = []
        
        if posture_score < 0.6:
            recommendations.append("Improve your overall posture - stand/sit up straighter")
        
        if "closed_posture" in body_language:
            recommendations.append("Open up your posture - avoid crossing arms or hunching")
        
        if "hands_lowered" in body_language:
            recommendations.append("Use more hand gestures at chest level for better engagement")
        
        if "left_shoulder_raised" in body_language or "right_shoulder_raised" in body_language:
            recommendations.append("Keep your shoulders level and relaxed")
        
        if posture_score > 0.8:
            recommendations.append("Excellent posture! Keep it up")
        
        return recommendations
    
    def _calculate_enhanced_scores(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive scores from all analyses"""
        scores = {
            "gesture_score": 0.0,
            "emotion_score": 0.0,
            "posture_score": 0.0,
            "overall_score": 0.0,
            "pitch_readiness": 0.0
        }
        
        try:
            # Hand gesture score
            hand_analysis = analysis.get("hand_analysis", {})
            if hand_analysis.get("hands_detected", 0) > 0:
                scores["gesture_score"] = hand_analysis.get("gesture_effectiveness", 0.0)
            
            # Emotion score
            emotion_analysis = analysis.get("emotion_analysis", {})
            if emotion_analysis.get("faces_detected", 0) > 0:
                confidence = emotion_analysis.get("confidence_score", 0.0)
                suitability_map = {
                    "excellent": 1.0,
                    "good": 0.8,
                    "moderate": 0.6,
                    "needs_improvement": 0.3
                }
                suitability = emotion_analysis.get("pitch_suitability", "neutral")
                emotion_score = confidence * suitability_map.get(suitability, 0.5)
                scores["emotion_score"] = emotion_score
            
            # Posture score
            pose_analysis = analysis.get("pose_analysis", {})
            if pose_analysis.get("pose_detected", False):
                scores["posture_score"] = pose_analysis.get("posture_score", 0.0)
            
            # Overall score (weighted average)
            valid_scores = [s for s in [scores["gesture_score"], scores["emotion_score"], scores["posture_score"]] if s > 0]
            if valid_scores:
                scores["overall_score"] = np.mean(valid_scores)
            
            # Pitch readiness (special composite score)
            scores["pitch_readiness"] = self._calculate_pitch_readiness(analysis)
            
        except Exception as e:
            logger.error(f"Error calculating enhanced scores: {e}")
        
        return scores
    
    def _calculate_pitch_readiness(self, analysis: Dict[str, Any]) -> float:
        """Calculate how ready the person is for pitching"""
        readiness_factors = []
        
        # Gesture readiness
        hand_analysis = analysis.get("hand_analysis", {})
        if hand_analysis.get("hands_detected", 0) > 0:
            gesture_effectiveness = hand_analysis.get("gesture_effectiveness", 0.0)
            readiness_factors.append(gesture_effectiveness * 0.3)  # 30% weight
        
        # Emotional readiness
        emotion_analysis = analysis.get("emotion_analysis", {})
        if emotion_analysis.get("faces_detected", 0) > 0:
            suitability = emotion_analysis.get("pitch_suitability", "neutral")
            suitability_scores = {
                "excellent": 1.0,
                "good": 0.8,
                "moderate": 0.6,
                "needs_improvement": 0.3
            }
            readiness_factors.append(suitability_scores.get(suitability, 0.5) * 0.4)  # 40% weight
        
        # Posture readiness
        pose_analysis = analysis.get("pose_analysis", {})
        if pose_analysis.get("pose_detected", False):
            posture_score = pose_analysis.get("posture_score", 0.0)
            engagement = pose_analysis.get("engagement_level", "neutral")
            engagement_scores = {
                "highly_engaged": 1.0,
                "engaged": 0.8,
                "moderately_engaged": 0.6,
                "disengaged": 0.3
            }
            combined_posture = (posture_score + engagement_scores.get(engagement, 0.5)) / 2
            readiness_factors.append(combined_posture * 0.3)  # 30% weight
        
        return sum(readiness_factors) if readiness_factors else 0.5
    
    def _generate_intelligent_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on comprehensive analysis"""
        recommendations = []
        
        try:
            scores = analysis.get("overall_scores", {})
            
            # Gesture recommendations
            if scores.get("gesture_score", 0) < 0.6:
                hand_analysis = analysis.get("hand_analysis", {})
                if hand_analysis.get("hands_detected", 0) == 0:
                    recommendations.append("🤲 Use hand gestures to emphasize key points - your hands aren't visible")
                else:
                    recommendations.append("🎯 Make your hand gestures more purposeful and expressive")
            
            # Emotion recommendations
            if scores.get("emotion_score", 0) < 0.6:
                emotion_analysis = analysis.get("emotion_analysis", {})
                if emotion_analysis.get("dominant_emotion") in ["sad", "angry", "fear"]:
                    recommendations.append("😊 Project more positive energy - smile and show enthusiasm")
                else:
                    recommendations.append("💪 Show more confidence in your facial expression")
            
            # Posture recommendations
            if scores.get("posture_score", 0) < 0.6:
                recommendations.append("🏃‍♂️ Improve your posture - stand tall and keep shoulders back")
            
            # Overall pitch readiness
            pitch_readiness = scores.get("pitch_readiness", 0)
            if pitch_readiness > 0.8:
                recommendations.append("🌟 Excellent! You're ready to deliver a compelling pitch")
            elif pitch_readiness > 0.6:
                recommendations.append("👍 Good presentation skills - minor adjustments will make you shine")
            elif pitch_readiness > 0.4:
                recommendations.append("📈 You're on the right track - focus on the areas highlighted above")
            else:
                recommendations.append("🎯 Practice your presentation skills - focus on body language and confidence")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _update_enhanced_metrics(self, analysis: Dict[str, Any]):
        """Update comprehensive metrics tracking"""
        try:
            # Update gesture metrics
            hand_analysis = analysis.get("hand_analysis", {})
            if hand_analysis.get("hands_detected", 0) > 0:
                self.gesture_metrics["total_gestures"] += 1
                for gesture in hand_analysis.get("gestures", []):
                    if gesture.get("effectiveness", 0) > 0.7:
                        self.gesture_metrics["effective_gestures"] += 1
                    self.gesture_metrics["gesture_variety"].add(gesture.get("name", "unknown"))
                
                self.gesture_metrics["hand_visibility_score"] = hand_analysis.get("hand_visibility", 0.0)
            
            # Update emotion metrics
            emotion_analysis = analysis.get("emotion_analysis", {})
            if emotion_analysis.get("faces_detected", 0) > 0:
                dominant_emotion = emotion_analysis.get("dominant_emotion")
                if dominant_emotion:
                    self.emotion_metrics["dominant_emotions"][dominant_emotion] = \
                        self.emotion_metrics["dominant_emotions"].get(dominant_emotion, 0) + 1
                
                confidence = emotion_analysis.get("confidence_score", 0.0)
                self.emotion_metrics["confidence_levels"].append(confidence)
            
            # Update posture metrics
            pose_analysis = analysis.get("pose_analysis", {})
            if pose_analysis.get("pose_detected", False):
                posture_score = pose_analysis.get("posture_score", 0.0)
                self.posture_metrics["posture_scores"].append(posture_score)
                
                engagement = pose_analysis.get("engagement_level", "neutral")
                self.posture_metrics["engagement_levels"].append(engagement)
                
                body_language = pose_analysis.get("body_language", [])
                self.posture_metrics["body_language_indicators"].extend(body_language)
            
        except Exception as e:
            logger.error(f"Error updating enhanced metrics: {e}")
    
    def get_enhanced_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary with professional insights"""
        try:
            if not self.analysis_history:
                return {"message": "No analysis data available"}
            
            # Calculate comprehensive statistics
            summary = {
                "session_id": self.current_session_id,
                "analysis_duration": self._get_analysis_duration(),
                "total_frames_analyzed": len(self.analysis_history),
                
                # Gesture insights
                "gesture_insights": {
                    "total_gestures": self.gesture_metrics["total_gestures"],
                    "effective_gestures": self.gesture_metrics["effective_gestures"],
                    "gesture_variety_count": len(self.gesture_metrics["gesture_variety"]),
                    "gesture_types_used": list(self.gesture_metrics["gesture_variety"]),
                    "effectiveness_rate": (self.gesture_metrics["effective_gestures"] / 
                                         max(1, self.gesture_metrics["total_gestures"])) * 100,
                    "average_hand_visibility": self.gesture_metrics["hand_visibility_score"]
                },
                
                # Emotion insights
                "emotion_insights": {
                    "dominant_emotions": dict(self.emotion_metrics["dominant_emotions"]),
                    "average_confidence": np.mean(self.emotion_metrics["confidence_levels"]) if self.emotion_metrics["confidence_levels"] else 0,
                    "emotion_stability": self._calculate_emotion_stability(),
                    "most_frequent_emotion": max(self.emotion_metrics["dominant_emotions"], 
                                               key=self.emotion_metrics["dominant_emotions"].get) if self.emotion_metrics["dominant_emotions"] else "unknown"
                },
                
                # Posture insights
                "posture_insights": {
                    "average_posture_score": np.mean(self.posture_metrics["posture_scores"]) if self.posture_metrics["posture_scores"] else 0,
                    "engagement_distribution": self._calculate_engagement_distribution(),
                    "common_body_language": self._get_common_body_language(),
                    "posture_consistency": np.std(self.posture_metrics["posture_scores"]) if len(self.posture_metrics["posture_scores"]) > 1 else 0
                },
                
                # Overall assessment
                "overall_assessment": self._generate_final_assessment(),
                "improvement_plan": self._generate_improvement_plan(),
                "strengths": self._identify_strengths(),
                "priority_areas": self._identify_priority_areas()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating enhanced session summary: {e}")
            return {"error": str(e)}
    
    def _get_analysis_duration(self) -> float:
        """Calculate analysis duration"""
        if len(self.analysis_history) < 2:
            return 0.0
        return self.analysis_history[-1]["timestamp"] - self.analysis_history[0]["timestamp"]
    
    def _calculate_emotion_stability(self) -> float:
        """Calculate how stable emotions were throughout the session"""
        if len(self.emotion_metrics["confidence_levels"]) < 2:
            return 0.0
        return 1.0 - np.std(self.emotion_metrics["confidence_levels"])
    
    def _calculate_engagement_distribution(self) -> Dict[str, int]:
        """Calculate distribution of engagement levels"""
        from collections import Counter
        return dict(Counter(self.posture_metrics["engagement_levels"]))
    
    def _get_common_body_language(self) -> List[str]:
        """Get most common body language indicators"""
        from collections import Counter
        counter = Counter(self.posture_metrics["body_language_indicators"])
        return [item for item, count in counter.most_common(5)]
    
    def _generate_final_assessment(self) -> Dict[str, Any]:
        """Generate final comprehensive assessment"""
        # Calculate overall scores from recent analyses
        recent_analyses = list(self.analysis_history)[-10:]  # Last 10 analyses
        
        gesture_scores = []
        emotion_scores = []
        posture_scores = []
        pitch_readiness_scores = []
        
        for analysis in recent_analyses:
            scores = analysis.get("overall_scores", {})
            if scores.get("gesture_score", 0) > 0:
                gesture_scores.append(scores["gesture_score"])
            if scores.get("emotion_score", 0) > 0:
                emotion_scores.append(scores["emotion_score"])
            if scores.get("posture_score", 0) > 0:
                posture_scores.append(scores["posture_score"])
            if scores.get("pitch_readiness", 0) > 0:
                pitch_readiness_scores.append(scores["pitch_readiness"])
        
        return {
            "final_gesture_score": np.mean(gesture_scores) if gesture_scores else 0.0,
            "final_emotion_score": np.mean(emotion_scores) if emotion_scores else 0.0,
            "final_posture_score": np.mean(posture_scores) if posture_scores else 0.0,
            "final_pitch_readiness": np.mean(pitch_readiness_scores) if pitch_readiness_scores else 0.0,
            "overall_grade": self._calculate_overall_grade(gesture_scores, emotion_scores, posture_scores)
        }
    
    def _calculate_overall_grade(self, gesture_scores: List[float], emotion_scores: List[float], posture_scores: List[float]) -> str:
        """Calculate letter grade for overall performance"""
        all_scores = gesture_scores + emotion_scores + posture_scores
        if not all_scores:
            return "N/A"
        
        avg_score = np.mean(all_scores)
        
        if avg_score >= 0.9:
            return "A+"
        elif avg_score >= 0.8:
            return "A"
        elif avg_score >= 0.7:
            return "B+"
        elif avg_score >= 0.6:
            return "B"
        elif avg_score >= 0.5:
            return "C+"
        elif avg_score >= 0.4:
            return "C"
        else:
            return "D"
    
    def _generate_improvement_plan(self) -> List[str]:
        """Generate personalized improvement plan"""
        plan = []
        
        # Analyze weakest areas
        gesture_avg = np.mean(self.posture_metrics["posture_scores"]) if self.posture_metrics["posture_scores"] else 0
        emotion_avg = np.mean(self.emotion_metrics["confidence_levels"]) if self.emotion_metrics["confidence_levels"] else 0
        
        if gesture_avg < 0.6:
            plan.append("📚 Study effective presentation gestures and practice in front of a mirror")
            plan.append("🎯 Focus on using purposeful hand movements that support your message")
        
        if emotion_avg < 0.6:
            plan.append("😊 Practice projecting confidence and positive emotions")
            plan.append("🎭 Work on facial expression exercises to appear more engaging")
        
        if len(self.gesture_metrics["gesture_variety"]) < 3:
            plan.append("🤹 Expand your gesture vocabulary - learn different types of hand movements")
        
        return plan
    
    def _identify_strengths(self) -> List[str]:
        """Identify user's strengths"""
        strengths = []
        
        if self.gesture_metrics["effective_gestures"] / max(1, self.gesture_metrics["total_gestures"]) > 0.7:
            strengths.append("Effective use of hand gestures")
        
        if len(self.gesture_metrics["gesture_variety"]) >= 4:
            strengths.append("Good variety in gesture types")
        
        if np.mean(self.emotion_metrics["confidence_levels"]) > 0.7 if self.emotion_metrics["confidence_levels"] else False:
            strengths.append("Confident facial expressions")
        
        if np.mean(self.posture_metrics["posture_scores"]) > 0.7 if self.posture_metrics["posture_scores"] else False:
            strengths.append("Good posture and body alignment")
        
        return strengths
    
    def _identify_priority_areas(self) -> List[str]:
        """Identify priority areas for improvement"""
        priority_areas = []
        
        if self.gesture_metrics["total_gestures"] == 0:
            priority_areas.append("Hand gesture usage - currently not using gestures")
        elif self.gesture_metrics["effective_gestures"] / max(1, self.gesture_metrics["total_gestures"]) < 0.5:
            priority_areas.append("Gesture effectiveness - improve quality of hand movements")
        
        if np.mean(self.emotion_metrics["confidence_levels"]) < 0.5 if self.emotion_metrics["confidence_levels"] else True:
            priority_areas.append("Emotional expression - work on projecting confidence")
        
        if np.mean(self.posture_metrics["posture_scores"]) < 0.5 if self.posture_metrics["posture_scores"] else True:
            priority_areas.append("Posture and body language - improve overall presence")
        
        return priority_areas


# Global enhanced video analyzer instance
enhanced_video_analyzer: Optional[EnhancedVideoAnalyzer] = None

def get_enhanced_video_analyzer() -> Optional[EnhancedVideoAnalyzer]:
    """Get the global enhanced video analyzer instance"""
    return enhanced_video_analyzer

def initialize_enhanced_video_analyzer(config: Optional[Dict] = None) -> EnhancedVideoAnalyzer:
    """Initialize the enhanced video analyzer"""
    global enhanced_video_analyzer
    enhanced_video_analyzer = EnhancedVideoAnalyzer(config)
    logger.info("Enhanced Video Analyzer initialized")
    return enhanced_video_analyzer