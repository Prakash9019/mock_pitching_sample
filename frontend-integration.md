# Frontend Integration Guide

## 🚀 Complete Integration Guide for React/Vue/Angular

This comprehensive guide provides everything you need to integrate the AI Mock Investor Pitch backend with any modern frontend framework, including real-time audio streaming, database integration, and advanced features.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [API Overview](#api-overview)
3. [Database Integration](#database-integration)
4. [WebSocket Integration](#websocket-integration)
5. [Real-Time Audio Streaming](#real-time-audio-streaming)
6. [**Video Analysis Integration**](#video-analysis-integration) 🆕
7. [Speech-to-Text Integration](#speech-to-text-integration)
8. [Text-to-Speech Integration](#text-to-speech-integration)
9. [Session Management](#session-management)
10. [Session Ending & Analysis](#session-ending--analysis)
11. [Analytics & Reporting](#analytics--reporting)
12. [Step-by-Step Implementation](#step-by-step-implementation)
13. [React Complete Example](#react-complete-example)
14. [Vue.js Complete Example](#vuejs-complete-example)
15. [Angular Complete Example](#angular-complete-example)
16. [Error Handling](#error-handling)
17. [Performance Optimization](#performance-optimization)
18. [Security Considerations](#security-considerations)
19. [Testing Guide](#testing-guide)
20. [Deployment Guide](#deployment-guide)
21. [Best Practices](#best-practices)

---

## Prerequisites

### Backend Requirements
- Backend server running on `https://ai-mock-pitching-427457295403.europe-west1.run.app` (deployed) or your local development URL
- Socket.IO server enabled and accessible
- MongoDB database connected
- All API endpoints functional
- Audio processing capabilities enabled

### Frontend Requirements
- Modern JavaScript framework (React 18+, Vue 3+, Angular 15+)
- Node.js 16+ and npm/yarn
- Modern browser with WebRTC support
- **Camera, microphone and speaker access** 🆕
- **Canvas API support for video frame capture** 🆕

### Required Dependencies
```bash
# Core dependencies
npm install socket.io-client axios

# Audio processing
npm install recordrtc web-audio-api

# **Video processing (NEW)** 🆕
npm install canvas-capture html2canvas

# UI components (optional)
npm install @mui/material @emotion/react @emotion/styled  # React
npm install vuetify  # Vue
npm install @angular/material  # Angular

# State management (optional)
npm install redux @reduxjs/toolkit  # React
npm install vuex  # Vue
npm install @ngrx/store  # Angular

# Utilities
npm install lodash moment uuid
```

---

## API Overview

### REST Endpoints

#### Core Endpoints
| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `GET` | `/api/personas` | Get all available investor personas | - | `{personas: {...}}` |
| `POST` | `/api/pitch/end/{session_id}` | **End pitch session and generate comprehensive analysis** | `{reason?: string}` | `{success: boolean, message: string, analysis: object}` |
| `GET` | `/api/pitch/analytics/{session_id}` | Get session analytics | - | `{analytics: {...}}` |
| `GET` | `/api/pitch/analysis/{session_id}` | Get detailed analysis | - | `{analysis: {...}}` |
| `GET` | `/api/pitch/report/{session_id}` | Get formatted report | - | `{report: {...}}` |
| `GET` | `/download/{filename}` | Download audio files | - | Audio file |

#### Database Management Endpoints
| Method | Endpoint | Description | Query Params | Response |
|--------|----------|-------------|--------------|----------|
| `GET` | `/api/sessions` | List all sessions | `page`, `limit`, `search` | `{sessions: [...], total: number}` |
| `GET` | `/api/sessions/{session_id}` | Get session details | - | `{session: {...}}` |
| `GET` | `/api/analyses` | Get recent analyses | `limit` | `{analyses: [...]}` |
| `GET` | `/api/stats` | Get database statistics | - | `{stats: {...}}` |
| `GET` | `/api/search` | Search sessions | `q`, `type` | `{results: [...]}` |

#### Health & Status Endpoints
| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| `GET` | `/health` | Health check | `{status: "healthy"}` |
| `GET` | `/api/status` | System status | `{database: boolean, ai: boolean}` |

### WebSocket Events

#### Client → Server Events
| Event | Data Structure | Description |
|-------|----------------|-------------|
| `connect` | - | Establish WebSocket connection |
| `text_message` | `{text: string, persona: string, session_id: string, system: string}` | Send text message to AI |
| `audio_chunk` | `{audio_data: string, session_id: string, persona: string, is_final: boolean, mime_type?: string}` | Send audio chunk for real-time STT |
| `start_recording` | `{session_id: string, persona: string, sample_rate: number}` | Start audio recording session |
| `stop_recording` | `{session_id: string}` | Stop audio recording session |
| **`start_video_analysis`** 🆕 | `{session_id: string}` | **Start real-time video analysis** |
| **`stop_video_analysis`** 🆕 | `{session_id: string}` | **Stop video analysis** |
| **`video_frame`** 🆕 | `{session_id: string, frame_data: string}` | **Send video frame for analysis (base64 JPEG)** |
| **`enable_video_analysis`** 🆕 | `{session_id: string, enabled: boolean}` | **Force enable/disable video analysis** |
| `join_session` | `{session_id: string}` | Join existing session |
| `leave_session` | `{session_id: string}` | Leave current session |

#### Server → Client Events
| Event | Data Structure | Description |
|-------|----------------|-------------|
| `response` | `{message: string, audio_url?: string, stage?: string, complete?: boolean, session_id: string}` | AI response with optional audio |
| `transcription` | `{text: string, is_final: boolean, session_id: string, confidence?: number}` | Real-time speech transcription |
| `session_started` | `{session_id: string, persona: string, system: string, timestamp: string}` | Session creation confirmation |
| `session_ended` | `{session_id: string, reason: string, analytics?: object}` | Session termination |
| **`video_analysis_started`** 🆕 | `{session_id: string, status: string, analyzer_type: string, message: string}` | **Video analysis activation confirmation** |
| **`video_analysis_stopped`** 🆕 | `{session_id: string, message: string}` | **Video analysis deactivation** |
| **`video_analysis_update`** 🆕 | `{session_id: string, analysis: object, analyzer_type: string}` | **Real-time video analysis results with gesture, emotion, and pose data** |
| **`video_insights`** 🆕 | `{session_id: string, insights: object, recommendations: array}` | **AI-generated video insights** |
| **`video_error`** 🆕 | `{error: string, session_id?: string}` | **Video analysis errors** |
| **`video_status`** 🆕 | `{session_id: string, status: string}` | **Video analysis status updates (enabled/disabled/analyzing)** |
| `error` | `{message: string, type: string, code?: number}` | Error notifications |
| `status_update` | `{session_id: string, status: string, stage?: string}` | Session status changes |
| `typing_indicator` | `{session_id: string, is_typing: boolean}` | AI typing indicator |

---

## Database Integration

### Session Management API

```javascript
// API service for database operations
class DatabaseService {
  constructor(baseURL = 'https://ai-mock-pitching-427457295403.europe-west1.run.app') {
    this.baseURL = baseURL;
    this.axios = axios.create({
      baseURL: this.baseURL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  // Get all sessions with pagination
  async getSessions(page = 1, limit = 10, search = '') {
    try {
      const response = await this.axios.get('/api/sessions', {
        params: { page, limit, search }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching sessions:', error);
      throw error;
    }
  }

  // Get specific session details
  async getSession(sessionId) {
    try {
      const response = await this.axios.get(`/api/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching session:', error);
      throw error;
    }
  }

  // Get recent analyses
  async getAnalyses(limit = 5) {
    try {
      const response = await this.axios.get('/api/analyses', {
        params: { limit }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching analyses:', error);
      throw error;
    }
  }

  // Get database statistics
  async getStats() {
    try {
      const response = await this.axios.get('/api/stats');
      return response.data;
    } catch (error) {
      console.error('Error fetching stats:', error);
      throw error;
    }
  }

  // Search sessions
  async searchSessions(query, type = 'all') {
    try {
      const response = await this.axios.get('/api/search', {
        params: { q: query, type }
      });
      return response.data;
    } catch (error) {
      console.error('Error searching sessions:', error);
      throw error;
    }
  }
}

// Usage example
const dbService = new DatabaseService();

// Load sessions for dashboard
const loadSessions = async () => {
  try {
    const data = await dbService.getSessions(1, 20);
    setSessions(data.sessions);
    setTotalSessions(data.total);
  } catch (error) {
    setError('Failed to load sessions');
  }
};

// Load analytics dashboard
const loadDashboard = async () => {
  try {
    const [stats, recentAnalyses] = await Promise.all([
      dbService.getStats(),
      dbService.getAnalyses(10)
    ]);
    
    setDashboardStats(stats);
    setRecentAnalyses(recentAnalyses.analyses);
  } catch (error) {
    setError('Failed to load dashboard data');
  }
};
```

---

## WebSocket Integration

### Advanced Connection Setup

```javascript
import io from 'socket.io-client';

class SocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.eventHandlers = new Map();
  }

  connect(serverURL = 'https://ai-mock-pitching-427457295403.europe-west1.run.app') {
    if (this.socket) {
      this.disconnect();
    }

    this.socket = io(serverURL, {
      transports: ['websocket', 'polling'],
      timeout: 20000,
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      maxReconnectionAttempts: this.maxReconnectAttempts,
      forceNew: true
    });

    this.setupEventHandlers();
    return this.socket;
  }

  setupEventHandlers() {
    // Connection events
    this.socket.on('connect', () => {
      console.log('✅ Connected to server');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.emit('connection_status', { connected: true });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('❌ Disconnected from server:', reason);
      this.isConnected = false;
      this.emit('connection_status', { connected: false, reason });
    });

    this.socket.on('connect_error', (error) => {
      console.error('🔴 Connection error:', error);
      this.reconnectAttempts++;
      this.emit('connection_error', { error, attempts: this.reconnectAttempts });
    });

    this.socket.on('reconnect', (attemptNumber) => {
      console.log('🔄 Reconnected after', attemptNumber, 'attempts');
      this.emit('reconnected', { attempts: attemptNumber });
    });

    this.socket.on('reconnect_failed', () => {
      console.error('💥 Failed to reconnect after maximum attempts');
      this.emit('reconnect_failed');
    });

    // Application events
    this.socket.on('response', (data) => {
      this.emit('ai_response', data);
    });

    this.socket.on('transcription', (data) => {
      this.emit('transcription', data);
    });

    this.socket.on('session_started', (data) => {
      this.emit('session_started', data);
    });

    this.socket.on('session_ended', (data) => {
      this.emit('session_ended', data);
    });

    this.socket.on('error', (error) => {
      this.emit('error', error);
    });

    this.socket.on('status_update', (data) => {
      this.emit('status_update', data);
    });

    this.socket.on('typing_indicator', (data) => {
      this.emit('typing_indicator', data);
    });
  }

  // Event emitter pattern for clean event handling
  on(event, handler) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event).push(handler);
  }

  off(event, handler) {
    if (this.eventHandlers.has(event)) {
      const handlers = this.eventHandlers.get(event);
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  emit(event, data) {
    if (this.eventHandlers.has(event)) {
      this.eventHandlers.get(event).forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in event handler for ${event}:`, error);
        }
      });
    }
  }

  // Send message to AI
  sendMessage(text, persona, sessionId, system = 'workflow') {
    if (!this.isConnected) {
      throw new Error('Socket not connected');
    }

    this.socket.emit('text_message', {
      text: text.trim(),
      persona: persona,
      session_id: sessionId,
      system: system
    });
  }

  // Join session
  joinSession(sessionId) {
    if (!this.isConnected) {
      throw new Error('Socket not connected');
    }

    this.socket.emit('join_session', {
      session_id: sessionId
    });
  }

  // Leave session
  leaveSession(sessionId) {
    if (!this.isConnected) {
      throw new Error('Socket not connected');
    }

    this.socket.emit('leave_session', {
      session_id: sessionId
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
    }
  }

  getConnectionStatus() {
    return {
      connected: this.isConnected,
      socket: this.socket,
      id: this.socket?.id
    };
  }
}

// Global socket service instance
const socketService = new SocketService();

// Usage in React component
const useSocket = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);

  useEffect(() => {
    // Connect to server
    socketService.connect();

    // Set up event handlers
    const handleConnectionStatus = ({ connected, reason }) => {
      setIsConnected(connected);
      if (!connected && reason) {
        setConnectionError(reason);
      } else {
        setConnectionError(null);
      }
    };

    const handleConnectionError = ({ error, attempts }) => {
      setConnectionError(`Connection failed (attempt ${attempts}): ${error.message}`);
    };

    socketService.on('connection_status', handleConnectionStatus);
    socketService.on('connection_error', handleConnectionError);

    // Cleanup
    return () => {
      socketService.off('connection_status', handleConnectionStatus);
      socketService.off('connection_error', handleConnectionError);
      socketService.disconnect();
    };
  }, []);

  return {
    isConnected,
    connectionError,
    socketService
  };
};
```

---

## Video Analysis Integration

### 🎥 Enhanced Video Analysis System

The platform now includes **professional-grade video analysis** using state-of-the-art computer vision libraries:

- **CVZone**: Advanced hand gesture recognition and tracking
- **FER (Facial Emotion Recognition)**: Real-time emotion detection and analysis
- **MediaPipe**: Professional pose estimation and body language analysis

### Video Analysis Capabilities

#### 🤲 Hand Gesture Analysis
- **Gesture Recognition**: Pointing, open palm, closed fist, thumbs up, peace sign
- **Gesture Effectiveness**: Scoring based on pitch context and timing
- **Hand Position Tracking**: Real-time hand movement and positioning
- **Gesture Confidence**: AI confidence scores for detected gestures

#### 😊 Facial Emotion Recognition
- **Emotion Detection**: Happy, sad, angry, fear, surprise, disgust, neutral
- **Pitch Suitability**: Emotion appropriateness for pitch context
- **Confidence Indicators**: Facial expression confidence analysis
- **Eye Contact Estimation**: Gaze direction and engagement scoring

#### 🧍 Pose and Body Language
- **Posture Analysis**: Upright, leaning forward/back, slouching detection
- **Engagement Level**: Body language-based engagement scoring
- **Professional Presence**: Overall body language assessment
- **Movement Tracking**: Gesture coordination and natural movement

#### 🔄 LangGraph Workflow Integration
- **Real-time State Updates**: Video analysis data is integrated into the LangGraph workflow state
- **Comprehensive Analysis**: Video insights are incorporated into the final pitch analysis
- **Gesture Feedback**: Hand gesture effectiveness is tracked throughout the pitch
- **Expression Tracking**: Emotional state is monitored during different pitch stages
- **Posture Assessment**: Body language is evaluated for professional presence
- **Stage-specific Analysis**: Video performance is analyzed for each pitch stage
- **Explicit Activation**: Use `enable_video_analysis` event to ensure video data is included in analysis
- **Reliable Integration**: Send frames consistently to maintain video analysis throughout the session

### Video Integration Service

```javascript
class VideoAnalysisService {
  constructor(socketService) {
    this.socketService = socketService;
    this.mediaStream = null;
    this.videoElement = null;
    this.analysisInterval = null;
    this.isAnalyzing = false;
    
    // Analysis settings
    this.frameRate = 0.5; // Analyze every 0.5 seconds
    this.frameQuality = 0.8; // JPEG quality
    this.frameWidth = 320;
    this.frameHeight = 240;
    
    // Analysis state
    this.currentAnalysis = null;
    this.analysisHistory = [];
    
    // Event handlers
    this.onAnalysisUpdate = null;
    this.onInsights = null;
    this.onError = null;
  }

  // Initialize video capture
  async initializeVideo() {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: 640, 
          height: 480, 
          facingMode: 'user',
          frameRate: { ideal: 30, max: 60 }
        },
        audio: false // Audio handled separately
      });
      
      // Set up video element
      this.videoElement = document.createElement('video');
      this.videoElement.srcObject = this.mediaStream;
      this.videoElement.autoplay = true;
      this.videoElement.muted = true;
      this.videoElement.playsInline = true;
      
      return new Promise((resolve) => {
        this.videoElement.onloadedmetadata = () => {
          console.log('✅ Video initialized for analysis');
          resolve(true);
        };
      });
      
    } catch (error) {
      console.error('❌ Video initialization failed:', error);
      throw new Error(`Video access denied: ${error.message}`);
    }
  }

  // Start video analysis
  async startAnalysis(sessionId) {
    if (!this.mediaStream || !this.socketService.isConnected) {
      throw new Error('Video not initialized or socket not connected');
    }

    try {
      // Start video analysis on server
      this.socketService.socket.emit('start_video_analysis', {
        session_id: sessionId
      });
      
      // Explicitly enable video analysis in LangGraph workflow
      this.socketService.socket.emit('enable_video_analysis', {
        session_id: sessionId,
        enabled: true
      });

      // Start frame capture
      this.isAnalyzing = true;
      this.analysisInterval = setInterval(() => {
        this.captureAndSendFrame(sessionId);
      }, this.frameRate * 1000);
      
      // Send initial frame immediately to ensure analysis starts
      setTimeout(() => {
        this.captureAndSendFrame(sessionId);
      }, 500);

      console.log('🎥 Video analysis started with LangGraph integration');
      
    } catch (error) {
      console.error('❌ Failed to start video analysis:', error);
      throw error;
    }
  }

  // Stop video analysis
  async stopAnalysis(sessionId) {
    try {
      this.isAnalyzing = false;
      
      if (this.analysisInterval) {
        clearInterval(this.analysisInterval);
        this.analysisInterval = null;
      }

      if (this.socketService.isConnected) {
        this.socketService.socket.emit('stop_video_analysis', {
          session_id: sessionId
        });
      }

      console.log('⏹️ Video analysis stopped');
      
    } catch (error) {
      console.error('❌ Error stopping video analysis:', error);
    }
  }

  // Capture and send video frame
  captureAndSendFrame(sessionId) {
    if (!this.isAnalyzing || !this.videoElement || !this.socketService.isConnected) {
      return;
    }

    try {
      // Create canvas for frame capture
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      // Use actual video dimensions or fallback to configured dimensions
      canvas.width = this.videoElement.videoWidth || this.frameWidth;
      canvas.height = this.videoElement.videoHeight || this.frameHeight;
      
      // Ensure valid dimensions
      if (canvas.width === 0 || canvas.height === 0) {
        canvas.width = this.frameWidth;
        canvas.height = this.frameHeight;
      }
      
      // Draw video frame to canvas
      ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);
      
      // Convert to base64 JPEG with quality setting
      const frameData = canvas.toDataURL('image/jpeg', this.frameQuality);
      
      // Send frame to server
      this.socketService.socket.emit('video_frame', {
        session_id: sessionId,
        frame_data: frameData
      });
      
      // Force enable video analysis flag to ensure proper integration
      this.socketService.socket.emit('enable_video_analysis', {
        session_id: sessionId,
        enabled: true
      });
      
    } catch (error) {
      console.error('❌ Error capturing video frame:', error);
    }
  }

  // Set up video analysis event listeners
  setupEventListeners() {
    if (!this.socketService.socket) return;

    // Video analysis started
    this.socketService.socket.on('video_analysis_started', (data) => {
      console.log(`🎥 Video analysis started: ${data.analyzer_type}`);
      if (this.onAnalysisUpdate) {
        this.onAnalysisUpdate({
          type: 'started',
          analyzerType: data.analyzer_type,
          message: data.message
        });
      }
    });

    // Video analysis stopped
    this.socketService.socket.on('video_analysis_stopped', (data) => {
      console.log('⏹️ Video analysis stopped');
      if (this.onAnalysisUpdate) {
        this.onAnalysisUpdate({
          type: 'stopped',
          message: data.message
        });
      }
    });

    // Real-time analysis updates
    this.socketService.socket.on('video_analysis_update', (data) => {
      this.currentAnalysis = data.analysis;
      this.analysisHistory.push({
        timestamp: Date.now(),
        analysis: data.analysis,
        analyzerType: data.analyzer_type
      });

      if (this.onAnalysisUpdate) {
        this.onAnalysisUpdate({
          type: 'update',
          analysis: data.analysis,
          analyzerType: data.analyzer_type
        });
      }
    });

    // Video insights
    this.socketService.socket.on('video_insights', (data) => {
      if (this.onInsights) {
        this.onInsights({
          insights: data.insights,
          recommendations: data.recommendations,
          sessionId: data.session_id
        });
      }
    });

    // Video errors
    this.socketService.socket.on('video_error', (data) => {
      console.error('❌ Video analysis error:', data.error);
      if (this.onError) {
        this.onError(data.error);
      }
    });
  }

  // Get current analysis metrics
  getCurrentMetrics() {
    if (!this.currentAnalysis) return null;

    const analysis = this.currentAnalysis;
    
    return {
      // Hand gesture metrics
      handGestures: {
        detected: analysis.hand_analysis?.hands_detected || 0,
        gestures: analysis.hand_analysis?.gestures || [],
        effectiveness: analysis.hand_analysis?.gesture_effectiveness || 0,
        confidence: analysis.hand_analysis?.gesture_confidence || 0
      },
      
      // Emotion metrics
      emotions: {
        dominant: analysis.emotion_analysis?.dominant_emotion || 'neutral',
        confidence: analysis.emotion_analysis?.emotion_confidence || 0,
        suitability: analysis.emotion_analysis?.pitch_suitability || 0,
        allEmotions: analysis.emotion_analysis?.emotions || {}
      },
      
      // Pose metrics
      pose: {
        detected: analysis.pose_analysis?.pose_detected || false,
        posture: analysis.pose_analysis?.posture_score || 0,
        engagement: analysis.pose_analysis?.engagement_level || 'neutral',
        bodyLanguage: analysis.pose_analysis?.body_language || []
      },
      
      // Overall scores
      overall: {
        gestureScore: analysis.overall_scores?.gesture_score || 0,
        emotionScore: analysis.overall_scores?.emotion_score || 0,
        poseScore: analysis.overall_scores?.pose_score || 0,
        overallScore: analysis.overall_scores?.overall_score || 0
      }
    };
  }

  // Cleanup resources
  cleanup() {
    this.stopAnalysis();
    
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    
    if (this.videoElement) {
      this.videoElement.srcObject = null;
      this.videoElement = null;
    }
  }
}

// Usage example
const videoService = new VideoAnalysisService(socketService);

// Initialize video analysis
const initializeVideoAnalysis = async () => {
  try {
    await videoService.initializeVideo();
    videoService.setupEventListeners();
    
    // Set up event handlers
    videoService.onAnalysisUpdate = (data) => {
      if (data.type === 'update') {
        updateVideoMetrics(data.analysis);
      }
    };
    
    videoService.onInsights = (data) => {
      displayVideoInsights(data.insights, data.recommendations);
    };
    
    videoService.onError = (error) => {
      console.error('Video analysis error:', error);
      showErrorMessage(`Video analysis error: ${error}`);
    };
    
  } catch (error) {
    console.error('Failed to initialize video analysis:', error);
  }
};

// Start video analysis for session
const startVideoAnalysis = async (sessionId) => {
  try {
    await videoService.startAnalysis(sessionId);
    console.log('🎥 Video analysis started successfully');
  } catch (error) {
    console.error('Failed to start video analysis:', error);
  }
};

// Stop video analysis
const stopVideoAnalysis = async (sessionId) => {
  try {
    await videoService.stopAnalysis(sessionId);
    console.log('⏹️ Video analysis stopped successfully');
  } catch (error) {
    console.error('Failed to stop video analysis:', error);
  }
};
```

### Video Analysis Data Structure

#### Enhanced Analysis Response
```javascript
{
  "session_id": "session_123",
  "analyzer_type": "enhanced", // "basic" or "enhanced"
  "analysis": {
    // Hand analysis (CVZone)
    "hand_analysis": {
      "hands_detected": 2,
      "gestures": [
        {
          "type": "pointing",
          "confidence": 0.85,
          "effectiveness": 0.9,
          "description": "Effective pointing gesture"
        }
      ],
      "gesture_effectiveness": 0.87,
      "gesture_confidence": 0.82
    },
    
    // Emotion analysis (FER)
    "emotion_analysis": {
      "dominant_emotion": "happy",
      "emotion_confidence": 0.91,
      "pitch_suitability": 0.88,
      "emotions": {
        "happy": 0.65,
        "neutral": 0.25,
        "confident": 0.10
      }
    },
    
    // Pose analysis (MediaPipe)
    "pose_analysis": {
      "pose_detected": true,
      "posture_score": 0.85,
      "engagement_level": "high",
      "body_language": [
        "upright_posture",
        "open_stance",
        "good_eye_contact"
      ]
    },
    
    // Overall scoring
    "overall_scores": {
      "gesture_score": 0.87,
      "emotion_score": 0.91,
      "pose_score": 0.85,
      "overall_score": 0.88
    }
  }
}
```

### React Video Analysis Hook

```javascript
import { useState, useEffect, useRef } from 'react';

const useVideoAnalysis = (socketService) => {
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentMetrics, setCurrentMetrics] = useState(null);
  const [videoInsights, setVideoInsights] = useState([]);
  const [error, setError] = useState(null);
  
  const videoServiceRef = useRef(null);
  const videoElementRef = useRef(null);

  useEffect(() => {
    // Initialize video service
    videoServiceRef.current = new VideoAnalysisService(socketService);
    
    return () => {
      // Cleanup on unmount
      if (videoServiceRef.current) {
        videoServiceRef.current.cleanup();
      }
    };
  }, [socketService]);

  const initializeVideo = async () => {
    try {
      setError(null);
      await videoServiceRef.current.initializeVideo();
      
      // Set video element for display
      if (videoElementRef.current) {
        videoElementRef.current.srcObject = videoServiceRef.current.mediaStream;
      }
      
      // Set up event handlers
      videoServiceRef.current.onAnalysisUpdate = (data) => {
        if (data.type === 'update') {
          const metrics = videoServiceRef.current.getCurrentMetrics();
          setCurrentMetrics(metrics);
        }
      };
      
      videoServiceRef.current.onInsights = (data) => {
        setVideoInsights(prev => [...prev, data]);
      };
      
      videoServiceRef.current.onError = (error) => {
        setError(error);
        setIsAnalyzing(false);
      };
      
      videoServiceRef.current.setupEventListeners();
      setIsVideoReady(true);
      
    } catch (error) {
      setError(error.message);
      setIsVideoReady(false);
    }
  };

  const startAnalysis = async (sessionId) => {
    try {
      setError(null);
      await videoServiceRef.current.startAnalysis(sessionId);
      setIsAnalyzing(true);
    } catch (error) {
      setError(error.message);
    }
  };

  const stopAnalysis = async (sessionId) => {
    try {
      await videoServiceRef.current.stopAnalysis(sessionId);
      setIsAnalyzing(false);
    } catch (error) {
      setError(error.message);
    }
  };

  return {
    // State
    isVideoReady,
    isAnalyzing,
    currentMetrics,
    videoInsights,
    error,
    
    // Refs
    videoElementRef,
    
    // Actions
    initializeVideo,
    startAnalysis,
    stopAnalysis
  };
};

// Usage in React component
const PitchPracticeComponent = () => {
  const {
    isVideoReady,
    isAnalyzing,
    currentMetrics,
    videoInsights,
    error,
    videoElementRef,
    initializeVideo,
    startAnalysis,
    stopAnalysis
  } = useVideoAnalysis(socketService);

  const [sessionId, setSessionId] = useState(null);

  useEffect(() => {
    initializeVideo();
  }, []);

  const handleStartSession = async () => {
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    await startAnalysis(newSessionId);
  };

  const handleStopSession = async () => {
    if (sessionId) {
      await stopAnalysis(sessionId);
      setSessionId(null);
    }
  };

  return (
    <div className="pitch-practice-container">
      {/* Video Preview */}
      <div className="video-section">
        <h3>📹 Video Preview</h3>
        <video
          ref={videoElementRef}
          autoPlay
          muted
          playsInline
          className="video-preview"
          style={{ width: '100%', maxWidth: '640px', height: 'auto' }}
        />
        <div className="video-status">
          {isVideoReady ? (
            <span className="status-ready">✅ Camera Ready</span>
          ) : (
            <span className="status-loading">⏳ Initializing Camera...</span>
          )}
        </div>
      </div>

      {/* Video Analysis Metrics */}
      {currentMetrics && (
        <div className="video-metrics">
          <h3>📊 Real-time Video Analysis</h3>
          
          {/* Hand Gestures */}
          <div className="metric-group">
            <h4>🤲 Hand Gestures</h4>
            <div className="metrics">
              <div>Hands Detected: {currentMetrics.handGestures.detected}</div>
              <div>Gesture Effectiveness: {Math.round(currentMetrics.handGestures.effectiveness * 100)}%</div>
              <div>Current Gestures: {currentMetrics.handGestures.gestures.map(g => g.type).join(', ')}</div>
            </div>
          </div>

          {/* Emotions */}
          <div className="metric-group">
            <h4>😊 Facial Emotions</h4>
            <div className="metrics">
              <div>Dominant Emotion: {currentMetrics.emotions.dominant}</div>
              <div>Emotion Confidence: {Math.round(currentMetrics.emotions.confidence * 100)}%</div>
              <div>Pitch Suitability: {Math.round(currentMetrics.emotions.suitability * 100)}%</div>
            </div>
          </div>

          {/* Pose & Body Language */}
          <div className="metric-group">
            <h4>🧍 Pose & Body Language</h4>
            <div className="metrics">
              <div>Posture Score: {Math.round(currentMetrics.pose.posture * 100)}%</div>
              <div>Engagement Level: {currentMetrics.pose.engagement}</div>
              <div>Body Language: {currentMetrics.pose.bodyLanguage.join(', ')}</div>
            </div>
          </div>

          {/* Overall Score */}
          <div className="metric-group overall-score">
            <h4>🎯 Overall Performance</h4>
            <div className="score-display">
              {Math.round(currentMetrics.overall.overallScore * 100)}%
            </div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="controls">
        {!isAnalyzing ? (
          <button 
            onClick={handleStartSession}
            disabled={!isVideoReady}
            className="start-button"
          >
            🎥 Start Video Analysis
          </button>
        ) : (
          <button 
            onClick={handleStopSession}
            className="stop-button"
          >
            ⏹️ Stop Analysis
          </button>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      {/* Video Insights */}
      {videoInsights.length > 0 && (
        <div className="video-insights">
          <h3>💡 AI Video Insights</h3>
          {videoInsights.map((insight, index) => (
            <div key={index} className="insight-item">
              <div className="insight-content">
                {insight.insights}
              </div>
              <div className="recommendations">
                {insight.recommendations.map((rec, i) => (
                  <div key={i} className="recommendation">
                    💡 {rec}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
```

---

## Real-Time Audio Streaming

### Advanced Audio Recording System

```javascript
class AdvancedAudioRecorder {
  constructor() {
    this.mediaRecorder = null;
    this.audioContext = null;
    this.processor = null;
    this.analyser = null;
    this.stream = null;
    this.isRecording = false;
    this.socket = null;
    this.sessionId = null;
    this.persona = null;
    this.audioChunks = [];
    this.recordingStartTime = null;
    this.silenceDetection = true;
    this.silenceThreshold = 0.01;
    this.silenceTimeout = 2000; // 2 seconds
    this.lastSoundTime = 0;
    this.vadEnabled = true; // Voice Activity Detection
  }

  async initialize(socket, sessionId, persona, options = {}) {
    this.socket = socket;
    this.sessionId = sessionId;
    this.persona = persona;
    
    // Merge options
    const config = {
      sampleRate: 16000,
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      silenceDetection: true,
      vadEnabled: true,
      ...options
    };

    this.silenceDetection = config.silenceDetection;
    this.vadEnabled = config.vadEnabled;

    try {
      // Request microphone access with advanced constraints
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: config.sampleRate,
          channelCount: config.channelCount,
          echoCancellation: config.echoCancellation,
          noiseSuppression: config.noiseSuppression,
          autoGainControl: config.autoGainControl,
          // Advanced constraints for better quality
          latency: 0.01,
          volume: 1.0
        }
      });

      // Create audio context with specified sample rate
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: config.sampleRate,
        latencyHint: 'interactive'
      });

      // Resume audio context if suspended (required by some browsers)
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      const source = this.audioContext.createMediaStreamSource(this.stream);
      
      // Create analyser for voice activity detection
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.8;
      
      // Create script processor for real-time audio processing
      this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
      
      this.processor.onaudioprocess = (event) => {
        if (this.isRecording) {
          const audioData = event.inputBuffer.getChannelData(0);
          this.processAudioData(audioData);
        }
      };

      // Connect audio nodes
      source.connect(this.analyser);
      source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);

      console.log('✅ Advanced audio recorder initialized');
      return true;
    } catch (error) {
      console.error('❌ Error initializing audio recorder:', error);
      throw error;
    }
  }

  processAudioData(audioData) {
    // Voice Activity Detection
    if (this.vadEnabled) {
      const volume = this.calculateVolume(audioData);
      const hasVoice = volume > this.silenceThreshold;
      
      if (hasVoice) {
        this.lastSoundTime = Date.now();
      }
      
      // Skip sending if silence detected for too long
      if (this.silenceDetection && 
          Date.now() - this.lastSoundTime > this.silenceTimeout) {
        return;
      }
    }

    this.sendAudioChunk(audioData);
  }

  calculateVolume(audioData) {
    let sum = 0;
    for (let i = 0; i < audioData.length; i++) {
      sum += audioData[i] * audioData[i];
    }
    return Math.sqrt(sum / audioData.length);
  }

  startRecording() {
    if (!this.socket || !this.sessionId) {
      throw new Error('Socket or session not initialized');
    }

    this.isRecording = true;
    this.recordingStartTime = Date.now();
    this.lastSoundTime = Date.now();
    this.audioChunks = [];
    
    // Notify server about recording start
    this.socket.emit('start_recording', {
      session_id: this.sessionId,
      persona: this.persona,
      sample_rate: this.audioContext.sampleRate,
      channels: 1,
      format: 'pcm_f32le'
    });

    console.log('🎤 Recording started');
  }

  stopRecording() {
    this.isRecording = false;
    
    // Send final audio chunk
    this.sendFinalAudioChunk();
    
    // Notify server about recording stop
    if (this.socket && this.sessionId) {
      this.socket.emit('stop_recording', {
        session_id: this.sessionId,
        duration: Date.now() - this.recordingStartTime
      });
    }

    console.log('⏹️ Recording stopped');
  }

  sendAudioChunk(audioData) {
    if (!this.socket || !this.isRecording) return;

    try {
      // Convert Float32Array to Int16Array for better compression
      const int16Data = new Int16Array(audioData.length);
      for (let i = 0; i < audioData.length; i++) {
        // Clamp values to prevent distortion
        const sample = Math.max(-1, Math.min(1, audioData[i]));
        int16Data[i] = sample * 32767;
      }

      // Convert to base64 for transmission
      const audioBase64 = this.arrayBufferToBase64(int16Data.buffer);

      this.socket.emit('audio_chunk', {
        audio_data: audioBase64,
        session_id: this.sessionId,
        persona: this.persona,
        is_final: false,
        timestamp: Date.now(),
        chunk_size: audioData.length,
        sample_rate: this.audioContext.sampleRate
      });

      // Store chunk for potential replay
      this.audioChunks.push({
        data: audioBase64,
        timestamp: Date.now()
      });

    } catch (error) {
      console.error('Error sending audio chunk:', error);
    }
  }

  sendFinalAudioChunk() {
    if (!this.socket) return;

    this.socket.emit('audio_chunk', {
      audio_data: '',
      session_id: this.sessionId,
      persona: this.persona,
      is_final: true,
      total_chunks: this.audioChunks.length,
      total_duration: Date.now() - this.recordingStartTime
    });
  }

  // Get real-time audio levels for UI visualization
  getAudioLevel() {
    if (!this.analyser) return 0;

    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.analyser.getByteFrequencyData(dataArray);

    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
      sum += dataArray[i];
    }
    return sum / bufferLength / 255; // Normalize to 0-1
  }

  // Get frequency data for spectrum visualization
  getFrequencyData() {
    if (!this.analyser) return new Uint8Array(0);

    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.analyser.getByteFrequencyData(dataArray);
    return dataArray;
  }

  arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  // Test microphone and audio setup
  async testAudio() {
    try {
      const testStream = await navigator.mediaDevices.getUserMedia({
        audio: true
      });
      
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(testStream);
      const analyser = audioContext.createAnalyser();
      
      source.connect(analyser);
      
      // Test for 2 seconds
      return new Promise((resolve) => {
        let maxLevel = 0;
        const checkLevel = () => {
          const bufferLength = analyser.frequencyBinCount;
          const dataArray = new Uint8Array(bufferLength);
          analyser.getByteFrequencyData(dataArray);
          
          const level = dataArray.reduce((sum, value) => sum + value, 0) / bufferLength;
          maxLevel = Math.max(maxLevel, level);
        };
        
        const interval = setInterval(checkLevel, 100);
        
        setTimeout(() => {
          clearInterval(interval);
          testStream.getTracks().forEach(track => track.stop());
          audioContext.close();
          resolve({
            working: maxLevel > 10,
            maxLevel: maxLevel,
            message: maxLevel > 10 ? 'Microphone working' : 'No audio detected'
          });
        }, 2000);
      });
    } catch (error) {
      return {
        working: false,
        error: error.message,
        message: 'Microphone access denied or not available'
      };
    }
  }

  cleanup() {
    this.stopRecording();
    
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    
    if (this.analyser) {
      this.analyser.disconnect();
      this.analyser = null;
    }
    
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
      this.audioContext = null;
    }
    
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    console.log('🧹 Audio recorder cleaned up');
  }
}
```

---

## Speech-to-Text Integration

### Real-Time Transcription Handler

```javascript
class TranscriptionManager {
  constructor() {
    this.currentTranscription = '';
    this.finalTranscriptions = [];
    this.isTranscribing = false;
    this.confidenceThreshold = 0.7;
    this.onTranscriptionUpdate = null;
    this.onFinalTranscription = null;
  }

  handleTranscription(data) {
    const { text, is_final, confidence, session_id } = data;

    if (is_final) {
      // Final transcription
      this.finalTranscriptions.push({
        text: text,
        confidence: confidence || 1.0,
        timestamp: Date.now(),
        session_id: session_id
      });

      this.currentTranscription = '';
      
      if (this.onFinalTranscription) {
        this.onFinalTranscription({
          text: text,
          confidence: confidence,
          allTranscriptions: this.finalTranscriptions
        });
      }
    } else {
      // Interim transcription
      this.currentTranscription = text;
      
      if (this.onTranscriptionUpdate) {
        this.onTranscriptionUpdate({
          text: text,
          confidence: confidence,
          isFinal: false
        });
      }
    }
  }

  setTranscriptionHandlers(onUpdate, onFinal) {
    this.onTranscriptionUpdate = onUpdate;
    this.onFinalTranscription = onFinal;
  }

  getCurrentTranscription() {
    return this.currentTranscription;
  }

  getFinalTranscriptions() {
    return this.finalTranscriptions;
  }

  clearTranscriptions() {
    this.currentTranscription = '';
    this.finalTranscriptions = [];
  }
}

// Usage in React component
const useTranscription = () => {
  const [currentTranscription, setCurrentTranscription] = useState('');
  const [finalTranscriptions, setFinalTranscriptions] = useState([]);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const transcriptionManager = useRef(new TranscriptionManager());

  useEffect(() => {
    const manager = transcriptionManager.current;
    
    manager.setTranscriptionHandlers(
      // On interim transcription
      ({ text, confidence, isFinal }) => {
        setCurrentTranscription(text);
        setIsTranscribing(!isFinal && text.length > 0);
      },
      // On final transcription
      ({ text, confidence, allTranscriptions }) => {
        setFinalTranscriptions([...allTranscriptions]);
        setCurrentTranscription('');
        setIsTranscribing(false);
        
        // Add to chat if confidence is high enough
        if (confidence >= manager.confidenceThreshold) {
          addMessageToChat(text, 'user');
        }
      }
    );

    return () => {
      manager.clearTranscriptions();
    };
  }, []);

  const handleTranscriptionData = (data) => {
    transcriptionManager.current.handleTranscription(data);
  };

  return {
    currentTranscription,
    finalTranscriptions,
    isTranscribing,
    handleTranscriptionData,
    clearTranscriptions: () => transcriptionManager.current.clearTranscriptions()
  };
};
```

---

## Text-to-Speech Integration

### Audio Playback System

```javascript
class AudioPlaybackManager {
  constructor() {
    this.audioQueue = [];
    this.currentAudio = null;
    this.isPlaying = false;
    this.volume = 1.0;
    this.playbackRate = 1.0;
    this.onPlaybackStart = null;
    this.onPlaybackEnd = null;
    this.onPlaybackError = null;
  }

  async playAudio(audioUrl, options = {}) {
    const config = {
      volume: this.volume,
      playbackRate: this.playbackRate,
      autoplay: true,
      preload: true,
      ...options
    };

    try {
      // Create audio element
      const audio = new Audio();
      audio.src = audioUrl;
      audio.volume = config.volume;
      audio.playbackRate = config.playbackRate;
      audio.preload = config.preload ? 'auto' : 'none';

      // Set up event listeners
      audio.addEventListener('loadstart', () => {
        console.log('🔄 Loading audio...');
      });

      audio.addEventListener('canplay', () => {
        console.log('✅ Audio ready to play');
      });

      audio.addEventListener('play', () => {
        this.isPlaying = true;
        this.currentAudio = audio;
        if (this.onPlaybackStart) {
          this.onPlaybackStart({ audio, url: audioUrl });
        }
      });

      audio.addEventListener('ended', () => {
        this.isPlaying = false;
        this.currentAudio = null;
        if (this.onPlaybackEnd) {
          this.onPlaybackEnd({ audio, url: audioUrl });
        }
        this.playNext();
      });

      audio.addEventListener('error', (error) => {
        console.error('❌ Audio playback error:', error);
        this.isPlaying = false;
        this.currentAudio = null;
        if (this.onPlaybackError) {
          this.onPlaybackError({ error, audio, url: audioUrl });
        }
        this.playNext();
      });

      // Play audio
      if (config.autoplay) {
        await audio.play();
      }

      return audio;
    } catch (error) {
      console.error('Error playing audio:', error);
      throw error;
    }
  }

  queueAudio(audioUrl, options = {}) {
    this.audioQueue.push({ url: audioUrl, options });
    
    if (!this.isPlaying) {
      this.playNext();
    }
  }

  async playNext() {
    if (this.audioQueue.length === 0 || this.isPlaying) {
      return;
    }

    const { url, options } = this.audioQueue.shift();
    try {
      await this.playAudio(url, options);
    } catch (error) {
      console.error('Error playing queued audio:', error);
      this.playNext(); // Try next in queue
    }
  }

  stopCurrent() {
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.isPlaying = false;
      this.currentAudio = null;
    }
  }

  clearQueue() {
    this.audioQueue = [];
  }

  setVolume(volume) {
    this.volume = Math.max(0, Math.min(1, volume));
    if (this.currentAudio) {
      this.currentAudio.volume = this.volume;
    }
  }

  setPlaybackRate(rate) {
    this.playbackRate = Math.max(0.25, Math.min(4, rate));
    if (this.currentAudio) {
      this.currentAudio.playbackRate = this.playbackRate;
    }
  }

  setEventHandlers(onStart, onEnd, onError) {
    this.onPlaybackStart = onStart;
    this.onPlaybackEnd = onEnd;
    this.onPlaybackError = onError;
  }

  getStatus() {
    return {
      isPlaying: this.isPlaying,
      queueLength: this.audioQueue.length,
      currentAudio: this.currentAudio,
      volume: this.volume,
      playbackRate: this.playbackRate
    };
  }
}

// Usage in React component
const useAudioPlayback = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [queueLength, setQueueLength] = useState(0);
  const [volume, setVolumeState] = useState(1.0);
  const audioManager = useRef(new AudioPlaybackManager());

  useEffect(() => {
    const manager = audioManager.current;
    
    manager.setEventHandlers(
      // On playback start
      ({ audio, url }) => {
        setIsPlaying(true);
        console.log('🔊 Playing audio:', url);
      },
      // On playback end
      ({ audio, url }) => {
        setIsPlaying(false);
        setQueueLength(manager.audioQueue.length);
        console.log('⏹️ Audio finished:', url);
      },
      // On playback error
      ({ error, audio, url }) => {
        setIsPlaying(false);
        setQueueLength(manager.audioQueue.length);
        console.error('Audio error:', error);
      }
    );
  }, []);

  const playAudio = async (audioUrl, options = {}) => {
    try {
      await audioManager.current.playAudio(audioUrl, options);
    } catch (error) {
      console.error('Failed to play audio:', error);
    }
  };

  const queueAudio = (audioUrl, options = {}) => {
    audioManager.current.queueAudio(audioUrl, options);
    setQueueLength(audioManager.current.audioQueue.length);
  };

  const stopAudio = () => {
    audioManager.current.stopCurrent();
    setIsPlaying(false);
  };

  const clearQueue = () => {
    audioManager.current.clearQueue();
    setQueueLength(0);
  };

  const setVolume = (newVolume) => {
    audioManager.current.setVolume(newVolume);
    setVolumeState(newVolume);
  };

  const setPlaybackRate = (rate) => {
    audioManager.current.setPlaybackRate(rate);
  };

  return {
    isPlaying,
    queueLength,
    volume,
    playAudio,
    queueAudio,
    stopAudio,
    clearQueue,
    setVolume,
    setPlaybackRate
  };
};
```

---

## Session Management

### Complete Session Handler

```javascript
class SessionManager {
  constructor() {
    this.currentSession = null;
    this.sessionHistory = [];
    this.isSessionActive = false;
    this.sessionStartTime = null;
    this.sessionData = {
      messages: [],
      transcriptions: [],
      audioFiles: [],
      analysis: null
    };
  }

  async createSession(founderName, companyName, persona, socketService) {
    try {
      const sessionId = this.generateSessionId();
      
      // Initialize session data
      this.currentSession = {
        id: sessionId,
        founderName: founderName,
        companyName: companyName,
        persona: persona,
        startTime: new Date(),
        status: 'active'
      };

      this.sessionStartTime = Date.now();
      this.isSessionActive = true;
      this.sessionData = {
        messages: [],
        transcriptions: [],
        audioFiles: [],
        analysis: null
      };

      // Join session via socket
      socketService.joinSession(sessionId);

      // Send initial message to start the session
      socketService.sendMessage(
        `Hello, I'm ${founderName} from ${companyName}. I'd like to practice my pitch.`,
        persona,
        sessionId,
        'workflow'
      );

      console.log('✅ Session created:', sessionId);
      return sessionId;
    } catch (error) {
      console.error('❌ Error creating session:', error);
      throw error;
    }
  }

  async endSession(socketService, dbService) {
    if (!this.currentSession) {
      throw new Error('No active session to end');
    }

    try {
      const sessionId = this.currentSession.id;
      
      // Leave session via socket
      socketService.leaveSession(sessionId);

      // End session via API and get comprehensive analysis
      const response = await fetch(`/api/pitch/end/${sessionId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ reason: 'user_ended' })
      });

      if (!response.ok) {
        throw new Error('Failed to end session');
      }

      const data = await response.json();
      
      // The analysis is included in the response
      const analysis = data.analysis;
      
      // Update session data
      this.currentSession.endTime = new Date();
      this.currentSession.duration = Date.now() - this.sessionStartTime;
      this.currentSession.status = 'completed';
      this.sessionData.analysis = analysis;

      // Add to history
      this.sessionHistory.push({
        ...this.currentSession,
        data: { ...this.sessionData }
      });

      // Reset current session
      this.currentSession = null;
      this.isSessionActive = false;
      this.sessionStartTime = null;

      console.log('✅ Session ended:', sessionId);
      return {
        success: true,
        analysis: analysis,
        sessionId: sessionId
      };
    } catch (error) {
      console.error('❌ Error ending session:', error);
      throw error;
    }
  }

  addMessage(message, type, metadata = {}) {
    if (!this.isSessionActive) return;

    const messageData = {
      id: this.generateMessageId(),
      content: message,
      type: type, // 'user', 'ai', 'system'
      timestamp: new Date(),
      sessionId: this.currentSession?.id,
      ...metadata
    };

    this.sessionData.messages.push(messageData);
    return messageData;
  }

  addTranscription(text, isFinal, confidence = 1.0) {
    if (!this.isSessionActive) return;

    const transcriptionData = {
      id: this.generateTranscriptionId(),
      text: text,
      isFinal: isFinal,
      confidence: confidence,
      timestamp: new Date(),
      sessionId: this.currentSession?.id
    };

    if (isFinal) {
      this.sessionData.transcriptions.push(transcriptionData);
    }

    return transcriptionData;
  }

  addAudioFile(audioUrl, type = 'ai_response') {
    if (!this.isSessionActive) return;

    const audioData = {
      id: this.generateAudioId(),
      url: audioUrl,
      type: type, // 'ai_response', 'user_recording'
      timestamp: new Date(),
      sessionId: this.currentSession?.id
    };

    this.sessionData.audioFiles.push(audioData);
    return audioData;
  }

  async getSessionAnalytics(sessionId) {
    try {
      const response = await fetch(`/api/pitch/analytics/${sessionId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch analytics');
      }
      const data = await response.json();
      return data.analytics;
    } catch (error) {
      console.error('Error fetching analytics:', error);
      return null;
    }
  }

  async getSessionAnalysis(sessionId) {
    try {
      const response = await fetch(`/api/pitch/analysis/${sessionId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch analysis');
      }
      const data = await response.json();
      return data.analysis;
    } catch (error) {
      console.error('Error fetching analysis:', error);
      return null;
    }
  }

  async getSessionReport(sessionId) {
    try {
      const response = await fetch(`/api/pitch/report/${sessionId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch report');
      }
      const data = await response.json();
      return data.report;
    } catch (error) {
      console.error('Error fetching report:', error);
      return null;
    }
  }

  getCurrentSession() {
    return this.currentSession;
  }

  getSessionData() {
    return this.sessionData;
  }

  getSessionHistory() {
    return this.sessionHistory;
  }

  isActive() {
    return this.isSessionActive;
  }

  getSessionDuration() {
    if (!this.sessionStartTime) return 0;
    return Date.now() - this.sessionStartTime;
  }

  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  generateMessageId() {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  generateTranscriptionId() {
    return `trans_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  generateAudioId() {
    return `audio_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

// React hook for session management
const useSessionManager = () => {
  const [currentSession, setCurrentSession] = useState(null);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [sessionData, setSessionData] = useState(null);
  const [sessionDuration, setSessionDuration] = useState(0);
  const sessionManager = useRef(new SessionManager());

  // Update duration every second
  useEffect(() => {
    let interval;
    if (isSessionActive) {
      interval = setInterval(() => {
        setSessionDuration(sessionManager.current.getSessionDuration());
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isSessionActive]);

  const createSession = async (founderName, companyName, persona, socketService) => {
    try {
      const sessionId = await sessionManager.current.createSession(
        founderName, companyName, persona, socketService
      );
      setCurrentSession(sessionManager.current.getCurrentSession());
      setIsSessionActive(true);
      setSessionData(sessionManager.current.getSessionData());
      return sessionId;
    } catch (error) {
      console.error('Failed to create session:', error);
      throw error;
    }
  };

  const endSession = async (socketService, dbService) => {
    try {
      const analytics = await sessionManager.current.endSession(socketService, dbService);
      setCurrentSession(null);
      setIsSessionActive(false);
      setSessionData(null);
      setSessionDuration(0);
      return analytics;
    } catch (error) {
      console.error('Failed to end session:', error);
      throw error;
    }
  };

  const addMessage = (message, type, metadata = {}) => {
    const messageData = sessionManager.current.addMessage(message, type, metadata);
    setSessionData({ ...sessionManager.current.getSessionData() });
    return messageData;
  };

  const addTranscription = (text, isFinal, confidence) => {
    const transcriptionData = sessionManager.current.addTranscription(text, isFinal, confidence);
    setSessionData({ ...sessionManager.current.getSessionData() });
    return transcriptionData;
  };

  const addAudioFile = (audioUrl, type) => {
    const audioData = sessionManager.current.addAudioFile(audioUrl, type);
    setSessionData({ ...sessionManager.current.getSessionData() });
    return audioData;
  };

  return {
    currentSession,
    isSessionActive,
    sessionData,
    sessionDuration,
    createSession,
    endSession,
    addMessage,
    addTranscription,
    addAudioFile,
    getSessionHistory: () => sessionManager.current.getSessionHistory(),
    getSessionAnalytics: (sessionId) => sessionManager.current.getSessionAnalytics(sessionId),
    getSessionAnalysis: (sessionId) => sessionManager.current.getSessionAnalysis(sessionId),
    getSessionReport: (sessionId) => sessionManager.current.getSessionReport(sessionId)
  };
};
```

---

## Session Ending & Analysis

### 🎯 Complete Guide to Ending Sessions and Getting Analysis

This section provides a comprehensive guide on how to properly end pitch sessions, manage WebSocket connections, and retrieve detailed analysis reports.

#### Understanding the Session Ending Process

When ending a pitch session, there are **two separate operations** you need to handle:

1. **WebSocket Connection Management** - Cleanly disconnect from real-time communication
2. **Session Analysis Generation** - End the session and get comprehensive analysis via REST API

#### 🔄 The Complete Session Ending Workflow

```javascript
// Complete session ending workflow
const endPitchSession = async (sessionId, reason = 'user_ended') => {
  try {
    console.log('🔄 Starting session ending process...');
    
    // Step 1: Disconnect WebSocket (stops real-time communication)
    if (socket && socket.connected) {
      socket.disconnect();
      console.log('✅ WebSocket disconnected');
    }
    
    // Step 2: End session and get comprehensive analysis via REST API
    const response = await fetch(`https://ai-mock-pitching-427457295403.europe-west1.run.app/api/pitch/end/${sessionId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ reason: reason })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to end session: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('✅ Session ended successfully');
    
    // Step 3: Process the comprehensive analysis
    if (data.success && data.analysis) {
      console.log('📊 Analysis received:', data.analysis);
      
      // The analysis contains:
      // - Overall performance metrics
      // - Detailed feedback and insights
      // - Stage-by-stage breakdown
      // - Recommendations for improvement
      // - Session statistics and duration
      
      return {
        success: true,
        analysis: data.analysis,
        message: data.message
      };
    }
    
    throw new Error('No analysis data received');
    
  } catch (error) {
    console.error('❌ Error ending session:', error);
    throw error;
  }
};
```

#### 📡 WebSocket Session Management

```javascript
// WebSocket session management
class WebSocketSessionManager {
  constructor(socketService) {
    this.socketService = socketService;
    this.currentSessionId = null;
    this.isInSession = false;
  }
  
  // Join a session for real-time communication
  joinSession(sessionId) {
    try {
      if (!this.socketService.isConnected) {
        throw new Error('Socket not connected');
      }
      
      this.socketService.joinSession(sessionId);
      this.currentSessionId = sessionId;
      this.isInSession = true;
      
      console.log(`✅ Joined WebSocket session: ${sessionId}`);
    } catch (error) {
      console.error('❌ Failed to join session:', error);
      throw error;
    }
  }
  
  // Disconnect from current session (stops real-time communication)
  leaveSession() {
    try {
      if (this.socketService.socket && this.socketService.socket.connected) {
        this.socketService.socket.disconnect();
        console.log(`✅ WebSocket disconnected from session: ${this.currentSessionId}`);
      }
      
      this.currentSessionId = null;
      this.isInSession = false;
    } catch (error) {
      console.error('❌ Failed to disconnect from session:', error);
    }
  }
  
  // Clean disconnect (call this when component unmounts)
  disconnect() {
    this.leaveSession();
    this.socketService.disconnect();
    console.log('✅ WebSocket disconnected');
  }
}
```

#### 🎯 The `/api/pitch/end/{session_id}` Endpoint

**Purpose**: This endpoint is the **primary way to end a pitch session and get comprehensive analysis**.

**What it does**:
- ✅ Formally ends the pitch session
- ✅ Generates detailed performance analysis
- ✅ Saves analysis to database
- ✅ Updates session status
- ✅ Returns complete analysis report

**What it does NOT do**:
- ❌ Does NOT handle WebSocket disconnections (that's separate)
- ❌ Does NOT just return success/failure (it returns full analysis)

#### 📊 Request & Response Details

**Request**:
```javascript
POST /api/pitch/end/{session_id}
Content-Type: application/json

{
  "reason": "user_ended"  // Optional: "user_ended", "completed", "timeout", etc.
}
```

**Response**:
```javascript
{
  "success": true,
  "message": "Session ended successfully",
  "analysis": {
    "session_id": "session_123...",
    "session_duration_minutes": 15.5,
    "overall_score": 8.2,
    "performance_metrics": {
      "clarity_score": 8.5,
      "confidence_score": 7.8,
      "engagement_score": 8.0,
      "structure_score": 8.5
    },
    "detailed_feedback": {
      "strengths": ["Clear value proposition", "Good market analysis"],
      "areas_for_improvement": ["Financial projections need more detail"],
      "specific_recommendations": ["Practice the closing more"]
    },
    "stage_breakdown": {
      "introduction": { "score": 8.0, "feedback": "..." },
      "problem_statement": { "score": 8.5, "feedback": "..." },
      "solution": { "score": 7.5, "feedback": "..." }
      // ... more stages
    },
    "conversation_summary": "...",
    "key_insights": ["...", "..."],
    "next_steps": ["...", "..."]
  }
}
```

#### 🔧 Complete Implementation Examples

**React Hook Implementation**:
```javascript
const usePitchSession = () => {
  const [sessionId, setSessionId] = useState(null);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const socketService = useRef(new SocketService());
  const wsManager = useRef(null);
  
  // Initialize WebSocket session manager
  useEffect(() => {
    wsManager.current = new WebSocketSessionManager(socketService.current);
    
    return () => {
      // Cleanup on unmount
      wsManager.current?.disconnect();
    };
  }, []);
  
  const startSession = async (founderName, companyName, persona) => {
    try {
      setIsLoading(true);
      
      // Connect to WebSocket
      socketService.current.connect();
      
      // Generate session ID
      const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Join WebSocket session
      wsManager.current.joinSession(newSessionId);
      
      setSessionId(newSessionId);
      setIsSessionActive(true);
      
      console.log('✅ Session started:', newSessionId);
      
    } catch (error) {
      console.error('❌ Failed to start session:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };
  
  const endSession = async (reason = 'user_ended') => {
    if (!sessionId) return null;
    
    try {
      setIsLoading(true);
      
      // Step 1: Leave WebSocket session
      wsManager.current.leaveSession();
      
      // Step 2: End session and get analysis
      const result = await endPitchSession(sessionId, reason);
      
      // Step 3: Update state
      setIsSessionActive(false);
      setAnalysis(result.analysis);
      
      console.log('✅ Session ended with analysis');
      return result;
      
    } catch (error) {
      console.error('❌ Failed to end session:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };
  
  return {
    sessionId,
    isSessionActive,
    analysis,
    isLoading,
    startSession,
    endSession,
    socketService: socketService.current
  };
};
```

**Vue.js Composition API Implementation**:
```javascript
import { ref, onUnmounted } from 'vue';

export function usePitchSession() {
  const sessionId = ref(null);
  const isSessionActive = ref(false);
  const analysis = ref(null);
  const isLoading = ref(false);
  
  let socketService = null;
  let wsManager = null;
  
  const initializeServices = () => {
    socketService = new SocketService();
    wsManager = new WebSocketSessionManager(socketService);
  };
  
  const startSession = async (founderName, companyName, persona) => {
    try {
      isLoading.value = true;
      
      if (!socketService) initializeServices();
      
      socketService.connect();
      
      const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      wsManager.joinSession(newSessionId);
      
      sessionId.value = newSessionId;
      isSessionActive.value = true;
      
    } catch (error) {
      console.error('Failed to start session:', error);
      throw error;
    } finally {
      isLoading.value = false;
    }
  };
  
  const endSession = async (reason = 'user_ended') => {
    if (!sessionId.value) return null;
    
    try {
      isLoading.value = true;
      
      wsManager.leaveSession();
      const result = await endPitchSession(sessionId.value, reason);
      
      isSessionActive.value = false;
      analysis.value = result.analysis;
      
      return result;
    } catch (error) {
      console.error('Failed to end session:', error);
      throw error;
    } finally {
      isLoading.value = false;
    }
  };
  
  // Cleanup on component unmount
  onUnmounted(() => {
    if (wsManager) {
      wsManager.disconnect();
    }
  });
  
  return {
    sessionId,
    isSessionActive,
    analysis,
    isLoading,
    startSession,
    endSession
  };
}
```

#### ⚠️ Important Best Practices

1. **Always handle both WebSocket and REST operations**:
   ```javascript
   // ✅ Correct approach
   wsManager.leaveSession();           // Stop real-time communication
   const analysis = await endPitchSession(sessionId);  // Get analysis
   
   // ❌ Wrong approach
   socketService.disconnect();         // Only disconnects WebSocket
   // Missing: No analysis retrieval
   ```

2. **Handle errors gracefully**:
   ```javascript
   try {
     wsManager.leaveSession();
   } catch (wsError) {
     console.warn('WebSocket leave failed:', wsError);
     // Continue with REST API call even if WebSocket fails
   }
   
   try {
     const analysis = await endPitchSession(sessionId);
     return analysis;
   } catch (apiError) {
     console.error('Failed to get analysis:', apiError);
     throw apiError;
   }
   ```

3. **Clean up resources**:
   ```javascript
   // In React useEffect cleanup
   useEffect(() => {
     return () => {
       wsManager.current?.disconnect();
     };
   }, []);
   
   // In Vue onUnmounted
   onUnmounted(() => {
     wsManager?.disconnect();
   });
   ```

4. **Use the analysis data effectively**:
   ```javascript
   const result = await endSession();
   if (result?.analysis) {
     // Display overall score
     setOverallScore(result.analysis.overall_score);
     
     // Show detailed feedback
     setFeedback(result.analysis.detailed_feedback);
     
     // Display stage-by-stage breakdown
     setStageBreakdown(result.analysis.stage_breakdown);
     
     // Show recommendations
     setRecommendations(result.analysis.detailed_feedback.specific_recommendations);
   }
   ```

---

## Analytics & Reporting

### Analytics Dashboard Integration

```javascript
class AnalyticsManager {
  constructor() {
    this.cache = new Map();
    this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
  }

  async getSessionAnalytics(sessionId, useCache = true) {
    const cacheKey = `analytics_${sessionId}`;
    
    if (useCache && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey);
      if (Date.now() - cached.timestamp < this.cacheTimeout) {
        return cached.data;
      }
    }

    try {
      const response = await fetch(`/api/pitch/analytics/${sessionId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Cache the result
      this.cache.set(cacheKey, {
        data: data.analytics,
        timestamp: Date.now()
      });
      
      return data.analytics;
    } catch (error) {
      console.error('Error fetching analytics:', error);
      throw error;
    }
  }

  async getDetailedAnalysis(sessionId, useCache = true) {
    const cacheKey = `analysis_${sessionId}`;
    
    if (useCache && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey);
      if (Date.now() - cached.timestamp < this.cacheTimeout) {
        return cached.data;
      }
    }

    try {
      const response = await fetch(`/api/pitch/analysis/${sessionId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Cache the result
      this.cache.set(cacheKey, {
        data: data.analysis,
        timestamp: Date.now()
      });
      
      return data.analysis;
    } catch (error) {
      console.error('Error fetching analysis:', error);
      throw error;
    }
  }

  async getFormattedReport(sessionId, format = 'json') {
    try {
      const response = await fetch(`/api/pitch/report/${sessionId}?format=${format}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      if (format === 'pdf') {
        return await response.blob();
      } else {
        const data = await response.json();
        return data.report;
      }
    } catch (error) {
      console.error('Error fetching report:', error);
      throw error;
    }
  }

  async getDashboardStats() {
    try {
      const response = await fetch('/api/stats');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.stats;
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
      throw error;
    }
  }

  processAnalyticsForChart(analytics) {
    if (!analytics || !analytics.stage_scores) {
      return null;
    }

    // Convert stage scores to chart data
    const chartData = Object.entries(analytics.stage_scores).map(([stage, score]) => ({
      stage: this.formatStageName(stage),
      score: score,
      color: this.getScoreColor(score)
    }));

    return {
      chartData,
      overallScore: analytics.overall_score,
      confidenceLevel: analytics.confidence_level,
      pitchReadiness: analytics.pitch_readiness
    };
  }

  formatStageName(stage) {
    return stage.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  }

  getScoreColor(score) {
    if (score >= 80) return '#4CAF50'; // Green
    if (score >= 60) return '#FF9800'; // Orange
    if (score >= 40) return '#FFC107'; // Yellow
    return '#F44336'; // Red
  }

  clearCache() {
    this.cache.clear();
  }
}

// React hook for analytics
const useAnalytics = () => {
  const [analytics, setAnalytics] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [dashboardStats, setDashboardStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const analyticsManager = useRef(new AnalyticsManager());

  const loadSessionAnalytics = async (sessionId, useCache = true) => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await analyticsManager.current.getSessionAnalytics(sessionId, useCache);
      setAnalytics(data);
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const loadDetailedAnalysis = async (sessionId, useCache = true) => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await analyticsManager.current.getDetailedAnalysis(sessionId, useCache);
      setAnalysis(data);
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const loadDashboardStats = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await analyticsManager.current.getDashboardStats();
      setDashboardStats(data);
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async (sessionId, format = 'pdf') => {
    try {
      const report = await analyticsManager.current.getFormattedReport(sessionId, format);
      
      if (format === 'pdf') {
        // Create download link for PDF
        const url = window.URL.createObjectURL(report);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pitch-report-${sessionId}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
      
      return report;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  const getChartData = (analyticsData = analytics) => {
    if (!analyticsData) return null;
    return analyticsManager.current.processAnalyticsForChart(analyticsData);
  };

  return {
    analytics,
    analysis,
    dashboardStats,
    loading,
    error,
    loadSessionAnalytics,
    loadDetailedAnalysis,
    loadDashboardStats,
    downloadReport,
    getChartData,
    clearCache: () => analyticsManager.current.clearCache()
  };
};
```

---

## React Complete Example

### Main Application Component

```jsx
import React, { useState, useEffect, useRef } from 'react';
import { 
  Container, 
  Paper, 
  Typography, 
  Button, 
  TextField, 
  Select, 
  MenuItem, 
  FormControl, 
  InputLabel,
  Box,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Alert
} from '@mui/material';
import { 
  Mic, 
  MicOff, 
  Send, 
  Stop, 
  PlayArrow, 
  Pause,
  Download,
  Analytics,
  Videocam,
  VideocamOff
} from '@mui/icons-material';

// Import our custom hooks and services
import { useSocket } from './hooks/useSocket';
import { useSessionManager } from './hooks/useSessionManager';
import { useAudioRecording } from './hooks/useAudioRecording';
import { useAudioPlayback } from './hooks/useAudioPlayback';
import { useTranscription } from './hooks/useTranscription';
import { useAnalytics } from './hooks/useAnalytics';
import { useVideoAnalysis } from './hooks/useVideoAnalysis'; // 🆕 Video analysis hook
import { DatabaseService } from './services/DatabaseService';

// Components
import ChatInterface from './components/ChatInterface';
import AudioVisualizer from './components/AudioVisualizer';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import SessionHistory from './components/SessionHistory';

const MokePitchApp = () => {
  // State management
  const [personas, setPersonas] = useState({});
  const [selectedPersona, setSelectedPersona] = useState('skeptical');
  const [founderName, setFounderName] = useState('');
  const [companyName, setCompanyName] = useState('');
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('pitch'); // pitch, analytics, history

  // Custom hooks
  const { isConnected, connectionError, socketService } = useSocket();
  const {
    currentSession,
    isSessionActive,
    sessionData,
    sessionDuration,
    createSession,
    endSession,
    addMessage,
    addTranscription,
    addAudioFile
  } = useSessionManager();

  const {
    isRecording,
    audioLevel,
    startRecording,
    stopRecording,
    testMicrophone
  } = useAudioRecording(socketService, currentSession?.id, selectedPersona);

  const {
    isPlaying,
    queueLength,
    volume,
    playAudio,
    queueAudio,
    stopAudio,
    setVolume
  } = useAudioPlayback();

  const {
    currentTranscription,
    finalTranscriptions,
    isTranscribing,
    handleTranscriptionData,
    clearTranscriptions
  } = useTranscription();

  const {
    analytics,
    analysis,
    dashboardStats,
    loading: analyticsLoading,
    loadSessionAnalytics,
    loadDetailedAnalysis,
    downloadReport,
    getChartData
  } = useAnalytics();

  // 🆕 Video Analysis Hook
  const {
    isVideoReady,
    isAnalyzing: isVideoAnalyzing,
    currentMetrics: videoMetrics,
    videoInsights,
    error: videoError,
    videoElementRef,
    initializeVideo,
    startAnalysis: startVideoAnalysis,
    stopAnalysis: stopVideoAnalysis
  } = useVideoAnalysis(socketService);

  // Services
  const dbService = useRef(new DatabaseService());

  // Load personas and initialize video on component mount
  useEffect(() => {
    loadPersonas();
    loadDashboardStats();
    initializeVideo(); // 🆕 Initialize video analysis
  }, []);

  // Socket event handlers
  useEffect(() => {
    if (!socketService) return;

    // AI response handler
    const handleAIResponse = (data) => {
      const { message, audio_url, stage, complete, session_id } = data;
      
      if (message) {
        addMessage(message, 'ai', { stage, complete });
      }
      
      if (audio_url) {
        addAudioFile(audio_url, 'ai_response');
        queueAudio(audio_url);
      }
      
      if (complete) {
        handleSessionComplete(session_id);
      }
      
      setIsLoading(false);
    };

    // Transcription handler
    const handleTranscription = (data) => {
      handleTranscriptionData(data);
      
      if (data.is_final) {
        addTranscription(data.text, true, data.confidence);
      }
    };

    // Session started handler
    const handleSessionStarted = (data) => {
      console.log('Session started:', data.session_id);
    };

    // Error handler
    const handleError = (error) => {
      setError(error.message);
      setIsLoading(false);
    };

    // Register event handlers
    socketService.on('ai_response', handleAIResponse);
    socketService.on('transcription', handleTranscription);
    socketService.on('session_started', handleSessionStarted);
    socketService.on('error', handleError);

    // Cleanup
    return () => {
      socketService.off('ai_response', handleAIResponse);
      socketService.off('transcription', handleTranscription);
      socketService.off('session_started', handleSessionStarted);
      socketService.off('error', handleError);
    };
  }, [socketService, addMessage, addTranscription, addAudioFile, queueAudio]);

  // Load personas from API
  const loadPersonas = async () => {
    try {
      const response = await fetch('/api/personas');
      if (!response.ok) throw new Error('Failed to load personas');
      const data = await response.json();
      setPersonas(data.personas);
    } catch (err) {
      setError('Failed to load personas: ' + err.message);
    }
  };

  // Load dashboard statistics
  const loadDashboardStats = async () => {
    try {
      await loadDashboardStats();
    } catch (err) {
      console.error('Failed to load dashboard stats:', err);
    }
  };

  // Start new pitch session
  const handleStartSession = async () => {
    if (!founderName.trim() || !companyName.trim()) {
      setError('Please enter founder name and company name');
      return;
    }

    if (!isConnected) {
      setError('Not connected to server. Please wait...');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      
      const sessionId = await createSession(
        founderName.trim(),
        companyName.trim(),
        selectedPersona,
        socketService
      );
      
      // 🆕 Start video analysis if available
      if (isVideoReady) {
        await startVideoAnalysis(sessionId);
      }
      
      console.log('Session started:', sessionId);
    } catch (err) {
      setError('Failed to start session: ' + err.message);
      setIsLoading(false);
    }
  };

  // End current session
  const handleEndSession = async () => {
    if (!isSessionActive) return;

    try {
      setIsLoading(true);
      
      // 🆕 Stop video analysis if active
      if (isVideoAnalyzing && currentSession?.id) {
        await stopVideoAnalysis(currentSession.id);
      }
      
      const analytics = await endSession(socketService, dbService.current);
      
      // Load analytics for the completed session
      if (analytics) {
        setActiveTab('analytics');
      }
    } catch (err) {
      setError('Failed to end session: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle session completion
  const handleSessionComplete = async (sessionId) => {
    try {
      // Load final analytics
      await loadSessionAnalytics(sessionId);
      await loadDetailedAnalysis(sessionId);
      
      // Switch to analytics tab
      setActiveTab('analytics');
    } catch (err) {
      console.error('Failed to load session analytics:', err);
    }
  };

  // Send text message
  const handleSendMessage = () => {
    if (!currentMessage.trim() || !isSessionActive) return;

    try {
      setIsLoading(true);
      
      // Add user message to chat
      addMessage(currentMessage.trim(), 'user');
      
      // Send to AI
      socketService.sendMessage(
        currentMessage.trim(),
        selectedPersona,
        currentSession.id,
        'workflow'
      );
      
      setCurrentMessage('');
    } catch (err) {
      setError('Failed to send message: ' + err.message);
      setIsLoading(false);
    }
  };

  // Handle voice recording
  const handleVoiceToggle = async () => {
    if (!isSessionActive) {
      setError('Please start a session first');
      return;
    }

    try {
      if (isRecording) {
        stopRecording();
      } else {
        await startRecording();
      }
    } catch (err) {
      setError('Voice recording error: ' + err.message);
    }
  };

  // Format session duration
  const formatDuration = (ms) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}:${(minutes % 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
    }
    return `${minutes}:${(seconds % 60).toString().padStart(2, '0')}`;
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          🎯 AI Mock Investor Pitch
        </Typography>
        
        {/* Connection Status */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Chip 
            label={isConnected ? 'Connected' : 'Disconnected'} 
            color={isConnected ? 'success' : 'error'}
            size="small"
          />
          
          {isSessionActive && (
            <>
              <Chip 
                label={`Session Active: ${formatDuration(sessionDuration)}`}
                color="primary"
                size="small"
              />
              <Chip 
                label={`${currentSession?.founderName} - ${currentSession?.companyName}`}
                variant="outlined"
                size="small"
              />
            </>
          )}
        </Box>

        {/* Error Display */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Connection Error */}
        {connectionError && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Connection issue: {connectionError}
          </Alert>
        )}
      </Paper>

      {/* Tab Navigation */}
      <Paper elevation={1} sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', borderBottom: 1, borderColor: 'divider' }}>
          {['pitch', 'analytics', 'history'].map((tab) => (
            <Button
              key={tab}
              onClick={() => setActiveTab(tab)}
              variant={activeTab === tab ? 'contained' : 'text'}
              sx={{ m: 1 }}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </Button>
          ))}
        </Box>
      </Paper>

      {/* Main Content */}
      {activeTab === 'pitch' && (
        <Box sx={{ display: 'flex', gap: 3 }}>
          {/* Left Panel - Session Setup */}
          <Paper elevation={2} sx={{ p: 3, width: 350, height: 'fit-content' }}>
            <Typography variant="h6" gutterBottom>
              Session Setup
            </Typography>

            {!isSessionActive ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  label="Founder Name"
                  value={founderName}
                  onChange={(e) => setFounderName(e.target.value)}
                  fullWidth
                  required
                />
                
                <TextField
                  label="Company Name"
                  value={companyName}
                  onChange={(e) => setCompanyName(e.target.value)}
                  fullWidth
                  required
                />
                
                <FormControl fullWidth>
                  <InputLabel>Investor Persona</InputLabel>
                  <Select
                    value={selectedPersona}
                    onChange={(e) => setSelectedPersona(e.target.value)}
                    label="Investor Persona"
                  >
                    {Object.entries(personas).map(([key, persona]) => (
                      <MenuItem key={key} value={key}>
                        {persona.name} - {persona.description}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                
                <Button
                  variant="contained"
                  onClick={handleStartSession}
                  disabled={isLoading || !isConnected}
                  fullWidth
                  size="large"
                >
                  {isLoading ? 'Starting...' : 'Start Pitch Session'}
                </Button>
              </Box>
            ) : (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Typography variant="body1">
                  <strong>Session Active</strong>
                </Typography>
                <Typography variant="body2">
                  Founder: {currentSession?.founderName}
                </Typography>
                <Typography variant="body2">
                  Company: {currentSession?.companyName}
                </Typography>
                <Typography variant="body2">
                  Persona: {personas[selectedPersona]?.name}
                </Typography>
                <Typography variant="body2">
                  Duration: {formatDuration(sessionDuration)}
                </Typography>
                
                <Button
                  variant="outlined"
                  color="error"
                  onClick={handleEndSession}
                  disabled={isLoading}
                  fullWidth
                >
                  End Session
                </Button>
              </Box>
            )}

            {/* Audio Controls */}
            {isSessionActive && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Voice Controls
                </Typography>
                
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Button
                    variant={isRecording ? 'contained' : 'outlined'}
                    color={isRecording ? 'error' : 'primary'}
                    onClick={handleVoiceToggle}
                    startIcon={isRecording ? <MicOff /> : <Mic />}
                    fullWidth
                  >
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                  </Button>
                  
                  {/* Audio Level Indicator */}
                  {isRecording && (
                    <Box>
                      <Typography variant="caption">Audio Level</Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={audioLevel * 100} 
                        sx={{ height: 8, borderRadius: 4 }}
                      />
                    </Box>
                  )}
                  
                  {/* Volume Control */}
                  <Box>
                    <Typography variant="caption">Volume</Typography>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={volume}
                      onChange={(e) => setVolume(parseFloat(e.target.value))}
                      style={{ width: '100%' }}
                    />
                  </Box>
                </Box>
              </Box>
            )}

            {/* 🆕 Video Analysis Section */}
            {isSessionActive && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                  📹 Video Analysis
                </Typography>
                
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {/* Video Preview */}
                  <Box sx={{ position: 'relative' }}>
                    <video
                      ref={videoElementRef}
                      autoPlay
                      muted
                      playsInline
                      style={{
                        width: '100%',
                        maxHeight: '200px',
                        backgroundColor: '#000',
                        borderRadius: '8px'
                      }}
                    />
                    <Chip
                      label={isVideoReady ? (isVideoAnalyzing ? '🔴 Analyzing' : '✅ Ready') : '⏳ Loading'}
                      color={isVideoReady ? (isVideoAnalyzing ? 'error' : 'success') : 'default'}
                      size="small"
                      sx={{ position: 'absolute', top: 8, right: 8 }}
                    />
                  </Box>

                  {/* Video Controls */}
                  <Button
                    variant={isVideoAnalyzing ? 'contained' : 'outlined'}
                    color={isVideoAnalyzing ? 'error' : 'primary'}
                    onClick={() => isVideoAnalyzing ? 
                      stopVideoAnalysis(currentSession?.id) : 
                      startVideoAnalysis(currentSession?.id)
                    }
                    startIcon={isVideoAnalyzing ? <VideocamOff /> : <Videocam />}
                    disabled={!isVideoReady}
                    fullWidth
                  >
                    {isVideoAnalyzing ? 'Stop Video Analysis' : 'Start Video Analysis'}
                  </Button>

                  {/* Video Error */}
                  {videoError && (
                    <Alert severity="warning" size="small">
                      Video: {videoError}
                    </Alert>
                  )}

                  {/* Real-time Video Metrics */}
                  {videoMetrics && isVideoAnalyzing && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        📊 Live Analysis
                      </Typography>
                      
                      {/* Overall Score */}
                      <Box sx={{ mb: 1 }}>
                        <Typography variant="caption">Overall Performance</Typography>
                        <LinearProgress 
                          variant="determinate" 
                          value={videoMetrics.overall.overallScore * 100}
                          sx={{ height: 6, borderRadius: 3 }}
                          color="primary"
                        />
                        <Typography variant="caption" sx={{ fontSize: '0.7rem' }}>
                          {Math.round(videoMetrics.overall.overallScore * 100)}%
                        </Typography>
                      </Box>

                      {/* Gesture Score */}
                      <Box sx={{ mb: 1 }}>
                        <Typography variant="caption">Hand Gestures</Typography>
                        <LinearProgress 
                          variant="determinate" 
                          value={videoMetrics.overall.gestureScore * 100}
                          sx={{ height: 4, borderRadius: 2 }}
                          color="secondary"
                        />
                      </Box>

                      {/* Emotion Score */}
                      <Box sx={{ mb: 1 }}>
                        <Typography variant="caption">Facial Expression</Typography>
                        <LinearProgress 
                          variant="determinate" 
                          value={videoMetrics.overall.emotionScore * 100}
                          sx={{ height: 4, borderRadius: 2 }}
                          color="info"
                        />
                      </Box>

                      {/* Current Status */}
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 1 }}>
                        <Chip 
                          label={`😊 ${videoMetrics.emotions.dominant}`}
                          size="small"
                          variant="outlined"
                        />
                        <Chip 
                          label={`🤲 ${videoMetrics.handGestures.detected} hands`}
                          size="small"
                          variant="outlined"
                        />
                        <Chip 
                          label={`🧍 ${videoMetrics.pose.engagement}`}
                          size="small"
                          variant="outlined"
                        />
                      </Box>
                    </Box>
                  )}
                </Box>
              </Box>
            )}
          </Paper>

          {/* Right Panel - Chat Interface */}
          <Paper elevation={2} sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            {/* Chat Messages */}
            <Box sx={{ flex: 1, p: 2, maxHeight: '60vh', overflow: 'auto' }}>
              <ChatInterface 
                messages={sessionData?.messages || []}
                currentTranscription={currentTranscription}
                isTranscribing={isTranscribing}
                isLoading={isLoading}
              />
            </Box>

            {/* Message Input */}
            {isSessionActive && (
              <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    placeholder="Type your message..."
                    fullWidth
                    multiline
                    maxRows={3}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSendMessage();
                      }
                    }}
                  />
                  <Button
                    variant="contained"
                    onClick={handleSendMessage}
                    disabled={!currentMessage.trim() || isLoading}
                    sx={{ minWidth: 'auto', px: 2 }}
                  >
                    <Send />
                  </Button>
                </Box>
              </Box>
            )}
          </Paper>
        </Box>
      )}

      {/* Analytics Tab */}
      {activeTab === 'analytics' && (
        <AnalyticsDashboard
          analytics={analytics}
          analysis={analysis}
          dashboardStats={dashboardStats}
          loading={analyticsLoading}
          onDownloadReport={downloadReport}
          chartData={getChartData()}
        />
      )}

      {/* History Tab */}
      {activeTab === 'history' && (
        <SessionHistory
          dbService={dbService.current}
          onSessionSelect={(sessionId) => {
            loadSessionAnalytics(sessionId);
            setActiveTab('analytics');
          }}
        />
      )}
    </Container>
  );
};

export default MokePitchApp;
```

### Chat Interface Component

```jsx
import React from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Avatar, 
  Chip,
  CircularProgress
} from '@mui/material';
import { Person, SmartToy, Mic } from '@mui/icons-material';

const ChatInterface = ({ 
  messages, 
  currentTranscription, 
  isTranscribing, 
  isLoading 
}) => {
  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getMessageColor = (type) => {
    switch (type) {
      case 'user': return '#e3f2fd';
      case 'ai': return '#f3e5f5';
      case 'system': return '#fff3e0';
      default: return '#f5f5f5';
    }
  };

  const getMessageIcon = (type) => {
    switch (type) {
      case 'user': return <Person />;
      case 'ai': return <SmartToy />;
      case 'system': return <Mic />;
      default: return null;
    }
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Messages */}
      <Box sx={{ flex: 1, overflow: 'auto', pb: 2 }}>
        {messages.length === 0 ? (
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            height: '100%',
            color: 'text.secondary'
          }}>
            <Typography variant="body1">
              Start a session to begin your pitch practice
            </Typography>
          </Box>
        ) : (
          messages.map((message, index) => (
            <Box key={message.id || index} sx={{ mb: 2 }}>
              <Paper
                elevation={1}
                sx={{
                  p: 2,
                  backgroundColor: getMessageColor(message.type),
                  maxWidth: '80%',
                  ml: message.type === 'user' ? 'auto' : 0,
                  mr: message.type === 'ai' ? 'auto' : 0
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                  <Avatar sx={{ width: 32, height: 32 }}>
                    {getMessageIcon(message.type)}
                  </Avatar>
                  
                  <Box sx={{ flex: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        {message.type === 'user' ? 'You' : 'AI Investor'}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {formatTimestamp(message.timestamp)}
                      </Typography>
                      {message.stage && (
                        <Chip 
                          label={message.stage} 
                          size="small" 
                          variant="outlined"
                        />
                      )}
                    </Box>
                    
                    <Typography variant="body1">
                      {message.content}
                    </Typography>
                  </Box>
                </Box>
              </Paper>
            </Box>
          ))
        )}
      </Box>

      {/* Current Transcription */}
      {isTranscribing && currentTranscription && (
        <Paper 
          elevation={1} 
          sx={{ 
            p: 2, 
            backgroundColor: '#f0f0f0',
            border: '2px dashed #ccc'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CircularProgress size={16} />
            <Typography variant="body2" color="text.secondary">
              Transcribing: {currentTranscription}
            </Typography>
          </Box>
        </Paper>
      )}

      {/* Loading Indicator */}
      {isLoading && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 2 }}>
          <CircularProgress size={20} />
          <Typography variant="body2" color="text.secondary">
            AI is thinking...
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default ChatInterface;
```

---

## Error Handling

### Comprehensive Error Management

```javascript
class ErrorHandler {
  constructor() {
    this.errorLog = [];
    this.maxLogSize = 100;
    this.onError = null;
  }

  handleError(error, context = 'Unknown') {
    const errorInfo = {
      id: Date.now(),
      message: error.message || 'Unknown error',
      context: context,
      timestamp: new Date(),
      stack: error.stack,
      type: error.name || 'Error'
    };

    // Add to log
    this.errorLog.unshift(errorInfo);
    if (this.errorLog.length > this.maxLogSize) {
      this.errorLog.pop();
    }

    // Log to console
    console.error(`[${context}] ${error.message}`, error);

    // Call error handler if set
    if (this.onError) {
      this.onError(errorInfo);
    }

    return errorInfo;
  }

  handleSocketError(error) {
    return this.handleError(error, 'Socket');
  }

  handleAudioError(error) {
    return this.handleError(error, 'Audio');
  }

  handleAPIError(error, endpoint) {
    return this.handleError(error, `API: ${endpoint}`);
  }

  getErrorLog() {
    return this.errorLog;
  }

  clearErrorLog() {
    this.errorLog = [];
  }

  setErrorHandler(handler) {
    this.onError = handler;
  }
}

// React hook for error handling
const useErrorHandler = () => {
  const [errors, setErrors] = useState([]);
  const errorHandler = useRef(new ErrorHandler());

  useEffect(() => {
    errorHandler.current.setErrorHandler((errorInfo) => {
      setErrors(prev => [errorInfo, ...prev.slice(0, 9)]); // Keep last 10 errors
    });
  }, []);

  const handleError = (error, context) => {
    return errorHandler.current.handleError(error, context);
  };

  const clearErrors = () => {
    setErrors([]);
    errorHandler.current.clearErrorLog();
  };

  return {
    errors,
    handleError,
    clearErrors,
    handleSocketError: (error) => errorHandler.current.handleSocketError(error),
    handleAudioError: (error) => errorHandler.current.handleAudioError(error),
    handleAPIError: (error, endpoint) => errorHandler.current.handleAPIError(error, endpoint)
  };
};
```

---

## Performance Optimization

### Optimization Strategies

```javascript
// 1. Audio Chunk Optimization
class OptimizedAudioRecorder extends AdvancedAudioRecorder {
  constructor() {
    super();
    this.compressionEnabled = true;
    this.adaptiveBitrate = true;
    this.chunkBuffer = [];
    this.bufferSize = 4; // Buffer 4 chunks before sending
  }

  sendAudioChunk(audioData) {
    if (!this.socket || !this.isRecording) return;

    // Add to buffer
    this.chunkBuffer.push(audioData);

    // Send when buffer is full or on silence detection
    if (this.chunkBuffer.length >= this.bufferSize || this.shouldFlushBuffer()) {
      this.flushAudioBuffer();
    }
  }

  flushAudioBuffer() {
    if (this.chunkBuffer.length === 0) return;

    // Combine chunks
    const combinedLength = this.chunkBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
    const combinedData = new Float32Array(combinedLength);
    
    let offset = 0;
    for (const chunk of this.chunkBuffer) {
      combinedData.set(chunk, offset);
      offset += chunk.length;
    }

    // Compress if enabled
    const finalData = this.compressionEnabled ? 
      this.compressAudio(combinedData) : combinedData;

    // Send combined chunk
    super.sendAudioChunk(finalData);

    // Clear buffer
    this.chunkBuffer = [];
  }

  compressAudio(audioData) {
    // Simple compression: reduce sample rate for silence
    const volume = this.calculateVolume(audioData);
    if (volume < this.silenceThreshold) {
      // Downsample silent audio
      const downsampledData = new Float32Array(Math.floor(audioData.length / 2));
      for (let i = 0; i < downsampledData.length; i++) {
        downsampledData[i] = audioData[i * 2];
      }
      return downsampledData;
    }
    return audioData;
  }

  shouldFlushBuffer() {
    // Flush on silence detection or timeout
    return Date.now() - this.lastSoundTime > 1000; // 1 second timeout
  }
}

// 2. Message Virtualization for Large Chat History
const VirtualizedChatInterface = ({ messages, height = 400 }) => {
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 20 });
  const containerRef = useRef(null);
  const itemHeight = 80; // Approximate height per message

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const scrollTop = container.scrollTop;
      const start = Math.floor(scrollTop / itemHeight);
      const end = Math.min(start + Math.ceil(height / itemHeight) + 5, messages.length);
      
      setVisibleRange({ start, end });
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, [messages.length, height]);

  const visibleMessages = messages.slice(visibleRange.start, visibleRange.end);
  const totalHeight = messages.length * itemHeight;
  const offsetY = visibleRange.start * itemHeight;

  return (
    <div 
      ref={containerRef}
      style={{ height, overflow: 'auto' }}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        <div style={{ transform: `translateY(${offsetY}px)` }}>
          {visibleMessages.map((message, index) => (
            <MessageComponent 
              key={message.id} 
              message={message}
              style={{ height: itemHeight }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

// 3. Debounced API Calls
const useDebouncedAPI = (apiFunction, delay = 300) => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const timeoutRef = useRef(null);

  const debouncedCall = useCallback((...args) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(async () => {
      try {
        setLoading(true);
        setError(null);
        const result = await apiFunction(...args);
        setData(result);
      } catch (err) {
        setError(err);
      } finally {
        setLoading(false);
      }
    }, delay);
  }, [apiFunction, delay]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return { debouncedCall, loading, data, error };
};

// 4. Memory Management
const useMemoryOptimization = () => {
  const [memoryUsage, setMemoryUsage] = useState(0);

  useEffect(() => {
    const checkMemory = () => {
      if (performance.memory) {
        setMemoryUsage(performance.memory.usedJSHeapSize / 1024 / 1024); // MB
      }
    };

    const interval = setInterval(checkMemory, 5000);
    return () => clearInterval(interval);
  }, []);

  const cleanupMemory = useCallback(() => {
    // Force garbage collection if available
    if (window.gc) {
      window.gc();
    }
    
    // Clear caches
    if ('caches' in window) {
      caches.keys().then(names => {
        names.forEach(name => {
          if (name.includes('audio-cache')) {
            caches.delete(name);
          }
        });
      });
    }
  }, []);

  return { memoryUsage, cleanupMemory };
};
```

---

## Security Considerations

### Security Best Practices

```javascript
// 1. Input Sanitization
class InputSanitizer {
  static sanitizeText(input) {
    if (typeof input !== 'string') return '';
    
    return input
      .trim()
      .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '') // Remove scripts
      .replace(/javascript:/gi, '') // Remove javascript: URLs
      .replace(/on\w+\s*=/gi, '') // Remove event handlers
      .substring(0, 1000); // Limit length
  }

  static sanitizeSessionData(data) {
    return {
      founderName: this.sanitizeText(data.founderName),
      companyName: this.sanitizeText(data.companyName),
      persona: this.sanitizeText(data.persona)
    };
  }

  static validateAudioData(audioData) {
    // Validate audio data format and size
    if (!audioData || typeof audioData !== 'string') {
      throw new Error('Invalid audio data format');
    }
    
    // Check size limits (e.g., 1MB)
    if (audioData.length > 1024 * 1024) {
      throw new Error('Audio data too large');
    }
    
    // Validate base64 format
    try {
      atob(audioData);
    } catch (error) {
      throw new Error('Invalid base64 audio data');
    }
    
    return true;
  }
}

// 2. Rate Limiting
class RateLimiter {
  constructor(maxRequests = 10, windowMs = 60000) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.requests = new Map();
  }

  isAllowed(identifier = 'default') {
    const now = Date.now();
    const windowStart = now - this.windowMs;
    
    // Get existing requests for this identifier
    let userRequests = this.requests.get(identifier) || [];
    
    // Remove old requests outside the window
    userRequests = userRequests.filter(timestamp => timestamp > windowStart);
    
    // Check if limit exceeded
    if (userRequests.length >= this.maxRequests) {
      return false;
    }
    
    // Add current request
    userRequests.push(now);
    this.requests.set(identifier, userRequests);
    
    return true;
  }

  getRemainingRequests(identifier = 'default') {
    const userRequests = this.requests.get(identifier) || [];
    return Math.max(0, this.maxRequests - userRequests.length);
  }
}

// 3. Secure WebSocket Connection
class SecureSocketService extends SocketService {
  connect(serverURL) {
    // Ensure HTTPS/WSS in production
    if (process.env.NODE_ENV === 'production' && !serverURL.startsWith('https://')) {
      throw new Error('HTTPS required in production');
    }

    // Add authentication token if available
    const token = localStorage.getItem('auth_token');
    const options = {
      transports: ['websocket', 'polling'],
      timeout: 20000,
      auth: token ? { token } : undefined,
      // Additional security options
      forceNew: true,
      upgrade: true,
      rememberUpgrade: false
    };

    return super.connect(serverURL, options);
  }

  sendMessage(text, persona, sessionId, system) {
    // Sanitize inputs
    const sanitizedData = {
      text: InputSanitizer.sanitizeText(text),
      persona: InputSanitizer.sanitizeText(persona),
      session_id: InputSanitizer.sanitizeText(sessionId),
      system: InputSanitizer.sanitizeText(system)
    };

    // Rate limiting
    if (!this.rateLimiter.isAllowed('messages')) {
      throw new Error('Rate limit exceeded for messages');
    }

    return super.sendMessage(
      sanitizedData.text,
      sanitizedData.persona,
      sanitizedData.session_id,
      sanitizedData.system
    );
  }
}

// 4. Content Security Policy Helper
const CSPHelper = {
  // Generate nonce for inline scripts
  generateNonce() {
    const array = new Uint8Array(16);
    crypto.getRandomValues(array);
    return btoa(String.fromCharCode.apply(null, array));
  },

  // Validate external URLs
  isAllowedURL(url) {
    const allowedDomains = [
      'localhost',
      '127.0.0.1',
      process.env.REACT_APP_API_DOMAIN
    ];

    try {
      const urlObj = new URL(url);
      return allowedDomains.some(domain => 
        urlObj.hostname === domain || urlObj.hostname.endsWith('.' + domain)
      );
    } catch {
      return false;
    }
  },

  // Sanitize URLs
  sanitizeURL(url) {
    if (!this.isAllowedURL(url)) {
      throw new Error('URL not allowed by security policy');
    }
    return url;
  }
};

// Usage in components
const useSecureAPI = () => {
  const rateLimiter = useRef(new RateLimiter(20, 60000)); // 20 requests per minute

  const secureAPICall = useCallback(async (endpoint, options = {}) => {
    // Rate limiting
    if (!rateLimiter.current.isAllowed('api')) {
      throw new Error('API rate limit exceeded');
    }

    // Sanitize endpoint
    const sanitizedEndpoint = InputSanitizer.sanitizeText(endpoint);
    
    // Validate URL
    const fullURL = `${process.env.REACT_APP_API_BASE_URL}${sanitizedEndpoint}`;
    CSPHelper.sanitizeURL(fullURL);

    // Add security headers
    const secureOptions = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'X-Requested-With': 'XMLHttpRequest',
        ...options.headers
      }
    };

    return fetch(fullURL, secureOptions);
  }, []);

  return { secureAPICall };
};
```

---

## Testing Guide

### Comprehensive Testing Strategy

```javascript
// 1. Unit Tests for Services
import { SocketService } from '../services/SocketService';
import { AudioRecorder } from '../services/AudioRecorder';

describe('SocketService', () => {
  let socketService;

  beforeEach(() => {
    socketService = new SocketService();
  });

  afterEach(() => {
    socketService.disconnect();
  });

  test('should connect to server', async () => {
    const mockSocket = {
      on: jest.fn(),
      emit: jest.fn(),
      disconnect: jest.fn()
    };

    // Mock socket.io
    jest.mock('socket.io-client', () => ({
      __esModule: true,
      default: jest.fn(() => mockSocket)
    }));

    socketService.connect('https://ai-mock-pitching-427457295403.europe-west1.run.app');
    expect(socketService.socket).toBeDefined();
  });

  test('should handle connection errors', () => {
    const errorHandler = jest.fn();
    socketService.on('connection_error', errorHandler);

    // Simulate connection error
    socketService.emit('connection_error', { error: new Error('Connection failed') });
    
    expect(errorHandler).toHaveBeenCalledWith({
      error: expect.any(Error),
      attempts: expect.any(Number)
    });
  });

  test('should send messages correctly', () => {
    const mockSocket = { emit: jest.fn() };
    socketService.socket = mockSocket;
    socketService.isConnected = true;

    socketService.sendMessage('Hello', 'friendly', 'session123', 'workflow');

    expect(mockSocket.emit).toHaveBeenCalledWith('text_message', {
      text: 'Hello',
      persona: 'friendly',
      session_id: 'session123',
      system: 'workflow'
    });
  });
});

describe('AudioRecorder', () => {
  let audioRecorder;
  let mockMediaDevices;

  beforeEach(() => {
    audioRecorder = new AudioRecorder();
    
    // Mock MediaDevices API
    mockMediaDevices = {
      getUserMedia: jest.fn().mockResolvedValue({
        getTracks: () => [{ stop: jest.fn() }]
      })
    };
    
    Object.defineProperty(navigator, 'mediaDevices', {
      value: mockMediaDevices,
      writable: true
    });

    // Mock AudioContext
    global.AudioContext = jest.fn().mockImplementation(() => ({
      createMediaStreamSource: jest.fn().mockReturnValue({
        connect: jest.fn()
      }),
      createScriptProcessor: jest.fn().mockReturnValue({
        connect: jest.fn(),
        onaudioprocess: null
      }),
      createAnalyser: jest.fn().mockReturnValue({
        connect: jest.fn(),
        fftSize: 2048,
        getByteFrequencyData: jest.fn()
      }),
      sampleRate: 16000,
      state: 'running',
      resume: jest.fn().mockResolvedValue(),
      close: jest.fn().mockResolvedValue()
    }));
  });

  afterEach(() => {
    audioRecorder.cleanup();
  });

  test('should initialize audio recorder', async () => {
    const mockSocket = { emit: jest.fn() };
    
    await audioRecorder.initialize(mockSocket, 'session123', 'friendly');
    
    expect(mockMediaDevices.getUserMedia).toHaveBeenCalledWith({
      audio: expect.objectContaining({
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      })
    });
  });

  test('should start recording', () => {
    const mockSocket = { emit: jest.fn() };
    audioRecorder.socket = mockSocket;
    audioRecorder.sessionId = 'session123';
    audioRecorder.persona = 'friendly';

    audioRecorder.startRecording();

    expect(audioRecorder.isRecording).toBe(true);
    expect(mockSocket.emit).toHaveBeenCalledWith('start_recording', {
      session_id: 'session123',
      persona: 'friendly',
      sample_rate: expect.any(Number),
      channels: 1,
      format: 'pcm_f32le'
    });
  });

  test('should handle microphone permission denied', async () => {
    mockMediaDevices.getUserMedia.mockRejectedValue(
      new Error('Permission denied')
    );

    await expect(
      audioRecorder.initialize({}, 'session123', 'friendly')
    ).rejects.toThrow('Permission denied');
  });
});

// 2. Integration Tests
describe('Integration Tests', () => {
  test('should complete full session flow', async () => {
    const mockSocket = {
      on: jest.fn(),
      emit: jest.fn(),
      disconnect: jest.fn()
    };

    const sessionManager = new SessionManager();
    const socketService = new SocketService();
    socketService.socket = mockSocket;
    socketService.isConnected = true;

    // Start session
    const sessionId = await sessionManager.createSession(
      'John Doe',
      'TechCorp',
      'friendly',
      socketService
    );

    expect(sessionId).toBeDefined();
    expect(sessionManager.isActive()).toBe(true);

    // Send message
    sessionManager.addMessage('Hello', 'user');
    expect(sessionManager.getSessionData().messages).toHaveLength(1);

    // End session
    await sessionManager.endSession(socketService, {});
    expect(sessionManager.isActive()).toBe(false);
  });
});

// 3. E2E Tests with Cypress
// cypress/integration/pitch_session.spec.js
describe('Pitch Session E2E', () => {
  beforeEach(() => {
    cy.visit('/');
    cy.intercept('GET', '/api/personas', { fixture: 'personas.json' });
  });

  it('should complete a full pitch session', () => {
    // Fill session form
    cy.get('[data-testid="founder-name"]').type('John Doe');
    cy.get('[data-testid="company-name"]').type('TechCorp');
    cy.get('[data-testid="persona-select"]').select('friendly');

    // Start session
    cy.get('[data-testid="start-session"]').click();

    // Wait for session to start
    cy.get('[data-testid="session-active"]').should('be.visible');

    // Send a message
    cy.get('[data-testid="message-input"]').type('Hello, I need help with my pitch');
    cy.get('[data-testid="send-message"]').click();

    // Check message appears in chat
    cy.get('[data-testid="chat-messages"]')
      .should('contain', 'Hello, I need help with my pitch');

    // Wait for AI response
    cy.get('[data-testid="ai-message"]', { timeout: 10000 })
      .should('be.visible');

    // End session
    cy.get('[data-testid="end-session"]').click();

    // Check analytics tab
    cy.get('[data-testid="analytics-tab"]').should('be.visible');
  });

  it('should handle voice recording', () => {
    // Mock microphone permissions
    cy.window().then((win) => {
      cy.stub(win.navigator.mediaDevices, 'getUserMedia').resolves({
        getTracks: () => [{ stop: cy.stub() }]
      });
    });

    // Start session first
    cy.get('[data-testid="founder-name"]').type('John Doe');
    cy.get('[data-testid="company-name"]').type('TechCorp');
    cy.get('[data-testid="start-session"]').click();

    // Test voice recording
    cy.get('[data-testid="voice-record"]').click();
    cy.get('[data-testid="recording-indicator"]').should('be.visible');

    cy.get('[data-testid="voice-record"]').click();
    cy.get('[data-testid="recording-indicator"]').should('not.exist');
  });
});

// 4. Performance Tests
describe('Performance Tests', () => {
  test('should handle large message history efficiently', () => {
    const messages = Array.from({ length: 1000 }, (_, i) => ({
      id: i,
      content: `Message ${i}`,
      type: i % 2 === 0 ? 'user' : 'ai',
      timestamp: new Date()
    }));

    const startTime = performance.now();
    
    render(<ChatInterface messages={messages} />);
    
    const endTime = performance.now();
    const renderTime = endTime - startTime;

    // Should render within 100ms
    expect(renderTime).toBeLessThan(100);
  });

  test('should not cause memory leaks', async () => {
    const initialMemory = performance.memory?.usedJSHeapSize || 0;
    
    // Create and destroy multiple components
    for (let i = 0; i < 100; i++) {
      const { unmount } = render(<MokePitchApp />);
      unmount();
    }

    // Force garbage collection
    if (global.gc) {
      global.gc();
    }

    await new Promise(resolve => setTimeout(resolve, 1000));

    const finalMemory = performance.memory?.usedJSHeapSize || 0;
    const memoryIncrease = finalMemory - initialMemory;

    // Memory increase should be minimal (less than 10MB)
    expect(memoryIncrease).toBeLessThan(10 * 1024 * 1024);
  });
});

// 5. Test Utilities
export const TestUtils = {
  // Mock socket service
  createMockSocket() {
    return {
      on: jest.fn(),
      off: jest.fn(),
      emit: jest.fn(),
      disconnect: jest.fn(),
      connect: jest.fn(),
      id: 'mock-socket-id'
    };
  },

  // Mock audio context
  createMockAudioContext() {
    return {
      createMediaStreamSource: jest.fn().mockReturnValue({
        connect: jest.fn()
      }),
      createScriptProcessor: jest.fn().mockReturnValue({
        connect: jest.fn(),
        disconnect: jest.fn(),
        onaudioprocess: null
      }),
      createAnalyser: jest.fn().mockReturnValue({
        connect: jest.fn(),
        disconnect: jest.fn(),
        fftSize: 2048,
        getByteFrequencyData: jest.fn()
      }),
      sampleRate: 16000,
      state: 'running',
      resume: jest.fn().mockResolvedValue(),
      close: jest.fn().mockResolvedValue()
    };
  },

  // Wait for async operations
  async waitFor(condition, timeout = 5000) {
    const start = Date.now();
    while (Date.now() - start < timeout) {
      if (await condition()) {
        return true;
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw new Error('Timeout waiting for condition');
  },

  // Generate test data
  generateTestMessages(count = 10) {
    return Array.from({ length: count }, (_, i) => ({
      id: `msg_${i}`,
      content: `Test message ${i}`,
      type: i % 2 === 0 ? 'user' : 'ai',
      timestamp: new Date(Date.now() - (count - i) * 1000)
    }));
  }
};
```

---

## Deployment Guide

### Production Deployment

```javascript
// 1. Environment Configuration
// .env.production
REACT_APP_API_BASE_URL=https://your-api-domain.com
REACT_APP_SOCKET_URL=https://your-api-domain.com
REACT_APP_ENVIRONMENT=production
REACT_APP_VERSION=1.0.0

// 2. Build Optimization
// webpack.config.js (for custom builds)
const path = require('path');

module.exports = {
  mode: 'production',
  optimization: {
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          chunks: 'all',
        },
        audio: {
          test: /[\\/]services[\\/](Audio|Socket)/,
          name: 'audio-services',
          chunks: 'all',
        }
      }
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  }
};

// 3. Service Worker for Offline Support
// public/sw.js
const CACHE_NAME = 'moke-pitch-v1';
const urlsToCache = [
  '/',
  '/static/js/bundle.js',
  '/static/css/main.css',
  '/api/personas'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // Return cached version or fetch from network
        return response || fetch(event.request);
      })
  );
});

// 4. Docker Configuration
// Dockerfile
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

// nginx.conf
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    server {
        listen 80;
        server_name localhost;
        root /usr/share/nginx/html;
        index index.html;

        # Gzip compression
        gzip on;
        gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # Handle client-side routing
        location / {
            try_files $uri $uri/ /index.html;
        }

        # Proxy API requests
        location /api/ {
            proxy_pass http://backend:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket proxy
        location /socket.io/ {
            proxy_pass http://backend:8000;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}

// 5. CI/CD Pipeline (GitHub Actions)
// .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Node.js
      uses: actions/setup-node@v2
      with:
        node-version: '18'
    - name: Install dependencies
      run: npm ci
    - name: Run tests
      run: npm test -- --coverage --watchAll=false
    - name: Run E2E tests
      run: npm run test:e2e

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: docker build -t moke-pitch-frontend .
    
    - name: Deploy to production
      run: |
        # Your deployment script here
        echo "Deploying to production..."

// 6. Monitoring and Analytics
// src/utils/monitoring.js
class MonitoringService {
  constructor() {
    this.metrics = {
      sessionCount: 0,
      errorCount: 0,
      averageSessionDuration: 0,
      audioQuality: 0
    };
  }

  trackEvent(eventName, properties = {}) {
    // Send to analytics service
    if (window.gtag) {
      window.gtag('event', eventName, properties);
    }

    // Send to custom analytics
    this.sendToAnalytics({
      event: eventName,
      properties: properties,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    });
  }

  trackError(error, context = {}) {
    this.metrics.errorCount++;
    
    this.trackEvent('error', {
      message: error.message,
      stack: error.stack,
      context: JSON.stringify(context)
    });

    // Send to error tracking service
    if (window.Sentry) {
      window.Sentry.captureException(error, { extra: context });
    }
  }

  trackPerformance(metric, value) {
    this.trackEvent('performance', {
      metric: metric,
      value: value,
      timestamp: Date.now()
    });
  }

  async sendToAnalytics(data) {
    try {
      await fetch('/api/analytics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      });
    } catch (error) {
      console.error('Failed to send analytics:', error);
    }
  }
}

export const monitoring = new MonitoringService();

// Usage in components
const useMonitoring = () => {
  const trackSessionStart = (sessionData) => {
    monitoring.trackEvent('session_start', {
      persona: sessionData.persona,
      founderName: sessionData.founderName,
      companyName: sessionData.companyName
    });
  };

  const trackSessionEnd = (sessionData, duration) => {
    monitoring.trackEvent('session_end', {
      sessionId: sessionData.id,
      duration: duration,
      messageCount: sessionData.messages?.length || 0
    });
  };

  const trackError = (error, context) => {
    monitoring.trackError(error, context);
  };

  return {
    trackSessionStart,
    trackSessionEnd,
    trackError
  };
};
```

---

## Best Practices

### Development Best Practices

```javascript
// 1. Code Organization
src/
├── components/           # Reusable UI components
│   ├── common/          # Generic components
│   ├── chat/            # Chat-specific components
│   └── analytics/       # Analytics components
├── hooks/               # Custom React hooks
├── services/            # Business logic services
├── utils/               # Utility functions
├── constants/           # Application constants
├── types/               # TypeScript type definitions
└── __tests__/           # Test files

// 2. Component Design Patterns
// Higher-Order Component for error boundaries
const withErrorBoundary = (Component) => {
  return class extends React.Component {
    constructor(props) {
      super(props);
      this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
      return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
      console.error('Error caught by boundary:', error, errorInfo);
      monitoring.trackError(error, { errorInfo });
    }

    render() {
      if (this.state.hasError) {
        return (
          <div className="error-fallback">
            <h2>Something went wrong</h2>
            <p>{this.state.error?.message}</p>
            <button onClick={() => this.setState({ hasError: false, error: null })}>
              Try again
            </button>
          </div>
        );
      }

      return <Component {...this.props} />;
    }
  };
};

// 3. Custom Hooks Best Practices
const useAsyncOperation = (asyncFunction, dependencies = []) => {
  const [state, setState] = useState({
    data: null,
    loading: false,
    error: null
  });

  const execute = useCallback(async (...args) => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const result = await asyncFunction(...args);
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      setState(prev => ({ ...prev, loading: false, error }));
      throw error;
    }
  }, dependencies);

  return { ...state, execute };
};

// 4. State Management Patterns
// Context for global state
const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  const value = {
    state,
    dispatch,
    // Action creators
    startSession: (sessionData) => dispatch({ type: 'START_SESSION', payload: sessionData }),
    endSession: () => dispatch({ type: 'END_SESSION' }),
    addMessage: (message) => dispatch({ type: 'ADD_MESSAGE', payload: message }),
    setError: (error) => dispatch({ type: 'SET_ERROR', payload: error }),
    clearError: () => dispatch({ type: 'CLEAR_ERROR' })
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return context;
};

// 5. Performance Optimization Patterns
// Memoized components
const MemoizedChatMessage = React.memo(({ message }) => {
  return (
    <div className="chat-message">
      <span className="timestamp">{formatTime(message.timestamp)}</span>
      <p>{message.content}</p>
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function
  return prevProps.message.id === nextProps.message.id &&
         prevProps.message.content === nextProps.message.content;
});

// Lazy loading
const AnalyticsDashboard = lazy(() => import('./components/AnalyticsDashboard'));
const SessionHistory = lazy(() => import('./components/SessionHistory'));

// Usage with Suspense
<Suspense fallback={<div>Loading...</div>}>
  <AnalyticsDashboard />
</Suspense>

// 6. Accessibility Best Practices
const AccessibleButton = ({ onClick, children, disabled, ariaLabel, ...props }) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      role="button"
      tabIndex={disabled ? -1 : 0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick?.(e);
        }
      }}
      {...props}
    >
      {children}
    </button>
  );
};

// Screen reader announcements
const useScreenReader = () => {
  const announce = useCallback((message, priority = 'polite') => {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', priority);
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    
    document.body.appendChild(announcement);
    
    setTimeout(() => {
      document.body.removeChild(announcement);
    }, 1000);
  }, []);

  return { announce };
};

// 7. Internationalization
const useTranslation = () => {
  const [language, setLanguage] = useState('en');
  const [translations, setTranslations] = useState({});

  useEffect(() => {
    loadTranslations(language).then(setTranslations);
  }, [language]);

  const t = useCallback((key, params = {}) => {
    let translation = translations[key] || key;
    
    // Replace parameters
    Object.entries(params).forEach(([param, value]) => {
      translation = translation.replace(`{{${param}}}`, value);
    });
    
    return translation;
  }, [translations]);

  return { t, language, setLanguage };
};

// 8. Configuration Management
const config = {
  development: {
    apiBaseURL: 'https://ai-mock-pitching-427457295403.europe-west1.run.app',
    socketURL: 'https://ai-mock-pitching-427457295403.europe-west1.run.app',
    logLevel: 'debug'
  },
  production: {
    apiBaseURL: process.env.REACT_APP_API_BASE_URL,
    socketURL: process.env.REACT_APP_SOCKET_URL,
    logLevel: 'error'
  }
};

export const getConfig = () => {
  const env = process.env.NODE_ENV || 'development';
  return config[env];
};

// 9. Logging Utility
class Logger {
  constructor(level = 'info') {
    this.level = level;
    this.levels = { debug: 0, info: 1, warn: 2, error: 3 };
  }

  log(level, message, ...args) {
    if (this.levels[level] >= this.levels[this.level]) {
      console[level](`[${new Date().toISOString()}] ${message}`, ...args);
    }
  }

  debug(message, ...args) { this.log('debug', message, ...args); }
  info(message, ...args) { this.log('info', message, ...args); }
  warn(message, ...args) { this.log('warn', message, ...args); }
  error(message, ...args) { this.log('error', message, ...args); }
}

export const logger = new Logger(getConfig().logLevel);
```
    const response = await fetch('https://ai-mock-pitching-427457295403.europe-west1.run.app/api/personas');
    const data = await response.json();
    
    if (data.success) {
      setPersonas(data.personas);
      console.log('Personas loaded:', data.available_personas);
    }
  } catch (error) {
    console.error('Error loading personas:', error);
  }
};

// Call on component mount
useEffect(() => {
  loadPersonas();
}, []);
```

### Step 3: Start Session

```javascript
const startSession = () => {
  // Generate unique session ID
  const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  setSessionId(newSessionId);
  setIsSessionActive(true);
  setMessages([]);
  
  console.log('Session started:', newSessionId);
};
```

### Step 4: Send Messages

```javascript
const sendMessage = () => {
  if (!currentMessage.trim() || !sessionId) return;
  
  // Add user message to chat
  const userMessage = {
    id: Date.now(),
    text: currentMessage,
    sender: 'user',
    timestamp: new Date()
  };
  
  setMessages(prev => [...prev, userMessage]);
  setIsLoading(true);
  
  // Send to backend
  socket.emit('text_message', {
    text: currentMessage.trim(),
    persona: selectedPersona,
    session_id: sessionId,
    system: 'workflow'
  });
  
  setCurrentMessage('');
};
```

### Step 5: Handle AI Responses

```javascript
useEffect(() => {
  socket.on('response', (data) => {
    setIsLoading(false);
    
    // Add AI message to chat
    const aiMessage = {
      id: Date.now(),
      text: data.message,
      sender: 'ai',
      timestamp: new Date(),
      audioUrl: data.audio_url,
      stage: data.stage,
      complete: data.complete
    };
    
    setMessages(prev => [...prev, aiMessage]);
    
    // Play audio if available
    if (data.audio_url) {
      playAudio(`https://ai-mock-pitching-427457295403.europe-west1.run.app${data.audio_url}`);
    }
    
    // Check if session is complete
    if (data.complete) {
      setIsSessionActive(false);
      getAnalysis();
    }
  });
  
  return () => {
    socket.off('response');
  };
}, []);
```

### Step 6: Audio Playback

```javascript
const playAudio = (audioUrl) => {
  const audio = new Audio();
  audio.src = audioUrl + `?t=${Date.now()}`; // Cache busting
  
  audio.play().catch(error => {
    console.error('Error playing audio:', error);
  });
  
  audio.onended = () => {
    console.log('Audio playback finished');
  };
};
```

### Step 7: End Session

```javascript
const endSession = async () => {
  if (!sessionId) return;
  
  try {
    setIsLoading(true);
    
    const response = await fetch(`https://ai-mock-pitching-427457295403.europe-west1.run.app/api/pitch/end/${sessionId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ reason: 'user_ended' })
    });
    
    const data = await response.json();
    
    if (data.success) {
      setIsSessionActive(false);
      setAnalysis(data.analysis);
      console.log('Session ended with analysis');
    }
  } catch (error) {
    console.error('Error ending session:', error);
  } finally {
    setIsLoading(false);
  }
};
```

### Step 8: Get Analysis

```javascript
const getAnalysis = async () => {
  if (!sessionId) return;
  
  try {
    const response = await fetch(`https://ai-mock-pitching-427457295403.europe-west1.run.app/api/pitch/analysis/${sessionId}`);
    const data = await response.json();
    
    if (data.success) {
      setAnalysis(data.analysis);
      console.log('Analysis loaded:', data.analysis);
    }
  } catch (error) {
    console.error('Error getting analysis:', error);
  }
};

const getQuickAnalytics = async () => {
  if (!sessionId) return;
  
  try {
    const response = await fetch(`https://ai-mock-pitching-427457295403.europe-west1.run.app/api/pitch/analytics/${sessionId}`);
    const data = await response.json();
    
    if (data.success) {
      return data.analytics;
    }
  } catch (error) {
    console.error('Error getting analytics:', error);
  }
};
```

---

## React Example

### Complete React Component with Audio Recording

```jsx
import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';

const PitchApp = () => {
  // State management
  const [socket, setSocket] = useState(null);
  const [personas, setPersonas] = useState({});
  const [selectedPersona, setSelectedPersona] = useState('skeptical');
  const [sessionId, setSessionId] = useState(null);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  
  // Audio recording state
  const [isRecording, setIsRecording] = useState(false);
  const [currentTranscription, setCurrentTranscription] = useState('');
  const [audioRecorder, setAudioRecorder] = useState(null);
  const [isAudioInitialized, setIsAudioInitialized] = useState(false);

  // Initialize socket connection
  useEffect(() => {
    const newSocket = io('https://ai-mock-pitching-427457295403.europe-west1.run.app/');
    setSocket(newSocket);
    
    // Load personas
    loadPersonas();
    
    return () => newSocket.close();
  }, []);

  // Socket event listeners
  useEffect(() => {
    if (!socket) return;

    socket.on('response', handleAIResponse);
    socket.on('session_started', handleSessionStarted);
    socket.on('error', handleError);
    socket.on('transcription', handleTranscription);

    return () => {
      socket.off('response');
      socket.off('session_started');
      socket.off('error');
      socket.off('transcription');
    };
  }, [socket]);

  // Load personas from API
  const loadPersonas = async () => {
    try {
      const response = await fetch('https://ai-mock-pitching-427457295403.europe-west1.run.app/api/personas');
      const data = await response.json();
      if (data.success) {
        setPersonas(data.personas);
      }
    } catch (error) {
      console.error('Error loading personas:', error);
    }
  };

  // Handle AI response
  const handleAIResponse = (data) => {
    setIsLoading(false);
    
    const aiMessage = {
      id: Date.now(),
      text: data.message,
      sender: 'ai',
      timestamp: new Date(),
      audioUrl: data.audio_url
    };
    
    setMessages(prev => [...prev, aiMessage]);
    
    if (data.audio_url) {
      playAudio(`https://ai-mock-pitching-427457295403.europe-west1.run.app${data.audio_url}`);
    }
    
    if (data.complete) {
      setIsSessionActive(false);
      getAnalysis();
    }
  };

  // Handle session started
  const handleSessionStarted = (data) => {
    console.log('Session confirmed:', data.session_id);
  };

  // Handle errors
  const handleError = (error) => {
    console.error('Socket error:', error);
    setIsLoading(false);
  };

  // Handle transcription updates
  const handleTranscription = (data) => {
    if (data.is_final) {
      // Final transcription - add as user message
      const userMessage = {
        id: Date.now(),
        text: data.text,
        sender: 'user',
        timestamp: new Date(),
        isTranscribed: true
      };
      setMessages(prev => [...prev, userMessage]);
      setCurrentTranscription('');
      setIsLoading(true); // Wait for AI response
    } else {
      // Interim transcription - show as typing
      setCurrentTranscription(data.text);
    }
  };

  // Play audio
  const playAudio = (audioUrl) => {
    const audio = new Audio();
    audio.src = audioUrl + `?t=${Date.now()}`;
    audio.play().catch(console.error);
  };

  // Initialize audio recorder
  const initializeAudio = async () => {
    try {
      const recorder = new AudioRecorder();
      await recorder.initialize(socket, sessionId, selectedPersona);
      setAudioRecorder(recorder);
      setIsAudioInitialized(true);
      console.log('Audio recorder initialized');
    } catch (error) {
      console.error('Failed to initialize audio:', error);
      alert('Microphone access required for voice chat');
    }
  };

  // Start new session
  const startSession = async () => {
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    setIsSessionActive(true);
    setMessages([]);
    setAnalysis(null);
    
    // Initialize audio recording
    if (socket) {
      await initializeAudio();
    }
  };

  // Send message
  const sendMessage = () => {
    if (!currentMessage.trim() || !sessionId || !socket) return;
    
    const userMessage = {
      id: Date.now(),
      text: currentMessage,
      sender: 'user',
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    
    socket.emit('text_message', {
      text: currentMessage.trim(),
      persona: selectedPersona,
      session_id: sessionId,
      system: 'workflow'
    });
    
    setCurrentMessage('');
  };

  // End session
  const endSession = async () => {
    if (!sessionId) return;
    
    try {
      setIsLoading(true);
      const response = await fetch(`https://ai-mock-pitching-427457295403.europe-west1.run.app/api/pitch/end/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'user_ended' })
      });
      
      const data = await response.json();
      if (data.success) {
        setIsSessionActive(false);
        setAnalysis(data.analysis);
      }
    } catch (error) {
      console.error('Error ending session:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Get analysis
  const getAnalysis = async () => {
    if (!sessionId) return;
    
    try {
      const response = await fetch(`https://ai-mock-pitching-427457295403.europe-west1.run.app/api/pitch/analysis/${sessionId}`);
      const data = await response.json();
      if (data.success) {
        setAnalysis(data.analysis);
      }
    } catch (error) {
      console.error('Error getting analysis:', error);
    }
  };

  // Voice recording functions
  const startVoiceRecording = () => {
    if (audioRecorder && isAudioInitialized) {
      audioRecorder.startRecording();
      setIsRecording(true);
    } else {
      alert('Audio not initialized. Please check microphone permissions.');
    }
  };

  const stopVoiceRecording = () => {
    if (audioRecorder && isRecording) {
      audioRecorder.stopRecording();
      audioRecorder.sendFinalAudioChunk();
      setIsRecording(false);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (audioRecorder) {
        audioRecorder.cleanup();
      }
    };
  }, [audioRecorder]);

  return (
    <div className="pitch-app">
      {/* Persona Selection */}
      <div className="persona-section">
        <h3>Select Investor Persona</h3>
        <select 
          value={selectedPersona} 
          onChange={(e) => setSelectedPersona(e.target.value)}
          disabled={isSessionActive}
        >
          {Object.keys(personas).map(key => (
            <option key={key} value={key}>
              {personas[key].name} - {personas[key].title}
            </option>
          ))}
        </select>
        
        {personas[selectedPersona] && (
          <div className="persona-info">
            <p>{personas[selectedPersona].description}</p>
            <div className="traits">
              {personas[selectedPersona].personality_traits?.map(trait => (
                <span key={trait} className="trait-tag">{trait}</span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Session Controls */}
      <div className="controls">
        {!isSessionActive ? (
          <button onClick={startSession}>Start Pitch Session</button>
        ) : (
          <div className="session-controls">
            <button onClick={endSession}>End Session</button>
            {isAudioInitialized && (
              <div className="voice-controls">
                {!isRecording ? (
                  <button onClick={startVoiceRecording} className="voice-btn">
                    🎤 Start Speaking
                  </button>
                ) : (
                  <button onClick={stopVoiceRecording} className="voice-btn recording">
                    ⏹️ Stop Speaking
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Chat Interface */}
      {isSessionActive && (
        <div className="chat-section">
          <div className="messages">
            {messages.map(message => (
              <div key={message.id} className={`message ${message.sender}`}>
                <div className="message-content">
                  <p>{message.text}</p>
                  {message.isTranscribed && (
                    <span className="transcribed-badge">🎤 Voice</span>
                  )}
                  {message.audioUrl && (
                    <button onClick={() => playAudio(`https://ai-mock-pitching-427457295403.europe-west1.run.app${message.audioUrl}`)}>
                      🔊 Play Audio
                    </button>
                  )}
                </div>
                <small>{message.timestamp.toLocaleTimeString()}</small>
              </div>
            ))}
            
            {/* Show current transcription */}
            {currentTranscription && (
              <div className="message user interim">
                <div className="message-content">
                  <p className="transcribing">{currentTranscription}</p>
                  <span className="transcribing-indicator">🎤 Listening...</span>
                </div>
              </div>
            )}
            
            {isLoading && <div className="loading">AI is thinking...</div>}
          </div>
          
          <div className="input-section">
            <input
              type="text"
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Type your message or use voice..."
              disabled={isLoading || isRecording}
            />
            <button 
              onClick={sendMessage} 
              disabled={isLoading || !currentMessage.trim() || isRecording}
            >
              Send
            </button>
          </div>
        </div>
      )}

      {/* Analysis Results */}
      {analysis && (
        <div className="analysis-section">
          <h3>Pitch Analysis</h3>
          <div className="analysis-summary">
            <div className="score">Overall Score: {analysis.overall_score}/100</div>
            <div className="rating">Rating: {analysis.overall_rating}</div>
            <div className="completion">Completion: {analysis.completion_percentage}%</div>
            <div className="readiness">Readiness: {analysis.pitch_readiness}</div>
          </div>
          
          <div className="overall-description">
            <p>{analysis.overall_description}</p>
          </div>
          
          {/* Category Scores */}
          <div className="category-scores">
            <h4>Category Breakdown</h4>
            <div className="categories-grid">
              {Object.entries(analysis.category_scores || {}).map(([key, category]) => (
                <div key={key} className={`category-card ${category.rating.toLowerCase().replace(' ', '-')}`}>
                  <h5>{key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</h5>
                  <div className="category-score">{category.score}/100</div>
                  <div className="category-rating">{category.rating}</div>
                  <p className="category-description">{category.description}</p>
                </div>
              ))}
            </div>
          </div>
          
          <div className="analysis-details">
            <div className="strengths">
              <h4>Strengths</h4>
              <ul>
                {analysis.strengths?.map((strength, index) => (
                  <li key={index}>
                    <strong>{strength.area}:</strong> {strength.description}
                    <span className="score-badge">Score: {strength.score}/10</span>
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="weaknesses">
              <h4>Areas for Improvement</h4>
              <ul>
                {analysis.weaknesses?.map((weakness, index) => (
                  <li key={index}>
                    <strong>{weakness.area}:</strong> {weakness.description}
                    {weakness.improvement && (
                      <div className="improvement-suggestion">
                        💡 <em>{weakness.improvement}</em>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="recommendations">
              <h4>Key Recommendations</h4>
              <ul>
                {analysis.key_recommendations?.map((rec, index) => (
                  <li key={index}>{rec}</li>
                ))}
              </ul>
            </div>
            
            <div className="investor-perspective">
              <h4>Investor Perspective</h4>
              <p>{analysis.investor_perspective}</p>
            </div>
            
            <div className="next-steps">
              <h4>Next Steps</h4>
              <ul>
                {analysis.next_steps?.map((step, index) => (
                  <li key={index}>{step}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PitchApp;
```

### CSS Styles for Audio Features

```css
/* Voice controls */
.voice-controls {
  margin-left: 10px;
  display: inline-block;
}

.voice-btn {
  padding: 10px 15px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s ease;
}

.voice-btn:not(.recording) {
  background-color: #4CAF50;
  color: white;
}

.voice-btn:not(.recording):hover {
  background-color: #45a049;
}

.voice-btn.recording {
  background-color: #f44336;
  color: white;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}

/* Transcription styles */
.message.interim {
  opacity: 0.7;
  border-left: 3px solid #2196F3;
}

.transcribing {
  font-style: italic;
  color: #666;
}

.transcribing-indicator {
  font-size: 12px;
  color: #2196F3;
  margin-left: 10px;
}

.transcribed-badge {
  background-color: #4CAF50;
  color: white;
  padding: 2px 6px;
  border-radius: 10px;
  font-size: 10px;
  margin-left: 8px;
}

/* Session controls */
.session-controls {
  display: flex;
  gap: 10px;
  align-items: center;
}

/* Analysis styles */
.analysis-summary {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  padding: 15px;
  background-color: #f5f5f5;
  border-radius: 8px;
}

.analysis-summary > div {
  text-align: center;
}

.overall-description {
  margin-bottom: 20px;
  padding: 15px;
  background-color: #e3f2fd;
  border-radius: 8px;
  border-left: 4px solid #2196F3;
}

.categories-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 15px;
  margin-bottom: 30px;
}

.category-card {
  padding: 15px;
  border-radius: 8px;
  border: 2px solid #ddd;
  background-color: #fff;
}

.category-card.vertx-assured {
  border-color: #4CAF50;
  background-color: #f1f8e9;
}

.category-card.good {
  border-color: #8BC34A;
  background-color: #f9fbe7;
}

.category-card.satisfactory {
  border-color: #FFC107;
  background-color: #fffde7;
}

.category-card.below-average {
  border-color: #FF9800;
  background-color: #fff3e0;
}

.category-card.need-to-improve {
  border-color: #f44336;
  background-color: #ffebee;
}

.category-card h5 {
  margin: 0 0 10px 0;
  font-size: 16px;
  font-weight: bold;
}

.category-score {
  font-size: 24px;
  font-weight: bold;
  margin: 5px 0;
}

.category-rating {
  font-size: 14px;
  font-weight: bold;
  margin: 5px 0;
  padding: 3px 8px;
  border-radius: 12px;
  display: inline-block;
}

.category-card.vertx-assured .category-rating {
  background-color: #4CAF50;
  color: white;
}

.category-card.good .category-rating {
  background-color: #8BC34A;
  color: white;
}

.category-card.satisfactory .category-rating {
  background-color: #FFC107;
  color: black;
}

.category-card.below-average .category-rating {
  background-color: #FF9800;
  color: white;
}

.category-card.need-to-improve .category-rating {
  background-color: #f44336;
  color: white;
}

.category-description {
  font-size: 14px;
  color: #666;
  margin-top: 10px;
}

.score-badge {
  background-color: #2196F3;
  color: white;
  padding: 2px 6px;
  border-radius: 10px;
  font-size: 12px;
  margin-left: 10px;
}

.improvement-suggestion {
  margin-top: 8px;
  padding: 8px;
  background-color: #fff3cd;
  border-radius: 4px;
  font-size: 14px;
}

.investor-perspective {
  background-color: #e8f5e8;
  padding: 15px;
  border-radius: 8px;
  margin: 20px 0;
  border-left: 4px solid #4CAF50;
}

.next-steps {
  background-color: #fff3e0;
  padding: 15px;
  border-radius: 8px;
  border-left: 4px solid #FF9800;
}

/* Responsive design */
@media (max-width: 768px) {
  .session-controls {
    flex-direction: column;
    gap: 5px;
  }
  
  .voice-controls {
    margin-left: 0;
    margin-top: 5px;
  }
  
  .analysis-summary {
    flex-direction: column;
    gap: 10px;
  }
  
  .categories-grid {
    grid-template-columns: 1fr;
  }
}
```

---

## Vue.js Example

### Vue Component

```vue
<template>
  <div class="pitch-app">
    <!-- Persona Selection -->
    <div class="persona-section">
      <h3>Select Investor Persona</h3>
      <select v-model="selectedPersona" :disabled="isSessionActive">
        <option v-for="(persona, key) in personas" :key="key" :value="key">
          {{ persona.name }} - {{ persona.title }}
        </option>
      </select>
      
      <div v-if="personas[selectedPersona]" class="persona-info">
        <p>{{ personas[selectedPersona].description }}</p>
        <div class="traits">
          <span 
            v-for="trait in personas[selectedPersona].personality_traits" 
            :key="trait" 
            class="trait-tag"
          >
            {{ trait }}
          </span>
        </div>
      </div>
    </div>

    <!-- Session Controls -->
    <div class="controls">
      <button v-if="!isSessionActive" @click="startSession">
        Start Pitch Session
      </button>
      <button v-else @click="endSession">
        End Session
      </button>
    </div>

    <!-- Chat Interface -->
    <div v-if="isSessionActive" class="chat-section">
      <div class="messages">
        <div 
          v-for="message in messages" 
          :key="message.id" 
          :class="['message', message.sender]"
        >
          <div class="message-content">
            <p>{{ message.text }}</p>
            <button 
              v-if="message.audioUrl" 
              @click="playAudio(`https://ai-mock-pitching-427457295403.europe-west1.run.app${message.audioUrl}`)"
            >
              🔊 Play Audio
            </button>
          </div>
          <small>{{ formatTime(message.timestamp) }}</small>
        </div>
        <div v-if="isLoading" class="loading">AI is thinking...</div>
      </div>
      
      <div class="input-section">
        <input
          v-model="currentMessage"
          @keyup.enter="sendMessage"
          :disabled="isLoading"
          placeholder="Type your message..."
        />
        <button 
          @click="sendMessage" 
          :disabled="isLoading || !currentMessage.trim()"
        >
          Send
        </button>
      </div>
    </div>

    <!-- Analysis Results -->
    <div v-if="analysis" class="analysis-section">
      <h3>Pitch Analysis</h3>
      <div class="analysis-summary">
        <div class="score">Overall Score: {{ analysis.overall_score }}/100</div>
        <div class="completion">Completion: {{ analysis.completion_percentage }}%</div>
        <div class="readiness">Readiness: {{ analysis.pitch_readiness }}</div>
      </div>
      
      <div class="analysis-details">
        <div class="strengths">
          <h4>Strengths</h4>
          <ul>
            <li v-for="(strength, index) in analysis.strengths" :key="index">
              <strong>{{ strength.area }}:</strong> {{ strength.description }}
            </li>
          </ul>
        </div>
        
        <div class="weaknesses">
          <h4>Areas for Improvement</h4>
          <ul>
            <li v-for="(weakness, index) in analysis.weaknesses" :key="index">
              <strong>{{ weakness.area }}:</strong> {{ weakness.description }}
            </li>
          </ul>
        </div>
        
        <div class="recommendations">
          <h4>Key Recommendations</h4>
          <ul>
            <li v-for="(rec, index) in analysis.key_recommendations" :key="index">
              {{ rec }}
            </li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import io from 'socket.io-client';

export default {
  name: 'PitchApp',
  data() {
    return {
      socket: null,
      personas: {},
      selectedPersona: 'skeptical',
      sessionId: null,
      isSessionActive: false,
      messages: [],
      currentMessage: '',
      isLoading: false,
      analysis: null
    };
  },
  
  mounted() {
    this.initSocket();
    this.loadPersonas();
  },
  
  beforeUnmount() {
    if (this.socket) {
      this.socket.close();
    }
  },
  
  methods: {
    initSocket() {
      this.socket = io('https://ai-mock-pitching-427457295403.europe-west1.run.app/');
      
      this.socket.on('response', this.handleAIResponse);
      this.socket.on('session_started', this.handleSessionStarted);
      this.socket.on('error', this.handleError);
    },
    
    async loadPersonas() {
      try {
        const response = await fetch('https://ai-mock-pitching-427457295403.europe-west1.run.app/api/personas');
        const data = await response.json();
        if (data.success) {
          this.personas = data.personas;
        }
      } catch (error) {
        console.error('Error loading personas:', error);
      }
    },
    
    handleAIResponse(data) {
      this.isLoading = false;
      
      const aiMessage = {
        id: Date.now(),
        text: data.message,
        sender: 'ai',
        timestamp: new Date(),
        audioUrl: data.audio_url
      };
      
      this.messages.push(aiMessage);
      
      if (data.audio_url) {
        this.playAudio(`https://ai-mock-pitching-427457295403.europe-west1.run.app${data.audio_url}`);
      }
      
      if (data.complete) {
        this.isSessionActive = false;
        this.getAnalysis();
      }
    },
    
    handleSessionStarted(data) {
      console.log('Session confirmed:', data.session_id);
    },
    
    handleError(error) {
      console.error('Socket error:', error);
      this.isLoading = false;
    },
    
    playAudio(audioUrl) {
      const audio = new Audio();
      audio.src = audioUrl + `?t=${Date.now()}`;
      audio.play().catch(console.error);
    },
    
    startSession() {
      const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      this.sessionId = newSessionId;
      this.isSessionActive = true;
      this.messages = [];
      this.analysis = null;
    },
    
    sendMessage() {
      if (!this.currentMessage.trim() || !this.sessionId || !this.socket) return;
      
      const userMessage = {
        id: Date.now(),
        text: this.currentMessage,
        sender: 'user',
        timestamp: new Date()
      };
      
      this.messages.push(userMessage);
      this.isLoading = true;
      
      this.socket.emit('text_message', {
        text: this.currentMessage.trim(),
        persona: this.selectedPersona,
        session_id: this.sessionId,
        system: 'workflow'
      });
      
      this.currentMessage = '';
    },
    
    async endSession() {
      if (!this.sessionId) return;
      
      try {
        this.isLoading = true;
        const response = await fetch(`https://ai-mock-pitching-427457295403.europe-west1.run.app/api/pitch/end/${this.sessionId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason: 'user_ended' })
        });
        
        const data = await response.json();
        if (data.success) {
          this.isSessionActive = false;
          this.analysis = data.analysis;
        }
      } catch (error) {
        console.error('Error ending session:', error);
      } finally {
        this.isLoading = false;
      }
    },
    
    async getAnalysis() {
      if (!this.sessionId) return;
      
      try {
        const response = await fetch(`https://ai-mock-pitching-427457295403.europe-west1.run.app/api/pitch/analysis/${this.sessionId}`);
        const data = await response.json();
        if (data.success) {
          this.analysis = data.analysis;
        }
      } catch (error) {
        console.error('Error getting analysis:', error);
      }
    },
    
    formatTime(timestamp) {
      return timestamp.toLocaleTimeString();
    }
  }
};
</script>
```

---

## Error Handling

### Common Error Scenarios

```javascript
// Connection errors
socket.on('connect_error', (error) => {
  console.error('Connection failed:', error);
  // Show user-friendly message
  showError('Unable to connect to server. Please try again.');
});

// API errors
const handleApiError = (response) => {
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  return response.json();
};

// Usage
fetch('/api/personas')
  .then(handleApiError)
  .then(data => {
    // Handle success
  })
  .catch(error => {
    console.error('API Error:', error);
    showError('Failed to load data. Please refresh the page.');
  });

// Session errors
const validateSession = () => {
  if (!sessionId) {
    throw new Error('No active session');
  }
  if (!socket || !socket.connected) {
    throw new Error('Not connected to server');
  }
};
```

---

## Best Practices

### 1. State Management
- Use proper state management (Redux, Vuex, etc.) for complex apps
- Keep session state synchronized
- Handle reconnection scenarios

### 2. Performance
- Implement message pagination for long conversations
- Clean up old audio files on frontend
- Use audio preloading for better UX
- **🆕 Optimize video frame capture rate (0.5-2 seconds interval)**
- **🆕 Use appropriate video resolution (320x240 for analysis)**
- **🆕 Implement video frame compression (JPEG quality 0.7-0.8)**

### 3. User Experience
- Show loading states during API calls
- Provide audio controls (play/pause/volume)
- Implement typing indicators
- Add message timestamps
- **🆕 Show real-time video analysis metrics**
- **🆕 Provide video preview with status indicators**
- **🆕 Display video analysis insights and recommendations**

### 4. Security
- Validate all user inputs
- Implement rate limiting on frontend
- Use HTTPS in production
- Sanitize displayed content
- **🆕 Secure video frame transmission (base64 encoding)**
- **🆕 Validate camera permissions before starting analysis**

### 5. Error Recovery
- Implement automatic reconnection
- Provide manual retry options
- Show meaningful error messages
- Log errors for debugging
- **🆕 Handle camera access failures gracefully**
- **🆕 Provide fallback when video analysis is unavailable**

### 6. 🆕 Video Analysis Best Practices
- **Camera Setup**: Request camera permissions early in the user flow
- **Frame Rate**: Use 0.5-2 second intervals for optimal performance
- **Resolution**: 320x240 is sufficient for analysis while maintaining performance
- **Quality**: JPEG quality of 0.7-0.8 balances file size and analysis accuracy
- **Error Handling**: Always provide fallback when camera is unavailable
- **User Privacy**: Clearly communicate that video is used for analysis only
- **Performance**: Monitor frame capture performance and adjust rates accordingly
- **Cleanup**: Always stop video streams and clear intervals on component unmount

---

## Testing

### Unit Tests
```javascript
// Test persona loading
test('should load personas from API', async () => {
  const mockPersonas = { success: true, personas: {} };
  fetch.mockResolvedValue({
    ok: true,
    json: () => Promise.resolve(mockPersonas)
  });
  
  const result = await loadPersonas();
  expect(result).toEqual(mockPersonas);
});

// Test message sending
test('should send message via socket', () => {
  const mockSocket = { emit: jest.fn() };
  const message = 'Hello';
  const persona = 'friendly';
  const sessionId = 'test-session';
  
  sendMessage(mockSocket, message, persona, sessionId);
  
  expect(mockSocket.emit).toHaveBeenCalledWith('text_message', {
    text: message,
    persona: persona,
    session_id: sessionId,
    system: 'workflow'
  });
});
```

### Integration Tests
```javascript
// Test full flow
test('complete pitch session flow', async () => {
  // 1. Load personas
  const personas = await loadPersonas();
  expect(personas.success).toBe(true);
  
  // 2. Start session
  const sessionId = startSession();
  expect(sessionId).toBeDefined();
  
  // 3. Send message
  await sendMessage('Hello', 'friendly', sessionId);
  
  // 4. End session
  const analysis = await endSession(sessionId);
  expect(analysis.success).toBe(true);
});
```

---

## Deployment Considerations

### Environment Variables
```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://ai-mock-pitching-427457295403.europe-west1.run.app/';
const SOCKET_URL = process.env.REACT_APP_SOCKET_URL || 'https://ai-mock-pitching-427457295403.europe-west1.run.app/';
```

### Production Settings
```javascript
const socket = io(SOCKET_URL, {
  transports: ['websocket', 'polling'],
  timeout: 20000,
  forceNew: true
});
```

### CORS Configuration
The backend is already configured to allow all origins for development:
```python
# In main.py
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Already configured for all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Quick Start Checklist

### ✅ Pre-Integration Checklist

**Backend Verification:**
- [ ] Backend server running and accessible at deployed URL
- [ ] MongoDB database connected and accessible
- [ ] All API endpoints responding correctly
- [ ] WebSocket server enabled and functional
- [ ] Audio processing services working
- [ ] CORS configured for your frontend domain

**Frontend Setup:**
- [ ] Node.js 16+ installed
- [ ] Required dependencies installed (`socket.io-client`, `axios`, `canvas-capture`)
- [ ] Environment variables configured
- [ ] Build tools configured (Webpack/Vite)
- [ ] HTTPS enabled for production (required for camera/microphone access)

**Browser Requirements:**
- [ ] Modern browser with WebRTC support
- [ ] **Camera permissions granted** 🆕
- [ ] Microphone permissions granted
- [ ] **Canvas API support** 🆕
- [ ] JavaScript enabled
- [ ] WebSocket support enabled

### 🚀 Integration Steps

1. **Install Dependencies**
   ```bash
   npm install socket.io-client axios recordrtc canvas-capture html2canvas
   ```

2. **Configure Environment**
   ```bash
   # .env
   REACT_APP_API_BASE_URL=https://ai-mock-pitching-427457295403.europe-west1.run.app
   REACT_APP_SOCKET_URL=https://ai-mock-pitching-427457295403.europe-west1.run.app
   ```

3. **Initialize Services**
   ```javascript
   import { SocketService } from './services/SocketService';
   import { AudioRecorder } from './services/AudioRecorder';
   import { VideoAnalysisService } from './services/VideoAnalysisService'; // 🆕
   
   const socketService = new SocketService();
   const audioRecorder = new AudioRecorder();
   const videoService = new VideoAnalysisService(socketService); // 🆕
   ```

4. **Connect to Backend**
   ```javascript
   socketService.connect('https://ai-mock-pitching-427457295403.europe-west1.run.app');
   ```

5. **Initialize Video Analysis** 🆕
   ```javascript
   await videoService.initializeVideo();
   videoService.setupEventListeners();
   ```

6. **Handle Events**
   ```javascript
   socketService.on('ai_response', handleAIResponse);
   socketService.on('transcription', handleTranscription);
   socketService.on('video_analysis_update', handleVideoUpdate); // 🆕
   ```

7. **Test Integration**
   - Start a session
   - Send text message
   - Test voice recording
   - **Test video analysis** 🆕
   - Verify analytics

---

## Troubleshooting

### Common Issues and Solutions

#### 🔴 Connection Issues

**Problem:** Cannot connect to WebSocket server
```
Error: WebSocket connection failed
```

**Solutions:**
1. Verify backend server is running: `curl https://ai-mock-pitching-427457295403.europe-west1.run.app/health`
2. Check CORS configuration in backend
3. Ensure WebSocket port is not blocked by firewall
4. Try different transport: `transports: ['polling']`

**Problem:** CORS errors when making API calls
```
Access to fetch at 'https://ai-mock-pitching-427457295403.europe-west1.run.app/api/personas' from origin 'http://localhost:3000' has been blocked by CORS policy
```

**Solutions:**
1. Add your frontend URL to backend CORS configuration
2. Use proxy in development: `"proxy": "https://ai-mock-pitching-427457295403.europe-west1.run.app"` in package.json
3. Configure webpack dev server proxy

#### 🎤 Audio Issues

**Problem:** Microphone access denied
```
Error: Permission denied
```

**Solutions:**
1. Ensure HTTPS in production (required for microphone access)
2. Check browser permissions: Settings > Privacy > Microphone
3. Test with different browser
4. Use `navigator.mediaDevices.getUserMedia()` test

#### 📹 Video Analysis Issues

**Problem:** Camera access denied
```
Error: Video access denied: NotAllowedError
```

**Solutions:**
1. Ensure HTTPS in production (required for camera access)
2. Check browser permissions: Settings > Privacy > Camera
3. Test camera access: `navigator.mediaDevices.getUserMedia({ video: true })`
4. Clear browser cache and permissions
5. Try different browser or incognito mode

**Problem:** Video analysis not starting
```
Video analysis error: Video analyzer not available
```

**Solutions:**
1. Verify backend video analysis services are running
2. Check if enhanced video analyzer is properly initialized
3. Test with basic video analyzer as fallback
4. Ensure WebSocket connection is stable
5. Check browser console for detailed error messages

**Problem:** Poor video analysis accuracy
```
Low confidence scores or incorrect gesture detection
```

**Solutions:**
1. Improve lighting conditions (avoid backlighting)
2. Ensure clear view of face and hands
3. Reduce background clutter and movement
4. Check camera resolution and quality
5. Adjust frame capture rate (try 1-2 second intervals)
6. Ensure stable internet connection for real-time processing

**Problem:** Video analysis not appearing in final pitch analysis
```
"Video Analysis: Not available for this session" in pitch analysis report
```

**Solutions:**
1. Explicitly enable video analysis with `enable_video_analysis` event
2. Ensure video frames are being sent regularly during the session
3. Check that video analysis data is being integrated into LangGraph workflow
4. Verify that `video_analysis_enabled` flag is set to true in session state
5. Monitor server logs for video analysis integration messages
6. Try sending frames at a lower resolution (320x240) to ensure processing speed

**Problem:** No audio transcription received
```
Audio chunks sent but no transcription events
```

**Solutions:**
1. Verify audio format compatibility (16kHz, mono, PCM)
2. Check audio chunk size (not too large/small)
3. Ensure `is_final: true` is sent to complete transcription
4. Test with different audio input devices

#### 📡 Real-time Issues

**Problem:** Delayed or missing AI responses
```
Message sent but no response received
```

**Solutions:**
1. Check backend logs for processing errors
2. Verify session_id is correctly passed
3. Ensure persona exists in backend
4. Test with simpler messages first

**Problem:** Transcription lag or inaccuracy
```
Transcription is slow or incorrect
```

**Solutions:**
1. Reduce audio chunk size for faster processing
2. Improve audio quality (noise cancellation, better microphone)
3. Check network latency
4. Adjust voice activity detection settings

#### 💾 Database Issues

**Problem:** Session data not persisting
```
Sessions created but not found in database
```

**Solutions:**
1. Verify MongoDB connection in backend
2. Check database write permissions
3. Ensure session_id is unique and valid
4. Test database endpoints directly

### Debug Tools

#### 1. Network Debugging
```javascript
// Enable Socket.IO debugging
localStorage.debug = 'socket.io-client:socket';

// Monitor WebSocket traffic
const originalEmit = socket.emit;
socket.emit = function(...args) {
  console.log('Socket emit:', args);
  return originalEmit.apply(this, args);
};

socket.onAny((event, ...args) => {
  console.log('Socket receive:', event, args);
});
```

#### 2. Audio Debugging
```javascript
// Test microphone access
const testMicrophone = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    console.log('✅ Microphone access granted');
    
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    source.connect(analyser);
    
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    const checkAudio = () => {
      analyser.getByteFrequencyData(dataArray);
      const volume = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
      console.log('Audio level:', volume);
      
      if (volume > 10) {
        console.log('✅ Audio detected');
      }
    };
    
    setInterval(checkAudio, 1000);
  } catch (error) {
    console.error('❌ Microphone test failed:', error);
  }
};
```

#### 3. Video Analysis Testing
```javascript
// Test video analysis capabilities
const testVideoAnalysis = async () => {
  try {
    // Test camera access
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 320, height: 240 } 
    });
    console.log('✅ Camera access granted');
    
    // Test video element
    const video = document.createElement('video');
    video.srcObject = stream;
    video.autoplay = true;
    video.muted = true;
    
    // Test frame capture
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 320;
    canvas.height = 240;
    
    video.onloadedmetadata = () => {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frameData = canvas.toDataURL('image/jpeg', 0.8);
      console.log('✅ Frame capture working, size:', frameData.length);
      
      // Test WebSocket video events
      if (socket && socket.connected) {
        socket.emit('start_video_analysis', { session_id: 'test' });
        socket.emit('video_frame', { session_id: 'test', frame_data: frameData });
        console.log('✅ Video WebSocket events sent');
      }
    };
    
    // Cleanup
    setTimeout(() => {
      stream.getTracks().forEach(track => track.stop());
      console.log('✅ Video test completed');
    }, 5000);
    
  } catch (error) {
    console.error('❌ Video analysis test failed:', error);
  }
};
```

#### 4. API Testing
```javascript
// Test all API endpoints
const testAPI = async () => {
  const endpoints = [
    '/health',
    '/api/personas',
    '/api/stats'
  ];
  
  for (const endpoint of endpoints) {
    try {
      const response = await fetch(`https://ai-mock-pitching-427457295403.europe-west1.run.app${endpoint}`);
      console.log(`✅ ${endpoint}:`, response.status);
    } catch (error) {
      console.error(`❌ ${endpoint}:`, error.message);
    }
  }
};
```

---

## Performance Optimization Tips

### 🚀 Frontend Optimization

1. **Bundle Splitting**
   ```javascript
   // Lazy load heavy components
   const AnalyticsDashboard = lazy(() => import('./AnalyticsDashboard'));
   
   // Code splitting by route
   const routes = [
     { path: '/', component: lazy(() => import('./Home')) },
     { path: '/analytics', component: lazy(() => import('./Analytics')) }
   ];
   ```

2. **Audio Optimization**
   ```javascript
   // Reduce audio chunk frequency for better performance
   const CHUNK_INTERVAL = 250; // ms (instead of 100ms)
   
   // Use compression for audio data
   const compressAudioData = (audioData) => {
     // Implement audio compression logic
     return compressedData;
   };
   ```

3. **Memory Management**
   ```javascript
   // Clean up resources
   useEffect(() => {
     return () => {
       audioRecorder?.cleanup();
       socketService?.disconnect();
       // Clear large data structures
       setMessages([]);
     };
   }, []);
   ```

### 📊 Monitoring Performance

```javascript
// Performance monitoring
const usePerformanceMonitoring = () => {
  useEffect(() => {
    // Monitor render times
    const observer = new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => {
        if (entry.entryType === 'measure') {
          console.log(`${entry.name}: ${entry.duration}ms`);
        }
      });
    });
    
    observer.observe({ entryTypes: ['measure'] });
    
    return () => observer.disconnect();
  }, []);
};

// Usage
performance.mark('component-render-start');
// ... component rendering
performance.mark('component-render-end');
performance.measure('component-render', 'component-render-start', 'component-render-end');
```

---

## Security Best Practices

### 🔒 Frontend Security

1. **Input Validation**
   ```javascript
   const validateInput = (input) => {
     // Sanitize HTML
     const sanitized = DOMPurify.sanitize(input);
     
     // Length limits
     if (sanitized.length > 1000) {
       throw new Error('Input too long');
     }
     
     // Content validation
     if (/<script|javascript:|on\w+=/i.test(input)) {
       throw new Error('Invalid content detected');
     }
     
     return sanitized;
   };
   ```

2. **Secure Communication**
   ```javascript
   // Always use HTTPS in production
   const getSocketURL = () => {
     const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
     return `${protocol}//${window.location.host}`;
   };
   
   // Validate server certificates
   const socketOptions = {
     rejectUnauthorized: process.env.NODE_ENV === 'production'
   };
   ```

3. **Data Protection**
   ```javascript
   // Don't store sensitive data in localStorage
   const secureStorage = {
     set: (key, value) => {
       sessionStorage.setItem(key, JSON.stringify(value));
     },
     get: (key) => {
       const item = sessionStorage.getItem(key);
       return item ? JSON.parse(item) : null;
     },
     remove: (key) => {
       sessionStorage.removeItem(key);
     }
   };
   ```

---

## Support & Resources

### 📚 Documentation Links

- **Backend API Documentation**: `/docs` endpoint on your backend server
- **Socket.IO Client Documentation**: https://socket.io/docs/v4/client-api/
- **WebRTC Documentation**: https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API
- **React Documentation**: https://react.dev/
- **Vue.js Documentation**: https://vuejs.org/
- **Angular Documentation**: https://angular.io/

### 🛠️ Development Tools

- **Browser DevTools**: Network tab for WebSocket inspection
- **React DevTools**: Component debugging
- **Vue DevTools**: Vue component inspection
- **Socket.IO Admin UI**: Real-time socket monitoring
- **Postman**: API endpoint testing

### 🐛 Getting Help

**For Backend Issues:**
1. Check backend server logs
2. Test API endpoints with curl/Postman
3. Verify database connectivity
4. Check environment variables

**For Frontend Issues:**
1. Open browser developer console
2. Check network tab for failed requests
3. Verify WebSocket connection status
4. Test microphone permissions

**For Integration Issues:**
1. Verify CORS configuration
2. Check WebSocket transport compatibility
3. Test with minimal example first
4. Compare with working examples in this guide

### 📞 Support Channels

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check this guide for common solutions
- **Community**: Join developer community discussions
- **Direct Support**: Contact development team

---

## Changelog & Updates

### Version 1.1.0 (Current) 🆕
- ✅ Complete WebSocket integration
- ✅ Real-time audio streaming
- ✅ Speech-to-text transcription
- ✅ Text-to-speech playback
- ✅ **🆕 Professional video analysis (CVZone, FER, MediaPipe)**
- ✅ **🆕 Real-time hand gesture recognition**
- ✅ **🆕 Facial emotion analysis**
- ✅ **🆕 Pose and body language assessment**
- ✅ **🆕 Multimodal pitch analysis (Audio + Video + Text)**
- ✅ Session management
- ✅ Analytics dashboard
- ✅ Database integration
- ✅ Error handling
- ✅ Performance optimization
- ✅ Security features
- ✅ Testing framework
- ✅ Deployment guide

### Version 1.0.0 (Previous)
- ✅ Complete WebSocket integration
- ✅ Real-time audio streaming
- ✅ Speech-to-text transcription
- ✅ Text-to-speech playback
- ✅ Session management
- ✅ Analytics dashboard
- ✅ Database integration
- ✅ Error handling
- ✅ Performance optimization
- ✅ Security features
- ✅ Testing framework
- ✅ Deployment guide

### Upcoming Features
- 🔄 Multi-language support
- 🔄 Advanced analytics with video insights
- 🔄 Mobile app integration
- 🔄 Offline mode support
- 🔄 Advanced audio processing
- 🔄 Custom persona creation
- 🔄 **Video analysis improvements (gesture library expansion)**
- 🔄 **Real-time coaching suggestions**
- 🔄 **Advanced emotion recognition**

---

## 🆕 Video Analysis Integration Summary

### 🎥 Professional Video Analysis Capabilities

The platform now includes **state-of-the-art video analysis** using professional computer vision libraries:

#### 🤲 Hand Gesture Analysis (CVZone)
- **Advanced Recognition**: Pointing, open palm, closed fist, thumbs up, peace sign, OK sign
- **Effectiveness Scoring**: AI-powered gesture effectiveness based on pitch context and timing
- **Real-time Tracking**: Continuous hand position, movement, and gesture coordination analysis
- **Confidence Metrics**: Accuracy scores and reliability indicators for detected gestures
- **Contextual Analysis**: Gesture appropriateness for different pitch stages

#### 😊 Facial Emotion Recognition (FER)
- **Emotion Detection**: Happy, sad, angry, fear, surprise, disgust, neutral with confidence scores
- **Pitch Suitability**: Context-aware emotion appropriateness analysis for investor presentations
- **Micro-expression Analysis**: Subtle facial expression changes and authenticity indicators
- **Eye Contact Estimation**: Gaze direction, engagement level, and audience connection scoring
- **Emotional Consistency**: Tracking emotional alignment with pitch content

#### 🧍 Pose & Body Language (MediaPipe)
- **Posture Analysis**: Upright, leaning forward/back, slouching detection with scoring
- **Engagement Scoring**: Body language-based engagement levels and professional presence
- **Movement Coordination**: Natural gesture flow and body-hand coordination analysis
- **Spatial Awareness**: Positioning, stance, and use of space evaluation
- **Professional Presence**: Overall body language assessment for business presentations

### 🔧 Technical Integration Points

#### New WebSocket Events
```javascript
// Client → Server Events
socket.emit('start_video_analysis', { session_id: 'session_123' });
socket.emit('video_frame', { 
  session_id: 'session_123', 
  frame_data: 'data:image/jpeg;base64,/9j/4AAQ...' 
});
socket.emit('stop_video_analysis', { session_id: 'session_123' });

// Server → Client Events
socket.on('video_analysis_started', (data) => {
  // { session_id, status, analyzer_type: 'enhanced', message }
});

socket.on('video_analysis_update', (data) => {
  // Real-time analysis results with hand, emotion, and pose data
});

socket.on('video_insights', (data) => {
  // AI-generated insights and recommendations
});

socket.on('video_error', (data) => {
  // Error handling for video analysis issues
});
```

#### Video Analysis Data Structure
```javascript
{
  "session_id": "session_123",
  "analyzer_type": "enhanced",
  "analysis": {
    "hand_analysis": {
      "hands_detected": 2,
      "gestures": [
        {
          "type": "pointing",
          "confidence": 0.85,
          "effectiveness": 0.9,
          "description": "Effective pointing gesture"
        }
      ],
      "gesture_effectiveness": 0.87,
      "gesture_confidence": 0.82
    },
    "emotion_analysis": {
      "dominant_emotion": "happy",
      "emotion_confidence": 0.91,
      "pitch_suitability": 0.88,
      "emotions": {
        "happy": 0.65,
        "neutral": 0.25,
        "confident": 0.10
      }
    },
    "pose_analysis": {
      "pose_detected": true,
      "posture_score": 0.85,
      "engagement_level": "high",
      "body_language": ["upright_posture", "open_stance", "good_eye_contact"]
    },
    "overall_scores": {
      "gesture_score": 0.87,
      "emotion_score": 0.91,
      "pose_score": 0.85,
      "overall_score": 0.88
    }
  }
}
```

#### Video Service Integration
```javascript
// Initialize Video Analysis Service
const videoService = new VideoAnalysisService(socketService);

// Setup and start analysis
await videoService.initializeVideo();
videoService.setupEventListeners();
await videoService.startAnalysis(sessionId);

// Get real-time metrics
const metrics = videoService.getCurrentMetrics();
// Returns: handGestures, emotions, pose, overall scores

// Handle insights
videoService.onInsights = (data) => {
  displayVideoInsights(data.insights, data.recommendations);
};
```

#### React Hook Integration
```javascript
const {
  isVideoReady,           // Camera initialized and ready
  isAnalyzing,           // Video analysis active
  currentMetrics,        // Real-time analysis data
  videoInsights,         // AI-generated insights
  error,                 // Video-related errors
  videoElementRef,       // Video element reference
  initializeVideo,       // Initialize camera
  startAnalysis,         // Start video analysis
  stopAnalysis          // Stop video analysis
} = useVideoAnalysis(socketService);
```

### 📊 Real-time Metrics Display

```javascript
// Example metrics display component
const VideoMetricsDisplay = ({ metrics }) => {
  if (!metrics) return null;

  return (
    <div className="video-metrics">
      {/* Overall Performance */}
      <div className="metric-group">
        <h4>🎯 Overall Performance</h4>
        <div className="score-display">
          {Math.round(metrics.overall.overallScore * 100)}%
        </div>
      </div>

      {/* Hand Gestures */}
      <div className="metric-group">
        <h4>🤲 Hand Gestures</h4>
        <div>Hands Detected: {metrics.handGestures.detected}</div>
        <div>Effectiveness: {Math.round(metrics.handGestures.effectiveness * 100)}%</div>
        <div>Current: {metrics.handGestures.gestures.map(g => g.type).join(', ')}</div>
      </div>

      {/* Emotions */}
      <div className="metric-group">
        <h4>😊 Facial Expression</h4>
        <div>Emotion: {metrics.emotions.dominant}</div>
        <div>Confidence: {Math.round(metrics.emotions.confidence * 100)}%</div>
        <div>Suitability: {Math.round(metrics.emotions.suitability * 100)}%</div>
      </div>

      {/* Pose & Body Language */}
      <div className="metric-group">
        <h4>🧍 Body Language</h4>
        <div>Posture: {Math.round(metrics.pose.posture * 100)}%</div>
        <div>Engagement: {metrics.pose.engagement}</div>
        <div>Indicators: {metrics.pose.bodyLanguage.join(', ')}</div>
      </div>
    </div>
  );
};
```

### 🚀 Performance & Optimization

#### Recommended Settings
```javascript
const videoConfig = {
  frameRate: 0.5,           // Analyze every 0.5 seconds
  frameQuality: 0.8,        // JPEG quality (0.7-0.8 recommended)
  frameWidth: 320,          // Optimal for analysis
  frameHeight: 240,         // Maintains performance
  analysisInterval: 500     // Milliseconds between captures
};
```

#### Browser Requirements
- **Camera Access**: Required for video analysis
- **Canvas API**: For frame capture and processing
- **WebRTC Support**: Modern browser compatibility
- **HTTPS**: Required for camera permissions in production

#### Performance Considerations
- **Bandwidth**: ~10-50KB per frame depending on quality settings
- **CPU Usage**: Minimal impact with optimized frame rates
- **Memory**: Efficient cleanup prevents memory leaks
- **Battery**: Optimized for mobile device usage

### 🔒 Privacy & Security

- **Local Processing**: Video frames processed for analysis only, not stored
- **No Recording**: No video recording or permanent storage
- **Base64 Transmission**: Secure frame encoding over WebSocket
- **User Consent**: Clear communication about video usage and purpose
- **Data Retention**: Analysis results stored temporarily for session duration only

### 📱 Mobile Compatibility

```javascript
// Mobile-optimized video configuration
const mobileVideoConfig = {
  video: {
    width: { ideal: 320 },
    height: { ideal: 240 },
    facingMode: 'user',      // Front camera
    frameRate: { ideal: 15, max: 30 }
  }
};

// Responsive video display
const videoStyle = {
  width: '100%',
  maxWidth: '400px',
  height: 'auto',
  borderRadius: '8px'
};
```

### 🛠️ Quick Start with Video Analysis

1. **Update Dependencies**
   ```bash
   npm install canvas-capture html2canvas
   ```

2. **Initialize Video Service**
   ```javascript
   import { VideoAnalysisService } from './services/VideoAnalysisService';
   
   const videoService = new VideoAnalysisService(socketService);
   await videoService.initializeVideo();
   ```

3. **Add Video Component**
   ```jsx
   <video 
     ref={videoElementRef} 
     autoPlay 
     muted 
     playsInline 
     style={{ width: '100%', maxHeight: '300px' }}
   />
   ```

4. **Handle Video Events**
   ```javascript
   videoService.onAnalysisUpdate = (data) => {
     if (data.type === 'update') {
       updateVideoMetrics(data.analysis);
     }
   };
   
   videoService.onInsights = (data) => {
     displayInsights(data.insights, data.recommendations);
   };
   ```

5. **Start Analysis**
   ```javascript
   // Start video analysis when session begins
   await videoService.startAnalysis(sessionId);
   
   // Stop when session ends
   await videoService.stopAnalysis(sessionId);
   ```

### 🎯 Video Analysis Benefits

- **Enhanced Feedback**: Comprehensive multimodal analysis combining audio, text, and video
- **Professional Insights**: Industry-standard computer vision analysis
- **Real-time Coaching**: Immediate feedback on presentation skills
- **Objective Metrics**: Quantified scores for gestures, emotions, and body language
- **Actionable Recommendations**: AI-generated suggestions for improvement

---

**🎯 Congratulations! You now have everything needed to build a complete, production-ready frontend with advanced video analysis for the AI Mock Investor Pitch application.**

This comprehensive guide now covers:
- ✅ **Complete API Integration** - All REST and WebSocket endpoints
- ✅ **Real-time Audio** - Recording, streaming, and playback
- ✅ **🆕 Professional Video Analysis** - CVZone, FER, and MediaPipe integration
- ✅ **Database Management** - Session storage and analytics
- ✅ **Multiple Frameworks** - React, Vue.js, and Angular examples
- ✅ **Production Ready** - Security, performance, and deployment
- ✅ **Testing & Debugging** - Comprehensive testing strategies
- ✅ **Best Practices** - Industry-standard development patterns
- ✅ **🆕 Multimodal Analysis** - Audio + Video + Text integration

**Ready to start building with advanced video analysis? Pick your framework and follow the step-by-step examples above!** 🚀