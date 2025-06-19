# Frontend Integration Guide

## 🚀 Complete Integration Guide for React/Vue/Angular

This guide provides step-by-step instructions to integrate the AI Mock Investor Pitch backend with any modern frontend framework.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [API Overview](#api-overview)
3. [WebSocket Integration](#websocket-integration)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [React Example](#react-example)
6. [Vue.js Example](#vuejs-example)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)

---

## Prerequisites

### Backend Requirements
- Backend server running on `https://ai-mock-pitching-427457295403.europe-west1.run.app/`
- Socket.IO server enabled
- All API endpoints functional

### Frontend Requirements
- Modern JavaScript framework (React, Vue, Angular)
- Socket.IO client library
- HTTP client (axios, fetch)
- Audio playback capability

### Installation
```bash
# For React
npm install socket.io-client axios

# For Vue
npm install socket.io-client axios

# For Angular
npm install socket.io-client axios
```

---

## API Overview

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/personas` | Get all available investor personas |
| `POST` | `/api/pitch/end/{session_id}` | End pitch session |
| `GET` | `/api/pitch/analytics/{session_id}` | Get session analytics |
| `GET` | `/api/pitch/analysis/{session_id}` | Get detailed analysis |
| `GET` | `/api/pitch/report/{session_id}` | Get formatted report |
| `GET` | `/download/{filename}` | Download audio files |

### WebSocket Events

| Event | Direction | Data | Description |
|-------|-----------|------|-------------|
| `connect` | Client → Server | - | Establish connection |
| `text_message` | Client → Server | `{text, persona, session_id, system}` | Send user message |
| `response` | Server → Client | `{message, audio_url, stage, complete}` | AI response |
| `session_started` | Server → Client | `{session_id, persona, system}` | Session confirmation |
| `error` | Server → Client | `{message, type}` | Error notification |

---

## WebSocket Integration

### Connection Setup
```javascript
import io from 'socket.io-client';

const socket = io('https://ai-mock-pitching-427457295403.europe-west1.run.app/', {
  transports: ['websocket', 'polling']
});

// Connection events
socket.on('connect', () => {
  console.log('Connected to server');
});

socket.on('disconnect', () => {
  console.log('Disconnected from server');
});
```

### Message Handling
```javascript
// Send message to AI
const sendMessage = (text, persona, sessionId) => {
  socket.emit('text_message', {
    text: text.trim(),
    persona: persona,
    session_id: sessionId,
    system: 'workflow'
  });
};

// Listen for AI responses
socket.on('response', (data) => {
  console.log('AI Response:', data);
  
  // Handle text response
  if (data.message) {
    displayMessage(data.message, 'ai');
  }
  
  // Handle audio response
  if (data.audio_url) {
    playAudio(data.audio_url);
  }
  
  // Check if session is complete
  if (data.complete) {
    handleSessionComplete();
  }
});

// Listen for session started
socket.on('session_started', (data) => {
  console.log('Session started:', data.session_id);
});

// Listen for errors
socket.on('error', (error) => {
  console.error('Socket error:', error);
});
```

---

## Step-by-Step Implementation

### Step 1: Initialize Application State

```javascript
// Application state
const [personas, setPersonas] = useState({});
const [selectedPersona, setSelectedPersona] = useState('skeptical');
const [sessionId, setSessionId] = useState(null);
const [isSessionActive, setIsSessionActive] = useState(false);
const [messages, setMessages] = useState([]);
const [currentMessage, setCurrentMessage] = useState('');
const [isLoading, setIsLoading] = useState(false);
const [analysis, setAnalysis] = useState(null);
```

### Step 2: Load Personas

```javascript
const loadPersonas = async () => {
  try {
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

### Complete React Component

```jsx
import React, { useState, useEffect } from 'react';
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

    return () => {
      socket.off('response');
      socket.off('session_started');
      socket.off('error');
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

  // Play audio
  const playAudio = (audioUrl) => {
    const audio = new Audio();
    audio.src = audioUrl + `?t=${Date.now()}`;
    audio.play().catch(console.error);
  };

  // Start new session
  const startSession = () => {
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    setIsSessionActive(true);
    setMessages([]);
    setAnalysis(null);
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
          <button onClick={endSession}>End Session</button>
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
                  {message.audioUrl && (
                    <button onClick={() => playAudio(`https://ai-mock-pitching-427457295403.europe-west1.run.app${message.audioUrl}`)}>
                      🔊 Play Audio
                    </button>
                  )}
                </div>
                <small>{message.timestamp.toLocaleTimeString()}</small>
              </div>
            ))}
            {isLoading && <div className="loading">AI is thinking...</div>}
          </div>
          
          <div className="input-section">
            <input
              type="text"
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Type your message..."
              disabled={isLoading}
            />
            <button onClick={sendMessage} disabled={isLoading || !currentMessage.trim()}>
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
            <div className="completion">Completion: {analysis.completion_percentage}%</div>
            <div className="readiness">Readiness: {analysis.pitch_readiness}</div>
          </div>
          
          <div className="analysis-details">
            <div className="strengths">
              <h4>Strengths</h4>
              <ul>
                {analysis.strengths?.map((strength, index) => (
                  <li key={index}>
                    <strong>{strength.area}:</strong> {strength.description}
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
          </div>
        </div>
      )}
    </div>
  );
};

export default PitchApp;
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

### 3. User Experience
- Show loading states during API calls
- Provide audio controls (play/pause/volume)
- Implement typing indicators
- Add message timestamps

### 4. Security
- Validate all user inputs
- Implement rate limiting on frontend
- Use HTTPS in production
- Sanitize displayed content

### 5. Error Recovery
- Implement automatic reconnection
- Provide manual retry options
- Show meaningful error messages
- Log errors for debugging

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

## Support

For issues or questions:
1. Check the console for error messages
2. Verify backend server is running
3. Test API endpoints directly
4. Check network connectivity
5. Review WebSocket connection status

---

**🎯 You're now ready to integrate the AI Mock Investor Pitch backend with any modern frontend framework!**