import React, { useState, useEffect, useRef } from 'react';

/**
 * Voice Activity Detection React Component
 * Provides automatic speech detection and transcription
 */
const VoiceActivityDetection = ({
  onTranscription,
  onAIResponse,
  onError,
  onStatusChange,
  persona = 'friendly',
  autoStart = false,
  debug = false
}) => {
  // State
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [vadStatus, setVadStatus] = useState('idle');
  const [currentMessage, setCurrentMessage] = useState('');
  
  // Refs
  const vadRef = useRef(null);
  const sessionIdRef = useRef(null);
  
  // Initialize VAD system
  useEffect(() => {
    const initializeVAD = async () => {
      try {
        // Import VAD class (assuming it's available globally or as a module)
        const VoiceActivityDetection = window.VoiceActivityDetection;
        
        if (!VoiceActivityDetection) {
          throw new Error('VoiceActivityDetection not loaded. Make sure to include the script.');
        }
        
        // Create VAD instance
        vadRef.current = new VoiceActivityDetection({
          debug: debug,
          socketUrl: window.location.origin
        });
        
        // Set up callbacks
        vadRef.current.setCallbacks({
          onConnectionChange: (data) => {
            setIsConnected(data.connected);
            onStatusChange?.({ type: 'connection', connected: data.connected });
          },
          
          onSessionStart: (data) => {
            setSessionId(data.session_id);
            sessionIdRef.current = data.session_id;
            setCurrentMessage(data.message);
            onStatusChange?.({ type: 'session_start', data });
          },
          
          onSessionEnd: (data) => {
            setSessionId(null);
            sessionIdRef.current = null;
            setIsRecording(false);
            setVadStatus('idle');
            onStatusChange?.({ type: 'session_end', data });
          },
          
          onSpeechStart: (data) => {
            setVadStatus('speaking');
            setIsRecording(true);
            onStatusChange?.({ type: 'speech_start', data });
          },
          
          onSpeechEnd: (data) => {
            setVadStatus('processing');
            setIsRecording(false);
            onStatusChange?.({ type: 'speech_end', data });
          },
          
          onTranscription: (data) => {
            setVadStatus('listening');
            onTranscription?.(data.transcript, data);
          },
          
          onAIResponse: (data) => {
            setCurrentMessage(data.message);
            setVadStatus('listening');
            onAIResponse?.(data.message, data);
          },
          
          onError: (data) => {
            console.error('VAD Error:', data);
            onError?.(data);
          },
          
          onVADStatus: (data) => {
            if (debug) {
              console.log('VAD Status:', data);
            }
          }
        });
        
        // Initialize the system
        const initialized = await vadRef.current.initialize();
        
        if (!initialized) {
          throw new Error('Failed to initialize VAD system');
        }
        
        // Auto-start if requested
        if (autoStart) {
          startSession();
        }
        
      } catch (error) {
        console.error('Failed to initialize VAD:', error);
        onError?.({ type: 'initialization', error: error.message });
      }
    };
    
    initializeVAD();
    
    // Cleanup on unmount
    return () => {
      if (vadRef.current) {
        vadRef.current.cleanup();
      }
    };
  }, [debug, autoStart]);
  
  // Functions
  const startSession = async () => {
    if (!vadRef.current || !isConnected) {
      onError?.({ type: 'session_start', error: 'VAD not ready or not connected' });
      return;
    }
    
    try {
      const newSessionId = generateSessionId();
      await vadRef.current.startSession(newSessionId, persona);
    } catch (error) {
      onError?.({ type: 'session_start', error: error.message });
    }
  };
  
  const stopSession = async () => {
    if (!vadRef.current) return;
    
    try {
      await vadRef.current.stopSession();
    } catch (error) {
      onError?.({ type: 'session_stop', error: error.message });
    }
  };
  
  const sendManualText = (text) => {
    if (!vadRef.current || !sessionIdRef.current) {
      onError?.({ type: 'manual_text', error: 'No active session' });
      return;
    }
    
    vadRef.current.sendManualText(text);
  };
  
  const generateSessionId = () => {
    return 'react_session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
  };
  
  // Status indicator component
  const StatusIndicator = ({ status, label }) => {
    const getStatusColor = () => {
      switch (status) {
        case 'connected': return '#4CAF50';
        case 'recording': return '#ff9800';
        case 'processing': return '#2196F3';
        case 'error': return '#f44336';
        default: return '#9e9e9e';
      }
    };
    
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span>{label}</span>
        <div
          style={{
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            backgroundColor: getStatusColor(),
            animation: status === 'recording' ? 'pulse 1s infinite' : 'none'
          }}
        />
      </div>
    );
  };
  
  // VAD Circle component
  const VADCircle = () => {
    const getCircleStyle = () => {
      const baseStyle = {
        width: '80px',
        height: '80px',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '2em',
        margin: '0 auto',
        transition: 'all 0.3s ease',
        cursor: sessionId ? 'default' : 'pointer'
      };
      
      switch (vadStatus) {
        case 'speaking':
          return {
            ...baseStyle,
            background: 'linear-gradient(45deg, #ff9800, #f57c00)',
            animation: 'pulse 0.5s infinite'
          };
        case 'listening':
          return {
            ...baseStyle,
            background: 'linear-gradient(45deg, #2196F3, #0b7dda)',
            animation: 'pulse 1s infinite'
          };
        case 'processing':
          return {
            ...baseStyle,
            background: 'linear-gradient(45deg, #9c27b0, #7b1fa2)'
          };
        default:
          return {
            ...baseStyle,
            background: 'linear-gradient(45deg, #9e9e9e, #757575)'
          };
      }
    };
    
    const getEmoji = () => {
      switch (vadStatus) {
        case 'speaking': return '🗣️';
        case 'listening': return '👂';
        case 'processing': return '⚙️';
        default: return '🎤';
      }
    };
    
    const getText = () => {
      switch (vadStatus) {
        case 'speaking': return 'Speaking detected...';
        case 'listening': return 'Listening for your voice...';
        case 'processing': return 'Processing speech...';
        default: return sessionId ? 'Session active' : 'Click Start to begin';
      }
    };
    
    return (
      <div style={{ textAlign: 'center', margin: '20px 0' }}>
        <div style={getCircleStyle()} onClick={!sessionId ? startSession : undefined}>
          {getEmoji()}
        </div>
        <div style={{ marginTop: '10px', fontSize: '14px' }}>
          {getText()}
        </div>
      </div>
    );
  };
  
  return (
    <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
      {/* Status Panel */}
      <div style={{ 
        background: 'rgba(0,0,0,0.05)', 
        padding: '15px', 
        borderRadius: '10px', 
        marginBottom: '20px' 
      }}>
        <h3 style={{ margin: '0 0 15px 0' }}>Voice Activity Status</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <StatusIndicator 
            status={isConnected ? 'connected' : 'disconnected'} 
            label={`Connection: ${isConnected ? 'Connected' : 'Disconnected'}`} 
          />
          <StatusIndicator 
            status={sessionId ? 'connected' : 'idle'} 
            label={`Session: ${sessionId ? 'Active' : 'Inactive'}`} 
          />
          <StatusIndicator 
            status={isRecording ? 'recording' : vadStatus} 
            label={`Voice: ${vadStatus.charAt(0).toUpperCase() + vadStatus.slice(1)}`} 
          />
        </div>
      </div>
      
      {/* VAD Circle */}
      <VADCircle />
      
      {/* Controls */}
      <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', margin: '20px 0' }}>
        <button
          onClick={startSession}
          disabled={!isConnected || !!sessionId}
          style={{
            padding: '12px 24px',
            borderRadius: '25px',
            border: 'none',
            background: sessionId ? '#ccc' : 'linear-gradient(45deg, #4CAF50, #45a049)',
            color: 'white',
            cursor: sessionId ? 'not-allowed' : 'pointer',
            fontWeight: 'bold'
          }}
        >
          Start Session
        </button>
        
        <button
          onClick={stopSession}
          disabled={!sessionId}
          style={{
            padding: '12px 24px',
            borderRadius: '25px',
            border: 'none',
            background: !sessionId ? '#ccc' : 'linear-gradient(45deg, #f44336, #da190b)',
            color: 'white',
            cursor: !sessionId ? 'not-allowed' : 'pointer',
            fontWeight: 'bold'
          }}
        >
          End Session
        </button>
      </div>
      
      {/* Current AI Message */}
      {currentMessage && (
        <div style={{
          background: 'linear-gradient(45deg, #667eea, #764ba2)',
          color: 'white',
          padding: '15px',
          borderRadius: '15px',
          margin: '20px 0'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>🤖 AI Investor:</div>
          <div>{currentMessage}</div>
        </div>
      )}
      
      {/* Manual Input (for fallback) */}
      <ManualInput onSend={sendManualText} disabled={!sessionId} />
    </div>
  );
};

// Manual Input Component
const ManualInput = ({ onSend, disabled }) => {
  const [text, setText] = useState('');
  const [showInput, setShowInput] = useState(false);
  
  const handleSend = () => {
    if (text.trim() && !disabled) {
      onSend(text.trim());
      setText('');
    }
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };
  
  return (
    <div>
      <button
        onClick={() => setShowInput(!showInput)}
        style={{
          padding: '8px 16px',
          borderRadius: '15px',
          border: '1px solid #ccc',
          background: 'white',
          cursor: 'pointer',
          fontSize: '12px',
          display: 'block',
          margin: '0 auto'
        }}
      >
        {showInput ? 'Hide' : 'Show'} Manual Input
      </button>
      
      {showInput && (
        <div style={{ marginTop: '15px' }}>
          <div style={{ display: 'flex', gap: '10px' }}>
            <input
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here..."
              disabled={disabled}
              style={{
                flex: 1,
                padding: '12px',
                borderRadius: '25px',
                border: '1px solid #ccc',
                fontSize: '14px'
              }}
            />
            <button
              onClick={handleSend}
              disabled={disabled || !text.trim()}
              style={{
                padding: '12px 20px',
                borderRadius: '25px',
                border: 'none',
                background: disabled || !text.trim() ? '#ccc' : 'linear-gradient(45deg, #2196F3, #0b7dda)',
                color: 'white',
                cursor: disabled || !text.trim() ? 'not-allowed' : 'pointer',
                fontWeight: 'bold'
              }}
            >
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default VoiceActivityDetection;