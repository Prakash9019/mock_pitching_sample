/**
 * Hybrid Voice Activity Detection System
 * - Real-time transcription display (see text as you speak)
 * - Smart pause detection (send to AI when you pause)
 * - Best of both worlds: immediate feedback + accurate conversation flow
 */

class HybridVoiceActivityDetection {
    constructor(config = {}) {
        this.config = {
            sampleRate: 16000,
            bufferSize: 4096,
            socketUrl: window.location.origin,
            pauseThreshold: 2000,        // 2 seconds pause to send (faster than 5s)
            minSpeechDuration: 1000,     // Minimum 1 second of speech
            debug: false,
            ...config
        };
        
        // State management
        this.isInitialized = false;
        this.isConnected = false;
        this.currentSessionId = null;
        this.isRecording = false;
        this.audioContext = null;
        this.mediaStream = null;
        this.socket = null;
        this.speechRecognition = null;
        
        // Hybrid transcription state
        this.currentTranscript = '';
        this.finalTranscript = '';
        this.lastSpeechTime = 0;
        this.pauseTimer = null;
        this.speechStartTime = 0;
        
        // Callbacks
        this.callbacks = {
            onConnectionChange: null,
            onSessionStart: null,
            onSessionEnd: null,
            onRealtimeTranscript: null,    // New: real-time text updates
            onFinalTranscript: null,       // New: when sending to AI
            onAIResponse: null,
            onError: null,
            onStatusChange: null
        };
        
        this.log('Hybrid VAD initialized');
    }
    
    log(message, level = 'info') {
        if (this.config.debug) {
            console.log(`[HybridVAD] ${message}`);
        }
    }
    
    /**
     * Initialize the hybrid system
     */
    async initialize() {
        try {
            this.log('Initializing hybrid VAD system...');
            
            // Initialize Socket.IO connection
            await this.initializeSocket();
            
            // Initialize speech recognition for real-time transcription
            await this.initializeSpeechRecognition();
            
            // Initialize audio context for VAD
            await this.initializeAudioContext();
            
            this.isInitialized = true;
            this.log('Hybrid VAD system initialized successfully');
            return true;
            
        } catch (error) {
            this.log(`Initialization failed: ${error.message}`, 'error');
            this.callbacks.onError?.({ type: 'initialization', error: error.message });
            return false;
        }
    }
    
    /**
     * Initialize Socket.IO connection
     */
    async initializeSocket() {
        return new Promise((resolve, reject) => {
            if (typeof io === 'undefined') {
                reject(new Error('Socket.IO not loaded'));
                return;
            }
            
            this.socket = io(this.config.socketUrl, {
                transports: ['websocket', 'polling'],
                timeout: 20000,
                forceNew: true,
                reconnection: true,
                reconnectionDelay: 1000,
                reconnectionAttempts: 5
            });
            
            this.socket.on('connect', () => {
                this.isConnected = true;
                this.log('Socket.IO connected');
                this.callbacks.onConnectionChange?.({ connected: true });
                resolve();
            });
            
            this.socket.on('disconnect', () => {
                this.isConnected = false;
                this.log('Socket.IO disconnected');
                this.callbacks.onConnectionChange?.({ connected: false });
            });
            
            this.socket.on('connect_error', (error) => {
                this.log(`Connection error: ${error.message}`, 'error');
                reject(error);
            });
            
            // Audio session events
            this.socket.on('audio_session_started', (data) => {
                this.log(`Session started: ${data.session_id}`);
                this.callbacks.onSessionStart?.(data);
            });
            
            this.socket.on('audio_session_stopped', (data) => {
                this.log(`Session stopped: ${data.session_id}`);
                this.callbacks.onSessionEnd?.(data);
            });
            
            this.socket.on('transcription_result', (data) => {
                this.log(`Final transcription: ${data.transcript}`);
                this.callbacks.onFinalTranscript?.(data);
            });
            
            this.socket.on('ai_response', (data) => {
                this.log(`AI response: ${data.message}`);
                this.callbacks.onAIResponse?.(data);
            });
            
            this.socket.on('audio_error', (data) => {
                this.log(`Audio error: ${data.error}`, 'error');
                this.callbacks.onError?.(data);
            });
        });
    }
    
    /**
     * Initialize speech recognition for real-time transcription
     */
    async initializeSpeechRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            throw new Error('Speech recognition not supported in this browser');
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.speechRecognition = new SpeechRecognition();
        
        // Configure for hybrid mode
        this.speechRecognition.continuous = true;
        this.speechRecognition.interimResults = true;  // Show partial results
        this.speechRecognition.lang = 'en-US';
        
        this.speechRecognition.onstart = () => {
            this.log('Speech recognition started');
            this.callbacks.onStatusChange?.({ type: 'recognition_start' });
        };
        
        this.speechRecognition.onresult = (event) => {
            this.handleSpeechResult(event);
        };
        
        this.speechRecognition.onerror = (event) => {
            this.log(`Speech recognition error: ${event.error}`, 'error');
            this.callbacks.onError?.({ type: 'speech_recognition', error: event.error });
        };
        
        this.speechRecognition.onend = () => {
            this.log('Speech recognition ended');
            if (this.isRecording) {
                // Restart if we're still in a session
                setTimeout(() => {
                    if (this.isRecording) {
                        this.speechRecognition.start();
                    }
                }, 100);
            }
        };
        
        this.log('Speech recognition initialized');
    }
    
    /**
     * Handle speech recognition results (hybrid processing)
     */
    handleSpeechResult(event) {
        let interimTranscript = '';
        let finalTranscript = '';
        
        // Process all results
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            
            if (event.results[i].isFinal) {
                finalTranscript += transcript + ' ';
                this.lastSpeechTime = Date.now();
            } else {
                interimTranscript += transcript;
            }
        }
        
        // Update current transcript
        if (finalTranscript) {
            this.finalTranscript += finalTranscript;
        }
        this.currentTranscript = this.finalTranscript + interimTranscript;
        
        // Show real-time transcription
        this.callbacks.onRealtimeTranscript?.({
            current: this.currentTranscript,
            final: this.finalTranscript,
            interim: interimTranscript,
            isFinal: !!finalTranscript
        });
        
        // Handle pause detection for sending to AI
        if (finalTranscript) {
            this.handlePauseDetection();
        }
        
        this.log(`Transcript update - Final: "${this.finalTranscript}" | Interim: "${interimTranscript}"`);
    }
    
    /**
     * Handle pause detection and smart sending to AI
     */
    handlePauseDetection() {
        // Clear existing pause timer
        if (this.pauseTimer) {
            clearTimeout(this.pauseTimer);
        }
        
        // Set new pause timer
        this.pauseTimer = setTimeout(() => {
            this.sendTranscriptToAI();
        }, this.config.pauseThreshold);
        
        this.log(`Pause timer set for ${this.config.pauseThreshold}ms`);
    }
    
    /**
     * Send accumulated transcript to AI
     */
    sendTranscriptToAI() {
        if (!this.finalTranscript.trim()) {
            this.log('No final transcript to send');
            return;
        }
        
        const speechDuration = Date.now() - this.speechStartTime;
        if (speechDuration < this.config.minSpeechDuration) {
            this.log('Speech too short, not sending');
            return;
        }
        
        this.log(`Sending transcript to AI: "${this.finalTranscript}"`);
        
        // Send to backend for AI processing
        this.socket.emit('manual_transcription', {
            session_id: this.currentSessionId,
            text: this.finalTranscript.trim()
        });
        
        // Mark as sent and reset for next speech
        this.callbacks.onFinalTranscript?.({
            transcript: this.finalTranscript.trim(),
            action: 'sent_to_ai'
        });
        
        // Reset transcript for next speech segment
        this.finalTranscript = '';
        this.currentTranscript = '';
    }
    
    /**
     * Initialize audio context for additional VAD
     */
    async initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.log('Audio context initialized');
        } catch (error) {
            this.log(`Audio context initialization failed: ${error.message}`, 'error');
            // Not critical for hybrid mode
        }
    }
    
    /**
     * Start hybrid session
     */
    async startSession(sessionId, persona = 'friendly') {
        if (!this.isInitialized || !this.isConnected) {
            throw new Error('System not initialized or not connected');
        }
        
        this.currentSessionId = sessionId;
        this.isRecording = true;
        this.speechStartTime = Date.now();
        this.finalTranscript = '';
        this.currentTranscript = '';
        
        // Start audio session with backend
        this.socket.emit('start_audio_session', {
            session_id: sessionId,
            persona: persona
        });
        
        // Start speech recognition for real-time transcription
        try {
            this.speechRecognition.start();
            this.log(`Hybrid session started: ${sessionId}`);
        } catch (error) {
            this.log(`Failed to start speech recognition: ${error.message}`, 'error');
            throw error;
        }
    }
    
    /**
     * Stop hybrid session
     */
    async stopSession() {
        if (!this.currentSessionId) {
            return;
        }
        
        this.isRecording = false;
        
        // Send any remaining transcript
        if (this.finalTranscript.trim()) {
            this.sendTranscriptToAI();
        }
        
        // Stop speech recognition
        if (this.speechRecognition) {
            this.speechRecognition.stop();
        }
        
        // Clear timers
        if (this.pauseTimer) {
            clearTimeout(this.pauseTimer);
            this.pauseTimer = null;
        }
        
        // Stop backend session
        this.socket.emit('stop_audio_session', {
            session_id: this.currentSessionId
        });
        
        this.log(`Hybrid session stopped: ${this.currentSessionId}`);
        this.currentSessionId = null;
    }
    
    /**
     * Set callbacks for events
     */
    setCallbacks(callbacks) {
        this.callbacks = { ...this.callbacks, ...callbacks };
    }
    
    /**
     * Get current status
     */
    getStatus() {
        return {
            initialized: this.isInitialized,
            connected: this.isConnected,
            recording: this.isRecording,
            sessionId: this.currentSessionId,
            currentTranscript: this.currentTranscript,
            finalTranscript: this.finalTranscript
        };
    }
    
    /**
     * Cleanup resources
     */
    cleanup() {
        this.stopSession();
        
        if (this.socket) {
            this.socket.disconnect();
        }
        
        if (this.audioContext) {
            this.audioContext.close();
        }
        
        this.log('Hybrid VAD cleaned up');
    }
}

// Make it globally available
window.HybridVoiceActivityDetection = HybridVoiceActivityDetection;