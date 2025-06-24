/**
 * Enhanced Hybrid Voice Activity Detection System
 * Features:
 * - Real-time transcription display
 * - Smart pause detection (2-second)
 * - Text-to-Speech AI responses
 * - Conversation flow control (pause listening during AI speech)
 * - Auto-resume listening after AI finishes
 */

class EnhancedHybridVAD {
    constructor(config = {}) {
        this.config = {
            sampleRate: 16000,
            bufferSize: 4096,
            socketUrl: window.location.origin,
            pauseThreshold: 2000,        // 2 seconds pause to send
            minSpeechDuration: 1000,     // Minimum 1 second of speech
            ttsEnabled: true,            // Enable Text-to-Speech
            ttsVoice: null,              // Will be set to best available voice
            ttsRate: 1.0,                // Speech rate
            ttsPitch: 1.0,               // Speech pitch
            ttsVolume: 0.8,              // Speech volume
            debug: false,
            ...config
        };
        
        // State management
        this.isInitialized = false;
        this.isConnected = false;
        this.currentSessionId = null;
        this.isRecording = false;
        this.isAISpeaking = false;      // New: Track AI speaking state
        this.audioContext = null;
        this.mediaStream = null;
        this.socket = null;
        this.speechRecognition = null;
        this.speechSynthesis = null;    // New: TTS synthesis
        this.currentUtterance = null;   // New: Current TTS utterance
        
        // Hybrid transcription state
        this.currentTranscript = '';
        this.finalTranscript = '';
        this.lastSpeechTime = 0;
        this.pauseTimer = null;
        this.speechStartTime = 0;
        
        // Conversation flow state
        this.conversationState = 'idle'; // idle, listening, processing, ai_speaking
        this.pendingResume = false;
        
        // Callbacks
        this.callbacks = {
            onConnectionChange: null,
            onSessionStart: null,
            onSessionEnd: null,
            onRealtimeTranscript: null,
            onFinalTranscript: null,
            onAIResponse: null,
            onTTSStart: null,           // New: TTS started
            onTTSEnd: null,             // New: TTS finished
            onConversationStateChange: null, // New: State changes
            onError: null,
            onStatusChange: null
        };
        
        this.log('Enhanced Hybrid VAD initialized');
    }
    
    log(message, level = 'info') {
        if (this.config.debug) {
            console.log(`[EnhancedHybridVAD] ${message}`);
        }
    }
    
    /**
     * Initialize the enhanced hybrid system
     */
    async initialize() {
        try {
            this.log('Initializing enhanced hybrid VAD system...');
            
            // Initialize TTS first
            await this.initializeTTS();
            
            // Initialize Socket.IO connection
            await this.initializeSocket();
            
            // Initialize speech recognition for real-time transcription
            await this.initializeSpeechRecognition();
            
            // Initialize audio context for VAD
            await this.initializeAudioContext();
            
            this.isInitialized = true;
            this.log('Enhanced hybrid VAD system initialized successfully');
            return true;
            
        } catch (error) {
            this.log(`Initialization failed: ${error.message}`, 'error');
            this.callbacks.onError?.({ type: 'initialization', error: error.message });
            return false;
        }
    }
    
    /**
     * Initialize Text-to-Speech
     */
    async initializeTTS() {
        if (!('speechSynthesis' in window)) {
            this.log('TTS not supported in this browser', 'warning');
            this.config.ttsEnabled = false;
            return;
        }
        
        this.speechSynthesis = window.speechSynthesis;
        
        // Wait for voices to load
        return new Promise((resolve) => {
            const loadVoices = () => {
                const voices = this.speechSynthesis.getVoices();
                
                if (voices.length > 0) {
                    // Find the best English voice
                    this.config.ttsVoice = voices.find(voice => 
                        voice.lang.startsWith('en') && voice.name.includes('Google')
                    ) || voices.find(voice => 
                        voice.lang.startsWith('en')
                    ) || voices[0];
                    
                    this.log(`TTS initialized with voice: ${this.config.ttsVoice.name}`);
                    resolve();
                } else {
                    // Voices not loaded yet, wait a bit
                    setTimeout(loadVoices, 100);
                }
            };
            
            // Handle voice loading
            if (this.speechSynthesis.onvoiceschanged !== undefined) {
                this.speechSynthesis.onvoiceschanged = loadVoices;
            }
            
            loadVoices();
        });
    }
    
    /**
     * Initialize Socket.IO connection with enhanced events
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
                this.setConversationState('listening');
                this.callbacks.onSessionStart?.(data);
                
                // Speak initial AI message if provided
                if (data.message) {
                    this.handleAIResponse({ message: data.message });
                }
            });
            
            this.socket.on('audio_session_stopped', (data) => {
                this.log(`Session stopped: ${data.session_id}`);
                this.setConversationState('idle');
                this.callbacks.onSessionEnd?.(data);
            });
            
            this.socket.on('transcription_result', (data) => {
                this.log(`Final transcription: ${data.transcript}`);
                this.callbacks.onFinalTranscript?.(data);
            });
            
            // Enhanced AI response handling
            this.socket.on('ai_response', (data) => {
                this.log(`AI response: ${data.message}`);
                this.handleAIResponse(data);
            });
            
            this.socket.on('audio_error', (data) => {
                this.log(`Audio error: ${data.error}`, 'error');
                this.callbacks.onError?.(data);
            });
        });
    }
    
    /**
     * Handle AI response with TTS and conversation flow
     */
    async handleAIResponse(data) {
        this.callbacks.onAIResponse?.(data);
        
        if (this.config.ttsEnabled && data.message) {
            await this.speakAIResponse(data.message);
        } else {
            // If TTS is disabled, resume listening after a short delay
            setTimeout(() => {
                this.resumeListening();
            }, 1000);
        }
    }
    
    /**
     * Speak AI response using TTS
     */
    async speakAIResponse(message) {
        if (!this.config.ttsEnabled || !this.speechSynthesis) {
            return;
        }
        
        // Stop any current speech
        this.stopTTS();
        
        // Set conversation state to AI speaking
        this.setConversationState('ai_speaking');
        this.isAISpeaking = true;
        
        // Pause user speech recognition
        this.pauseListening();
        
        return new Promise((resolve) => {
            this.currentUtterance = new SpeechSynthesisUtterance(message);
            
            // Configure voice
            if (this.config.ttsVoice) {
                this.currentUtterance.voice = this.config.ttsVoice;
            }
            this.currentUtterance.rate = this.config.ttsRate;
            this.currentUtterance.pitch = this.config.ttsPitch;
            this.currentUtterance.volume = this.config.ttsVolume;
            
            // Handle TTS events
            this.currentUtterance.onstart = () => {
                this.log('TTS started');
                this.callbacks.onTTSStart?.({ message });
            };
            
            this.currentUtterance.onend = () => {
                this.log('TTS finished');
                this.isAISpeaking = false;
                this.callbacks.onTTSEnd?.({ message });
                
                // Resume listening after AI finishes speaking
                setTimeout(() => {
                    this.resumeListening();
                    resolve();
                }, 500); // Small delay before resuming
            };
            
            this.currentUtterance.onerror = (event) => {
                this.log(`TTS error: ${event.error}`, 'error');
                this.isAISpeaking = false;
                this.callbacks.onError?.({ type: 'tts', error: event.error });
                this.resumeListening();
                resolve();
            };
            
            // Start speaking
            this.speechSynthesis.speak(this.currentUtterance);
        });
    }
    
    /**
     * Stop current TTS
     */
    stopTTS() {
        if (this.speechSynthesis && this.speechSynthesis.speaking) {
            this.speechSynthesis.cancel();
        }
        this.currentUtterance = null;
        this.isAISpeaking = false;
    }
    
    /**
     * Set conversation state and notify callbacks
     */
    setConversationState(newState) {
        const oldState = this.conversationState;
        this.conversationState = newState;
        
        this.log(`Conversation state: ${oldState} → ${newState}`);
        this.callbacks.onConversationStateChange?.({ 
            oldState, 
            newState, 
            isAISpeaking: this.isAISpeaking 
        });
    }
    
    /**
     * Pause listening (when AI is speaking)
     */
    pauseListening() {
        if (this.speechRecognition && this.isRecording) {
            this.log('Pausing speech recognition (AI speaking)');
            try {
                this.speechRecognition.stop();
            } catch (error) {
                this.log(`Error stopping recognition: ${error.message}`, 'error');
            }
        }
    }
    
    /**
     * Resume listening (after AI finishes speaking)
     */
    resumeListening() {
        if (!this.isRecording || this.isAISpeaking) {
            return;
        }
        
        this.log('Resuming speech recognition');
        this.setConversationState('listening');
        
        // Clear any existing transcript for new input
        this.finalTranscript = '';
        this.currentTranscript = '';
        
        // Restart speech recognition
        try {
            if (this.speechRecognition) {
                this.speechRecognition.start();
            }
        } catch (error) {
            this.log(`Error restarting recognition: ${error.message}`, 'error');
            // Try again after a short delay
            setTimeout(() => {
                try {
                    if (this.speechRecognition && this.isRecording) {
                        this.speechRecognition.start();
                    }
                } catch (retryError) {
                    this.log(`Retry error: ${retryError.message}`, 'error');
                }
            }, 1000);
        }
    }
    
    /**
     * Initialize speech recognition with enhanced flow control
     */
    async initializeSpeechRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            throw new Error('Speech recognition not supported in this browser');
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.speechRecognition = new SpeechRecognition();
        
        // Configure for hybrid mode
        this.speechRecognition.continuous = true;
        this.speechRecognition.interimResults = true;
        this.speechRecognition.lang = 'en-US';
        
        this.speechRecognition.onstart = () => {
            this.log('Speech recognition started');
            this.callbacks.onStatusChange?.({ type: 'recognition_start' });
        };
        
        this.speechRecognition.onresult = (event) => {
            // Only process if not AI speaking
            if (!this.isAISpeaking) {
                this.handleSpeechResult(event);
            }
        };
        
        this.speechRecognition.onerror = (event) => {
            this.log(`Speech recognition error: ${event.error}`, 'error');
            this.callbacks.onError?.({ type: 'speech_recognition', error: event.error });
        };
        
        this.speechRecognition.onend = () => {
            this.log('Speech recognition ended');
            
            // Only restart if we're still recording and AI is not speaking
            if (this.isRecording && !this.isAISpeaking) {
                setTimeout(() => {
                    if (this.isRecording && !this.isAISpeaking) {
                        try {
                            this.speechRecognition.start();
                        } catch (error) {
                            this.log(`Error restarting recognition: ${error.message}`, 'error');
                        }
                    }
                }, 100);
            }
        };
        
        this.log('Speech recognition initialized with flow control');
    }
    
    /**
     * Handle speech recognition results (enhanced with flow control)
     */
    handleSpeechResult(event) {
        // Don't process if AI is speaking
        if (this.isAISpeaking) {
            return;
        }
        
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
     * Handle pause detection and smart sending to AI (enhanced)
     */
    handlePauseDetection() {
        // Don't process if AI is speaking
        if (this.isAISpeaking) {
            return;
        }
        
        // Clear existing pause timer
        if (this.pauseTimer) {
            clearTimeout(this.pauseTimer);
        }
        
        // Set new pause timer
        this.pauseTimer = setTimeout(() => {
            if (!this.isAISpeaking) {
                this.sendTranscriptToAI();
            }
        }, this.config.pauseThreshold);
        
        this.log(`Pause timer set for ${this.config.pauseThreshold}ms`);
    }
    
    /**
     * Send accumulated transcript to AI (enhanced)
     */
    sendTranscriptToAI() {
        if (!this.finalTranscript.trim() || this.isAISpeaking) {
            this.log('No transcript to send or AI is speaking');
            return;
        }
        
        const speechDuration = Date.now() - this.speechStartTime;
        if (speechDuration < this.config.minSpeechDuration) {
            this.log('Speech too short, not sending');
            return;
        }
        
        this.log(`Sending transcript to AI: "${this.finalTranscript}"`);
        
        // Set state to processing
        this.setConversationState('processing');
        
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
     * Initialize audio context
     */
    async initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.log('Audio context initialized');
        } catch (error) {
            this.log(`Audio context initialization failed: ${error.message}`, 'error');
        }
    }
    
    /**
     * Start enhanced hybrid session
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
        this.isAISpeaking = false;
        
        // Start audio session with backend
        this.socket.emit('start_audio_session', {
            session_id: sessionId,
            persona: persona
        });
        
        // Don't start speech recognition immediately - wait for AI greeting
        this.setConversationState('processing');
        
        this.log(`Enhanced hybrid session started: ${sessionId}`);
    }
    
    /**
     * Stop enhanced hybrid session
     */
    async stopSession() {
        if (!this.currentSessionId) {
            this.log('No active session to stop');
            return;
        }
        
        const sessionToStop = this.currentSessionId;
        this.log(`Stopping session: ${sessionToStop}`);
        
        // Set flags to stop recording
        this.isRecording = false;
        this.isAISpeaking = false;
        
        // Stop TTS immediately
        this.stopTTS();
        
        // Stop speech recognition immediately
        if (this.speechRecognition) {
            try {
                this.speechRecognition.stop();
                this.log('Speech recognition stopped');
            } catch (error) {
                this.log(`Error stopping speech recognition: ${error.message}`, 'error');
            }
        }
        
        // Clear all timers
        if (this.pauseTimer) {
            clearTimeout(this.pauseTimer);
            this.pauseTimer = null;
            this.log('Pause timer cleared');
        }
        
        // Send stop signal to backend
        if (this.socket && this.socket.connected) {
            this.socket.emit('stop_audio_session', {
                session_id: sessionToStop
            });
            this.log(`Stop signal sent to backend for session: ${sessionToStop}`);
        } else {
            this.log('Socket not connected, cannot send stop signal', 'warning');
        }
        
        // Reset conversation state
        this.setConversationState('idle');
        
        // Clear session data
        this.currentSessionId = null;
        this.finalTranscript = '';
        this.currentTranscript = '';
        this.lastSpeechTime = 0;
        this.speechStartTime = 0;
        this.pendingResume = false;
        
        this.log(`Enhanced hybrid session fully stopped: ${sessionToStop}`);
        
        // Notify callbacks
        this.callbacks.onSessionEnd?.({
            session_id: sessionToStop,
            status: 'stopped_by_user'
        });
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
            aiSpeaking: this.isAISpeaking,
            conversationState: this.conversationState,
            sessionId: this.currentSessionId,
            currentTranscript: this.currentTranscript,
            finalTranscript: this.finalTranscript,
            ttsEnabled: this.config.ttsEnabled
        };
    }
    
    /**
     * Toggle TTS on/off
     */
    toggleTTS() {
        this.config.ttsEnabled = !this.config.ttsEnabled;
        this.log(`TTS ${this.config.ttsEnabled ? 'enabled' : 'disabled'}`);
        
        if (!this.config.ttsEnabled) {
            this.stopTTS();
        }
        
        return this.config.ttsEnabled;
    }
    
    /**
     * Disconnect from server and cleanup
     */
    async disconnect() {
        this.log('Disconnecting from server...');
        
        // Stop any active session first
        if (this.currentSessionId) {
            await this.stopSession();
        }
        
        // Disconnect socket
        if (this.socket && this.socket.connected) {
            this.socket.disconnect();
            this.log('Socket disconnected');
        }
        
        // Update connection status
        this.isConnected = false;
        this.callbacks.onConnectionChange?.({ connected: false });
        
        this.log('Disconnected from server');
    }
    
    /**
     * Cleanup resources
     */
    cleanup() {
        this.log('Cleaning up Enhanced Hybrid VAD...');
        
        // Stop session and disconnect
        this.stopSession();
        this.stopTTS();
        
        // Disconnect socket
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
        
        // Close audio context
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        // Clear all state
        this.isInitialized = false;
        this.isConnected = false;
        this.isRecording = false;
        this.isAISpeaking = false;
        this.currentSessionId = null;
        this.speechRecognition = null;
        this.speechSynthesis = null;
        this.currentUtterance = null;
        
        this.log('Enhanced hybrid VAD fully cleaned up');
    }
}

// Make it globally available
window.EnhancedHybridVAD = EnhancedHybridVAD;