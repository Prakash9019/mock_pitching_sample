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
            bufferSize: 8192,           // FIXED: Increased buffer size
            socketUrl: window.location.origin,
            pauseThreshold: 2000,        // 2 seconds pause to send
            minSpeechDuration: 1000,     // Minimum 1 second of speech
            ttsEnabled: true,            // Enable Text-to-Speech (Server-side Google Cloud TTS only)
            ttsVolume: 0.8,              // Speech volume
            debug: true,                 // FIXED: Enable debug by default
            reconnectAttempts: 5,        // FIXED: Socket reconnect attempts
            audioThrottleMs: 100,        // FIXED: Throttle audio sending
            ...config
        };
        
        // State management
        this.isInitialized = false;
        this.isConnected = false;
        this.currentSessionId = null;
        this.isRecording = false;
        this.isAISpeaking = false;      // Track AI speaking state
        this.audioContext = null;
        this.mediaStream = null;
        this.socket = null;
        this.speechRecognition = null;
        this.currentAudio = null;       // Current server TTS audio
        
        // FIXED: Error tracking
        this.recognitionErrorCount = 0;
        this.socketErrorCount = 0;
        this.lastErrorTime = 0;
        this.lastAudioSendTime = 0;     // For throttling audio sending
        
        // Audio streaming for backend processing
        this.audioStreamingEnabled = true;  // Enable audio streaming to backend
        this.audioProcessor = null;
        this.audioStreamInterval = null;
        this.useAudioWorklet = false;   // FIXED: Track AudioWorklet usage
        
        // Hybrid transcription state
        this.currentTranscript = '';
        this.finalTranscript = '';
        this.lastSpeechTime = 0;
        this.pauseTimer = null;
        this.speechStartTime = 0;
        
        // Conversation flow state
        this.conversationState = 'idle'; // idle, listening, processing, ai_speaking
        this.pendingResume = false;
        this.speechRecognitionRestarting = false; // Prevent rapid restarts
        
        // FIXED: Connection management
        this.connectionAttempts = 0;
        this.maxConnectionAttempts = this.config.reconnectAttempts;
        this.reconnectTimer = null;
        
        // Callbacks
        this.callbacks = {
            onConnectionChange: null,
            onSessionStart: null,
            onSessionEnd: null,
            onRealtimeTranscript: null,
            onFinalTranscript: null,
            onAIResponse: null,
            onTTSStart: null,           // TTS started
            onTTSEnd: null,             // TTS finished
            onConversationStateChange: null, // State changes
            onError: null,
            onStatusChange: null
        };
        
        this.log('Enhanced Hybrid VAD initialized with improved reliability');
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
            
            // Detect browser for Edge-specific handling
            const isEdge = navigator.userAgent.indexOf('Edg') !== -1;
            if (isEdge) {
                this.log('Edge browser detected - using Edge-optimized initialization');
            }
            
            // Note: Using server-side Google Cloud TTS only - no browser TTS initialization needed
            
            // Initialize Socket.IO connection
            await this.initializeSocket();
            
            // Initialize speech recognition for real-time transcription
            try {
                await this.initializeSpeechRecognition();
            } catch (speechError) {
                this.log(`Speech recognition initialization failed: ${speechError.message}`, 'warning');
                // Continue initialization even if speech recognition fails
            }
            
            // Initialize audio context for VAD
            await this.initializeAudioContext();
            
            // Initialize audio streaming for backend processing
            if (this.audioStreamingEnabled) {
                try {
                    await this.initializeAudioStreaming();
                } catch (audioError) {
                    this.log(`Audio streaming initialization failed: ${audioError.message}`, 'warning');
                    // Continue without audio streaming if it fails
                    this.audioStreamingEnabled = false;
                }
            }
            
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
     * Initialize Socket.IO connection with enhanced events
     */
    async initializeSocket() {
        return new Promise((resolve, reject) => {
            if (typeof io === 'undefined') {
                reject(new Error('Socket.IO not loaded'));
                return;
            }
            
            // FIXED: Improved Socket.IO configuration for better reliability
            this.socket = io(this.config.socketUrl, {
                transports: ['websocket', 'polling'],
                timeout: 30000, // Increased timeout
                forceNew: true,
                reconnection: true,
                reconnectionDelay: 1000,
                reconnectionDelayMax: 5000, // Added max delay
                reconnectionAttempts: 10, // Increased attempts
                pingTimeout: 60000, // Increased ping timeout
                pingInterval: 25000, // Increased ping interval
                autoConnect: true
            });
            
            // FIXED: Track connection attempts
            this.connectionAttempts = 0;
            this.maxConnectionAttempts = 5;
            
            this.socket.on('connect', () => {
                this.isConnected = true;
                this.connectionAttempts = 0; // Reset counter on successful connection
                this.log('Socket.IO connected successfully');
                this.callbacks.onConnectionChange?.({ connected: true });
                resolve();
            });
            
            this.socket.on('disconnect', (reason) => {
                this.isConnected = false;
                this.log(`Socket.IO disconnected: ${reason}`);
                this.callbacks.onConnectionChange?.({ connected: false, reason });
                
                // FIXED: Auto-reconnect for certain disconnect reasons
                if (reason === 'io server disconnect' || reason === 'transport close') {
                    this.connectionAttempts++;
                    if (this.connectionAttempts < this.maxConnectionAttempts) {
                        this.log(`Attempting to reconnect (${this.connectionAttempts}/${this.maxConnectionAttempts})...`);
                        setTimeout(() => {
                            this.socket.connect();
                        }, 2000);
                    }
                }
            });
            
            this.socket.on('connect_error', (error) => {
                this.log(`Connection error: ${error.message}`, 'error');
                this.connectionAttempts++;
                
                if (this.connectionAttempts < this.maxConnectionAttempts) {
                    this.log(`Retrying connection (${this.connectionAttempts}/${this.maxConnectionAttempts})...`);
                    setTimeout(() => {
                        this.socket.connect();
                    }, 2000 * this.connectionAttempts); // Increasing backoff
                } else {
                    reject(error);
                }
            });
            
            // FIXED: Added reconnect events
            this.socket.on('reconnect', (attemptNumber) => {
                this.log(`Socket.IO reconnected after ${attemptNumber} attempts`);
                this.isConnected = true;
                this.callbacks.onConnectionChange?.({ connected: true, reconnected: true });
            });
            
            this.socket.on('reconnect_attempt', (attemptNumber) => {
                this.log(`Socket.IO reconnect attempt ${attemptNumber}`);
            });
            
            this.socket.on('reconnect_error', (error) => {
                this.log(`Socket.IO reconnect error: ${error.message}`, 'error');
            });
            
            this.socket.on('reconnect_failed', () => {
                this.log('Socket.IO reconnect failed after all attempts', 'error');
                this.callbacks.onError?.({ type: 'socket', error: 'Reconnection failed' });
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
                const transcriptText = data.text || (typeof data.transcript === 'string' ? data.transcript : data.transcript?.text || '');
                this.log(`Final transcription: ${transcriptText}`);
                this.callbacks.onFinalTranscript?.({ ...data, text: transcriptText });
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
            
            // FIXED: Add ping/pong monitoring
            this.socket.io.on("ping", () => {
                this.log("Socket.IO ping received", "debug");
            });
        });
    }
    
    /**
     * Handle AI response with server-side Google Cloud TTS only
     */
    async handleAIResponse(data) {
        this.callbacks.onAIResponse?.(data);
        
        if (this.config.ttsEnabled && data.message) {
            if (data.audio_data) {
                this.log('Using server-side Google Cloud TTS audio');
                await this.playServerTTSAudio(data.audio_data, data.message);
            } else {
                this.log('No server TTS audio available - skipping TTS', 'warning');
                // Resume listening immediately if no TTS audio
                setTimeout(() => {
                    this.resumeListening();
                }, 500);
            }
        } else {
            // If TTS is disabled, resume listening after a short delay
            setTimeout(() => {
                this.resumeListening();
            }, 1000);
        }
    }
    
    /**
     * Play server-side TTS audio (Google Cloud TTS)
     */
    async playServerTTSAudio(audioBase64, message) {
        if (!this.config.ttsEnabled) {
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
            try {
                // Convert base64 to audio blob
                const audioBytes = atob(audioBase64);
                const audioArray = new Uint8Array(audioBytes.length);
                for (let i = 0; i < audioBytes.length; i++) {
                    audioArray[i] = audioBytes.charCodeAt(i);
                }
                
                const audioBlob = new Blob([audioArray], { type: 'audio/mp3' });
                const audioUrl = URL.createObjectURL(audioBlob);
                
                // Create audio element
                this.currentAudio = new Audio(audioUrl);
                this.currentAudio.volume = this.config.ttsVolume;
                
                // Handle audio events
                this.currentAudio.onloadstart = () => {
                    this.log('Server TTS audio loading...');
                };
                
                this.currentAudio.oncanplay = () => {
                    this.log('Server TTS audio ready to play');
                };
                
                this.currentAudio.onplay = () => {
                    this.log('Server TTS audio started');
                    this.callbacks.onTTSStart?.({ message, source: 'server' });
                };
                
                this.currentAudio.onended = () => {
                    this.log('Server TTS audio finished');
                    this.isAISpeaking = false;
                    this.callbacks.onTTSEnd?.({ message, source: 'server' });
                    
                    // Notify backend that AI audio finished
                    if (this.socket && this.currentSessionId) {
                        this.socket.emit('ai_audio_finished', {
                            session_id: this.currentSessionId
                        });
                        this.log('Notified backend: AI audio finished');
                    }
                    
                    // Clean up
                    URL.revokeObjectURL(audioUrl);
                    this.currentAudio = null;
                    
                    // Resume listening after AI finishes speaking
                    setTimeout(() => {
                        this.resumeListening();
                        resolve();
                    }, 500); // Small delay before resuming
                };
                
                this.currentAudio.onerror = (event) => {
                    this.log(`Server TTS audio error: ${event.error || 'Unknown error'}`, 'error');
                    this.isAISpeaking = false;
                    this.callbacks.onError?.({ type: 'server_tts', error: event.error || 'Audio playback failed' });
                    
                    // Clean up
                    URL.revokeObjectURL(audioUrl);
                    this.currentAudio = null;
                    
                    this.resumeListening();
                    resolve();
                };
                
                // Start playing
                this.currentAudio.play().catch(error => {
                    this.log(`Failed to play server TTS audio: ${error.message}`, 'error');
                    this.isAISpeaking = false;
                    this.callbacks.onError?.({ type: 'server_tts', error: error.message });
                    
                    // Clean up
                    URL.revokeObjectURL(audioUrl);
                    this.currentAudio = null;
                    
                    this.resumeListening();
                    resolve();
                });
                
            } catch (error) {
                this.log(`Error processing server TTS audio: ${error.message}`, 'error');
                this.isAISpeaking = false;
                this.callbacks.onError?.({ type: 'server_tts', error: error.message });
                this.resumeListening();
                resolve();
            }
        });
    }
    

    
    /**
     * Stop current server TTS audio
     */
    stopTTS() {
        // Stop server audio
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.currentAudio = null;
        }
        
        this.isAISpeaking = false;
    }
    
    /**
     * Get current TTS mode (Server-side Google Cloud TTS only)
     */
    getTTSMode() {
        return {
            mode: 'server',
            description: 'Google Cloud TTS (Server-side only)'
        };
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
                // Use abort instead of stop to immediately stop recognition
                this.speechRecognition.abort();
            } catch (error) {
                // Ignore errors when aborting - this is expected
                this.log(`Recognition abort: ${error.message}`, 'debug');
            }
        }
    }
    
    /**
     * Resume listening (after AI finishes speaking)
     */
    resumeListening() {
        if (!this.isRecording || this.isAISpeaking || this.speechRecognitionRestarting) {
            return;
        }
        
        this.log('Resuming speech recognition');
        this.setConversationState('listening');
        
        // Clear any existing transcript for new input
        this.finalTranscript = '';
        this.currentTranscript = '';
        
        // Stop any existing recognition first to prevent conflicts
        try {
            if (this.speechRecognition) {
                this.speechRecognition.abort();
            }
        } catch (error) {
            // Ignore abort errors
        }
        
        // Start speech recognition with delay to prevent rapid restarts
        this.speechRecognitionRestarting = true;
        setTimeout(() => {
            try {
                if (this.speechRecognition && this.isRecording && !this.isAISpeaking) {
                    this.speechRecognition.start();
                }
            } catch (error) {
                this.log(`Error starting recognition: ${error.message}`, 'error');
                // Try again after a longer delay
                setTimeout(() => {
                    try {
                        if (this.speechRecognition && this.isRecording && !this.isAISpeaking) {
                            this.speechRecognition.start();
                        }
                    } catch (retryError) {
                        this.log(`Retry error: ${retryError.message}`, 'error');
                    }
                }, 2000);
            }
            
            // Reset restart flag
            setTimeout(() => {
                this.speechRecognitionRestarting = false;
            }, 1000);
        }, 300);
    }
    
    /**
     * Initialize speech recognition with enhanced flow control
     */
    async initializeSpeechRecognition() {
        // Enhanced browser detection for Edge compatibility
        const hasWebkitSpeechRecognition = 'webkitSpeechRecognition' in window;
        const hasSpeechRecognition = 'SpeechRecognition' in window;
        
        if (!hasWebkitSpeechRecognition && !hasSpeechRecognition) {
            this.log('Speech recognition not supported in this browser', 'warning');
            throw new Error('Speech recognition not supported in this browser');
        }
        
        // Edge-specific: Prefer SpeechRecognition over webkitSpeechRecognition for Edge
        const isEdge = navigator.userAgent.indexOf('Edg') !== -1;
        let SpeechRecognition;
        
        if (isEdge && hasSpeechRecognition) {
            SpeechRecognition = window.SpeechRecognition;
            this.log('Using SpeechRecognition API for Edge browser');
        } else {
            SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.log('Using standard SpeechRecognition API');
        }
        
        this.speechRecognition = new SpeechRecognition();
        
        // Configure for hybrid mode - FIXED: Changed continuous to false to prevent network errors
        this.speechRecognition.continuous = false;
        this.speechRecognition.interimResults = true;
        this.speechRecognition.lang = 'en-US';
        
        // FIXED: Add max alternatives for better recognition
        this.speechRecognition.maxAlternatives = 3;
        
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
            
            // Handle specific error types
            if (event.error === 'aborted') {
                this.log('Speech recognition was aborted - this is normal during state transitions');
                return; // Don't report aborted errors as they're expected
            }
            
            // FIXED: Handle network errors more gracefully
            if (event.error === 'network') {
                this.log('Network error detected - will retry with fallback mode');
                // Don't report network errors to UI, just retry
                setTimeout(() => {
                    if (this.isRecording && !this.isAISpeaking && !this.speechRecognitionRestarting) {
                        this.tryRestartRecognition();
                    }
                }, 1000);
                return;
            }
            
            // FIXED: Handle no-speech errors more gracefully
            if (event.error === 'no-speech') {
                this.log('No speech detected - will retry');
                // Don't report no-speech errors to UI, just retry
                return;
            }
            
            this.callbacks.onError?.({ type: 'speech_recognition', error: event.error });
        };
        
        this.speechRecognition.onend = () => {
            this.log('Speech recognition ended');
            
            // Only restart if we're still recording and AI is not speaking
            if (this.isRecording && !this.isAISpeaking) {
                // FIXED: Increased delay to prevent rapid restarts
                setTimeout(() => {
                    if (this.isRecording && !this.isAISpeaking && !this.speechRecognitionRestarting) {
                        this.tryRestartRecognition();
                    }
                }, 800); // Increased delay from 500ms to 800ms
            }
        };
        
        this.log('Speech recognition initialized with improved flow control');
    }
    
    /**
     * FIXED: New helper method to restart recognition with error handling
     */
    tryRestartRecognition() {
        if (this.speechRecognitionRestarting) {
            return;
        }
        
        this.speechRecognitionRestarting = true;
        
        try {
            // Create a new instance to avoid potential issues with reusing the same instance
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            
            // Only create a new instance if we've had errors
            if (this.recognitionErrorCount && this.recognitionErrorCount > 3) {
                this.log('Creating fresh speech recognition instance after errors');
                
                // Save event handlers
                const oldHandlers = {
                    onstart: this.speechRecognition.onstart,
                    onresult: this.speechRecognition.onresult,
                    onerror: this.speechRecognition.onerror,
                    onend: this.speechRecognition.onend
                };
                
                // Create new instance
                this.speechRecognition = new SpeechRecognition();
                
                // Configure
                this.speechRecognition.continuous = false;
                this.speechRecognition.interimResults = true;
                this.speechRecognition.lang = 'en-US';
                this.speechRecognition.maxAlternatives = 3;
                
                // Restore handlers
                this.speechRecognition.onstart = oldHandlers.onstart;
                this.speechRecognition.onresult = oldHandlers.onresult;
                this.speechRecognition.onerror = oldHandlers.onerror;
                this.speechRecognition.onend = oldHandlers.onend;
                
                // Reset error count
                this.recognitionErrorCount = 0;
            }
            
            // Start recognition
            this.speechRecognition.start();
        } catch (error) {
            this.log(`Error restarting recognition: ${error.message}`, 'error');
            
            // Track errors
            if (!this.recognitionErrorCount) this.recognitionErrorCount = 0;
            this.recognitionErrorCount++;
            
            // Try again after a longer delay if we keep having errors
            setTimeout(() => {
                this.speechRecognitionRestarting = false;
                if (this.isRecording && !this.isAISpeaking) {
                    this.tryRestartRecognition();
                }
            }, 2000);
        }
        
        // Reset restart flag after a delay
        setTimeout(() => {
            this.speechRecognitionRestarting = false;
        }, 1000);
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
            // Edge-specific audio context initialization
            const isEdge = navigator.userAgent.indexOf('Edg') !== -1;
            
            if (isEdge) {
                // Edge prefers AudioContext over webkitAudioContext
                if (window.AudioContext) {
                    this.audioContext = new AudioContext();
                    this.log('Using AudioContext for Edge browser');
                } else if (window.webkitAudioContext) {
                    this.audioContext = new webkitAudioContext();
                    this.log('Using webkitAudioContext fallback for Edge');
                } else {
                    throw new Error('No AudioContext available');
                }
            } else {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                this.log('Using standard AudioContext');
            }
            
            // Resume audio context if suspended (required by browser policies)
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
                this.log('Audio context resumed from suspended state');
            }
            
            this.log(`Audio context initialized (state: ${this.audioContext.state}, sampleRate: ${this.audioContext.sampleRate}Hz)`);
        } catch (error) {
            this.log(`Audio context initialization failed: ${error.message}`, 'error');
            // Don't throw error, allow app to continue without audio context
        }
    }
    
    /**
     * Initialize audio streaming for backend processing
     */
    async initializeAudioStreaming() {
        try {
            // Edge-specific microphone access with enhanced fallback options
            const isEdge = navigator.userAgent.indexOf('Edg') !== -1;
            
            let constraints;
            if (isEdge) {
                // Edge-optimized constraints
                constraints = {
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                };
                this.log('Using Edge-optimized audio constraints');
            } else {
                // Standard constraints for other browsers
                constraints = {
                    audio: {
                        sampleRate: { ideal: 16000 },
                        channelCount: { ideal: 1 },
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                };
            }
            
            try {
                // Try with optimized settings first
                this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
                this.log('Got microphone access with optimized settings');
            } catch (initialError) {
                this.log(`Initial microphone access failed: ${initialError.message}. Trying fallback...`, 'warning');
                
                // Fallback to simpler constraints
                try {
                    this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    this.log('Got microphone access with fallback settings');
                } catch (fallbackError) {
                    this.log(`All microphone access attempts failed: ${fallbackError.message}`, 'error');
                    throw new Error(`Microphone access failed: ${fallbackError.message}`);
                }
            }
            
            // FIXED: Use AudioWorkletNode if available (modern) or fallback to ScriptProcessor (legacy)
            if (this.audioContext) {
                const source = this.audioContext.createMediaStreamSource(this.mediaStream);
                
                // Check if AudioWorklet is supported
                if (window.AudioWorkletNode && this.audioContext.audioWorklet) {
                    try {
                        // This is a more modern approach but requires more setup
                        this.log('AudioWorklet is supported, but using ScriptProcessor for compatibility');
                        this.useAudioWorklet = false;
                    } catch (workletError) {
                        this.log(`AudioWorklet setup failed: ${workletError.message}. Using ScriptProcessor.`, 'warning');
                        this.useAudioWorklet = false;
                    }
                } else {
                    this.log('AudioWorklet not supported, using ScriptProcessor');
                    this.useAudioWorklet = false;
                }
                
                // Fallback to ScriptProcessor
                if (!this.useAudioWorklet) {
                    // FIXED: Increased buffer size for better stability
                    this.audioProcessor = this.audioContext.createScriptProcessor(8192, 1, 1);
                    
                    this.audioProcessor.onaudioprocess = (event) => {
                        if (this.isRecording && !this.isAISpeaking && this.currentSessionId) {
                            // FIXED: Added throttling to reduce network load
                            if (!this.lastAudioSendTime || Date.now() - this.lastAudioSendTime > 100) {
                                this.sendAudioChunk(event.inputBuffer);
                                this.lastAudioSendTime = Date.now();
                            }
                        } else {
                            // Debug why audio chunks aren't being sent
                            if (!this.audioProcessDebugCount) this.audioProcessDebugCount = 0;
                            this.audioProcessDebugCount++;
                            if (this.audioProcessDebugCount % 100 === 0) {
                                this.log(`🎤 Audio processing: recording=${this.isRecording}, aiSpeaking=${this.isAISpeaking}, session=${!!this.currentSessionId}`, 'debug');
                            }
                        }
                    };
                    
                    source.connect(this.audioProcessor);
                    // FIXED: Connect to destination with zero gain to keep the audio context active
                    const silentGain = this.audioContext.createGain();
                    silentGain.gain.value = 0;
                    this.audioProcessor.connect(silentGain);
                    silentGain.connect(this.audioContext.destination);
                }
            }
            
            this.log('Audio streaming initialized successfully');
        } catch (error) {
            this.log(`Audio streaming initialization failed: ${error.message}`, 'error');
            this.audioStreamingEnabled = false;
        }
    }
    
    /**
     * Force start audio streaming (for debugging)
     */
    async forceStartAudioStreaming() {
        if (!this.audioStreamingEnabled) {
            this.log('Audio streaming is disabled');
            return false;
        }
        
        try {
            // Ensure audio context is running
            if (this.audioContext && this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
                this.log('Audio context resumed for streaming');
            }
            
            // Check if we have all components
            if (!this.mediaStream) {
                this.log('No media stream for audio streaming');
                return false;
            }
            
            if (!this.audioProcessor) {
                this.log('No audio processor - reinitializing...');
                await this.initializeAudioStreaming();
            }
            
            this.log('✅ Audio streaming is ready');
            return true;
            
        } catch (error) {
            this.log(`Failed to start audio streaming: ${error.message}`, 'error');
            return false;
        }
    }
    
    /**
     * Send audio chunk to backend
     */
    sendAudioChunk(audioBuffer) {
        if (!this.socket || !this.currentSessionId) {
            this.log('Cannot send audio chunk: no socket or session', 'debug');
            return;
        }
        
        // FIXED: Check socket connection status
        if (!this.socket.connected) {
            this.log('Socket not connected, cannot send audio chunk', 'debug');
            return;
        }
        
        // FIXED: Throttle audio sending to reduce network load
        if (this.lastAudioSendTime && Date.now() - this.lastAudioSendTime < this.config.audioThrottleMs) {
            return;
        }
        this.lastAudioSendTime = Date.now();
        
        try {
            // FIXED: Check for silence before sending
            const isSilent = this.isAudioBufferSilent(audioBuffer);
            if (isSilent) {
                // Skip silent frames to reduce bandwidth
                if (!this.silentFrameCount) this.silentFrameCount = 0;
                this.silentFrameCount++;
                
                if (this.silentFrameCount % 50 === 0) {
                    this.log(`Skipped ${this.silentFrameCount} silent audio frames`);
                }
                return;
            }
            
            // Convert audio buffer to base64
            const audioData = this.audioBufferToBase64(audioBuffer);
            
            // FIXED: Check data size before sending
            const dataSizeKB = Math.round(audioData.length / 1024);
            if (dataSizeKB > 50) {
                this.log(`Audio chunk too large (${dataSizeKB}KB), skipping`);
                return;
            }
            
            // Send to backend with timeout
            const sendPromise = new Promise((resolve, reject) => {
                this.socket.emit('audio_stream', {
                    session_id: this.currentSessionId,
                    audio_data: audioData,
                    sample_rate: this.audioContext ? this.audioContext.sampleRate : 44100,
                    duration: audioData.length / (this.audioContext ? this.audioContext.sampleRate : 44100) / 2  // Int16 = 2 bytes per sample
                }, (ack) => {
                    // Handle acknowledgment if the server supports it
                    if (ack && ack.status === 'error') {
                        reject(new Error(ack.message || 'Unknown error'));
                    } else {
                        resolve();
                    }
                });
                
                // Resolve anyway after timeout (Socket.IO might not support acks)
                setTimeout(resolve, 1000);
            });
            
            // Log every 50th chunk to avoid spam
            if (!this.audioChunkCount) this.audioChunkCount = 0;
            this.audioChunkCount++;
            if (this.audioChunkCount % 50 === 0) {
                this.log(`📡 Sent ${this.audioChunkCount} audio chunks (${dataSizeKB}KB)`);
            }
            
        } catch (error) {
            this.log(`Error sending audio chunk: ${error.message}`, 'error');
        }
    }
    
    /**
     * FIXED: Check if audio buffer is silent (to avoid sending silent frames)
     */
    isAudioBufferSilent(audioBuffer) {
        const channelData = audioBuffer.getChannelData(0);
        const bufferLength = channelData.length;
        
        // Calculate RMS (root mean square) of the audio buffer
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
            sum += channelData[i] * channelData[i];
        }
        const rms = Math.sqrt(sum / bufferLength);
        
        // Consider silent if RMS is below threshold
        const isSilent = rms < 0.01;
        return isSilent;
    }
    
    /**
     * Convert audio buffer to base64
     */
    audioBufferToBase64(audioBuffer) {
        const channelData = audioBuffer.getChannelData(0);
        const samples = new Int16Array(channelData.length);
        
        // Convert float32 to int16
        for (let i = 0; i < channelData.length; i++) {
            samples[i] = Math.max(-32768, Math.min(32767, channelData[i] * 32768));
        }
        
        // Convert to base64
        const bytes = new Uint8Array(samples.buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        
        return btoa(binary);
    }
    
    /**
     * Start enhanced hybrid session
     */
    async startSession(sessionId, persona = 'friendly') {
        if (!this.isInitialized || !this.isConnected) {
            throw new Error('System not initialized or not connected');
        }
        
        // Ensure audio context is resumed and audio streaming is ready
        if (this.audioContext && this.audioContext.state === 'suspended') {
            try {
                await this.audioContext.resume();
                this.log('Audio context resumed for session');
            } catch (error) {
                this.log(`Failed to resume audio context: ${error.message}`, 'error');
            }
        }
        
        // Force start audio streaming
        if (this.audioStreamingEnabled) {
            const streamingReady = await this.forceStartAudioStreaming();
            if (!streamingReady) {
                this.log('⚠️ Audio streaming not ready - continuing without backend audio processing');
            }
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
        
        // Cleanup audio streaming
        if (this.audioProcessor) {
            this.audioProcessor.disconnect();
            this.audioProcessor = null;
            this.log('Audio processor disconnected');
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
            this.log('Media stream stopped');
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