/**
 * Voice Activity Detection Frontend
 * Handles real-time audio streaming and automatic speech detection
 * Compatible with both vanilla JS and React
 */

class VoiceActivityDetection {
    constructor(options = {}) {
        // Configuration
        this.config = {
            sampleRate: options.sampleRate || 16000,
            bufferSize: options.bufferSize || 4096,
            socketUrl: options.socketUrl || window.location.origin,
            debug: options.debug || false,
            ...options
        };
        
        // State
        this.isRecording = false;
        this.isConnected = false;
        this.sessionId = null;
        this.socket = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        
        // Callbacks
        this.callbacks = {
            onConnectionChange: null,
            onSessionStart: null,
            onSessionEnd: null,
            onSpeechStart: null,
            onSpeechEnd: null,
            onTranscription: null,
            onAIResponse: null,
            onError: null,
            onVADStatus: null
        };
        
        this.log('VAD initialized with config:', this.config);
    }
    
    /**
     * Set callback functions
     */
    setCallbacks(callbacks) {
        this.callbacks = { ...this.callbacks, ...callbacks };
    }
    
    /**
     * Initialize the VAD system
     */
    async initialize() {
        try {
            // Initialize Socket.IO connection
            await this.initializeSocket();
            
            // Initialize Web Audio API
            await this.initializeAudio();
            
            this.log('VAD system initialized successfully');
            return true;
            
        } catch (error) {
            this.error('Failed to initialize VAD system:', error);
            this.triggerCallback('onError', { type: 'initialization', error });
            return false;
        }
    }
    
    /**
     * Initialize Socket.IO connection
     */
    async initializeSocket() {
        return new Promise((resolve, reject) => {
            // Check if Socket.IO is loaded
            if (typeof io === 'undefined') {
                reject(new Error('Socket.IO not loaded. Make sure to include the Socket.IO script before this script.'));
                return;
            }
            
            this.connectSocket(resolve, reject);
        });
    }
    
    /**
     * Connect to Socket.IO server
     */
    connectSocket(resolve, reject) {
        try {
            this.socket = io(this.config.socketUrl, {
                transports: ['websocket', 'polling'],
                timeout: 20000,
                forceNew: true,
                reconnection: true,
                reconnectionDelay: 1000,
                reconnectionAttempts: 5,
                maxReconnectionAttempts: 5
            });
            
            this.socket.on('connect', () => {
                this.isConnected = true;
                this.log('Socket connected');
                this.triggerCallback('onConnectionChange', { connected: true });
                resolve();
            });
            
            this.socket.on('disconnect', () => {
                this.isConnected = false;
                this.log('Socket disconnected');
                this.triggerCallback('onConnectionChange', { connected: false });
            });
            
            this.socket.on('audio_session_started', (data) => {
                this.log('Audio session started:', data);
                this.triggerCallback('onSessionStart', data);
            });
            
            this.socket.on('audio_session_stopped', (data) => {
                this.log('Audio session stopped:', data);
                this.triggerCallback('onSessionEnd', data);
            });
            
            this.socket.on('vad_status', (data) => {
                this.log('VAD status:', data);
                this.triggerCallback('onVADStatus', data);
                
                if (data.action === 'speech_started') {
                    this.triggerCallback('onSpeechStart', data);
                } else if (data.action === 'speech_ended') {
                    this.triggerCallback('onSpeechEnd', data);
                }
            });
            
            this.socket.on('transcription_result', (data) => {
                this.log('Transcription result:', data);
                this.triggerCallback('onTranscription', data);
            });
            
            this.socket.on('ai_response', (data) => {
                this.log('AI response:', data);
                this.triggerCallback('onAIResponse', data);
            });
            
            this.socket.on('audio_error', (data) => {
                this.error('Audio error:', data);
                this.triggerCallback('onError', { type: 'audio', ...data });
            });
            
            this.socket.on('transcription_error', (data) => {
                this.error('Transcription error:', data);
                this.triggerCallback('onError', { type: 'transcription', ...data });
            });
            
            this.socket.on('ai_error', (data) => {
                this.error('AI error:', data);
                this.triggerCallback('onError', { type: 'ai', ...data });
            });
            
            // Handle connection errors
            this.socket.on('connect_error', (error) => {
                this.error('Socket connection error:', error);
                reject(error);
            });
            
        } catch (error) {
            reject(error);
        }
    }
    
    /**
     * Initialize Web Audio API
     */
    async initializeAudio() {
        try {
            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.config.sampleRate,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.config.sampleRate
            });
            
            // Create audio source
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            // Create script processor for audio data
            this.processor = this.audioContext.createScriptProcessor(
                this.config.bufferSize, 1, 1
            );
            
            // Handle audio processing
            this.processor.onaudioprocess = (event) => {
                if (this.isRecording && this.socket && this.isConnected) {
                    this.processAudioData(event.inputBuffer);
                }
            };
            
            // Connect audio nodes
            source.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            
            this.log('Audio system initialized');
            
        } catch (error) {
            throw new Error(`Failed to initialize audio: ${error.message}`);
        }
    }
    
    /**
     * Start a new voice session
     */
    async startSession(sessionId, persona = 'friendly') {
        try {
            if (!this.isConnected) {
                throw new Error('Not connected to server');
            }
            
            this.sessionId = sessionId;
            
            // Start audio session on server
            this.socket.emit('start_audio_session', {
                session_id: sessionId,
                persona: persona
            });
            
            // Start recording
            await this.startRecording();
            
            this.log(`Session started: ${sessionId} with persona: ${persona}`);
            
        } catch (error) {
            this.error('Failed to start session:', error);
            this.triggerCallback('onError', { type: 'session_start', error });
            throw error;
        }
    }
    
    /**
     * Stop the current session
     */
    async stopSession() {
        try {
            if (this.sessionId && this.socket && this.isConnected) {
                this.socket.emit('stop_audio_session', {
                    session_id: this.sessionId
                });
            }
            
            await this.stopRecording();
            this.sessionId = null;
            
            this.log('Session stopped');
            
        } catch (error) {
            this.error('Failed to stop session:', error);
            this.triggerCallback('onError', { type: 'session_stop', error });
        }
    }
    
    /**
     * Start recording audio
     */
    async startRecording() {
        try {
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            
            this.isRecording = true;
            this.log('Recording started');
            
        } catch (error) {
            throw new Error(`Failed to start recording: ${error.message}`);
        }
    }
    
    /**
     * Stop recording audio
     */
    async stopRecording() {
        this.isRecording = false;
        this.log('Recording stopped');
    }
    
    /**
     * Process audio data and send to server
     */
    processAudioData(inputBuffer) {
        try {
            // Get audio data
            const audioData = inputBuffer.getChannelData(0);
            
            // Convert to 16-bit PCM
            const pcmData = this.convertToPCM16(audioData);
            
            // Convert to base64
            const base64Data = this.arrayBufferToBase64(pcmData);
            
            // Send to server
            this.socket.emit('audio_stream', {
                session_id: this.sessionId,
                audio_data: base64Data
            });
            
        } catch (error) {
            this.error('Error processing audio data:', error);
        }
    }
    
    /**
     * Convert float32 audio to 16-bit PCM
     */
    convertToPCM16(float32Array) {
        const buffer = new ArrayBuffer(float32Array.length * 2);
        const view = new DataView(buffer);
        
        for (let i = 0; i < float32Array.length; i++) {
            const sample = Math.max(-1, Math.min(1, float32Array[i]));
            view.setInt16(i * 2, sample * 0x7FFF, true);
        }
        
        return buffer;
    }
    
    /**
     * Convert ArrayBuffer to base64
     */
    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
    
    /**
     * Send manual text (fallback)
     */
    sendManualText(text) {
        if (!this.sessionId || !this.socket || !this.isConnected) {
            this.error('Cannot send manual text: session not active');
            return;
        }
        
        this.socket.emit('manual_transcription', {
            session_id: this.sessionId,
            text: text
        });
        
        this.log('Manual text sent:', text);
    }
    
    /**
     * Clean up resources
     */
    cleanup() {
        try {
            this.stopRecording();
            
            if (this.processor) {
                this.processor.disconnect();
                this.processor = null;
            }
            
            if (this.audioContext) {
                this.audioContext.close();
                this.audioContext = null;
            }
            
            if (this.mediaStream) {
                this.mediaStream.getTracks().forEach(track => track.stop());
                this.mediaStream = null;
            }
            
            if (this.socket) {
                this.socket.disconnect();
                this.socket = null;
            }
            
            this.log('VAD system cleaned up');
            
        } catch (error) {
            this.error('Error during cleanup:', error);
        }
    }
    
    /**
     * Trigger callback function
     */
    triggerCallback(callbackName, data) {
        if (this.callbacks[callbackName] && typeof this.callbacks[callbackName] === 'function') {
            try {
                this.callbacks[callbackName](data);
            } catch (error) {
                this.error(`Error in callback ${callbackName}:`, error);
            }
        }
    }
    
    /**
     * Logging functions
     */
    log(...args) {
        if (this.config.debug) {
            console.log('[VAD]', ...args);
        }
    }
    
    error(...args) {
        console.error('[VAD ERROR]', ...args);
    }
    
    /**
     * Get current status
     */
    getStatus() {
        return {
            isConnected: this.isConnected,
            isRecording: this.isRecording,
            sessionId: this.sessionId,
            hasAudioContext: !!this.audioContext,
            hasMediaStream: !!this.mediaStream
        };
    }
}

// Export for both CommonJS and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceActivityDetection;
} else if (typeof window !== 'undefined') {
    window.VoiceActivityDetection = VoiceActivityDetection;
}

// React Hook version for React applications
if (typeof React !== 'undefined') {
    window.useVoiceActivityDetection = function(options = {}) {
        const [vad, setVad] = React.useState(null);
        const [status, setStatus] = React.useState({
            isConnected: false,
            isRecording: false,
            sessionId: null
        });
        
        React.useEffect(() => {
            const vadInstance = new VoiceActivityDetection({
                debug: true,
                ...options
            });
            
            vadInstance.setCallbacks({
                onConnectionChange: (data) => {
                    setStatus(prev => ({ ...prev, isConnected: data.connected }));
                },
                onSessionStart: (data) => {
                    setStatus(prev => ({ ...prev, sessionId: data.session_id }));
                },
                onSessionEnd: () => {
                    setStatus(prev => ({ ...prev, sessionId: null, isRecording: false }));
                },
                onVADStatus: (data) => {
                    if (data.action === 'speech_started') {
                        setStatus(prev => ({ ...prev, isRecording: true }));
                    } else if (data.action === 'speech_ended') {
                        setStatus(prev => ({ ...prev, isRecording: false }));
                    }
                }
            });
            
            vadInstance.initialize();
            setVad(vadInstance);
            
            return () => {
                vadInstance.cleanup();
            };
        }, []);
        
        return { vad, status };
    };
}