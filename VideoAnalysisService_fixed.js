/**
 * Fixed Video Analysis Service
 * This service properly integrates with the backend video analysis system
 */

class VideoAnalysisService {
    constructor(socketService) {
        this.socketService = socketService;
        this.isInitialized = false;
        this.isAnalyzing = false;
        this.videoElement = null;
        this.stream = null;
        this.frameInterval = null;
        this.frameRate = 1000; // Send frame every 1 second
        this.currentMetrics = null;
        this.insights = [];
        
        // Callbacks
        this.onAnalysisUpdate = null;
        this.onInsights = null;
        this.onError = null;
        this.onStatusChange = null;
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        if (!this.socketService || !this.socketService.socket) {
            console.error('Socket service not available');
            return;
        }
        
        const socket = this.socketService.socket;
        
        // Video analysis events
        socket.on('video_analysis_started', (data) => {
            console.log('✅ Video analysis started:', data);
            this.isAnalyzing = true;
            if (this.onStatusChange) {
                this.onStatusChange('started', data);
            }
        });
        
        socket.on('video_analysis_update', (data) => {
            console.log('📊 Video analysis update:', data);
            this.currentMetrics = data.analysis;
            if (this.onAnalysisUpdate) {
                this.onAnalysisUpdate(data);
            }
        });
        
        socket.on('video_insights', (data) => {
            console.log('💡 Video insights:', data);
            this.insights = data.insights || [];
            if (this.onInsights) {
                this.onInsights(data);
            }
        });
        
        socket.on('video_error', (data) => {
            console.error('❌ Video error:', data);
            if (this.onError) {
                this.onError(data.error);
            }
        });
        
        socket.on('video_analysis_stopped', (data) => {
            console.log('⏹️ Video analysis stopped:', data);
            this.isAnalyzing = false;
            if (this.onStatusChange) {
                this.onStatusChange('stopped', data);
            }
        });
    }
    
    async initializeVideo(videoElementId = 'videoElement') {
        try {
            console.log('🔄 Initializing video...');
            
            // Get video element
            this.videoElement = document.getElementById(videoElementId);
            if (!this.videoElement) {
                throw new Error(`Video element with id '${videoElementId}' not found`);
            }
            
            // Request camera access
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640, max: 1280 },
                    height: { ideal: 480, max: 720 },
                    facingMode: 'user'
                }
            });
            
            // Set video source
            this.videoElement.srcObject = this.stream;
            this.videoElement.autoplay = true;
            this.videoElement.muted = true;
            this.videoElement.playsInline = true;
            
            // Wait for video to be ready
            await new Promise((resolve, reject) => {
                this.videoElement.onloadedmetadata = () => {
                    console.log('✅ Video initialized successfully');
                    this.isInitialized = true;
                    resolve();
                };
                this.videoElement.onerror = reject;
                
                // Timeout after 10 seconds
                setTimeout(() => reject(new Error('Video initialization timeout')), 10000);
            });
            
            return true;
            
        } catch (error) {
            console.error('❌ Video initialization failed:', error);
            throw error;
        }
    }
    
    async startAnalysis(sessionId) {
        if (!this.isInitialized) {
            throw new Error('Video not initialized. Call initializeVideo() first.');
        }
        
        if (!this.socketService || !this.socketService.socket || !this.socketService.socket.connected) {
            throw new Error('Socket not connected');
        }
        
        console.log('🚀 Starting video analysis for session:', sessionId);
        
        // Start video analysis on server
        this.socketService.socket.emit('start_video_analysis', { 
            session_id: sessionId 
        });
        
        // Start sending frames
        this.frameInterval = setInterval(() => {
            this.captureAndSendFrame(sessionId);
        }, this.frameRate);
        
        console.log('📸 Frame capture started');
    }
    
    async stopAnalysis(sessionId) {
        console.log('⏹️ Stopping video analysis...');
        
        // Stop frame capture
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
        
        // Stop video analysis on server
        if (this.socketService && this.socketService.socket && this.socketService.socket.connected) {
            this.socketService.socket.emit('stop_video_analysis', { 
                session_id: sessionId 
            });
        }
        
        this.isAnalyzing = false;
        console.log('✅ Video analysis stopped');
    }
    
    captureAndSendFrame(sessionId) {
        if (!this.isAnalyzing || !this.videoElement || !this.stream) {
            return;
        }
        
        try {
            // Check if video is ready
            if (this.videoElement.readyState < 2) {
                console.warn('⚠️ Video not ready, skipping frame');
                return;
            }
            
            // Create canvas for frame capture
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size (optimized for analysis)
            canvas.width = 640;
            canvas.height = 480;
            
            // Draw current video frame
            ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);
            
            // Convert to base64 JPEG
            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send frame to server
            this.socketService.socket.emit('video_frame', {
                session_id: sessionId,
                frame_data: frameData
            });
            
            console.log(`📸 Frame sent (${Math.round(frameData.length / 1024)}KB)`);
            
        } catch (error) {
            console.error('❌ Error capturing frame:', error);
        }
    }
    
    getCurrentMetrics() {
        return this.currentMetrics;
    }
    
    getInsights() {
        return this.insights;
    }
    
    isVideoReady() {
        return this.isInitialized;
    }
    
    isAnalysisActive() {
        return this.isAnalyzing;
    }
    
    cleanup() {
        console.log('🧹 Cleaning up video analysis service...');
        
        // Stop analysis
        if (this.isAnalyzing) {
            this.stopAnalysis();
        }
        
        // Stop frame capture
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
        
        // Stop video stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        // Clear video element
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
        
        this.isInitialized = false;
        this.isAnalyzing = false;
        
        console.log('✅ Video analysis service cleaned up');
    }
}

// React Hook for Video Analysis
function useVideoAnalysis(socketService) {
    const [isVideoReady, setIsVideoReady] = React.useState(false);
    const [isAnalyzing, setIsAnalyzing] = React.useState(false);
    const [currentMetrics, setCurrentMetrics] = React.useState(null);
    const [videoInsights, setVideoInsights] = React.useState([]);
    const [error, setError] = React.useState(null);
    const videoElementRef = React.useRef(null);
    const videoServiceRef = React.useRef(null);
    
    React.useEffect(() => {
        // Initialize video service
        videoServiceRef.current = new VideoAnalysisService(socketService);
        
        // Setup callbacks
        videoServiceRef.current.onAnalysisUpdate = (data) => {
            setCurrentMetrics(data.analysis);
        };
        
        videoServiceRef.current.onInsights = (data) => {
            setVideoInsights(data.insights || []);
        };
        
        videoServiceRef.current.onError = (errorMessage) => {
            setError(errorMessage);
        };
        
        videoServiceRef.current.onStatusChange = (status, data) => {
            setIsAnalyzing(status === 'started');
        };
        
        // Cleanup on unmount
        return () => {
            if (videoServiceRef.current) {
                videoServiceRef.current.cleanup();
            }
        };
    }, [socketService]);
    
    const initializeVideo = async () => {
        try {
            setError(null);
            await videoServiceRef.current.initializeVideo('videoElement');
            setIsVideoReady(true);
        } catch (err) {
            setError(err.message);
            setIsVideoReady(false);
        }
    };
    
    const startAnalysis = async (sessionId) => {
        try {
            setError(null);
            await videoServiceRef.current.startAnalysis(sessionId);
        } catch (err) {
            setError(err.message);
        }
    };
    
    const stopAnalysis = async (sessionId) => {
        try {
            setError(null);
            await videoServiceRef.current.stopAnalysis(sessionId);
        } catch (err) {
            setError(err.message);
        }
    };
    
    return {
        isVideoReady,
        isAnalyzing,
        currentMetrics,
        videoInsights,
        error,
        videoElementRef,
        initializeVideo,
        startAnalysis,
        stopAnalysis
    };
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { VideoAnalysisService, useVideoAnalysis };
} else {
    window.VideoAnalysisService = VideoAnalysisService;
    window.useVideoAnalysis = useVideoAnalysis;
}