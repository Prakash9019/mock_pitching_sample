from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import os
import json
import uuid
import tempfile
from datetime import datetime
import shutil
from typing import Optional, Union, BinaryIO
import logging

# Socket.IO integration
import socketio
from fastapi.middleware.cors import CORSMiddleware

# Import service modules
from app.services.transcription import transcribe_audio
from app.services.intelligent_ai_agent import get_conversation_statistics
from app.services.enhanced_text_to_speech import convert_text_to_speech_with_persona

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
fastapi_app = FastAPI(
    title="AI Mock Investor Pitch",
    description="API for simulating investor pitch sessions with AI-generated responses",
    version="1.0.0"
)

# Add CORS middleware to FastAPI app
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True
)

# Create ASGI app with Socket.IO and FastAPI
app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=fastapi_app,
    socketio_path='socket.io'
)

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')
RESPONSE_DIR = os.path.join(DATA_DIR, 'responses')
SESSION_DIR = os.path.join(DATA_DIR, 'sessions')

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESPONSE_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
logger.info(f"Data directories created at: {DATA_DIR}")
logger.info(f"Upload directory: {UPLOAD_DIR}")
logger.info(f"Response directory: {RESPONSE_DIR}")
logger.info(f"Session directory: {SESSION_DIR}")

@fastapi_app.get("/")
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>🎙️ AI Investor Pitch</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
          }
          
          /* Audio level indicator */
          #audio-level-container {
            width: 100%;
            height: 10px;
            background: #f0f0f0;
            border-radius: 5px;
            margin: 10px 0;
            overflow: hidden;
          }
          
          #audio-level {
            height: 100%;
            width: 0%;
            background: #2196F3;
            transition: width 0.1s, background-color 0.3s;
          }
        .controls {
          margin: 30px 0;
        }
        button {
          padding: 12px 24px;
          margin: 0 10px;
          font-size: 16px;
          border: none;
          border-radius: 5px;
          cursor: pointer;
          transition: all 0.3s;
        }
        #startBtn {
          background-color: #4CAF50;
          color: white;
        }
        #recordBtn {
          background-color: #4CAF50;
          color: white;
          display: none;
        }
        #recordBtn.active {
          background-color: #f44336;
          animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.7); }
          70% { box-shadow: 0 0 0 10px rgba(244, 67, 54, 0); }
          100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
        }
        #pauseBtn {
          background-color: #ff9800;
          color: white;
          display: none;
        }
        #endBtn {
          background-color: #555555;
          color: white;
          display: none;
        }
        #status {
          margin: 20px 0;
          padding: 10px;
          border-radius: 5px;
          font-weight: bold;
        }
        .active {
          opacity: 0.8;
          transform: scale(0.95);
        }
        #audioPlayer {
          width: 100%;
          max-width: 500px;
          margin: 20px auto;
          display: block;
        }
      </style>
    </head>
    <body>
      <h1>🎤 AI Investor Pitch Practice</h1>
      
      <div id="status">Click 'Start Meeting' to begin your pitch session with real-time speech recognition</div>
      
      <div id="conversationProgress" style="
        display: none;
        margin: 15px 0;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        text-align: center;
      ">
        <div id="progressInfo" style="margin-bottom: 10px;">
          <strong id="investorName">Investor:</strong> <span id="currentStage">Getting started...</span>
        </div>
        <div id="progressBar" style="
          width: 100%;
          height: 8px;
          background-color: rgba(255,255,255,0.3);
          border-radius: 4px;
          overflow: hidden;
        ">
          <div id="progressFill" style="
            width: 0%;
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.5s ease;
          "></div>
        </div>
        <div id="stageCounter" style="margin-top: 8px; font-size: 12px; opacity: 0.9;">
          Stage 1 of 12
        </div>
      </div>
      
      <div class="controls">
        <div style="margin-bottom: 20px; text-align: center;">
          <label for="personaSelect" style="margin-right: 10px; font-size: 16px;">Investor Type:</label>
          <select id="personaSelect" style="padding: 8px; border-radius: 5px; border: 1px solid #ccc; font-size: 16px;">
            <option value="friendly">🤝 Friendly</option>
            <option value="skeptical" selected>🤨 Skeptical</option>
            <option value="technical">🔧 Technical</option>
          </select>
        </div>
        <div>
          <button id="startBtn">🚀 Start Meeting</button>
          <button id="recordBtn" style="display: none;">🎙️ Start Speaking</button>
          <button id="pauseBtn" style="display: none;">⏸️ Pause & Send</button>
          <button id="endBtn" style="display: none;">⏹️ End Meeting</button>
          <div id="audio-level-container" style="display: none;">
            <div id="audio-level"></div>
          </div>
        </div>
      </div>
      
      <div id="transcriptDisplay" style="
        margin: 20px 0;
        padding: 15px;
        border: 2px dashed #ccc;
        border-radius: 10px;
        min-height: 100px;
        background-color: #f9f9f9;
        font-style: italic;
        color: #666;
        display: none;
        text-align: left;
      ">
        Your speech will appear here in real-time...
      </div>
      
      <audio id="audioPlayer" controls></audio>

      <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
      <script>
        // DOM Elements
        const startBtn = document.getElementById('startBtn');
        const recordBtn = document.getElementById('recordBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        const endBtn = document.getElementById('endBtn');
        const audioPlayer = document.getElementById('audioPlayer');
        const statusElement = document.getElementById('status');
        const personaSelect = document.getElementById('personaSelect');
        const transcriptDisplay = document.getElementById('transcriptDisplay');
        const conversationProgress = document.getElementById('conversationProgress');
        const investorName = document.getElementById('investorName');
        const currentStage = document.getElementById('currentStage');
        const progressFill = document.getElementById('progressFill');
        const stageCounter = document.getElementById('stageCounter');
        
        // Speech recognition variables
        let recognition;
        let isListening = false;
        let currentTranscript = '';
        let interimTranscript = '';
        let audioStream;
        let socket;
        
        // Investor persona information
        const investorPersonas = {
          'skeptical': {
            name: 'Sarah Martinez',
            title: 'Senior Partner at Venture Capital',
            description: 'Analytical and thorough investor who asks tough questions'
          },
          'technical': {
            name: 'Dr. Alex Chen', 
            title: 'CTO-turned-Investor at TechVentures',
            description: 'Tech-focused investor interested in deep technical details'
          },
          'friendly': {
            name: 'Michael Thompson',
            title: 'Angel Investor & Former Entrepreneur',
            description: 'Supportive investor focused on founder journey'
          }
        };

        // Initialize socket connection
        function initSocket() {
          socket = io({
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 1000
          });

          socket.on('connect', () => {
            console.log('Connected to server');
            updateStatus('Connected. Ready to start the pitch session.', 'info');
          });

          socket.on('disconnect', () => {
            updateStatus('Disconnected from server. Please refresh the page.', 'error');
          });

          socket.on('ai_response', (data) => {
            try {
              const blob = new Blob([data], { type: 'audio/mp3' });
              const url = URL.createObjectURL(blob);
              audioPlayer.src = url;
              audioPlayer.play();
              updateStatus('AI response received. Click "Start Speaking" to respond.', 'success');
              
              // Hide the transcript display after receiving response
              setTimeout(() => {
                transcriptDisplay.style.display = 'none';
              }, 2000);
              
              // Update conversation progress
              updateConversationProgress(personaSelect.value);
            } catch (error) {
              console.error('Error playing AI response:', error);
              updateStatus('Error playing AI response', 'error');
            }
          });
        }

        // Update status message
        function updateStatus(message, type = 'info') {
          statusElement.textContent = message;
          statusElement.style.color = type === 'error' ? '#f44336' : 
                                     type === 'success' ? '#4CAF50' : '#2196F3';
        }
        
        // Update conversation progress
        function updateConversationProgress(persona) {
          const personaInfo = investorPersonas[persona];
          if (personaInfo) {
            investorName.textContent = `${personaInfo.name} - ${personaInfo.title}`;
            conversationProgress.style.display = 'block';
            
            // Fetch and update conversation stats
            if (socket && socket.id) {
              fetch(`/api/conversation/${socket.id}/stats`)
                .then(response => response.json())
                .then(stats => {
                  if (stats.topics_covered) {
                    const currentTopic = stats.topics_covered[stats.topics_covered.length - 1] || 'getting_started';
                    currentStage.textContent = currentTopic.replace('_', ' ').toUpperCase();
                    progressFill.style.width = stats.progress_percentage + '%';
                    stageCounter.textContent = `${stats.topics_covered.length} of 8 topics covered`;
                  }
                })
                .catch(error => console.log('Stats not available yet'));
            }
          }
        }

        // Start Meeting
        startBtn.addEventListener('click', async () => {
          try {
            updateStatus('Initializing meeting...', 'info');
            initSocket();
            
            // Check for speech recognition support
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
              throw new Error('Speech recognition not supported in this browser. Please use Chrome, Edge, or Safari.');
            }
            
            // Initialize speech recognition
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            
            // Configure speech recognition
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            recognition.maxAlternatives = 1;
            
            // Handle speech recognition results
            recognition.onresult = (event) => {
              let finalTranscript = '';
              let interimTranscript = '';
              
              for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                  finalTranscript += transcript + ' ';
                } else {
                  interimTranscript += transcript;
                }
              }
              
              if (finalTranscript) {
                currentTranscript += finalTranscript;
              }
              
              // Update transcript display with both final and interim results
              const displayText = currentTranscript + interimTranscript;
              if (displayText.trim()) {
                transcriptDisplay.innerHTML = `<strong>Your speech:</strong><br>"${displayText}"`;
                transcriptDisplay.style.color = '#333';
                transcriptDisplay.style.fontStyle = 'normal';
              }
              
              // Update status
              if (interimTranscript) {
                updateStatus('Listening... Keep speaking.', 'info');
              } else if (finalTranscript) {
                updateStatus('Listening... Speech captured.', 'info');
              }
            };
            
            recognition.onerror = (event) => {
              console.error('Speech recognition error:', event.error);
              updateStatus(`Speech recognition error: ${event.error}`, 'error');
            };
            
            recognition.onend = () => {
              if (isListening) {
                // Restart recognition if we're still supposed to be listening
                try {
                  recognition.start();
                } catch (error) {
                  console.error('Error restarting recognition:', error);
                }
              }
            };
            
            // Request microphone access for audio level monitoring
            try {
              audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: { echoCancellation: true, noiseSuppression: true } 
              });
              setupAudioMeter(audioStream);
            } catch (error) {
              console.warn('Could not access microphone for audio level monitoring:', error);
            }
            
            // Update UI
            startBtn.style.display = 'none';
            recordBtn.style.display = 'inline-block';
            endBtn.style.display = 'inline-block';
            updateStatus('Meeting started. Click "Start Speaking" to begin real-time speech recognition.', 'success');
            
            // Show conversation progress
            updateConversationProgress(personaSelect.value);
            
          } catch (error) {
            console.error('Error starting meeting:', error);
            updateStatus(`Error: ${error.message}`, 'error');
          }
        });

        // Start Speaking Button
        recordBtn.addEventListener('click', () => {
          if (!isListening) {
            // Start speech recognition
            try {
              currentTranscript = ''; // Reset transcript
              transcriptDisplay.style.display = 'block';
              transcriptDisplay.innerHTML = 'Listening... Start speaking now.';
              transcriptDisplay.style.color = '#666';
              transcriptDisplay.style.fontStyle = 'italic';
              
              recognition.start();
              isListening = true;
              recordBtn.classList.add('active');
              recordBtn.textContent = '🎙️ Listening...';
              pauseBtn.style.display = 'inline-block';
              document.getElementById('audio-level-container').style.display = 'block';
              updateStatus('Listening... Start speaking now.', 'info');
            } catch (error) {
              console.error('Error starting speech recognition:', error);
              updateStatus('Error starting speech recognition', 'error');
            }
          }
        });

        // Pause & Send Button
        pauseBtn.addEventListener('click', () => {
          if (isListening) {
            // Stop speech recognition and send the transcript
            recognition.stop();
            isListening = false;
            recordBtn.classList.remove('active');
            recordBtn.textContent = '🎙️ Start Speaking';
            pauseBtn.style.display = 'none';
            document.getElementById('audio-level-container').style.display = 'none';
            
            if (currentTranscript.trim()) {
              transcriptDisplay.innerHTML = `<strong>Sent:</strong><br>"${currentTranscript.trim()}"`;
              transcriptDisplay.style.color = '#2196F3';
              updateStatus('Sending your message to AI...', 'info');
              socket.emit('text_message', {
                text: currentTranscript.trim(),
                persona: personaSelect.value
              });
            } else {
              transcriptDisplay.style.display = 'none';
              updateStatus('No speech detected. Please try again.', 'error');
            }
          }
        });

        // End Session
        endBtn.addEventListener('click', () => {
          if (isListening) {
            recognition.stop();
            isListening = false;
          }
          
          // Stop all tracks in the stream
          if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
          }
          
          // Reset UI
          startBtn.style.display = 'inline-block';
          recordBtn.style.display = 'none';
          pauseBtn.style.display = 'none';
          endBtn.style.display = 'none';
          recordBtn.textContent = '🎙️ Start Speaking';
          document.getElementById('audio-level-container').style.display = 'none';
          transcriptDisplay.style.display = 'none';
          conversationProgress.style.display = 'none';
          
          updateStatus('Session ended. Click "Start Meeting" to begin a new session.', 'info');
          
          // Disconnect socket
          if (socket) {
            socket.disconnect();
          }
        });

        // Audio level monitoring function
        function setupAudioMeter(stream) {
          try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            source.connect(analyser);
            
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            
            function checkLevel() {
              analyser.getByteFrequencyData(dataArray);
              const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
              // Visual feedback for audio level
              const levelIndicator = document.getElementById('audio-level');
              if (levelIndicator) {
                levelIndicator.style.width = `${Math.min(100, average)}%`;
                levelIndicator.style.backgroundColor = average > 50 ? '#4CAF50' : '#2196F3';
              }
              requestAnimationFrame(checkLevel);
            }
            checkLevel();
          } catch (error) {
            console.warn('Audio level monitoring not available:', error);
          }
        }

        // Check for browser support
        if (!navigator.mediaDevices || !window.MediaRecorder) {
          updateStatus('Your browser does not support audio recording. Please use Chrome, Firefox, or Edge.', 'error');
          startBtn.disabled = true;
        }
      </script>
    </body>
    </html>
    """)

@fastapi_app.post("/pitch")
async def process_pitch(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    investor_persona: Optional[str] = "skeptical"
):
    if not audio_file.filename.endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 files are supported")

    try:
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        upload_path = os.path.join(UPLOAD_DIR, f"{session_id}_{audio_file.filename}")
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        transcript = transcribe_audio(upload_path)
        # Use the global ai_agent to generate a response
        if ai_agent is None:
            raise HTTPException(status_code=500, detail="AI Agent not initialized")
        investor_response = ai_agent.generate_response(str(uuid.uuid4()), transcript)
        response_audio_path = os.path.join(RESPONSE_DIR, f"{session_id}_response.mp3")
        convert_text_to_speech(investor_response, response_audio_path)

        session_data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "founder_transcript": transcript,
            "investor_response": investor_response,
            "investor_persona": investor_persona,
            "original_audio": upload_path,
            "response_audio": response_audio_path
        }

        session_file = os.path.join(SESSION_DIR, f"{session_id}.json")
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)

        return {
            "session_id": session_id,
            "timestamp": timestamp,
            "founder_transcript": transcript,
            "investor_response": investor_response,
            "response_audio_url": f"/download/{session_id}"
        }

    except Exception as e:
        logger.error(f"Error processing pitch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing pitch: {str(e)}")

@fastapi_app.get("/download/{session_id}")
async def download_response(session_id: str):
    response_path = os.path.join(RESPONSE_DIR, f"{session_id}_response.mp3")
    if not os.path.exists(response_path):
        raise HTTPException(status_code=404, detail="Response audio not found")
    return FileResponse(
        path=response_path,
        filename=f"investor_response_{session_id}.mp3",
        media_type="audio/mpeg"
    )

@fastapi_app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session_path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")
    with open(session_path, "r") as f:
        session_data = json.load(f)
    return session_data

@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint for Docker healthcheck and load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Mock Investor Pitch",
        "version": "1.0.0"
    }

@fastapi_app.get("/conversation/{session_id}/memory")
async def get_conversation_memory_stats(session_id: str):
    """Get memory statistics for a conversation."""
    try:
        if session_id in conversation_states:
            conversation = conversation_states[session_id]
            return {
                "memory_stats": conversation.get_memory_stats(),
                "langchain_context": conversation.get_langchain_context() if hasattr(conversation, 'get_langchain_context') else None,
                "basic_summary": conversation.get_conversation_summary()
            }
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        logger.error(f"Error getting memory stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting memory stats: {str(e)}")

@fastapi_app.get("/conversation/{session_id}/context")
async def get_conversation_context(session_id: str):
    """Get the full conversation context with LangChain enhancement."""
    try:
        if session_id in conversation_states:
            conversation = conversation_states[session_id]
            
            # Get enhanced context if available
            if hasattr(conversation, 'get_langchain_context'):
                context = conversation.get_langchain_context()
                using_langchain = conversation.use_langchain
            else:
                context = conversation.get_conversation_summary()
                using_langchain = False
            
            return {
                "conversation_id": session_id,
                "context": context,
                "using_langchain": using_langchain,
                "persona": conversation.persona,
                "message_count": len(conversation.conversation_history)
            }
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        logger.error(f"Error getting conversation context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting context: {str(e)}")

@fastapi_app.get("/langchain-status")
async def get_langchain_status():
    """Check if LangChain is available and working."""
    try:
        # Import the module to check status
        # Legacy code - LangChain integration is now handled in the intelligent agent
        
        status = {
            "langchain_available": True,  # Now handled by intelligent agent
            "google_api_key_set": bool(os.getenv("GEMINI_API_KEY")),
            "active_conversations": len(conversation_states)
        }
        
        if True:  # Enhanced AI agent is always available
            # Count conversations using LangChain
            langchain_conversations = sum(1 for conv in conversation_states.values() 
                                        if hasattr(conv, 'use_langchain') and conv.use_langchain)
            status["conversations_using_langchain"] = langchain_conversations
        
        return status
        
    except Exception as e:
        logger.error(f"Error checking LangChain status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking status: {str(e)}")

@fastapi_app.get("/api/conversation/{conversation_id}/stats")
async def get_conversation_stats(conversation_id: str):
    """Get statistics for a specific conversation.
    
    Args:
        conversation_id: The ID of the conversation to get stats for
        
    Returns:
        dict: Conversation statistics or error details
        
    Raises:
        HTTPException: With appropriate status code and error details
    """
    global ai_agent
    
    try:
        if not conversation_id:
            logger.warning("Empty conversation_id provided")
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "conversation_id is required", 
                    "status": "error",
                    "message": "Please provide a valid conversation ID"
                }
            )
            
        if ai_agent is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service Unavailable",
                    "status": "error",
                    "message": "AI Agent is not initialized. Please try again later."
                }
            )
            
        # Get conversation statistics directly from the agent
        try:
            stats = ai_agent.get_conversation_stats(conversation_id)
            logger.debug(f"Conversation stats response: {stats}")
            
            # Handle different status responses
            status = stats.get("status", "active")
            
            if status == "active":
                return {
                    "status": "success",
                    "data": {
                        "conversation_id": conversation_id,
                        "current_stage": stats.get("current_stage"),
                        "stages_completed": stats.get("stages_completed", []),
                        "next_stage": stats.get("next_stage"),
                        "messages_exchanged": stats.get("messages_exchanged", 0),
                        "last_activity": stats.get("last_activity")
                    }
                }
            else:
                # Handle other statuses
                return {
                    "status": "success",
                    "data": stats
                }
                
        except KeyError:
            # Handle case where conversation doesn't exist
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Conversation not found",
                    "status": "not_found",
                    "message": f"No conversation found with ID: {conversation_id}"
                }
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions as is
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error getting conversation stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "status": "error",
                "type": type(e).__name__,
                "message": str(e)
            }
        )

# Global instance of the AI agent
ai_agent = None

@fastapi_app.on_event("startup")
async def startup_event():
    """Initialize the AI agent when the application starts"""
    global ai_agent
    try:
        from app.services.intelligent_ai_agent import IntelligentAIAgent
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Initialize the AI agent
        ai_agent = IntelligentAIAgent(llm)
        logger.info("AI Agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI agent: {str(e)}", exc_info=True)
        raise

@fastapi_app.post("/api/conversation/start")
async def start_conversation(persona: str = "friendly"):
    """Start a new conversation with the AI investor
    
    Args:
        persona: The investor persona to use (e.g., 'skeptical', 'friendly', 'technical')
        
    Returns:
        dict: Conversation details including the conversation_id and initial greeting
    """
    try:
        if ai_agent is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "AI Agent not initialized",
                    "status": "error"
                }
            )
            
        # Create a new conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Start a new conversation with the AI agent
        context = ai_agent.start_conversation(conversation_id, persona)
        
        # The initial greeting is already added to the chat history
        initial_greeting = context.chat_history[-1].replace("Investor: ", "")
        
        logger.info(f"Started new conversation: {conversation_id} with persona: {persona}")
        
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "persona": persona,
            "initial_greeting": initial_greeting,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to start conversation",
                "details": str(e),
                "status": "error"
            }
        )

@fastapi_app.get("/api/tts/voices")
async def get_tts_voices():
    """Get available TTS voices for all personas"""
    try:
        from app.services.enhanced_text_to_speech import list_available_voices
        return list_available_voices()
    except Exception as e:
        logger.error(f"Error getting TTS voices: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get TTS voices")

@fastapi_app.get("/api/tts/test/{persona}")
async def test_persona_voice(persona: str):
    """Test a specific persona voice"""
    try:
        from app.services.enhanced_text_to_speech import test_all_personas, get_persona_voice_info
        
        if persona not in ["skeptical", "technical", "friendly"]:
            raise HTTPException(status_code=400, detail="Invalid persona. Choose from: skeptical, technical, friendly")
        
        # Get voice info
        voice_info = get_persona_voice_info(persona)
        
        # Test the voice
        test_results = test_all_personas()
        
        return {
            "persona": persona,
            "voice_info": voice_info,
            "test_result": test_results.get(persona, False),
            "message": f"Voice test for {persona} persona completed"
        }
    except Exception as e:
        logger.error(f"Error testing persona voice: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to test persona voice")

@fastapi_app.get("/api/debug/conversations")
async def debug_conversations():
    """Debug: Show all active conversations"""
    try:
        debug_info = {}
        for conv_id, context in conversation_states.items():
            if hasattr(context, 'conversation_history'):
                debug_info[conv_id] = {
                    "persona": context.persona,
                    "founder_name": getattr(context, 'founder_name', None),
                    "company_name": getattr(context, 'company_name', None),
                    "topics_covered": getattr(context, 'covered_topics', []),
                    "message_count": len(context.conversation_history),
                    "last_activity": str(context.last_activity)
                }
        return {"active_conversations": debug_info}
    except Exception as e:
        logger.error(f"Error getting debug info: {str(e)}")
        return {"error": str(e)}

@sio.event
async def connect(sid, environ):
    logger.info(f"WebSocket connected: {sid}")

@sio.event
async def text_message(sid, data):
    """Handle real-time transcribed text messages"""
    try:
        global ai_agent
        
        if ai_agent is None:
            logger.error("AI Agent not initialized")
            await sio.emit("error", {"message": "AI Agent not initialized"}, to=sid)
            return
            
        logger.info(f"Received text message from {sid}")
        
        # Extract text and persona from the incoming message
        if isinstance(data, dict):
            transcript_text = data.get('text', '')
            persona = data.get('persona', 'skeptical')
        else:
            # Backward compatibility
            transcript_text = str(data)
            persona = 'skeptical'
        
        if not transcript_text.strip():
            logger.warning("Received empty text message")
            return
            
        logger.info(f"Processing text: {transcript_text[:100]}...")
        
        # Generate response using the AI agent
        try:
            # First, check if we have a conversation, start one if not
            try:
                # Check if this is the first message in a new conversation
                is_new_conversation = False
                try:
                    # This will raise ValueError if conversation doesn't exist
                    ai_agent.conversations[sid]
                except (KeyError, ValueError):
                    # Start a new conversation
                    logger.info(f"Starting new conversation for {sid} with {persona} persona")
                    await sio.emit("status", {"message": "Starting new conversation..."}, to=sid)
                    ai_agent.start_conversation(sid, persona)
                    is_new_conversation = True
                
                # Process the user's message
                investor_reply = ai_agent.generate_response(sid, transcript_text)
                
                # If this was the first message, use a more contextual greeting
                if is_new_conversation:
                    # If the user introduced themselves, respond accordingly
                    if any(word in transcript_text.lower() for word in ["i am", "i'm", "my name is"]):
                        # The AI will naturally respond to the introduction in generate_response
                        pass
                    else:
                        # Default greeting for new conversations
                        investor_reply = f"Hello! I'm your AI investor. {investor_reply}"
                
            except ValueError as ve:
                # Re-raise any unexpected ValueErrors
                logger.error(f"Error in conversation handling: {str(ve)}")
                raise
            
            logger.info("Converting response to speech with persona-specific voice...")
            # Get audio data directly without saving to file using persona-specific voice
            audio_data = convert_text_to_speech_with_persona(investor_reply, persona)
            
            if not audio_data:
                raise ValueError("Failed to generate audio response")
                
            logger.info(f"Generated {len(audio_data)} bytes of audio data")
            
            # Send the audio data directly to the client
            await sio.emit("ai_response", audio_data, to=sid)
            
        except Exception as e:
            logger.error(f"Error generating or sending response: {str(e)}", exc_info=True)
            await sio.emit("error", {"message": f"Error processing your message: {str(e)}"}, to=sid)
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        error_msg = f"Error processing message: {str(e)}"
        await sio.emit("error", {"message": error_msg}, to=sid)

@sio.event
async def audio_chunk(sid, data):
    """Handle incoming audio chunks, transcribe, and respond with AI-generated audio.
    
    This is a legacy endpoint that processes audio chunks directly. It's recommended
    to use the text-based WebSocket endpoint for new implementations.
    """
    global ai_agent
    
    if ai_agent is None:
        logger.error("AI Agent not initialized")
        await sio.emit("error", {"message": "AI Agent not initialized"}, to=sid)
        return
        
    try:
        logger.info(f"Received audio chunk from {sid} (legacy mode)")
        
        # Extract audio data and persona from the incoming message
        if isinstance(data, dict) and 'audio' in data:
            audio_data = data['audio']
            persona = data.get('persona', 'skeptical')  # Default to 'skeptical' if not provided
        else:
            # Backward compatibility with older clients
            audio_data = data
            persona = 'skeptical'
        
        # Save the incoming audio to a temporary file for transcription
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        try:
            logger.info("Starting transcription...")
            transcription_result = transcribe_audio(temp_audio_path)
            transcript_text = transcription_result.get('text', '')
            confidence = transcription_result.get('confidence', 0.0)
            
            if not transcript_text.strip():
                logger.warning("Empty transcription result")
                await sio.emit("error", {"message": "Could not transcribe audio. Please try again."}, to=sid)
                return
                
            logger.info(f"Transcription complete (confidence: {confidence:.2f}): {transcript_text[:100]}...")
            
            # Generate response using the AI agent
            try:
                investor_reply = ai_agent.generate_response(sid, transcript_text)
                
                logger.info("Converting response to speech with persona-specific voice...")
                # Get audio data directly without saving to file using persona-specific voice
                audio_data = convert_text_to_speech_with_persona(investor_reply, persona)
                
                if not audio_data:
                    raise ValueError("Failed to generate audio response")
                    
                logger.info(f"Generated {len(audio_data)} bytes of audio data")
                
                # Send the audio data directly to the client
                await sio.emit("ai_response", audio_data, to=sid)
                
            except KeyError:
                # If conversation doesn't exist, start a new one and retry
                logger.info(f"Starting new conversation for {sid} with {persona} persona")
                await sio.emit("status", {"message": "Starting new conversation..."}, to=sid)
                
                # Start a new conversation
                ai_agent.start_conversation(sid, persona)
                
                # Get the initial greeting
                initial_greeting = "Hello! I'm your AI investor. How can I help you today?"
                
                # Convert greeting to speech
                audio_data = convert_text_to_speech_with_persona(initial_greeting, persona)
                await sio.emit("ai_response", audio_data, to=sid)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}", exc_info=True)
            await sio.emit("error", {"message": f"Error processing audio: {str(e)}"}, to=sid)
            
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_audio_path}: {e}")
                
    except Exception as e:
        logger.error(f"WebSocket error in audio_chunk: {str(e)}", exc_info=True)
        await sio.emit("error", {"message": f"Error processing audio: {str(e)}"}, to=sid)


@sio.event
async def disconnect(sid):
    logger.info(f"WebSocket disconnected: {sid}")

# This block is only for direct execution with `python main.py`
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get host and port from environment variables with defaults
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))
    
    # Only enable reload in development
    reload = os.getenv("ENV", "production").lower() == "development"
    
    # Run the application
    uvicorn.run(
        "main:app",  # Changed from "app.main:app" to "main:app" for direct execution
        host=host, 
        port=port, 
        # reload=reload,
        workers=1,  # For development
        log_level="debug",
        timeout_keep_alive=120
    )