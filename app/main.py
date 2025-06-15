from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse
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
from services.transcription import transcribe_audio
from services.ai_response import generate_investor_response
from services.text_to_speech import convert_text_to_speech

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

fastapi_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
app = socketio.ASGIApp(sio, fastapi_app)

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
          background-color: #f44336;
          color: white;
          display: none;
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
      
      <div id="status">Click 'Start Meeting' to begin your pitch session</div>
      
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
          <button id="recordBtn" style="display: none;">🎙️ Record</button>
          <button id="pauseBtn" style="display: none;">⏸️ Pause</button>
          <button id="endBtn" style="display: none;">⏹️ End Meeting</button>
          <div id="audio-level-container" style="display: none;">
            <div id="audio-level"></div>
          </div>
        </div>
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
        
        // Audio recording variables
        let mediaRecorder;
        let audioChunks = [];
        let audioStream;
        let isRecording = false;
        let socket;

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
              updateStatus('AI response received. Click "Record" to respond.', 'success');
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

        // Start Meeting
        startBtn.addEventListener('click', async () => {
          try {
            updateStatus('Initializing meeting...', 'info');
            initSocket();
            
            // Audio constraints for better quality
            const audioConstraints = {
              audio: {
                channelCount: 1,  // Mono audio is sufficient for speech
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000,  // 16kHz sample rate is good for speech
                sampleSize: 16,     // 16-bit samples
                mimeType: 'audio/webm;codecs=opus'  // Use Opus codec for better quality
              },
              video: false
            };

            // MediaRecorder options
            const recorderOptions = {
              mimeType: 'audio/webm;codecs=opus',
              audioBitsPerSecond: 128000  // 128kbps for good quality
            };

            // Request microphone access with better settings
            audioStream = await navigator.mediaDevices.getUserMedia(audioConstraints);
            
            // Setup audio level monitoring
            setupAudioMeter(audioStream);
            
            // Initialize media recorder with better settings
            mediaRecorder = new MediaRecorder(audioStream, recorderOptions);
            
            // Handle data available event
            mediaRecorder.ondataavailable = (event) => {
              if (event.data.size > 0) {
                audioChunks.push(event.data);
              }
            };
            
            mediaRecorder.onstop = async () => {
              if (audioChunks.length === 0) {
                updateStatus('No audio recorded. Please try again.', 'error');
                return;
              }
              
              try {
                updateStatus('Sending audio to AI...', 'info');
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const arrayBuffer = await audioBlob.arrayBuffer();
                socket.emit('audio_chunk', {
                  audio: arrayBuffer,
                  persona: personaSelect.value
                });
                audioChunks = [];
              } catch (error) {
                console.error('Error processing audio:', error);
                updateStatus('Error processing audio', 'error');
              }
            };
            
            // Update UI
            startBtn.style.display = 'none';
            recordBtn.style.display = 'inline-block';
            endBtn.style.display = 'inline-block';
            updateStatus('Meeting started. Click "Record" to start speaking.', 'success');
            
          } catch (error) {
            console.error('Error starting meeting:', error);
            updateStatus(`Error: ${error.message}`, 'error');
          }
        });

        // Record Button
        recordBtn.addEventListener('click', () => {
          if (!isRecording) {
            audioChunks = [];
            mediaRecorder.start(100); // Collect data every 100ms
            isRecording = true;
            recordBtn.classList.add('active');
            updateStatus('Recording... Click "Pause & Send" when done.', 'info');
          }
        });

        // Pause & Send Button
        pauseBtn.addEventListener('click', () => {
          if (isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            recordBtn.classList.remove('active');
            updateStatus('Processing your response...', 'info');
          }
        });

        // Toggle between Record and Pause buttons
        recordBtn.addEventListener('click', () => {
          if (isRecording) {
            pauseBtn.style.display = 'inline-block';
            recordBtn.textContent = '🎙️ Recording...';
          } else {
            pauseBtn.style.display = 'none';
            recordBtn.textContent = '🎙️ Record';
          }
        });

        // End Session
        endBtn.addEventListener('click', () => {
          if (isRecording) {
            mediaRecorder.stop();
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
        investor_response = generate_investor_response(transcript, investor_persona)
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
        from services.ai_response import LANGCHAIN_AVAILABLE
        
        status = {
            "langchain_available": LANGCHAIN_AVAILABLE,
            "google_api_key_set": bool(os.getenv("GEMINI_API_KEY")),
            "active_conversations": len(conversation_states)
        }
        
        if LANGCHAIN_AVAILABLE:
            # Count conversations using LangChain
            langchain_conversations = sum(1 for conv in conversation_states.values() 
                                        if hasattr(conv, 'use_langchain') and conv.use_langchain)
            status["conversations_using_langchain"] = langchain_conversations
        
        return status
        
    except Exception as e:
        logger.error(f"Error checking LangChain status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking status: {str(e)}")

@sio.event
async def connect(sid, environ):
    logger.info(f"WebSocket connected: {sid}")

# Store conversation states
conversation_states = {}

@sio.event
async def audio_chunk(sid, data):
    try:
        logger.info(f"Received audio chunk from {sid}")
        
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
            
            logger.info(f"Transcription complete (confidence: {confidence:.2f}): {transcript_text[:100]}...")
            
            # Initialize conversation state if it doesn't exist or if persona has changed
            from services.ai_response import start_new_conversation
            if sid not in conversation_states or conversation_states[sid].persona != persona:
                conversation_states[sid] = start_new_conversation(sid, persona)
                logger.info(f"Started new conversation with {persona} persona")
            
            logger.info("Generating investor response...")
            investor_reply = generate_investor_response(conversation_states[sid], transcript_text)
            
            logger.info("Converting response to speech...")
            # Get audio data directly without saving to file
            audio_data = convert_text_to_speech(investor_reply)
            
            if not audio_data:
                raise ValueError("Failed to generate audio response")
                
            logger.info(f"Generated {len(audio_data)} bytes of audio data")
            
            # Send the audio data directly to the client
            await sio.emit("ai_response", audio_data, to=sid)
            
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_audio_path}: {e}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        await sio.emit("error", {"message": f"Error processing audio: {str(e)}"}, to=sid)


@sio.event
async def disconnect(sid):
    logger.info(f"WebSocket disconnected: {sid}")

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get host and port from environment variables with defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    
    # Only enable reload in development
    reload = os.getenv("ENV", "production").lower() == "development"
    
    # Run the application
    uvicorn.run("main:app", 
                host=host, 
                port=port, 
                reload=reload,
                workers=int(os.getenv("WEB_CONCURRENCY", "2")),
                limit_concurrency=int(os.getenv("WEB_CONCURRENCY", "2")),
                timeout_keep_alive=120)