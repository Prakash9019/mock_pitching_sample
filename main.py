from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
import os
import json
import uuid
import tempfile
import time
import asyncio
from datetime import datetime
import shutil
from typing import Optional, Union, BinaryIO
import logging
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Configure logging first
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Socket.IO integration
import socketio
from fastapi.middleware.cors import CORSMiddleware

# Import service modules
from app.services.transcription import transcribe_audio
from app.services.enhanced_text_to_speech import convert_text_to_speech_with_persona

# Import LangGraph workflow and improved AI agent
from app.services.langgraph_workflow import (
    start_pitch_session as start_practice_session,
    start_pitch_session_async,
    start_pitch_session_with_message,
    process_pitch_message as handle_practice_message,
    process_pitch_message_async,
    end_pitch_session_async,
    get_pitch_workflow,
    initialize_pitch_workflow,
    generate_pitch_analysis_report,
    end_pitch_session_with_analysis,
    get_pitch_analytics
)

# Import VAD and audio streaming services
try:
    from app.services.voice_activity_detection import initialize_vad_system
    from app.services.audio_websocket_handler import initialize_audio_websocket_handler
    VAD_AVAILABLE = True
    logger.info("VAD services imported successfully")
except ImportError as e:
    logger.warning(f"VAD services not available: {e}")
    VAD_AVAILABLE = False

# Import video analysis services
try:
    from app.services.video_analysis import initialize_video_analyzer
    from app.services.enhanced_video_analysis import initialize_enhanced_video_analyzer, get_enhanced_video_analyzer
    from app.services.video_websocket_handler import initialize_video_websocket_handler
    VIDEO_ANALYSIS_AVAILABLE = True
    logger.info("Video analysis services imported successfully")
except ImportError as e:
    logger.warning(f"Video analysis services not available: {e}")
    VIDEO_ANALYSIS_AVAILABLE = False

from app.services.intelligent_ai_agent_improved import (
    start_improved_conversation,
    generate_improved_response,
    initialize_improved_agent,
    improved_agent
)

# Import database components
from app.database import connect_to_mongo, close_mongo_connection, get_database, test_database_connection
from app.services.database_service import DatabaseService
from app.models import PitchAnalysis, PitchSession

# Import audio storage service
from app.services.audio_conversation_storage import finalize_session_recording, get_session_audio_info
from app.services.audio_websocket_handler import get_session_audio_url

# Integration example import removed - focusing on audio conversation

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
    engineio_logger=False,  # Reduce logging noise
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1000000  # 1MB for audio data
)

# Initialize VAD system (if available)
if VAD_AVAILABLE:
    try:
        initialize_vad_system()
        initialize_audio_websocket_handler(sio)
        logger.info("VAD system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize VAD system: {e}")
        VAD_AVAILABLE = False
else:
    logger.warning("VAD system not available - audio features disabled")

# Initialize video analysis system (if available)
if VIDEO_ANALYSIS_AVAILABLE:
    try:
        initialize_video_analyzer()
        initialize_enhanced_video_analyzer()  # Initialize enhanced analyzer
        initialize_video_websocket_handler(sio)
        logger.info("Video analysis system initialized successfully")
        logger.info("Enhanced video analysis with CVZone, FER, and MediaPipe ready")
    except Exception as e:
        logger.error(f"Failed to initialize video analysis system: {e}")
        VIDEO_ANALYSIS_AVAILABLE = False
else:
    logger.warning("Video analysis system not available - video features disabled")

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

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_PATH, "templates"))

# Global database service instance
db_service: Optional[DatabaseService] = None

def get_database_service() -> Optional[DatabaseService]:
    """Get the global database service instance"""
    return db_service

# Database startup and shutdown events
@fastapi_app.on_event("startup")
async def startup_database():
    """Initialize database connection on startup"""
    global db_service
    try:
        await connect_to_mongo()
        database = await get_database()
        if database is not None:
            db_service = DatabaseService(database)
            logger.info("✅ Database service initialized successfully")
        else:
            logger.warning("⚠️ Database not available - running without persistence")
            db_service = None
    except Exception as e:
        logger.warning(f"⚠️ Database connection failed: {e}")
        logger.info("📝 Application starting without database - TTS and conversation features will work normally")
        # Don't raise here to allow app to start even if DB is unavailable
        db_service = None

@fastapi_app.on_event("shutdown")
async def shutdown_database():
    """Close database connection on shutdown"""
    try:
        await close_mongo_connection()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")

# Mount static files
fastapi_app.mount("/static", StaticFiles(directory=os.path.join(BASE_PATH, "static")), name="static")

# Add Socket.IO client route
@fastapi_app.get("/socket.io/socket.io.js")
async def socket_io_js():
    """Serve Socket.IO client library"""
    try:
        import socketio
        # Get the Socket.IO client path from the package
        import pkg_resources
        socketio_path = pkg_resources.resource_filename('socketio', 'static/socket.io.js')
        
        if os.path.exists(socketio_path):
            return FileResponse(socketio_path, media_type="application/javascript")
        else:
            # Fallback: serve from CDN content
            socketio_js_content = """
            // Socket.IO client library fallback
            console.warn('Loading Socket.IO from embedded fallback');
            """
            return Response(content=socketio_js_content, media_type="application/javascript")
    except Exception as e:
        logger.error(f"Error serving Socket.IO client: {e}")
        # Return a minimal Socket.IO client loader
        fallback_content = """
        // Socket.IO client loader
        (function() {
            if (typeof io === 'undefined') {
                const script = document.createElement('script');
                script.src = 'https://cdn.socket.io/4.7.2/socket.io.min.js';
                script.onload = function() {
                    console.log('Socket.IO loaded from CDN');
                    window.dispatchEvent(new Event('socketio-loaded'));
                };
                script.onerror = function() {
                    console.error('Failed to load Socket.IO from CDN');
                };
                document.head.appendChild(script);
            }
        })();
        """
        return Response(content=fallback_content, media_type="application/javascript")

def cleanup_old_audio_files(session_id: str, keep_count: int = 5):
    """Clean up old audio files for a session, keeping only the most recent ones"""
    try:
        # Find all audio files for this session
        audio_files = []
        for filename in os.listdir(RESPONSE_DIR):
            if filename.startswith(f"{session_id}_response_") and filename.endswith('.mp3'):
                file_path = os.path.join(RESPONSE_DIR, filename)
                # Extract timestamp from filename
                try:
                    timestamp_str = filename.replace(f"{session_id}_response_", "").replace('.mp3', '')
                    timestamp = int(timestamp_str)
                    audio_files.append((timestamp, file_path, filename))
                except ValueError:
                    # Skip files with invalid timestamp format
                    continue
        
        # Sort by timestamp (newest first) and remove old files
        audio_files.sort(key=lambda x: x[0], reverse=True)
        
        # Remove files beyond keep_count
        for i, (timestamp, file_path, filename) in enumerate(audio_files):
            if i >= keep_count:
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up old audio file: {filename}")
                except OSError as e:
                    logger.warning(f"Failed to remove old audio file {filename}: {e}")
                    
    except Exception as e:
        logger.error(f"Error during audio cleanup: {e}")



@fastapi_app.get("/", response_class=HTMLResponse)
async def multimodal_clean_page(request: Request):
    """Serve the Clean Multimodal Pitch Analysis demo page"""
    return templates.TemplateResponse("multimodal_pitch_demo_clean.html", {"request": request})
@fastapi_app.get("/database-test", response_class=HTMLResponse)
async def database_test_page(request: Request):
    """Serve the Database Connection Test page"""
    return templates.TemplateResponse("database_test.html", {"request": request})

@fastapi_app.get("/debug-user-audio", response_class=HTMLResponse)
async def debug_user_audio_page(request: Request):
    """Serve the Debug User Audio page"""
    return templates.TemplateResponse("debug_user_audio.html", {"request": request})

@fastapi_app.get("/api/database/test")
async def test_database():
    """Test database connection"""
    try:
        success = await test_database_connection()
        if success:
            return {
                "status": "success",
                "message": "Database connection successful",
                "database_name": db_service.database.name if db_service else "Not connected",
                "connected": db_service is not None
            }
        else:
            return {
                "status": "error",
                "message": "Database connection failed",
                "connected": False
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Database test error: {str(e)}",
            "connected": False
        }

@fastapi_app.get("/api/database/status")
async def database_status():
    """Get current database status"""
    return {
        "connected": db_service is not None,
        "database_name": db_service.database.name if db_service else None,
        "connection_string": "mongodb://localhost:27017" if db_service else None,
        "status": "connected" if db_service else "disconnected"
    }

@fastapi_app.get("/api/audio/monitoring")
async def audio_monitoring():
    """Monitor audio system status and transcription usage"""
    try:
        from app.services.audio_websocket_handler import get_audio_websocket_handler
        from app.services.audio_conversation_storage import get_storage_stats, get_session_audio_info
        
        # Get active sessions from audio handler
        handler = get_audio_websocket_handler()
        active_sessions = handler.get_active_sessions() if handler else {}
        
        # Get storage statistics
        storage_stats = get_storage_stats()
        
        # Get detailed session audio info
        session_audio_details = {}
        for session_id in active_sessions.keys():
            audio_info = get_session_audio_info(session_id)
            if audio_info:
                session_audio_details[session_id] = audio_info
        
        return {
            "status": "healthy",
            "active_sessions": len(active_sessions),
            "sessions_detail": {
                session_id: {
                    "persona": data.get("persona"),
                    "status": data.get("status"),
                    "audio_recording": data.get("audio_recording_enabled", False),
                    "ai_speaking": data.get("ai_speaking", False),
                    "user_can_speak": data.get("user_can_speak", True),
                    "conversation_turn": data.get("conversation_turn", "unknown")
                }
                for session_id, data in active_sessions.items()
            },
            "session_audio_details": session_audio_details,
            "storage_stats": storage_stats,
            "websocket_handlers": {
                "audio_handler_active": handler is not None,
                "vad_available": VAD_AVAILABLE,
                "video_analysis_available": VIDEO_ANALYSIS_AVAILABLE
            }
        }
    except Exception as e:
        logger.error(f"Error getting audio monitoring data: {e}")
        return {
            "status": "error",
            "error": str(e),
            "active_sessions": 0
        }


@fastapi_app.get("/api/tts/test")
async def test_tts(
    text: str = "Hello! This is a test of the Google Cloud Text-to-Speech system.",
    persona: str = "friendly"
):
    """Test endpoint for Google Cloud TTS"""
    try:
        logger.info(f"Testing TTS with text: '{text}' and persona: '{persona}'")
        
        # Generate TTS audio
        audio_data = convert_text_to_speech_with_persona(text, persona)
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="TTS generation failed")
        
        # Convert to base64 for JSON response
        import base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return {
            "success": True,
            "message": "TTS generated successfully",
            "text": text,
            "persona": persona,
            "audio_data": audio_base64,
            "audio_size": len(audio_data)
        }
        
    except Exception as e:
        logger.error(f"TTS test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS test failed: {str(e)}")

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
        convert_text_to_speech_with_persona(investor_response, response_audio_path, investor_persona)

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

@fastapi_app.get("/download/{filename}")
async def download_response(filename: str):
    """Download audio response file
    
    Args:
        filename: The audio filename (e.g., 'session_id_latest_response.mp3' or just 'session_id')
    """
    # Handle both full filename and session_id patterns
    if filename.endswith('.mp3'):
        # Full filename provided
        file_path = os.path.join(RESPONSE_DIR, filename)
    else:
        # Just session_id provided, try different patterns
        possible_paths = [
            os.path.join(RESPONSE_DIR, f"{filename}_latest_response.mp3"),  # WebSocket pattern
            os.path.join(RESPONSE_DIR, f"{filename}_response.mp3"),         # Original pattern
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail=f"Response audio not found for session: {filename}")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")
    
    return FileResponse(
        path=file_path,
        filename=f"investor_response_{filename}",
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
    # Check database health
    db_healthy = False
    if db_service and db_service.db:
        try:
            from app.database import db_manager
            db_healthy = await db_manager.health_check()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
    
    return {
        "status": "healthy" if db_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Mock Investor Pitch",
        "version": "1.0.0",
        "database": "healthy" if db_healthy else "unavailable"
    }

# Memory stats endpoint removed - focusing on audio conversation

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

# Minimal stats endpoint for frontend compatibility
@fastapi_app.get("/api/conversation/{conversation_id}/stats")
async def get_conversation_stats(conversation_id: str):
    """Simple stats endpoint for frontend compatibility"""
    return {
        "status": "success",
        "conversation_id": conversation_id,
        "system": "workflow",
        "data": {
            "conversation_id": conversation_id,
            "current_stage": "active",
            "messages_exchanged": 0,
            "last_activity": datetime.now().isoformat(),
            "persona": "friendly"
        }
    }

# Global instances
ai_agent = None
pitch_workflow = None

# Global dictionary to store conversation states
conversation_states = {}

@fastapi_app.on_event("startup")
async def startup_ai_systems():
    """Initialize all AI agents and workflows when the application starts"""
    global ai_agent, pitch_workflow
    
    try:
        # Initialize the improved AI agent
        initialize_improved_agent()
        logger.info("Improved AI Agent initialized successfully")
        
        # Initialize the LangGraph workflow with database service
        initialize_pitch_workflow(db_service)
        pitch_workflow = get_pitch_workflow()
        logger.info("LangGraph workflow initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI systems: {str(e)}", exc_info=True)
        # Don't raise here to allow app to start even if AI systems fail
        ai_agent = None
        pitch_workflow = None

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

@fastapi_app.get("/api/personas")
async def get_personas():
    """Get all available investor personas"""
    try:
        personas = {
            "skeptical": {
                "name": "Sarah Martinez",
                "title": "Senior Partner at Venture Capital",
                "description": "Analytical and thorough investor who asks tough questions about market validation, financial projections, and competitive advantages. Expects detailed data and proof points.",
                "personality_traits": [
                    "Detail-oriented",
                    "Risk-averse", 
                    "Data-driven",
                    "Challenging"
                ],
                "focus_areas": [
                    "Market validation",
                    "Financial projections",
                    "Competitive analysis",
                    "Risk assessment"
                ],
                "typical_questions": [
                    "What's your customer acquisition cost?",
                    "How do you plan to defend against competitors?",
                    "What are your unit economics?",
                    "What evidence do you have of product-market fit?"
                ]
            },
            "technical": {
                "name": "Dr. Alex Chen",
                "title": "CTO-turned-Investor at TechVentures",
                "description": "Tech-focused investor with deep technical expertise. Interested in architecture, scalability, and technical innovation. Values technical depth and implementation details.",
                "personality_traits": [
                    "Technically savvy",
                    "Innovation-focused",
                    "Architecture-minded",
                    "Implementation-oriented"
                ],
                "focus_areas": [
                    "Technical architecture",
                    "Scalability",
                    "Innovation",
                    "Development process"
                ],
                "typical_questions": [
                    "How does your technology stack scale?",
                    "What's your technical differentiation?",
                    "How do you handle data security?",
                    "What's your development methodology?"
                ]
            },
            "friendly": {
                "name": "Michael Thompson",
                "title": "Angel Investor & Former Entrepreneur",
                "description": "Supportive investor focused on founder journey and team dynamics. Emphasizes mentorship and long-term relationship building. Interested in founder-market fit.",
                "personality_traits": [
                    "Supportive",
                    "Mentor-oriented",
                    "Relationship-focused",
                    "Encouraging"
                ],
                "focus_areas": [
                    "Founder journey",
                    "Team dynamics",
                    "Vision and passion",
                    "Market opportunity"
                ],
                "typical_questions": [
                    "What inspired you to start this company?",
                    "How did you identify this problem?",
                    "What's your long-term vision?",
                    "How can I help you succeed?"
                ]
            }
        }
        
        return {
            "success": True,
            "personas": personas,
            "total_count": len(personas),
            "available_personas": list(personas.keys())
        }
        
    except Exception as e:
        logger.error(f"Error getting personas: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get personas")

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

# Audio Conversation Management Endpoints
@fastapi_app.get("/api/audio/conversations")
async def get_audio_conversations(limit: int = 50):
    """Get list of all audio conversations"""
    try:
        from app.services.audio_conversation_storage import list_conversation_files
        
        files = list_conversation_files(limit)
        return {
            "conversations": files,
            "total": len(files),
            "message": f"Retrieved {len(files)} audio conversations"
        }
    except Exception as e:
        logger.error(f"Error getting audio conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get audio conversations")

@fastapi_app.get("/api/audio/conversation/{session_id}")
async def get_session_audio_conversation(session_id: str):
    """Get audio conversation data for a specific session"""
    try:
        db_service = get_database_service()
        if not db_service:
            raise HTTPException(status_code=500, detail="Database service not available")
        
        # Get audio conversation from database
        audio_conversation = await db_service.get_audio_conversation(session_id)
        if not audio_conversation:
            raise HTTPException(status_code=404, detail="Audio conversation not found")
        
        return {
            "session_id": session_id,
            "audio_conversation": audio_conversation,
            "message": "Audio conversation retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session audio conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get session audio conversation")

@fastapi_app.get("/api/audio/sessions-with-audio")
async def get_sessions_with_audio(limit: int = 50):
    """Get sessions that have audio recordings"""
    try:
        db_service = get_database_service()
        if not db_service:
            raise HTTPException(status_code=500, detail="Database service not available")
        
        sessions = await db_service.get_sessions_with_audio(limit)
        return {
            "sessions": sessions,
            "total": len(sessions),
            "message": f"Retrieved {len(sessions)} sessions with audio recordings"
        }
    except Exception as e:
        logger.error(f"Error getting sessions with audio: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get sessions with audio")

@fastapi_app.delete("/api/audio/conversation/{session_id}")
async def delete_session_audio_conversation(session_id: str):
    """Delete audio conversation for a specific session"""
    try:
        db_service = get_database_service()
        if not db_service:
            raise HTTPException(status_code=500, detail="Database service not available")
        
        # Get audio conversation data first
        audio_conversation = await db_service.get_audio_conversation(session_id)
        if not audio_conversation:
            raise HTTPException(status_code=404, detail="Audio conversation not found")
        
        # Delete from Google Cloud Storage
        from app.services.audio_conversation_storage import delete_conversation_file
        filename = audio_conversation.get('audio_filename')
        if filename:
            storage_deleted = delete_conversation_file(filename)
            if not storage_deleted:
                logger.warning(f"Failed to delete audio file from storage: {filename}")
        
        # Delete from database
        db_deleted = await db_service.delete_audio_conversation(session_id)
        if not db_deleted:
            raise HTTPException(status_code=500, detail="Failed to delete audio conversation from database")
        
        return {
            "session_id": session_id,
            "message": "Audio conversation deleted successfully",
            "storage_deleted": storage_deleted if filename else False,
            "database_deleted": db_deleted
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session audio conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete session audio conversation")

@fastapi_app.post("/api/audio/finalize/{session_id}")
async def finalize_session_audio(session_id: str):
    """Manually finalize audio recording for a session and get GCP URL"""
    try:
        logger.info(f"Manually finalizing audio for session {session_id}")
        
        # Finalize audio recording
        audio_url = finalize_session_recording(session_id, use_mp3=True)
        
        if audio_url:
            audio_info = get_session_audio_info(session_id)
            return {
                "success": True,
                "message": "Audio recording finalized successfully",
                "audio_url": audio_url,
                "audio_info": audio_info
            }
        else:
            raise HTTPException(status_code=404, detail="No audio recording found for this session")
            
    except Exception as e:
        logger.error(f"Error finalizing audio for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to finalize audio: {str(e)}")

@fastapi_app.get("/api/audio/storage-info")
async def get_audio_storage_info():
    """Get information about audio storage configuration"""
    try:
        from app.services.audio_conversation_storage import audio_storage_service
        
        # Check if storage is properly configured
        storage_available = audio_storage_service.client is not None and audio_storage_service.bucket is not None
        
        return {
            "storage_available": storage_available,
            "bucket_name": audio_storage_service.bucket_name,
            "storage_provider": "Google Cloud Storage",
            "audio_formats_supported": ["wav", "mp3"],
            "default_format": audio_storage_service.format,
            "sample_rate": audio_storage_service.sample_rate,
            "channels": audio_storage_service.channels,
            "message": "Audio storage configuration retrieved"
        }
    except Exception as e:
        logger.error(f"Error getting audio storage info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get audio storage info")

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

@fastapi_app.get("/api/debug/video-analysis/{session_id}")
async def debug_video_analysis_status(session_id: str):
    """Debug: Check video analysis status for a session"""
    try:
        from app.services.langgraph_workflow import get_pitch_workflow
        
        workflow = get_pitch_workflow()
        if not workflow:
            return {"error": "Workflow not available"}
        
        config = {"configurable": {"thread_id": session_id}}
        current_state = workflow.workflow.get_state(config)
        
        if not current_state.values:
            return {"error": "Session not found"}
        
        state = current_state.values
        
        return {
            "session_id": session_id,
            "video_analysis_enabled": state.get('video_analysis_enabled', False),
            "video_insights_count": len(state.get('video_insights', [])),
            "gesture_feedback_count": len(state.get('gesture_feedback', [])),
            "posture_feedback_count": len(state.get('posture_feedback', [])),
            "expression_feedback_count": len(state.get('expression_feedback', [])),
            "video_analysis_available": VIDEO_ANALYSIS_AVAILABLE,
            "recent_video_insights": state.get('video_insights', [])[-3:] if state.get('video_insights') else [],
            "fix_status": "Video analysis fix applied - scores should be reasonable even when video not available"
        }
        
    except Exception as e:
        logger.error(f"Error getting video analysis debug info: {str(e)}")
        return {"error": str(e)}

# ===== NEW IMPROVED AI SYSTEMS ENDPOINTS =====

@fastapi_app.post("/api/pitch/start")
async def start_pitch_practice(
    persona: str = "friendly",
    system: str = "workflow"
):
    """Start a new pitch practice session with improved AI systems
    
    Args:
        persona: Investor personality (friendly, skeptical, technical)
        system: AI system to use (workflow, improved)
    
    Returns:
        Session info with initial greeting
    """
    try:
        # Validate inputs
        valid_personas = ["friendly", "skeptical", "technical"]
        valid_systems = ["workflow", "improved"]
        
        if persona not in valid_personas:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid persona. Choose from: {', '.join(valid_personas)}"
            )
        
        if system not in valid_systems:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid system. Choose from: {', '.join(valid_systems)}"
            )
        
        # Generate a unique session ID
        import uuid
        session_id = str(uuid.uuid4())
        
        # Start the session based on system type
        if system == "workflow":
            # Use async workflow function for proper database integration
            session_data = await start_pitch_session_async(session_id, persona)
        else:
            # For improved system, use the old method and manually save to database
            session_data = start_improved_conversation(session_id, persona)
            
            # Create database record for improved system
            if db_service and session_id:
                try:
                    await db_service.create_session({
                        "session_id": session_id,
                        "persona_used": persona,
                        "status": "active"
                    })
                    logger.info(f"Created database record for session: {session_id}")
                except Exception as db_error:
                    logger.error(f"Failed to create database record for session: {db_error}")
        
        if "error" in session_data:
            raise HTTPException(status_code=500, detail=session_data["error"])
        
        logger.info(f"Started {system} pitch session with {persona} persona: {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "message": session_data.get("message"),
            "persona": persona,
            "system": system,
            "stage": session_data.get("stage"),
            "type": session_data.get("type")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting pitch practice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start pitch practice: {str(e)}")

@fastapi_app.post("/api/pitch/message")
async def process_pitch_message(
    session_id: str,
    message: str,
    background_tasks: BackgroundTasks,
    system: str = "workflow",
    persona: str = "friendly"
):
    """Process a founder's message in the pitch practice session
    
    Args:
        session_id: Session identifier
        message: Founder's message/response
        background_tasks: For generating audio response
        system: Which AI system to use ('workflow' or 'improved')
        persona: Investor persona for TTS generation
    
    Returns:
        Investor's response with session info
    """
    try:
        if not session_id or not message:
            raise HTTPException(status_code=400, detail="session_id and message are required")
        
        # Process the message using the appropriate system
        if system == "improved":
            # Use the improved AI agent
            response_data = generate_improved_response(session_id, message)
            investor_response = response_data.get("response", "")
            
            # Format response to match expected structure
            formatted_response = {
                "message": investor_response,
                "stage": response_data.get("current_stage"),
                "complete": response_data.get("is_complete", False),
                "type": "improved",
                "insights": {
                    "key_points": response_data.get("key_points", []),
                    "suggestions": response_data.get("suggestions", [])
                }
            }
        else:
            # Use the LangGraph workflow with database integration
            response_data = await process_pitch_message_async(session_id, message)
            if "error" in response_data:
                raise HTTPException(status_code=404, detail=response_data["error"])
            
            investor_response = response_data.get("message", "")
            formatted_response = response_data
        
        # Generate audio response in background
        if investor_response:
            response_audio_path = os.path.join(RESPONSE_DIR, f"{session_id}_latest_response.mp3")
            background_tasks.add_task(
                convert_text_to_speech_with_persona,
                investor_response,
                persona,
                response_audio_path
            )
        
        logger.info(f"Processed message for {system} session {session_id}")
        
        return {
            "success": True,
            "message": investor_response,
            "session_id": session_id,
            "stage": formatted_response.get("stage"),
            "complete": formatted_response.get("complete", False),
            "insights": formatted_response.get("insights", {}),
            "type": formatted_response.get("type", system),
            "audio_url": f"/download/{session_id}_latest_response.mp3" if investor_response else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing pitch message: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@fastapi_app.get("/api/pitch/analytics/{session_id}")
async def get_pitch_session_analytics(session_id: str):
    """Get analytics for a pitch practice session
    
    Args:
        session_id: Session identifier
    
    Returns:
        Session analytics including duration, stages completed, insights, etc.
    """
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        if not db_service:
            raise HTTPException(status_code=500, detail="Database service not available")
        
        # Try to get from database first
        quick_analytics = await db_service.get_quick_analytics(session_id)
        
        if quick_analytics:
            return {
                "success": True,
                "analytics": {
                    "overall_score": quick_analytics["overall_score"],
                    "key_insights": quick_analytics["key_insights"],
                    "completion_percentage": quick_analytics["completion_percentage"],
                    "current_topics": quick_analytics["current_topics"],
                    "generated_at": quick_analytics["generated_at"]
                }
            }
        
        # Fallback to original method if not in database
        analytics = get_pitch_analytics(session_id)
        
        if "error" in analytics:
            raise HTTPException(status_code=404, detail=analytics["error"])
        
        # Save to database for future use
        if analytics and "overall_score" in analytics:
            await db_service.save_quick_analytics({
                "session_id": session_id,
                "overall_score": analytics.get("overall_score", 0),
                "key_insights": analytics.get("key_insights", []),
                "completion_percentage": analytics.get("completion_percentage", 0),
                "current_topics": analytics.get("current_topics", [])
            })
        
        return {
            "success": True,
            "analytics": analytics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pitch analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@fastapi_app.get("/api/pitch/analysis/{session_id}")
async def get_pitch_analysis_report(session_id: str):
    """Generate comprehensive pitch analysis report
    
    Args:
        session_id: Session identifier
    
    Returns:
        Detailed pitch analysis with scores, strengths, weaknesses, recommendations
    """
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        if not db_service:
            raise HTTPException(status_code=500, detail="Database service not available")
        
        # Try to get from database first
        stored_analysis = await db_service.get_analysis(session_id)
        
        if stored_analysis:
            logger.info(f"Retrieved analysis from database for session: {session_id}")
            return {
                "success": True,
                "analysis": stored_analysis
            }
        
        # Generate new analysis if not in database
        analysis = generate_pitch_analysis_report(session_id)
        
        if "error" in analysis:
            if analysis["error"] == "conversation_too_short":
                # Special handling for short conversations
                return {
                    "success": False,
                    "error": "conversation_too_short",
                    "message": analysis.get("message", "The conversation is too short for meaningful analysis."),
                    "minimum_required": analysis.get("minimum_required", 3),
                    "current_count": analysis.get("founder_messages", 0)
                }
            else:
                raise HTTPException(status_code=404, detail=analysis["error"])
        
        # Save the analysis to database
        try:
            await db_service.save_analysis(analysis)
            logger.info(f"Saved new analysis to database for session: {session_id}")
        except Exception as save_error:
            logger.error(f"Failed to save analysis to database: {save_error}")
            # Continue anyway, return the analysis even if saving failed
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating pitch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate analysis: {str(e)}")

@fastapi_app.post("/api/pitch/end/{session_id}")
async def end_pitch_session(session_id: str, request: Request):
    """End pitch session and generate comprehensive analysis
    
    Args:
        session_id: Session identifier
        request: Request object containing optional reason in body
    
    Returns:
        Comprehensive pitch analysis report
    """
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        if not db_service:
            raise HTTPException(status_code=500, detail="Database service not available")
        
        # Parse request body to get reason (optional)
        reason = "user_ended"  # default
        try:
            body = await request.json()
            reason = body.get("reason", "user_ended")
        except:
            # If no JSON body or parsing fails, use default
            pass
        
        # End the session and generate analysis using async function
        analysis_result = await end_pitch_session_async(session_id, reason)
        
        if "error" in analysis_result:
            if analysis_result["error"] == "conversation_too_short":
                # Special handling for short conversations
                return {
                    "success": False,
                    "error": "conversation_too_short",
                    "message": analysis_result.get("message", "The conversation is too short for meaningful analysis."),
                    "minimum_required": analysis_result.get("minimum_required", 3),
                    "current_count": analysis_result.get("founder_messages", 0)
                }
            else:
                raise HTTPException(status_code=404, detail=analysis_result["error"])
        
        # The analysis_result IS the analysis (not wrapped in another dict)
        analysis = analysis_result
        if not analysis or "error" in analysis:
            raise HTTPException(status_code=404, detail="No analysis generated")
        
        # Database operations are already handled by end_pitch_session_async
        logger.info(f"Session {session_id} ended successfully via API")
        
        # Get audio URL - first try the cached URL from the websocket handler
        audio_url = None
        audio_info = None
        try:
            # First check if we have a cached URL from the websocket handler
            cached_url = get_session_audio_url(session_id)
            if cached_url:
                logger.info(f"Found cached audio URL for session {session_id}: {cached_url}")
                audio_url = cached_url
                # No need to get audio_info as the session is already finalized
            else:
                # If no cached URL, try to finalize the recording
                logger.info(f"No cached URL found, finalizing audio recording for session {session_id}")
                
                # Force MP3 format for smaller file size and better compatibility
                audio_url = finalize_session_recording(session_id, use_mp3=True)
                if audio_url:
                    logger.info(f"Audio recording finalized successfully: {audio_url}")
                    audio_info = get_session_audio_info(session_id)
                else:
                    # If no audio URL, try to create a fallback recording
                    logger.warning(f"No audio recording found for session {session_id}, attempting fallback...")
                    from app.services.audio_conversation_storage import start_session_recording, add_user_audio, add_ai_audio
                    
                    # Create a fallback recording with dummy data if needed
                    start_session_recording(session_id, "friendly")
                    
                    # Try to finalize again
                    audio_url = finalize_session_recording(session_id, use_mp3=True)
                    if audio_url:
                        logger.info(f"Fallback audio recording created: {audio_url}")
                    else:
                        logger.error(f"Failed to create fallback audio recording for session {session_id}")
        except Exception as e:
            logger.error(f"Error finalizing audio recording: {str(e)}")
            # Don't fail the entire request if audio finalization fails
        
        response_data = {
            "success": True,
            "message": "Session ended successfully",
            "analysis": analysis,
            # Always include audio field, even if URL is None
            "audio": {
                "url": audio_url,
                "info": audio_info
            }
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending pitch session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")

@fastapi_app.get("/api/pitch/report/{session_id}")
async def get_formatted_pitch_report(session_id: str):
    """Get formatted pitch analysis report for frontend display
    
    Args:
        session_id: Session identifier
    
    Returns:
        Formatted analysis report optimized for frontend display
    """
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        analysis = generate_pitch_analysis_report(session_id)
        
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        
        # Format the analysis for better frontend consumption
        formatted_report = {
            "session_info": {
                "session_id": analysis.get("session_id"),
                "founder_name": analysis.get("founder_name", "Unknown"),
                "company_name": analysis.get("company_name", "Unknown"),
                "duration_minutes": analysis.get("session_duration_minutes", 0),
                "completion_percentage": analysis.get("completion_percentage", 0),
                "stages_completed": analysis.get("stages_completed", 0),
                "total_stages": analysis.get("total_stages", 9),
                "persona_used": analysis.get("persona_used", "Unknown"),
                "generated_at": analysis.get("generated_at"),
                "end_reason": analysis.get("end_reason", "analysis_requested")
            },
            "scores": {
                "overall_score": analysis.get("overall_score", 0),
                "confidence_level": analysis.get("confidence_level", "Unknown"),
                "pitch_readiness": analysis.get("pitch_readiness", "Unknown"),
                "stage_scores": analysis.get("stage_scores", {})
            },
            "category_scores": analysis.get("category_scores", {}),
            "feedback": {
                "strengths": analysis.get("strengths", []),
                "weaknesses": analysis.get("weaknesses", []),
                "key_recommendations": analysis.get("key_recommendations", []),
                "next_steps": analysis.get("next_steps", []),
                "founder_performance": analysis.get("founder_performance", []),
                "what_worked": analysis.get("what_worked", []),
                "what_didnt_work": analysis.get("what_didnt_work", [])
            },
            "insights": {
                "investor_perspective": analysis.get("investor_perspective", ""),
                "summary": f"Completed {analysis.get('stages_completed', 0)} out of 9 pitch stages with an overall score of {analysis.get('overall_score', 0)}/100."
            }
        }
        
        return {
            "success": True,
            "report": formatted_report
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting formatted pitch report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")


# WebSocket handlers removed to prevent conflicts with audio_websocket_handler.py
# All real-time audio processing is now handled by the dedicated audio handler

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
