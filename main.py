from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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

# Socket.IO integration
import socketio
from fastapi.middleware.cors import CORSMiddleware

# Import service modules
from app.services.transcription import transcribe_audio
from app.services.enhanced_text_to_speech import convert_text_to_speech_with_persona

# Import LangGraph workflow and improved AI agent
from app.services.langgraph_workflow import (
    start_pitch_session as start_practice_session,
    start_pitch_session_with_message,
    process_pitch_message as handle_practice_message,
    get_pitch_workflow,
    initialize_pitch_workflow,
    generate_pitch_analysis_report,
    end_pitch_session_with_analysis,
    get_pitch_analytics
)

from app.services.intelligent_ai_agent_improved import (
    start_improved_conversation,
    generate_improved_response,
    initialize_improved_agent,
    improved_agent
)

# Import database components
from app.database import connect_to_mongo, close_mongo_connection, get_database
from app.services.database_service import DatabaseService
from app.models import PitchAnalysis, PitchSession

# Integration example import removed - focusing on audio conversation

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

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_PATH, "templates"))

# Global database service instance
db_service: Optional[DatabaseService] = None

# Database startup and shutdown events
@fastapi_app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    global db_service
    try:
        await connect_to_mongo()
        database = await get_database()
        db_service = DatabaseService(database)
        logger.info("Database service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

@fastapi_app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown"""
    try:
        await close_mongo_connection()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")

# Mount static files
fastapi_app.mount("/static", StaticFiles(directory=os.path.join(BASE_PATH, "static")), name="static")

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
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@fastapi_app.get("/pitch-analysis/{session_id}", response_class=HTMLResponse)
async def pitch_analysis_page(request: Request, session_id: str):
    """Serve the pitch analysis page"""
    return templates.TemplateResponse("pitch_analysis.html", {"request": request, "session_id": session_id})


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
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Mock Investor Pitch",
        "version": "1.0.0"
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
async def startup_event():
    """Initialize all AI agents and workflows when the application starts"""
    global ai_agent, pitch_workflow
    
    try:
        # Initialize the improved AI agent
        initialize_improved_agent()
        logger.info("Improved AI Agent initialized successfully")
        
        # Initialize the LangGraph workflow
        initialize_pitch_workflow()
        pitch_workflow = get_pitch_workflow()
        logger.info("LangGraph workflow initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI systems: {str(e)}", exc_info=True)
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
        
        # Start the session
        session_data = start_practice_session(system, persona)
        
        if "error" in session_data:
            raise HTTPException(status_code=500, detail=session_data["error"])
        
        session_id = session_data.get("session_id")
        
        # Create database record for the session
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
                # Continue anyway, don't fail the session start
        
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
            # Use the LangGraph workflow
            response_data = handle_practice_message(session_id, message)
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
async def end_pitch_session(session_id: str, reason: str = "user_ended"):
    """End pitch session and generate comprehensive analysis
    
    Args:
        session_id: Session identifier
        reason: Reason for ending session (user_ended, completed, timeout, etc.)
    
    Returns:
        Comprehensive pitch analysis report
    """
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        if not db_service:
            raise HTTPException(status_code=500, detail="Database service not available")
        
        # End the session and generate analysis
        analysis = end_pitch_session_with_analysis(session_id, reason)
        
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        
        # Save the analysis to database
        try:
            await db_service.save_analysis(analysis)
            logger.info(f"Saved final analysis to database for session: {session_id}")
        except Exception as save_error:
            logger.error(f"Failed to save final analysis to database: {save_error}")
        
        # Update session status in database
        try:
            session_duration = analysis.get("session_duration_minutes", 0)
            await db_service.end_session(session_id, session_duration)
            logger.info(f"Updated session status in database for session: {session_id}")
        except Exception as update_error:
            logger.error(f"Failed to update session status: {update_error}")
        
        return {
            "success": True,
            "message": "Session ended successfully",
            "analysis": analysis
        }
        
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
            "feedback": {
                "strengths": analysis.get("strengths", []),
                "weaknesses": analysis.get("weaknesses", []),
                "key_recommendations": analysis.get("key_recommendations", []),
                "next_steps": analysis.get("next_steps", [])
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


@sio.event
async def connect(sid, environ):
    logger.info(f"WebSocket connected: {sid}")

@sio.event
async def text_message(sid, data):
    """Handle real-time transcribed text messages
    
    Args:
        sid: Socket.IO session ID
        data: Message data (can be string or dict with text, system, persona)
    """
    try:
        logger.info(f"Received message from {sid}")
        
        # Extract message data
        if isinstance(data, dict):
            message_text = data.get('text', '').strip()
            system = data.get('system', 'workflow')
            persona = data.get('persona', 'friendly')
            session_id = data.get('session_id', sid)  # Use provided session_id or fallback to sid
        else:
            # Backward compatibility
            message_text = str(data).strip()
            system = 'workflow'
            persona = 'friendly'
            session_id = sid
        
        if not message_text:
            logger.warning("Received empty message")
            await sio.emit("error", {"message": "Empty message received"}, to=sid)
            return
        
        logger.info(f"Processing message for {session_id} using {system} system")
        
        try:
            # Check if we need to start a new session
            is_new_session = False
            
            if system == "improved":
                # Check if improved agent is available
                if not improved_agent:
                    raise Exception("Improved AI agent not initialized")
                
                # Check if conversation exists
                if not hasattr(improved_agent, 'conversations') or session_id not in improved_agent.conversations:
                    logger.info(f"Starting new improved conversation for {session_id}")
                    improved_agent.start_conversation(session_id, persona)
                    is_new_session = True
                
                # Process message with improved agent
                response_data = generate_improved_response(session_id, message_text)
                
                # Format response
                response = {
                    "message": response_data.get("response", ""),
                    "stage": response_data.get("current_stage", "introduction"),
                    "complete": response_data.get("is_complete", False),
                    "insights": {
                        "suggestions": response_data.get("suggestions", []),
                        "key_points": response_data.get("key_points", [])
                    },
                    "type": "improved"
                }
                
            else:  # Default to workflow
                # Check if workflow is available
                if not pitch_workflow:
                    raise Exception("LangGraph workflow not initialized")
                
                # Check if this is a new session by checking if we have the session in memory
                # For simplicity, we'll treat each WebSocket connection as potentially new
                # and let the workflow handle session state internally
                
                # Try to process the message - if session doesn't exist, it will return an error
                logger.info(f"Attempting to process message with existing session: {session_id}")
                response_data = handle_practice_message(session_id, message_text)
                logger.info(f"Response from handle_practice_message: {response_data}")
                
                # Check if we got an error response (session doesn't exist)
                if isinstance(response_data, dict) and "error" in response_data:
                    # Session doesn't exist, start a new one with the user's message
                    logger.info(f"Starting new workflow session with message for {session_id}: {response_data.get('error')}")
                    response_data = start_pitch_session_with_message(session_id, persona, message_text)
                    is_new_session = True
                    logger.info(f"Session started with message: {response_data}")
                    
                    # If we still get an error, fall back to greeting
                    if isinstance(response_data, dict) and "error" in response_data:
                        logger.warning(f"Still getting error after starting session with message: {response_data.get('error')}")
                        response_data = {
                            "message": "Hello! I'm excited to hear your pitch. Please introduce yourself and your company.",
                            "stage": "greeting",
                            "complete": False,
                            "insights": {}
                        }
                
                # Format response
                response = {
                    "message": response_data.get("message", ""),
                    "stage": response_data.get("stage", "introduction"),
                    "complete": response_data.get("complete", False),
                    "insights": response_data.get("insights", {}),
                    "type": "workflow"
                }
            
            # Generate audio response and wait for completion
            if response["message"]:
                # Use timestamp to ensure unique audio files and prevent caching issues
                timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
                audio_filename = f"{session_id}_response_{timestamp}.mp3"
                response_audio_path = os.path.join(RESPONSE_DIR, audio_filename)
                
                # Run TTS in thread pool and wait for completion
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    convert_text_to_speech_with_persona,
                    response["message"],
                    persona,
                    response_audio_path
                )
                response["audio_url"] = f"/download/{audio_filename}"
                logger.info(f"Audio generated and ready at: {response_audio_path}")
                
                # Clean up old audio files for this session (keep only last 5)
                try:
                    cleanup_old_audio_files(session_id)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup old audio files: {cleanup_error}")
            
            # Send response
            await sio.emit("response", response, to=sid)
            
            # Send session started event if new
            if is_new_session:
                await sio.emit("session_started", {
                    "session_id": session_id,
                    "system": system,
                    "persona": persona,
                    "message": "New session started"
                }, to=sid)
            
            logger.info(f"Sent response to {sid} for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            await sio.emit("error", {
                "message": f"Error processing message: {str(e)}",
                "type": "error"
            }, to=sid)
            
    except Exception as e:
        logger.error(f"Unexpected error in text_message: {str(e)}", exc_info=True)
        try:
            await sio.emit("error", {
                "message": "Internal server error",
                "type": "error"
            }, to=sid)
        except:
            pass  # Socket might be disconnected

@sio.event
async def audio_chunk(sid, data):
    """Handle incoming audio chunks, transcribe, and respond with AI-generated audio.
    
    This endpoint processes audio chunks and routes them to either the LangGraph workflow
    or improved AI agent based on the system parameter.
    
    For new implementations, it's recommended to use the text-based WebSocket endpoint.
    """
    try:
        logger.info(f"Received audio chunk from {sid}")
        
        # Extract audio data and parameters from the incoming message
        if isinstance(data, dict) and 'audio' in data:
            audio_data = data['audio']
            persona = data.get('persona', 'friendly')  # Default to 'friendly' if not provided
            system = data.get('system', 'workflow')    # Default to 'workflow' if not provided
            session_id = data.get('session_id', sid)   # Use provided session_id or fallback to sid
        else:
            # Backward compatibility with older clients (legacy mode)
            audio_data = data
            persona = 'friendly'
            system = 'workflow'
            session_id = sid
        
        # Save the incoming audio to a temporary file for transcription
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        try:
            logger.info("Starting transcription...")
            transcription_result = transcribe_audio(temp_audio_path)
            transcript_text = transcription_result.get('text', '').strip()
            confidence = transcription_result.get('confidence', 0.0)
            
            if not transcript_text:
                logger.warning("Empty transcription result")
                await sio.emit("error", {
                    "message": "Could not transcribe audio. Please try again.",
                    "type": "transcription_error"
                }, to=sid)
                return
                
            logger.info(f"Transcription complete (confidence: {confidence:.2f}): {transcript_text[:100]}...")
            
            # Process the transcribed text using the appropriate system
            try:
                if system == "improved":
                    if not improved_agent:
                        raise Exception("Improved AI agent not initialized")
                    
                    # Check if conversation exists for improved agent
                    if not hasattr(improved_agent, 'conversations') or session_id not in improved_agent.conversations:
                        logger.info(f"Starting new improved conversation for {session_id}")
                        improved_agent.start_conversation(session_id, persona)
                    
                    # Process message with improved agent
                    response_data = generate_improved_response(session_id, transcript_text)
                    
                    # Format response
                    response = {
                        "message": response_data.get("response", ""),
                        "stage": response_data.get("current_stage", "introduction"),
                        "complete": response_data.get("is_complete", False),
                        "insights": {
                            "suggestions": response_data.get("suggestions", []),
                            "key_points": response_data.get("key_points", [])
                        },
                        "type": "improved",
                        "transcript": transcript_text
                    }
                    
                else:  # Default to workflow
                    if not pitch_workflow:
                        raise Exception("LangGraph workflow not initialized")
                    
                    # Try to process the message - if session doesn't exist, it will return an error
                    response_data = handle_practice_message(session_id, transcript_text)
                    
                    # Check if we got an error response (session doesn't exist)
                    if isinstance(response_data, dict) and "error" in response_data:
                        # Session doesn't exist, start a new one
                        logger.info(f"Starting new workflow session for {session_id}: {response_data.get('error')}")
                        session_start_result = start_practice_session(session_id, persona)
                        
                        # Now process the user's message with the new session
                        response_data = handle_practice_message(session_id, transcript_text)
                        
                        # If we still get an error, fall back to greeting
                        if isinstance(response_data, dict) and "error" in response_data:
                            logger.warning(f"Still getting error after starting session: {response_data.get('error')}")
                            response_data = {
                                "message": session_start_result.get("message", "Hello! I'm excited to hear your pitch. Please introduce yourself and your company."),
                                "stage": session_start_result.get("stage", "greeting"),
                                "complete": False,
                                "insights": {}
                            }
                    
                    # Format response
                    response = {
                        "message": response_data.get("message", ""),
                        "stage": response_data.get("stage", "introduction"),
                        "complete": response_data.get("complete", False),
                        "insights": response_data.get("insights", {}),
                        "type": "workflow",
                        "transcript": transcript_text
                    }
                
                # Generate audio response and wait for completion
                if response["message"]:
                    response_audio_path = os.path.join(RESPONSE_DIR, f"{session_id}_latest_response.mp3")
                    # Run TTS in thread pool and wait for completion
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        convert_text_to_speech_with_persona,
                        response["message"],
                        persona,
                        response_audio_path
                    )
                    response["audio_url"] = f"/download/{session_id}_latest_response.mp3"
                    logger.info(f"Audio generated and ready at: {response_audio_path}")
                
                # Send the response back to the client
                await sio.emit("response", response, to=sid)
                
                logger.info(f"Sent response to {sid} for session {session_id}")
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}", exc_info=True)
                await sio.emit("error", {
                    "message": f"Error processing message: {str(e)}",
                    "type": "processing_error"
                }, to=sid)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}", exc_info=True)
            await sio.emit("error", {
                "message": f"Error processing audio: {str(e)}",
                "type": "audio_processing_error"
            }, to=sid)
            
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_audio_path}: {e}")
                
    except Exception as e:
        logger.error(f"WebSocket error in audio_chunk: {str(e)}", exc_info=True)
        try:
            await sio.emit("error", {
                "message": "Internal server error",
                "type": "server_error"
            }, to=sid)
        except:
            pass  # Socket might be disconnected


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