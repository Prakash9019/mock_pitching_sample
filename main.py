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
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Socket.IO integration
import socketio
from fastapi.middleware.cors import CORSMiddleware

# Import service modules
from app.services.transcription import transcribe_audio
from app.services.intelligent_ai_agent import get_conversation_statistics
from app.services.enhanced_text_to_speech import convert_text_to_speech_with_persona

# Import improved AI systems
from app.services.integration_example import (
    start_practice_session,
    handle_practice_message,
    get_practice_analytics,
    get_pitch_manager
)

# Import improved AI systems
from app.services.integration_example import (
    start_practice_session,
    handle_practice_message,
    get_practice_analytics,
    get_pitch_manager
)

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

fastapi_app.mount("/static", StaticFiles(directory=os.path.join(BASE_PATH, "static")), name="static")


@fastapi_app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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

# Global dictionary to store conversation states
conversation_states = {}

@fastapi_app.on_event("startup")
async def startup_event():
    """Initialize all AI agents when the application starts"""
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
        
        # Initialize the original AI agent (for backwards compatibility)
        ai_agent = IntelligentAIAgent(llm)
        logger.info("Original AI Agent initialized successfully")
        
        # Initialize the improved pitch manager and systems
        manager = get_pitch_manager()
        logger.info("Improved AI systems initialized successfully")
        
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
        
        logger.info(f"Started {system} pitch session with {persona} persona: {session_data.get('session_id')}")
        
        return {
            "success": True,
            "session_id": session_data.get("session_id"),
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
    background_tasks: BackgroundTasks
):
    """Process a founder's message in the pitch practice session
    
    Args:
        session_id: Session identifier
        message: Founder's message/response
        background_tasks: For generating audio response
    
    Returns:
        Investor's response with session info
    """
    try:
        if not session_id or not message:
            raise HTTPException(status_code=400, detail="session_id and message are required")
        
        # Process the message
        response_data = handle_practice_message(session_id, message)
        
        if "error" in response_data:
            raise HTTPException(status_code=404, detail=response_data["error"])
        
        # Generate audio response in background if system supports it
        investor_response = response_data.get("message", "")
        if investor_response:
            # Determine persona from session (you might want to store this separately)
            session_analytics = get_practice_analytics(session_id)
            persona = session_analytics.get("persona", "friendly")
            
            # Generate audio response
            response_audio_path = os.path.join(RESPONSE_DIR, f"{session_id}_latest_response.mp3")
            background_tasks.add_task(
                convert_text_to_speech_with_persona,
                investor_response,
                response_audio_path,
                persona
            )
        
        logger.info(f"Processed message for session {session_id}: {message[:50]}...")
        
        return {
            "success": True,
            "message": investor_response,
            "session_id": session_id,
            "stage": response_data.get("stage"),
            "complete": response_data.get("complete", False),
            "insights": response_data.get("insights", {}),
            "type": response_data.get("type"),
            "audio_url": f"/download/{session_id}_latest_response.mp3" if investor_response else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing pitch message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@fastapi_app.get("/api/pitch/analytics/{session_id}")
async def get_pitch_session_analytics(session_id: str):
    """Get comprehensive analytics for a pitch practice session
    
    Args:
        session_id: Session identifier
    
    Returns:
        Detailed session analytics and insights
    """
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        analytics = get_practice_analytics(session_id)
        
        if "error" in analytics:
            raise HTTPException(status_code=404, detail=analytics["error"])
        
        logger.info(f"Retrieved analytics for session {session_id}")
        
        return {
            "success": True,
            "analytics": analytics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pitch analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@fastapi_app.get("/api/pitch/sessions/active")
async def get_active_pitch_sessions():
    """Get all active pitch practice sessions
    
    Returns:
        List of active sessions with basic info
    """
    try:
        manager = get_pitch_manager()
        active_sessions = []
        
        for session_id, session_info in manager.active_sessions.items():
            try:
                analytics = get_practice_analytics(session_id)
                if "error" not in analytics:
                    active_sessions.append({
                        "session_id": session_id,
                        "type": session_info.get("type"),
                        "persona": session_info.get("persona"),
                        "founder_name": analytics.get("founder_name", ""),
                        "company_name": analytics.get("company_name", ""),
                        "current_stage": analytics.get("current_stage"),
                        "duration_minutes": analytics.get("duration_minutes", 0)
                    })
            except Exception as e:
                logger.warning(f"Error getting info for session {session_id}: {e}")
                continue
        
        return {
            "success": True,
            "active_sessions": active_sessions,
            "total_sessions": len(active_sessions)
        }
        
    except Exception as e:
        logger.error(f"Error getting active sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active sessions: {str(e)}")

@fastapi_app.post("/api/pitch/compare")
async def compare_ai_systems(
    test_messages: list = None
):
    """Compare the performance of different AI systems with the same conversation
    
    Args:
        test_messages: List of test messages to send to both systems
    
    Returns:
        Comparison results between improved agent and workflow systems
    """
    try:
        # Default test conversation if none provided
        if not test_messages:
            test_messages = [
                "Hi, I'm Alex and my startup is GreenTech Solutions",
                "We're solving climate change by helping companies reduce their carbon footprint",
                "Our target market is mid-size companies that want to go carbon neutral",
                "We make money through SaaS subscriptions and consulting services"
            ]
        
        manager = get_pitch_manager()
        comparison_results = manager.compare_systems(test_messages)
        
        if "error" in comparison_results:
            raise HTTPException(status_code=500, detail=comparison_results["error"])
        
        logger.info("Completed AI systems comparison")
        
        return {
            "success": True,
            "comparison": comparison_results,
            "test_messages": test_messages
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing AI systems: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to compare systems: {str(e)}")

# ===== END OF NEW ENDPOINTS =====

@fastapi_app.get("/api/pitch/status")
async def get_pitch_systems_status():
    """Get the status of all pitch practice systems
    
    Returns:
        Status of improved agent and LangGraph workflow systems
    """
    try:
        manager = get_pitch_manager()
        
        # Test both systems quickly
        status = {
            "success": True,
            "systems": {
                "improved_agent": {
                    "available": manager.improved_agent is not None,
                    "initialized": True if manager.improved_agent else False
                },
                "langgraph_workflow": {
                    "available": manager.workflow_agent is not None,
                    "initialized": True if manager.workflow_agent else False
                }
            },
            "active_sessions": len(manager.active_sessions),
            "supported_personas": ["friendly", "skeptical", "technical"],
            "supported_systems": ["improved", "workflow"]
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting pitch systems status: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "systems": {
                "improved_agent": {"available": False, "initialized": False},
                "langgraph_workflow": {"available": False, "initialized": False}
            }
        }

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