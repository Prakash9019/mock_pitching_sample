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
    initialize_pitch_workflow
)

from app.services.intelligent_ai_agent_improved import (
    start_improved_conversation,
    generate_improved_response,
    initialize_improved_agent,
    improved_agent
)

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