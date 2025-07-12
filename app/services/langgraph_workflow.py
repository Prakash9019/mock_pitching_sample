# langgraph_workflow.py
"""
LangGraph workflow for intelligent pitch practice sessions
"""

import os
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
import json
import statistics

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
from .intelligent_ai_agent_improved import INVESTOR_PERSONAS, PitchStage, PITCH_STAGES
from .database_service import DatabaseService

# Configure logging
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL","gemini-2.5-flash-lite-preview-06-17"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    convert_system_message_to_human=True
)

class PitchWorkflowState(TypedDict):
    """State for the pitch practice workflow"""
    # Conversation tracking
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_id: str
    
    # Current context
    current_stage: str
    stage_progress: Dict[str, Any]
    persona: str
    
    # Founder information
    founder_name: str
    company_name: str
    
    # Workflow control
    should_transition: bool
    workflow_complete: bool
    current_question: str
    question_answered: bool
    follow_up_count: int  # Track how many follow-up questions we've asked
    
    # Analytics
    session_start: str
    stage_durations: Dict[str, float]
    questions_asked: List[str]
    key_insights: Dict[str, List[str]]
    
    # Video analysis integration
    video_analysis_enabled: bool
    video_insights: List[str]
    gesture_feedback: List[str]
    posture_feedback: List[str]
    expression_feedback: List[str]

class PitchWorkflowAgent:
    """LangGraph-based pitch practice workflow"""
    
    def __init__(self, db_service: DatabaseService = None):
        """
        Initialize a new PitchWorkflowAgent instance
        
        Creates the LangGraph workflow and initializes the memory as an empty dictionary.
        Also creates a MemorySaver instance to save the state of the workflow.
        """
        # Initialize memory_saver first as it's used in _create_workflow
        self.memory_saver = MemorySaver()
        self.workflow = self._create_workflow()
        self.memory = {}  # Initialize memory as an empty dictionary
        self.db_service = db_service  # Database service for persistence
        
    async def _log_to_database(self, session_id: str, message_type: str, content: str, persona: str = None):
        """Log conversation message to database"""
        if self.db_service:
            try:
                await self.db_service.log_conversation({
                    "session_id": session_id,
                    "message_type": message_type,
                    "content": content,
                    "persona": persona
                })
            except Exception as e:
                logger.error(f"Failed to log conversation to database: {e}")
    
    async def _save_session_to_database(self, session_id: str, persona: str, founder_name: str = None, company_name: str = None):
        """Save or update session in database"""
        if self.db_service:
            try:
                session_data = {
                    "session_id": session_id,
                    "persona_used": persona,
                    "status": "active"
                }
                if founder_name:
                    session_data["founder_name"] = founder_name
                if company_name:
                    session_data["company_name"] = company_name
                
                logger.debug(f"Preparing to save session data: {session_data}")
                    
                # Try to update first, create if doesn't exist
                existing = await self.db_service.get_session(session_id)
                if existing:
                    logger.info(f"Updating existing session {session_id}")
                    await self.db_service.update_session(session_id, session_data)
                else:
                    logger.info(f"Creating new session {session_id}")
                    await self.db_service.create_session(session_data)
                    
                logger.info(f"Session {session_id} saved to database successfully")
            except Exception as e:
                logger.error(f"Failed to save session to database: {e}", exc_info=True)
    
    async def _save_analytics_to_database(self, session_id: str, analytics: dict):
        """Save quick analytics to database"""
        if self.db_service:
            try:
                # Convert key_insights from dict to list format
                key_insights = analytics.get("key_insights", {})
                if isinstance(key_insights, dict):
                    # Flatten the insights dict into a list
                    insights_list = []
                    for stage, insights in key_insights.items():
                        if insights:
                            insights_list.extend([f"{stage}: {insight}" for insight in insights])
                    key_insights = insights_list
                
                await self.db_service.save_quick_analytics({
                    "session_id": session_id,
                    "overall_score": analytics.get("overall_score", 0),
                    "key_insights": key_insights,
                    "completion_percentage": analytics.get("completion_percentage", 0),
                    "current_topics": analytics.get("current_topics", [])
                })
                logger.info(f"Analytics for session {session_id} saved to database")
            except Exception as e:
                logger.error(f"Failed to save analytics to database: {e}")
        
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Define the workflow graph
        workflow = StateGraph(PitchWorkflowState)
        
        # Add nodes
        workflow.add_node("initialize_session", self._initialize_session)
        workflow.add_node("assess_stage", self._assess_current_stage)
        workflow.add_node("generate_question", self._generate_stage_question)
        workflow.add_node("evaluate_response", self._evaluate_founder_response)
        workflow.add_node("decide_transition", self._decide_stage_transition)
        workflow.add_node("transition_stage", self._transition_to_next_stage)
        workflow.add_node("finalize_session", self._finalize_session)
        
        # Define edges
        workflow.add_edge(START, "initialize_session")
        workflow.add_edge("initialize_session", "assess_stage")
        workflow.add_edge("assess_stage", "generate_question")
        workflow.add_edge("generate_question", "evaluate_response")
        workflow.add_edge("evaluate_response", "decide_transition")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "decide_transition",
            self._should_transition_or_continue,
            {
                "transition": "transition_stage",
                "continue": "generate_question",
                "complete": "finalize_session",
                "wait": END  # End workflow after first question, wait for founder response
            }
        )
        
        workflow.add_edge("transition_stage", "assess_stage")
        workflow.add_edge("finalize_session", END)
        
        return workflow.compile(checkpointer=self.memory_saver)
    
    def _initialize_session(self, state: PitchWorkflowState) -> PitchWorkflowState:
        """Initialize a new pitch practice session"""
        
        # Check if session is already initialized (for continuing conversations)
        if state.get("session_start") and state.get("current_stage"):
            logger.info(f"Session already initialized for conversation {state['conversation_id']}, skipping initialization")
            return state
        
        logger.info(f"Initializing session for conversation {state['conversation_id']}")
        
        # Initialize session data
        state["current_stage"] = PitchStage.GREETING.value
        state["stage_progress"] = {stage: {"questions_asked": 0, "key_points": []} for stage in PITCH_STAGES}
        state["should_transition"] = False
        state["workflow_complete"] = False
        state["session_start"] = datetime.now().isoformat()
        state["stage_durations"] = {}
        state["questions_asked"] = []
        state["key_insights"] = {stage: [] for stage in PITCH_STAGES}
        state["question_answered"] = False
        state["follow_up_count"] = 0
        
        # Initialize video analysis fields
        state["video_analysis_enabled"] = False
        state["video_insights"] = []
        state["gesture_feedback"] = []
        state["posture_feedback"] = []
        state["expression_feedback"] = []
        
        # Set persona if not provided
        if "persona" not in state:
            state["persona"] = "friendly"
        
        # Generate initial greeting
        persona_info = INVESTOR_PERSONAS[state["persona"]]
        initial_greeting = f"Hello! I'm {persona_info['name']}, {persona_info['title']}. I'm excited to learn about your business today. Let's start with a quick introduction - what's your name?"
        
        state["messages"] = [AIMessage(content=initial_greeting)]
        state["current_question"] = initial_greeting
        
        return state
    
    def _assess_current_stage(self, state: PitchWorkflowState) -> PitchWorkflowState:
        """Assess the current stage and determine focus"""
        current_stage = state["current_stage"]
        logger.info(f"Assessing stage: {current_stage}")
        
        # Update stage progress
        if current_stage not in state["stage_progress"]:
            state["stage_progress"][current_stage] = {"questions_asked": 0, "key_points": []}
        
        # Record stage entry time if not already recorded
        if current_stage not in state["stage_durations"]:
            state["stage_durations"][current_stage] = datetime.now().timestamp()
        
        return state
    
    def _generate_stage_question(self, state: PitchWorkflowState) -> PitchWorkflowState:
        """Generate a focused question for the current stage"""
        current_stage = state["current_stage"]
        persona = state["persona"]
        persona_info = INVESTOR_PERSONAS[persona]
        
        # Get conversation history
        recent_messages = state["messages"][-6:]  # Last 6 messages for context
        conversation_context = "\n".join([
            f"{msg.__class__.__name__.replace('Message', '')}: {msg.content}"
            for msg in recent_messages
        ])
        
        # Create stage-specific question prompt
        question_prompt = self._create_question_prompt(current_stage, persona_info, conversation_context, state)
        
        try:
            # Generate question using LLM
            response = llm.invoke(question_prompt)
            question = response.content.strip()
            
            # Update state
            state["current_question"] = question
            state["stage_progress"][current_stage]["questions_asked"] += 1
            state["questions_asked"].append(question)
            state["question_answered"] = False
            state["follow_up_count"] = 0  # Reset follow-up count for new question
            
            # Add to messages
            state["messages"].append(AIMessage(content=question))
            
            logger.info(f"Generated question for {current_stage}: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            fallback_question = f"Could you tell me more about {current_stage.replace('_', ' ')}?"
            state["current_question"] = fallback_question
            state["follow_up_count"] = 0  # Reset follow-up count for new question
            state["messages"].append(AIMessage(content=fallback_question))
        
        return state
    
    def _create_question_prompt(self, stage: str, persona_info: Dict, conversation_context: str, state: PitchWorkflowState) -> str:
        """Create a stage-specific question prompt"""
        
        stage_objectives = {
            "greeting": "Get the founder's name, company name, and a brief introduction",
            "problem_solution": "Understand the core problem they're solving and their solution approach",
            "target_market": "Identify their target customers and market opportunity",
            "business_model": "Understand how they make money and their pricing strategy",
            "competition": "Learn about their competitive landscape and differentiation",
            "traction": "Assess their current progress, metrics, and growth",
            "team": "Evaluate the team's experience and composition",
            "funding_needs": "Understand their funding requirements and use of funds",
            "future_plans": "Discuss their vision and long-term strategy"
        }
        
        questions_asked_count = state["stage_progress"][stage]["questions_asked"]
        
        # Special handling for greeting stage to be more conversational
        if stage == "greeting" and questions_asked_count == 0:
            # Check if user has already provided some information in their first message
            user_intro = ""
            if state["messages"]:
                # Look for the most recent human message
                for msg in reversed(state["messages"]):
                    if msg.__class__.__name__ == "HumanMessage":
                        user_intro = msg.content
                        break
            
            prompt = f"""You are {persona_info['name']}, {persona_info['title']}.

INVESTOR PROFILE & CONTEXT:
- Background: {persona_info['personality']}
- Questioning Methodology: {persona_info['questioning_style']}
- Communication Style: {persona_info.get('communication_patterns', 'Professional and engaging')}
- Decision Framework: {persona_info.get('cognitive_approach', 'Systematic evaluation approach')}

CURRENT SITUATION ANALYSIS:
Meeting Context: Initial pitch meeting - greeting and rapport building phase
Founder's Opening: "{user_intro}"
Your Objective: Establish professional rapport while efficiently gathering foundational information

GREETING STAGE STRATEGIC FRAMEWORK:
Primary Goals:
1. Create positive first impression aligned with your personality
2. Gather essential identifiers (name, company, basic context)
3. Set conversational tone for the entire pitch session
4. Begin initial founder assessment (confidence, communication style)

Information Gathering Priorities:
- Founder's name (if not provided)
- Company name and basic description
- Context for the meeting/pitch
- Initial rapport establishment

CONVERSATIONAL INTELLIGENCE PROTOCOL:
1. ACKNOWLEDGMENT: Recognize and respond to information already provided
2. PERSONALIZATION: Use their name if provided, reference their company if mentioned
3. NATURAL FLOW: Make the interaction feel conversational, not interrogative
4. PERSONALITY CONSISTENCY: Every response must reflect your established character
5. SINGLE FOCUS: One clear objective per question

RESPONSE CONSTRAINTS:
- EXACTLY ONE question mark (?) per response
- Maximum 20 words for natural conversation flow
- No compound or multi-part questions
- Maintain your personality's characteristic tone

STAGE BOUNDARIES:
✓ Appropriate for Greeting Stage: Names, company identity, brief overview, meeting context
✗ Premature for Greeting Stage: Revenue, metrics, detailed business model, competition, funding

RESPONSE GENERATION INSTRUCTIONS:
1. Analyze what information the founder has already provided
2. Identify the most important missing piece for rapport and context
3. Craft a response that acknowledges their input while gathering needed information
4. Ensure the question reflects your personality's natural communication style
5. Maintain professional warmth appropriate to your character

Generate your response as {persona_info['name']}, balancing efficiency with authentic personality expression:"""
        else:
            # Enhanced prompt for systematic stage-based questioning
            prompt = f"""You are {persona_info['name']}, {persona_info['title']}.

INVESTOR IDENTITY & METHODOLOGY:
- Core Personality: {persona_info['personality']}
- Questioning Framework: {persona_info['questioning_style']}
- Analytical Approach: {persona_info.get('cognitive_approach', 'Systematic evaluation methodology')}
- Communication Patterns: {persona_info.get('communication_patterns', 'Professional and direct')}

CURRENT EVALUATION STAGE: {stage.replace('_', ' ').title()}
Stage Objective: {stage_objectives.get(stage, 'Gather relevant information')}
Question Sequence: #{questions_asked_count + 1} for this stage

CONVERSATION ANALYSIS:
{conversation_context}

FOUNDER PROFILE:
- Name: {state.get('founder_name', 'Not provided')}
- Company: {state.get('company_name', 'Not provided')}
- Stage Progress: Currently evaluating {stage.replace('_', ' ')} aspects

STAGE-SPECIFIC EVALUATION FRAMEWORK:
{stage.replace('_', ' ').title()} Assessment Criteria:
- Information completeness and specificity
- Evidence quality and validation
- Strategic thinking demonstration
- Execution capability indicators

QUESTIONING STRATEGY PROTOCOL:
1. RESPONSE ANALYSIS: Evaluate founder's previous answer for completeness and depth
2. GAP IDENTIFICATION: Identify the most critical missing information for this stage
3. QUESTION FORMULATION: Create a question that reflects your personality's analytical approach
4. DEPTH PROGRESSION: Each question should build deeper understanding within the stage
5. PERSONALITY CONSISTENCY: Maintain your characteristic decision triggers and concerns

RESPONSE QUALITY INDICATORS:
✓ Specific examples and concrete data
✓ Clear reasoning and strategic thinking
✓ Honest acknowledgment of challenges
✓ Evidence-based claims and validation

⚠ Red Flags Requiring Follow-up:
- Vague or generic responses
- Unsupported claims or assumptions
- Evasive answers to direct questions
- Lack of specific examples or metrics

STAGE BOUNDARY ENFORCEMENT:
Current Focus: {stage.replace('_', ' ')} ONLY
Do NOT ask about: {', '.join([s.replace('_', ' ') for s in stage_objectives.keys() if s != stage])}

RESPONSE CONSTRAINTS:
- EXACTLY ONE question mark (?) per response
- Maximum 25 words for clarity and focus
- No compound or multi-part questions
- Align with your personality's decision-making criteria

Generate your next question as {persona_info['name']}, applying your characteristic analytical rigor to {stage.replace('_', ' ')} evaluation:"""
        
        return prompt
    
    def _validate_answer_relevance(self, question: str, response: str, stage: str) -> Dict[str, Any]:
        """Validate if the user's response actually answers the question asked"""
        
        validation_prompt = f"""
        CONVERSATION RESPONSE ANALYSIS TASK
        
        You are an expert conversation analyst. Analyze the user's response to determine its type and appropriate handling.
        
        CONTEXT:
        Current Stage: {stage.replace('_', ' ').title()}
        Question Asked: "{question}"
        User's Response: "{response}"
        
        ANALYSIS CATEGORIES:
        
        1. REPEAT REQUEST DETECTION:
        Is the user asking to repeat/clarify the question? Look for:
        - Direct requests: "repeat", "again", "what was the question"
        - Confusion indicators: "what?", "huh?", "pardon?", "sorry?"
        - Clarification requests: "didn't catch that", "didn't hear", "come again"
        - Any indication they want the question restated
        
        2. ANSWER RELEVANCE (if not a repeat request):
        - Does the response directly address what was asked?
        - Is the user attempting to answer the specific question?
        - Or are they avoiding, deflecting, or changing the subject?
        
        3. COMPLETENESS CHECK (if answering):
        - If the question asks for specific information, is it provided?
        - If it's open-ended, does the response show genuine engagement?
        
        EXAMPLES:
        
        REPEAT REQUESTS:
        Q: "What's your name?" → A: "Can you repeat that?" (REPEAT REQUEST)
        Q: "Tell me about your business" → A: "What?" (REPEAT REQUEST)
        Q: "What's your company?" → A: "Sorry, didn't catch that" (REPEAT REQUEST)
        Q: "How do you make money?" → A: "Huh?" (REPEAT REQUEST)
        Q: "What problem do you solve?" → A: "Come again?" (REPEAT REQUEST)
        
        VALID ANSWERS:
        Q: "What's your name?" → A: "My name is John" (VALID ANSWER)
        Q: "What's your company?" → A: "TechCorp" (VALID ANSWER)
        
        INVALID ANSWERS:
        Q: "What's your name?" → A: "Guess my name!" (EVASIVE)
        Q: "What's your company?" → A: "That's secret!" (EVASIVE)
        
        OUTPUT FORMAT:
        REPEAT_REQUESTED: [true/false]
        VALID: [true/false - only relevant if not a repeat request]
        REASON: [Brief explanation]
        MISSING_INFO: [What specific information is missing if invalid]
        FOLLOW_UP_NEEDED: [true/false - whether we need to ask again]
        """
        
        try:
            validation_response = llm.invoke(validation_prompt)
            validation_text = validation_response.content
            
            # Parse validation result
            repeat_requested = "true" in validation_text.split("REPEAT_REQUESTED:")[1].split("\n")[0].lower() if "REPEAT_REQUESTED:" in validation_text else False
            is_valid = "true" in validation_text.split("VALID:")[1].split("\n")[0].lower() if "VALID:" in validation_text else False
            
            reason = ""
            if "REASON:" in validation_text:
                reason = validation_text.split("REASON:")[1].split("\n")[0].strip()
            
            missing_info = ""
            if "MISSING_INFO:" in validation_text:
                missing_info = validation_text.split("MISSING_INFO:")[1].split("\n")[0].strip()
            
            follow_up_needed = "true" in validation_text.split("FOLLOW_UP_NEEDED:")[1].split("\n")[0].lower() if "FOLLOW_UP_NEEDED:" in validation_text else not is_valid
            
            return {
                "is_valid": is_valid,
                "reason": reason,
                "missing_info": missing_info,
                "follow_up_needed": follow_up_needed,
                "repeat_requested": repeat_requested
            }
            
        except Exception as e:
            logger.error(f"Error validating answer relevance: {e}")
            # Default to valid to avoid getting stuck
            return {
                "is_valid": True,
                "reason": "Validation error - defaulting to valid",
                "missing_info": "",
                "follow_up_needed": False,
                "repeat_requested": False
            }

    def _generate_follow_up_question(self, original_question: str, response: str, validation_result: Dict, stage: str, persona_info: Dict) -> str:
        """Generate a follow-up question when the user didn't properly answer"""
        
        follow_up_prompt = f"""
        FOLLOW-UP QUESTION GENERATION - KEEP IT SHORT FOR TTS
        
        You are {persona_info['name']}, {persona_info['title']}.
        
        SITUATION:
        - You asked: "{original_question}"
        - User responded: "{response}"
        - Issue: {validation_result['reason']}
        
        CRITICAL REQUIREMENTS:
        - Maximum 1-2 sentences only
        - Keep it under 25 words if possible
        - Perfect for text-to-speech conversion
        - Sound natural when spoken aloud
        
        TASK:
        Generate a SHORT, polite follow-up that redirects to the original question.
        
        GOOD EXAMPLES (SHORT):
        "I appreciate the humor, but what's your name?"
        "That's interesting, but what's your company called?"
        "I understand, but could you tell me your business model?"
        "Thanks for that, but I still need to know your name."
        
        BAD EXAMPLES (TOO LONG):
        "I appreciate the humor and playfulness, it shows creativity, but I really need to know your name so I can address you properly and understand your venture better."
        
        Generate a SHORT follow-up question (max 25 words):
        """
        
        try:
            follow_up_response = llm.invoke(follow_up_prompt)
            return follow_up_response.content.strip()
        except Exception as e:
            logger.error(f"Error generating follow-up question: {e}")
            # Fallback follow-up
            return f"I'd like to get back to my question - {original_question}"

    def _evaluate_founder_response(self, state: PitchWorkflowState) -> PitchWorkflowState:
        """Evaluate the founder's response and extract key insights"""
        
        # Get the latest founder message
        latest_messages = state["messages"][-2:]  # Last 2 messages (question and response)
        if len(latest_messages) < 2:
            # During initialization, there's no response to evaluate yet
            # Set flags to stop the workflow after generating the first question
            state["question_answered"] = False
            state["should_transition"] = False
            state["workflow_complete"] = False
            return state
        
        founder_response = latest_messages[-1].content if latest_messages[-1].__class__.__name__ == "HumanMessage" else ""
        
        if not founder_response:
            # No founder response yet, stop workflow after first question
            state["question_answered"] = False
            state["should_transition"] = False
            state["workflow_complete"] = False
            return state
        
        current_stage = state["current_stage"]
        original_question = state["current_question"]
        
        # First, validate if the response actually answers the question
        validation_result = self._validate_answer_relevance(original_question, founder_response, current_stage)
        
        # If the answer is not valid, generate a follow-up question (with limits)
        if not validation_result["is_valid"] and validation_result["follow_up_needed"]:
            logger.info(f"Invalid response detected: {validation_result['reason']}")
            
            # Check follow-up limit to prevent infinite loops
            current_follow_ups = state.get("follow_up_count", 0)
            max_follow_ups = 2  # Maximum 2 follow-up attempts per question
            
            if current_follow_ups >= max_follow_ups:
                logger.info(f"Maximum follow-ups ({max_follow_ups}) reached, accepting response and moving on")
                # Accept the response and continue to avoid getting stuck
                state["question_answered"] = True
                state["follow_up_count"] = 0  # Reset for next question
            else:
                # Generate follow-up question
                persona_info = INVESTOR_PERSONAS[state["persona"]]
                follow_up_question = self._generate_follow_up_question(
                    original_question, founder_response, validation_result, current_stage, persona_info
                )
                
                # Update state with follow-up question
                state["current_question"] = follow_up_question
                state["question_answered"] = False
                state["should_transition"] = False
                
                # Add follow-up message to conversation
                state["messages"].append(AIMessage(content=follow_up_question))
                
                # Track that we had to ask a follow-up
                state["follow_up_count"] = current_follow_ups + 1
                
                logger.info(f"Generated follow-up question ({state['follow_up_count']}/{max_follow_ups}): {follow_up_question[:50]}...")
                return state
        
        # If we reach here, the answer is valid - proceed with normal evaluation
        current_stage = state["current_stage"]
        
        # Special handling for greeting stage to extract founder info
        if current_stage == "greeting":
            # Extract founder name and company from their response
            extraction_prompt = f"""
            Extract the founder's name and company from this message:
            "{founder_response}"
            
            Look for:
            - Their name (e.g., "I'm Alex", "My name is Sarah", "This is John")
            - Company name (e.g., "from Facebook", "I work at Google", "my company TechCorp")
            
            Format your response as:
            NAME: [extracted name or "Not mentioned"]
            COMPANY: [extracted company or "Not mentioned"]
            """
            
            try:
                extraction = llm.invoke(extraction_prompt)
                extract_text = extraction.content
                
                # Parse extracted info
                if "NAME:" in extract_text:
                    name_line = extract_text.split("NAME:")[1].split("\n")[0].strip()
                    if name_line and name_line.lower() != "not mentioned":
                        state["founder_name"] = name_line
                
                if "COMPANY:" in extract_text:
                    company_line = extract_text.split("COMPANY:")[1].split("\n")[0].strip()
                    if company_line and company_line.lower() != "not mentioned":
                        state["company_name"] = company_line
                        
            except Exception as e:
                logger.error(f"Error extracting founder info: {e}")
        
        # Enhanced response evaluation with detailed analysis framework
        evaluation_prompt = f"""
        RESPONSE EVALUATION FRAMEWORK
        
        CONTEXT ANALYSIS:
        Investor Question: {state['current_question']}
        Founder's Response: {founder_response}
        Current Stage: {current_stage.replace('_', ' ').title()}
        
        EVALUATION DIMENSIONS:
        
        1. COMPLETENESS ASSESSMENT:
        - Does the response directly address the question asked?
        - Are there specific examples, data points, or evidence provided?
        - Is the level of detail appropriate for an investor conversation?
        - Are there obvious gaps or evasive elements in the answer?
        
        2. QUALITY INDICATORS:
        ✓ Specific metrics, numbers, or concrete examples
        ✓ Clear reasoning and logical flow
        ✓ Honest acknowledgment of challenges or limitations
        ✓ Evidence of strategic thinking and market understanding
        ✓ Appropriate depth for the current stage
        
        3. RED FLAGS:
        ⚠ Vague or generic statements without specifics
        ⚠ Unsupported claims or unrealistic projections
        ⚠ Evasive answers that don't address the core question
        ⚠ Lack of evidence or validation for claims made
        ⚠ Overly technical jargon without clear business impact
        
        4. INSIGHT EXTRACTION:
        - What specific business insights can be derived from this response?
        - What does this reveal about the founder's understanding of their business?
        - What strengths or weaknesses are demonstrated?
        - What additional information would be most valuable to gather?
        
        5. FOLLOW-UP STRATEGY:
        - Is clarification needed on any specific points?
        - Are there logical follow-up questions within this stage?
        - Has sufficient information been gathered for this topic area?
        
        EVALUATION OUTPUT FORMAT:
        COMPLETE: [yes/no with brief justification]
        QUALITY_SCORE: [1-10 scale with reasoning]
        INSIGHTS:
        - [Specific business insight 1]
        - [Specific business insight 2]
        - [Specific business insight 3]
        STRENGTHS_DEMONSTRATED:
        - [Specific strength shown in response]
        GAPS_IDENTIFIED:
        - [Specific gap or weakness revealed]
        FOLLOW_UP_NEEDED: [yes/no with reasoning]
        RECOMMENDED_FOCUS: [What aspect needs deeper exploration]
        """
        
        try:
            evaluation = llm.invoke(evaluation_prompt)
            eval_text = evaluation.content
            
            # Parse evaluation
            complete = "yes" in eval_text.split("COMPLETE:")[1].split("\n")[0].lower() if "COMPLETE:" in eval_text else True
            
            # Extract insights
            if "INSIGHTS:" in eval_text:
                insights_section = eval_text.split("INSIGHTS:")[1].split("FOLLOW_UP_NEEDED:")[0] if "FOLLOW_UP_NEEDED:" in eval_text else eval_text.split("INSIGHTS:")[1]
                insights = [line.strip("- ").strip() for line in insights_section.split("\n") if line.strip().startswith("-")]
                state["key_insights"][current_stage].extend(insights)
            
            # For greeting stage, always mark as answered if we got a response
            if current_stage == "greeting" and founder_response.strip():
                state["question_answered"] = True
                # Extract founder info
                self._extract_founder_info(state, founder_response)
                logger.info(f"Greeting response received and processed")
            else:
                # For other stages, mark as answered if we got a substantive response
                # This is more lenient to allow stage transitions
                if founder_response.strip() and len(founder_response.strip()) > 10:
                    state["question_answered"] = True
                else:
                    state["question_answered"] = complete
            
            logger.info(f"Response evaluation - Complete: {complete}, Question answered: {state['question_answered']}")
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            # Default to considering response complete to avoid getting stuck
            state["question_answered"] = True
            logger.info(f"Defaulting to question_answered = True due to evaluation error")
        
        return state
    
    def _extract_founder_info(self, state: PitchWorkflowState, response: str) -> None:
        """Extract founder name and company from greeting stage"""
        
        # Enhanced name extraction with pattern recognition
        name_prompt = f"""
        FOUNDER NAME EXTRACTION TASK
        
        Input Text: "{response}"
        
        EXTRACTION METHODOLOGY:
        Analyze the text for name indicators using these patterns:
        
        1. DIRECT INTRODUCTIONS:
        - "I'm [Name]" / "I am [Name]"
        - "My name is [Name]"
        - "This is [Name]"
        - "[Name] here" / "It's [Name]"
        
        2. CONTEXTUAL PATTERNS:
        - "Hi, [Name] from [Company]"
        - "[Name], founder of [Company]"
        - "I'm [Name], and I..."
        
        3. VALIDATION CRITERIA:
        - Must be a proper noun (capitalized)
        - Typically 1-3 words
        - Not a company name or title
        - Sounds like a human name
        
        EXTRACTION RULES:
        - Return ONLY the person's name, no titles or descriptions
        - If multiple names mentioned, return the one being introduced
        - If unclear or no name present, return "Not provided"
        - Remove any titles (Mr., Dr., CEO, etc.)
        
        OUTPUT: [Extracted name or "Not provided"]
        """
        
        try:
            name_response = llm.invoke(name_prompt)
            name = name_response.content.strip()
            if name and name != "Not provided" and len(name) < 50 and not any(word in name.lower() for word in ['company', 'corp', 'inc', 'llc', 'ltd']):
                state["founder_name"] = name
        except Exception as e:
            logger.warning(f"Error extracting name: {e}")
        
        # Enhanced company extraction with business entity recognition
        company_prompt = f"""
        COMPANY NAME EXTRACTION TASK
        
        Input Text: "{response}"
        
        EXTRACTION METHODOLOGY:
        Identify company/business names using these indicators:
        
        1. EXPLICIT COMPANY MENTIONS:
        - "I work at [Company]"
        - "I'm from [Company]"
        - "My company [Company]"
        - "[Company] is our startup"
        - "We're building [Company]"
        
        2. BUSINESS ENTITY PATTERNS:
        - Names ending with Inc, Corp, LLC, Ltd, Co
        - Technology company patterns (ending with -ly, -ify, -tech)
        - Startup naming conventions
        
        3. CONTEXTUAL INDICATORS:
        - "founder of [Company]"
        - "CEO of [Company]"
        - "we started [Company]"
        - "our platform [Company]"
        
        4. VALIDATION CRITERIA:
        - Proper noun (capitalized)
        - Business-sounding name
        - Not a person's name
        - Not a generic term
        
        EXTRACTION RULES:
        - Return ONLY the company name, no descriptions
        - Remove articles (the, a, an) if present
        - Include entity suffixes if present (Inc, LLC, etc.)
        - If unclear or no company mentioned, return "Not provided"
        
        OUTPUT: [Extracted company name or "Not provided"]
        """
        
        try:
            company_response = llm.invoke(company_prompt)
            company = company_response.content.strip()
            if company and company != "Not provided" and len(company) < 100:
                state["company_name"] = company
        except Exception as e:
            logger.warning(f"Error extracting company: {e}")
    
    def _decide_stage_transition(self, state: PitchWorkflowState) -> PitchWorkflowState:
        """Decide whether to transition to the next stage"""
        current_stage = state["current_stage"]
        questions_asked = state["stage_progress"][current_stage]["questions_asked"]
        
        # Minimum questions per stage
        min_questions = {
            "greeting": 1,
            "problem_solution": 2,
            "target_market": 2,
            "business_model": 2,
            "competition": 2,
            "traction": 2,
            "team": 2,
            "funding_needs": 2,
            "future_plans": 2
        }
        
        min_required = min_questions.get(current_stage, 2)
        
        # Don't transition if we haven't asked minimum questions
        if questions_asked < min_required:
            state["should_transition"] = False
            logger.info(f"Transition decision for {current_stage}: False (not enough questions: {questions_asked}/{min_required})")
            return state
        
        # Special logic for greeting stage - transition if we have basic info
        if current_stage == "greeting":
            has_name = state.get("founder_name") and state["founder_name"] != "Not provided"
            has_company = state.get("company_name") and state["company_name"] != "Not provided"
            has_response = state.get("question_answered", False)
            
            # Transition if we have at least company info and a response
            if (has_company or has_name) and has_response and questions_asked >= 1:
                state["should_transition"] = True
                logger.info(f"Greeting stage transition: True (name: {has_name}, company: {has_company}, answered: {has_response})")
                return state
        
        # For other stages, check if we have enough information
        key_insights_count = len(state["key_insights"][current_stage])
        
        if key_insights_count >= 2 and state["question_answered"]:
            state["should_transition"] = True
        else:
            state["should_transition"] = False
        
        logger.info(f"Transition decision for {current_stage}: {state['should_transition']} (questions: {questions_asked}, insights: {key_insights_count})")
        
        return state
    
    def _should_transition_or_continue(self, state: PitchWorkflowState) -> str:
        """Determine the next step in the workflow"""
        
        # Check if this is initialization (no founder response yet)
        latest_messages = state["messages"]
        has_founder_response = False
        
        if len(latest_messages) >= 2:
            # Check if the last message is from founder (HumanMessage)
            last_msg = latest_messages[-1]
            has_founder_response = last_msg.__class__.__name__ == "HumanMessage"
        
        # If no founder response yet, stop workflow after generating first question
        if not has_founder_response and len(latest_messages) > 0:
            return "wait"  # Stop workflow, wait for founder response
        
        # Check if workflow is complete
        if state["current_stage"] == "future_plans" and state["should_transition"]:
            return "complete"
        
        # Check if should transition to next stage
        if state["should_transition"]:
            return "transition"
        
        # Continue with current stage
        return "continue"
    
    def _transition_to_next_stage(self, state: PitchWorkflowState) -> PitchWorkflowState:
        """Transition to the next stage"""
        current_stage = state["current_stage"]
        
        # Record stage completion time
        if current_stage in state["stage_durations"]:
            duration = datetime.now().timestamp() - state["stage_durations"][current_stage]
            state["stage_durations"][current_stage] = duration
        
        # Find next stage
        try:
            current_idx = PITCH_STAGES.index(current_stage)
            if current_idx + 1 < len(PITCH_STAGES):
                next_stage = PITCH_STAGES[current_idx + 1]
                state["current_stage"] = next_stage
                
                # Add transition message
                transition_message = f"Great! Now let's move on to discussing {next_stage.replace('_', ' ')}."
                state["messages"].append(AIMessage(content=transition_message))
                
                logger.info(f"Transitioned from {current_stage} to {next_stage}")
            else:
                state["workflow_complete"] = True
                
        except (ValueError, IndexError) as e:
            logger.error(f"Error transitioning stages: {e}")
            state["workflow_complete"] = True
        
        state["should_transition"] = False
        return state
    
    def _finalize_session(self, state: PitchWorkflowState) -> PitchWorkflowState:
        """Finalize the pitch practice session"""
        
        # Generate comprehensive session summary
        summary_prompt = f"""
        PITCH PRACTICE SESSION SUMMARY GENERATION
        
        SESSION DETAILS:
        Founder: {state.get('founder_name', 'The founder')}
        Company: {state.get('company_name', 'Their company')}
        Stages Completed: {list(state['key_insights'].keys())}
        Total Duration: {(datetime.now().timestamp() - datetime.fromisoformat(state['session_start']).timestamp()) / 60:.1f} minutes
        
        INSIGHTS GATHERED:
        {json.dumps(state['key_insights'], indent=2)}
        
        SUMMARY GENERATION FRAMEWORK:
        
        1. ACKNOWLEDGMENT & APPRECIATION:
        - Recognize their time investment and engagement
        - Acknowledge the effort put into the practice session
        - Express appreciation for their openness and responses
        
        2. STRENGTH IDENTIFICATION:
        - Identify 2-3 specific strengths demonstrated during the session
        - Focus on concrete examples from their responses
        - Highlight areas where they showed particular clarity or passion
        
        3. ENCOURAGEMENT & MOTIVATION:
        - Provide genuine, specific encouragement based on their performance
        - Connect their strengths to investor appeal
        - Reinforce confidence in their business potential
        
        4. FORWARD-LOOKING PERSPECTIVE:
        - Suggest next steps or areas for continued development
        - Frame the practice as preparation for real investor meetings
        - Maintain optimistic tone about their pitch readiness
        
        TONE REQUIREMENTS:
        - Warm and supportive, but professional
        - Specific rather than generic
        - Encouraging without being unrealistic
        - Personalized to their actual responses and company
        
        LENGTH: 4-6 sentences that feel substantial but not overwhelming
        
        Generate a personalized, encouraging summary that reflects their specific journey through this practice session:
        """
        
        try:
            summary_response = llm.invoke(summary_prompt)
            final_message = summary_response.content.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            final_message = f"Thank you for the great pitch practice session! You've covered all the key areas and I can see the passion and thought you've put into {state.get('company_name', 'your business')}. Keep refining your story and you'll do great with investors!"
        
        state["messages"].append(AIMessage(content=final_message))
        state["workflow_complete"] = True
        
        logger.info(f"Finalized session for {state['conversation_id']}")
        
        return state
    
    def start_session(self, conversation_id: str, persona: str = "friendly") -> Dict[str, Any]:
        """Start a new pitch practice session"""
        
        if persona not in INVESTOR_PERSONAS:
            raise ValueError(f"Unknown persona: {persona}")
        
        # Initialize state
        initial_state = PitchWorkflowState(
            messages=[],
            conversation_id=conversation_id,
            persona=persona,
            founder_name="",
            company_name="",
            current_stage="",
            stage_progress={},
            should_transition=False,
            workflow_complete=False,
            current_question="",
            question_answered=False,
            follow_up_count=0,
            session_start="",
            stage_durations={},
            questions_asked=[],
            key_insights={}
        )
        
        # Run initialization
        config = {"configurable": {"thread_id": conversation_id}}
        result = self.workflow.invoke(initial_state, config)
        
        # Return the greeting message
        return {
            "message": result["messages"][-1].content,
            "stage": result["current_stage"],
            "session_id": conversation_id
        }
    
    def start_session_with_message(self, conversation_id: str, persona: str, first_message: str) -> Dict[str, Any]:
        """Start a new pitch practice session with the user's first message"""
        
        if persona not in INVESTOR_PERSONAS:
            raise ValueError(f"Unknown persona: {persona}")
        
        # Initialize state with the user's first message
        initial_state = PitchWorkflowState(
            messages=[HumanMessage(content=first_message)],
            conversation_id=conversation_id,
            persona=persona,
            founder_name="",
            company_name="",
            current_stage="",
            stage_progress={},
            should_transition=False,
            workflow_complete=False,
            current_question="",
            question_answered=False,
            follow_up_count=0,
            session_start="",
            stage_durations={},
            questions_asked=[],
            key_insights={}
        )
        
        # Run initialization
        config = {"configurable": {"thread_id": conversation_id}}
        result = self.workflow.invoke(initial_state, config)
        
        # Return the response message
        return {
            "message": result["messages"][-1].content,
            "stage": result["current_stage"],
            "session_id": conversation_id,
            "complete": result.get("workflow_complete", False),
            "insights": result.get("key_insights", {})
        }
    
    def process_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """Process a founder's message and return the investor's response"""
        
        config = {"configurable": {"thread_id": conversation_id}}
        
        # Get current state
        try:
            current_state = self.workflow.get_state(config)
            if current_state.values:
                # Add the founder's message to existing state
                human_message = HumanMessage(content=message)
                current_state.values["messages"].append(human_message)
                
                # Manually run the evaluation and transition steps
                # 1. Evaluate the founder's response
                evaluated_state = self._evaluate_founder_response(current_state.values)
                
                # 2. Decide on stage transition
                transition_state = self._decide_stage_transition(evaluated_state)
                
                # 3. Check if we should transition or continue
                next_action = self._should_transition_or_continue(transition_state)
                
                if next_action == "transition":
                    # Transition to next stage
                    transitioned_state = self._transition_to_next_stage(transition_state)
                    # Assess the new stage
                    assessed_state = self._assess_current_stage(transitioned_state)
                    # Generate question for new stage
                    final_state = self._generate_stage_question(assessed_state)
                elif next_action == "continue":
                    # Continue with current stage - generate new question
                    final_state = self._generate_stage_question(transition_state)
                else:
                    # Complete or other action
                    final_state = transition_state
                
                # Update the workflow state by invoking with the final state
                # This ensures the state is properly persisted
                try:
                    self.workflow.update_state(config, final_state)
                except Exception as update_error:
                    logger.warning(f"State update failed, using invoke instead: {update_error}")
                    # Fallback: invoke the workflow with the final state
                    self.workflow.invoke(final_state, config)
                
                # Check if session is complete and generate analysis
                is_complete = final_state.get("workflow_complete", False)
                analysis = None
                
                if is_complete:
                    # Generate comprehensive analysis for completed session
                    try:
                        analysis = self.generate_pitch_analysis(conversation_id)
                        logger.info(f"Generated pitch analysis for completed session {conversation_id}")
                    except Exception as analysis_error:
                        logger.error(f"Failed to generate analysis: {analysis_error}")
                
                # Return the response
                response = {
                    "message": final_state["messages"][-1].content,
                    "stage": final_state["current_stage"],
                    "complete": is_complete,
                    "session_id": conversation_id,
                    "insights": final_state.get("key_insights", {})
                }
                
                # Include analysis if session is complete
                if analysis and "error" not in analysis:
                    response["analysis"] = analysis
                
                return response
            else:
                return {"error": "Session not found or expired"}
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"error": "Failed to process message"}
    
    def get_session_analytics(self, conversation_id: str) -> Dict[str, Any]:
        """Get analytics for a pitch practice session"""
        
        config = {"configurable": {"thread_id": conversation_id}}
        
        try:
            current_state = self.workflow.get_state(config)
            if not current_state.values:
                return {"error": "Session not found"}
            
            state = current_state.values
            
            # Calculate session duration
            session_start = datetime.fromisoformat(state["session_start"])
            duration_minutes = (datetime.now() - session_start).total_seconds() / 60
            
            # Compile analytics
            analytics = {
                "session_id": conversation_id,
                "duration_minutes": round(duration_minutes, 2),
                "persona": state["persona"],
                "founder_name": state.get("founder_name", ""),
                "company_name": state.get("company_name", ""),
                "current_stage": state["current_stage"],
                "completed_stages": list(state["stage_durations"].keys()),
                "total_questions": len(state["questions_asked"]),
                "key_insights": state["key_insights"],
                "stage_progress": state["stage_progress"],
                "workflow_complete": state.get("workflow_complete", False)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {"error": "Failed to get session analytics"}
    
    def generate_pitch_analysis(self, conversation_id: str) -> Dict[str, Any]:
        """Generate comprehensive pitch analysis report"""
        
        config = {"configurable": {"thread_id": conversation_id}}
        
        try:
            current_state = self.workflow.get_state(config)
            if not current_state.values:
                return {"error": "Session not found"}
            
            state = current_state.values
            
            # Get session analytics first
            analytics = self.get_session_analytics(conversation_id)
            if "error" in analytics:
                return analytics
            
            # Count founder messages to determine conversation depth
            # No minimum requirement - we'll analyze any conversation length
            founder_messages = [msg for msg in state['messages'] if msg.__class__.__name__ == "HumanMessage"]
            founder_message_count = len(founder_messages)
            
            # Calculate communication metrics
            comm_metrics = self._calculate_communication_metrics(state['messages'])
            
            # Generate comprehensive pitch analysis using advanced evaluation framework
            analysis_prompt = f"""
            COMPREHENSIVE PITCH ANALYSIS FRAMEWORK
            
            You are an expert pitch evaluation consultant with 15+ years of experience analyzing startup presentations for top-tier venture capital firms. Your analysis will be used by founders to improve their pitch effectiveness and by investors to make informed decisions.
            
            SESSION OVERVIEW:
            Founder: {state.get('founder_name', 'Unknown')}
            Company: {state.get('company_name', 'Unknown')}
            Session Duration: {analytics['duration_minutes']} minutes
            Completion Status: {len(analytics['completed_stages'])}/9 stages completed
            Current Stage: {analytics['current_stage']}
            Total Investor Questions: {analytics['total_questions']}
            Session Complete: {analytics['workflow_complete']}
            
            COMMUNICATION PERFORMANCE METRICS:
            Engagement Balance:
            - Founder Speaking: {comm_metrics['engagement']['talked_count']} instances ({comm_metrics['engagement']['talk_percentage']}%)
            - Investor Listening: {comm_metrics['engagement']['listened_count']} instances ({comm_metrics['engagement']['listen_percentage']}%)
            - Optimal Range: 70-80% founder speaking, 20-30% listening/responding
            
            Fluency Assessment:
            - Filler Words: {comm_metrics['fluency']['filler_count']} instances (um, uh, like, you know)
            - Grammar Issues: {comm_metrics['fluency']['grammar_issues']} detected
            - Vocabulary Sophistication: {comm_metrics['fluency']['vocabulary_richness']} score
            
            Conversational Dynamics:
            - Total Conversation Turns: {comm_metrics['interactivity']['conversation_turns']}
            - Turn Frequency: {comm_metrics['interactivity']['turn_frequency']} exchanges per minute
            - Founder Questions Asked: {comm_metrics['questions_asked']['founder_questions']}
            - Question Rate: {comm_metrics['questions_asked']['questions_per_minute']} questions per minute
            
            STAGE-BY-STAGE INSIGHTS ANALYSIS:
            {json.dumps(analytics['key_insights'], indent=2)}
            
            VIDEO ANALYSIS INTEGRATION:
            {self._generate_video_analysis_summary(state)}
            
            COMPLETE CONVERSATION TRANSCRIPT:
            {self._format_conversation_for_analysis(state['messages'])}
            
            EVALUATION METHODOLOGY:
            Apply the following analytical framework to generate a comprehensive assessment:
            
            1. CONTENT ANALYSIS (70% of overall score):
            - Evaluate each of the 10 core pitch categories
            - Assess strategic thinking and business acumen
            - Analyze market understanding and competitive positioning
            - Review financial projections and business model viability
            
            2. COMMUNICATION ANALYSIS (20% of overall score):
            - Engagement balance and conversational flow
            - Speaking fluency and confidence indicators
            - Interactive responsiveness and adaptability
            - Question-asking behavior and investor engagement
            
            3. VIDEO ANALYSIS INTEGRATION (10% of overall score):
            - Body language and posture assessment
            - Hand gesture effectiveness and engagement
            - Facial expression and confidence indicators
            - Overall presentation presence and charisma
            
            4. INVESTOR READINESS ASSESSMENT:
            - Overall pitch maturity and sophistication
            - Ability to handle challenging questions
            - Demonstration of coachability and learning
            - Market timing and opportunity assessment
            - Professional presentation presence and confidence
            
            SCORING METHODOLOGY:
            Use a rigorous 100-point scale with the following calibration:
            - 90-100 (Vertx Assured): Exceptional, investor-ready quality that would impress top-tier VCs
            - 75-89 (Good): Strong performance with minor improvements needed for investor meetings
            - 60-74 (Satisfactory): Meets basic requirements but needs enhancement for serious consideration
            - 40-59 (Below Average): Some elements present but requires significant improvement
            - 0-39 (Need to Improve): Major gaps requiring fundamental work before investor presentations
            
            CRITICAL VIDEO ANALYSIS SCORING RULES - MUST FOLLOW EXACTLY:
            
            When video analysis is NOT available:
            ❌ NEVER assign scores of 0 for body_language or presentation_presence
            ❌ NEVER make negative assumptions about body language based on limited verbal responses
            ❌ NEVER use phrases like: "lack of confidence", "hesitant body language", "passive presence", "unconvincing", "likely poor", "suggests nervousness"
            ❌ NEVER infer poor physical presence from uncertain verbal responses
            
            ✅ ALWAYS assign reasonable baseline scores (45-70 range) for body_language and presentation_presence
            ✅ ALWAYS base scores on communication confidence, engagement patterns, and vocal indicators
            ✅ ALWAYS assume baseline professional presentation skills unless conversation clearly indicates otherwise
            ✅ ALWAYS use neutral, professional language in descriptions
            
            REQUIRED DESCRIPTION TEMPLATES when video NOT available:
            - body_language description: "Body language assessment based on professional communication patterns and vocal confidence indicators. Standard assessment mode active with focus on verbal delivery quality."
            - presentation_presence description: "Presentation presence based on vocal delivery quality, communication clarity, and professional engagement patterns. Standard assessment mode active with focus on verbal presentation skills."
            - gesture_effectiveness: "Professional gesture baseline"
            - posture_assessment: "Standard professional posture"
            - overall_presence: "Professional presence with standard baseline assessment"
            
            EXAMPLE - WRONG WAY (DO NOT DO THIS):
            {{
                "body_language": {{
                    "score": 15,
                    "description": "The founder's uncertain responses suggest lack of confidence that likely translates to hesitant body language",
                    "overall_presence": "Likely passive and unconvincing"
                }}
            }}
            
            EXAMPLE - CORRECT WAY (DO THIS):
            {{
                "body_language": {{
                    "score": 65,
                    "description": "Body language assessment based on professional communication patterns and vocal confidence indicators. Standard assessment mode active with focus on verbal delivery quality.",
                    "overall_presence": "Professional presence with standard baseline assessment"
                }}
            }}
            
            Generate your analysis in the following detailed JSON format:
            {{
                "overall_score": <score out of 100>,
                "overall_rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                "overall_description": "<Detailed description of overall performance>",
                "confidence_level": "<Low/Medium/High>",
                "pitch_readiness": "<Not Ready/Partially Ready/Ready/Investor Ready>",
                "category_scores": {{
                    "hooks_story": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on opening hooks and storytelling>"
                    }},
                    "problem_urgency": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on problem identification and urgency>"
                    }},
                    "solution_fit": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on solution and product-market fit>"
                    }},
                    "market_opportunity": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on market size and opportunity>"
                    }},
                    "team_execution": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on team strength and execution capability>"
                    }},
                    "business_model": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on revenue model and monetization>"
                    }},
                    "competitive_edge": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on competitive advantage and differentiation>"
                    }},
                    "traction_vision": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on current traction and future vision>"
                    }},
                    "funding_ask": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on funding requirements and use of funds>"
                    }},
                    "closing_impact": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on closing statement and call to action>"
                    }},
                    "engagement": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on talk vs listen balance>",
                        "talked_count": {comm_metrics['engagement']['talked_count']},
                        "listened_count": {comm_metrics['engagement']['listened_count']},
                        "talk_percentage": {comm_metrics['engagement']['talk_percentage']},
                        "listen_percentage": {comm_metrics['engagement']['listen_percentage']}
                    }},
                    "fluency": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on speaking fluency and confidence>",
                        "fillers": {comm_metrics['fluency']['filler_count']},
                        "grammar": {comm_metrics['fluency']['grammar_issues']},
                        "vocabulary": {comm_metrics['fluency']['vocabulary_richness']}
                    }},
                    "interactivity": {{
                        "score": <score 0-10>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on conversation flow and interaction>",
                        "conversation_turns": {comm_metrics['interactivity']['conversation_turns']},
                        "turn_frequency": {comm_metrics['interactivity']['turn_frequency']}
                    }},
                    "body_language": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<When video available: detailed feedback on posture, gestures, and physical presence. When video NOT available: 'Body language assessment based on vocal confidence and communication patterns. Score reflects professional presentation baseline inferred from conversation quality.'>",
                        "gesture_effectiveness": "<When video available: assessment of hand gesture usage. When video NOT available: 'Professional gestures assumed based on communication style'>",
                        "posture_assessment": "<When video available: evaluation of body posture and engagement. When video NOT available: 'Standard professional posture assumed'>",
                        "overall_presence": "<When video available: overall physical presentation assessment. When video NOT available: 'Professional presence inferred from vocal confidence and engagement level'>"
                    }},
                    "questions_asked": {{
                        "score": <score 0-10>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on question asking frequency and quality>",
                        "total_questions": {comm_metrics['questions_asked']['founder_questions']},
                        "questions_per_minute": {comm_metrics['questions_asked']['questions_per_minute']}
                    }},
                    "presentation_presence": {{
                        "score": <score out of 100>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<When video available: detailed feedback on facial expressions, eye contact, and charisma. When video NOT available: 'Presentation presence based on vocal confidence, communication clarity, and engagement level. Score reflects professional presentation baseline inferred from conversation quality.'>",
                        "facial_expressions": "<When video available: assessment of facial expression effectiveness. When video NOT available: 'Professional expressions assumed based on vocal tone and engagement'>",
                        "eye_contact": "<When video available: evaluation of eye contact and engagement. When video NOT available: 'Standard eye contact assumed based on communication confidence'>",
                        "confidence_indicators": "<When video available: analysis of confidence through visual cues. When video NOT available: 'Confidence indicators based on vocal patterns and response quality'>"
                    }}
                }},
                "strengths": [
                    {{"area": "<strength area>", "description": "<detailed description>", "score": <1-10>}},
                    ...
                ],
                "weaknesses": [
                    {{"area": "<weakness area>", "description": "<detailed description>", "improvement": "<specific improvement suggestion>"}},
                    ...
                ],
                "key_recommendations": [
                    "<specific actionable recommendation>",
                    ...
                ],
                "investor_perspective": "<What an investor would think about this pitch>",
                "next_steps": [
                    "<immediate action item>",
                    ...
                ],
                "founder_performance": [
                    {{
                        "title": "<Performance aspect title>",
                        "description": "<Detailed description of founder's performance in this area>"
                    }},
                    ...
                ],
                "what_worked": [
                    "<Specific thing that worked well in the pitch>",
                    ...
                ],
                "what_didnt_work": [
                    "<Specific thing that didn't work well in the pitch>",
                    ...
                ]
            }}
            
            Base your analysis on these 16 key categories:
            
            CONTENT CATEGORIES:
            1. HOOKS & STORY: Opening engagement, storytelling ability, emotional connection
            2. PROBLEM & URGENCY: Problem identification, market pain points, urgency demonstration
            3. SOLUTION & FIT: Solution clarity, product-market fit, value proposition
            4. MARKET & OPPORTUNITY: Market size, opportunity assessment, target audience
            5. TEAM & EXECUTION: Team strength, execution capability, relevant experience
            6. BUSINESS MODEL: Revenue streams, monetization strategy, financial sustainability
            7. COMPETITIVE EDGE: Differentiation, competitive advantage, unique positioning
            8. TRACTION & VISION: Current progress, growth metrics, future roadmap
            9. FUNDING ASK: Funding requirements, use of funds, investment rationale
            10. CLOSING IMPACT: Call to action, memorable closing, investor engagement
            
            COMMUNICATION CATEGORIES:
            11. ENGAGEMENT: How much you talked vs. how much you listened. Ideal balance is 70-80% founder talking, 20-30% listening/responding to investor questions.
            12. FLUENCY: How smoothly and confidently you speak. Analyzed through fillers (um, uh, like), grammar issues, and vocabulary richness.
            13. INTERACTIVITY: How frequently the conversation shifted between you and the investor, rated on a 0–10 scale. Higher scores indicate better conversational flow.
            14. QUESTIONS ASKED: Number of questions asked per minute, rated on a 0–10 scale. Shows engagement and curiosity about investor perspective.
            
            VIDEO ANALYSIS CATEGORIES:
            15. BODY LANGUAGE: Posture, engagement level, and overall physical presence. Includes gesture effectiveness and professional demeanor.
            16. PRESENTATION PRESENCE: Facial expressions, eye contact, confidence indicators, and overall charisma during the pitch.
            
            ADDITIONAL ANALYSIS SECTIONS:
            - FOUNDER PERFORMANCE: Specific aspects of founder's performance with titles and detailed descriptions (e.g., "Opener Effectiveness", "Confidence Level", "Storytelling Ability", "Technical Knowledge", "Vision Communication")
            - WHAT WORKED: Specific positive elements that were effective in the pitch
            - WHAT DIDN'T WORK: Specific areas that were ineffective or problematic in the pitch
            
            Rating Scale:
            - Need to Improve (0-39): Significant gaps, requires major work
            - Below Average (40-59): Some elements present but needs improvement
            - Satisfactory (60-74): Meets basic requirements, room for enhancement
            - Good (75-89): Strong performance, minor improvements needed
            - Vertx Assured (90-100): Exceptional, investor-ready quality
            """
            
            try:
                # Debug: Log state video analysis info
                logger.info(f"Analysis Debug - Video enabled: {state.get('video_analysis_enabled', False)}, Video insights: {len(state.get('video_insights', []))}, Gesture feedback: {len(state.get('gesture_feedback', []))}")
                
                analysis_response = llm.invoke(analysis_prompt)
                analysis_text = analysis_response.content
                
                # Try to parse JSON from the response
                if "```json" in analysis_text:
                    json_start = analysis_text.find("```json") + 7
                    json_end = analysis_text.find("```", json_start)
                    analysis_json = analysis_text[json_start:json_end].strip()
                elif "{" in analysis_text and "}" in analysis_text:
                    json_start = analysis_text.find("{")
                    json_end = analysis_text.rfind("}") + 1
                    analysis_json = analysis_text[json_start:json_end]
                else:
                    analysis_json = analysis_text
                
                analysis_data = json.loads(analysis_json)
                
                # Validate and fix video analysis scores if AI ignored our guidance
                analysis_data = self._validate_and_fix_video_analysis(analysis_data, state)
                
                # Add session metadata
                analysis_data.update({
                    "session_id": conversation_id,
                    "generated_at": datetime.now().isoformat(),
                    "session_duration_minutes": analytics['duration_minutes'],
                    "stages_completed": len(analytics['completed_stages']),
                    "total_stages": 9,
                    "completion_percentage": round((len(analytics['completed_stages']) / 9) * 100, 1),
                    "founder_name": state.get('founder_name', ''),
                    "company_name": state.get('company_name', ''),
                    "persona_used": state['persona']
                })
                
                return analysis_data
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse analysis JSON: {e}")
                # Return a basic analysis if JSON parsing fails
                return self._generate_basic_analysis(state, analytics)
                
        except Exception as e:
            logger.error(f"Error generating pitch analysis: {e}")
            return {"error": "Failed to generate pitch analysis"}
    
    def _format_conversation_for_analysis(self, messages: List[BaseMessage]) -> str:
        """Format conversation messages for analysis"""
        formatted = []
        for i, msg in enumerate(messages[-20:]):  # Last 20 messages to avoid token limits
            role = "Founder" if msg.__class__.__name__ == "HumanMessage" else "Investor"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
    
    def _validate_and_fix_video_analysis(self, analysis_data: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate AI analysis and fix video analysis issues if present"""
        
        video_enabled = state.get('video_analysis_enabled', False)
        video_insights = state.get('video_insights', [])
        gesture_feedback = state.get('gesture_feedback', [])
        posture_feedback = state.get('posture_feedback', [])
        expression_feedback = state.get('expression_feedback', [])
        
        # Check if video analysis is actually available
        has_video_data = (video_insights or gesture_feedback or posture_feedback or expression_feedback)
        
        # Store video analysis status in state for use in final analysis
        state['video_analysis_available'] = has_video_data
        
        # Log the video analysis status
        logger.info(f"Video Analysis Available: {has_video_data} (Data: {len(video_insights) + len(gesture_feedback) + len(posture_feedback) + len(expression_feedback)} items)")
        
        # Check if video analysis is available in the state
        video_analysis_available = state.get('video_analysis_available', False)
        
        # Log the video analysis availability
        logger.info(f"Video Analysis Available (from state): {video_analysis_available}")
        
        if not video_analysis_available and 'category_scores' in analysis_data:
            # Video analysis not available - validate and fix scores/descriptions
            
            # Check body_language
            if 'body_language' in analysis_data['category_scores']:
                body_lang = analysis_data['category_scores']['body_language']
                
                # Fix low scores
                if body_lang.get('score', 0) < 40:
                    body_lang['score'] = max(45, min(65, 50 + len(state.get('messages', []))))
                
                # Fix problematic descriptions
                problematic_phrases = [
                    'lack of confidence', 'hesitant body language', 'passive presence',
                    'unconvincing', 'likely poor', 'suggests nervousness', 'appears disengaged',
                    'probably nervous', 'seems uncertain'
                ]
                
                description = body_lang.get('description', '')
                if any(phrase.lower() in description.lower() for phrase in problematic_phrases):
                    body_lang['description'] = "Body language assessment based on communication confidence and engagement patterns. Score reflects professional presentation baseline inferred from conversation quality."
                
                # Fix other fields
                if 'gesture_effectiveness' in body_lang:
                    if any(phrase.lower() in body_lang['gesture_effectiveness'].lower() for phrase in problematic_phrases):
                        body_lang['gesture_effectiveness'] = "Professional gestures assumed based on communication style"
                
                if 'posture_assessment' in body_lang:
                    if any(phrase.lower() in body_lang['posture_assessment'].lower() for phrase in problematic_phrases):
                        body_lang['posture_assessment'] = "Standard professional posture assumed"
                
                if 'overall_presence' in body_lang:
                    if any(phrase.lower() in body_lang['overall_presence'].lower() for phrase in problematic_phrases):
                        body_lang['overall_presence'] = "Professional presence inferred from vocal confidence and engagement level"
            
            # Check presentation_presence
            if 'presentation_presence' in analysis_data['category_scores']:
                pres_presence = analysis_data['category_scores']['presentation_presence']
                
                # Fix low scores
                if pres_presence.get('score', 0) < 40:
                    pres_presence['score'] = max(45, min(70, 52 + len(state.get('messages', []))))
                
                # Fix problematic descriptions
                description = pres_presence.get('description', '')
                if any(phrase.lower() in description.lower() for phrase in problematic_phrases):
                    pres_presence['description'] = "Presentation presence based on vocal confidence, communication clarity, and engagement level. Score reflects professional presentation baseline inferred from conversation quality."
                
                # Fix other fields
                if 'facial_expressions' in pres_presence:
                    if any(phrase.lower() in pres_presence['facial_expressions'].lower() for phrase in problematic_phrases):
                        pres_presence['facial_expressions'] = "Professional expressions assumed based on vocal tone and engagement"
                
                if 'eye_contact' in pres_presence:
                    if any(phrase.lower() in pres_presence['eye_contact'].lower() for phrase in problematic_phrases):
                        pres_presence['eye_contact'] = "Standard eye contact assumed based on communication confidence"
                
                if 'confidence_indicators' in pres_presence:
                    if any(phrase.lower() in pres_presence['confidence_indicators'].lower() for phrase in problematic_phrases):
                        pres_presence['confidence_indicators'] = "Confidence indicators based on vocal patterns and response quality"
        
        return analysis_data
    
    def _calculate_communication_metrics(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Calculate communication and interaction metrics"""
        
        founder_messages = [msg for msg in messages if msg.__class__.__name__ == "HumanMessage"]
        investor_messages = [msg for msg in messages if msg.__class__.__name__ == "AIMessage"]
        
        # 1. Engagement Analysis (Talk vs Listen ratio)
        founder_word_count = sum(len(msg.content.split()) for msg in founder_messages)
        investor_word_count = sum(len(msg.content.split()) for msg in investor_messages)
        total_words = founder_word_count + investor_word_count
        
        talk_percentage = (founder_word_count / total_words * 100) if total_words > 0 else 0
        listen_percentage = (investor_word_count / total_words * 100) if total_words > 0 else 0
        
        # 2. Fluency Analysis (estimate from text patterns)
        total_founder_text = " ".join([msg.content for msg in founder_messages])
        
        # Count potential fillers (basic estimation)
        filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally', 'so', 'well']
        filler_count = sum(total_founder_text.lower().count(filler) for filler in filler_words)
        
        # Estimate grammar issues (very basic - repeated words, incomplete sentences)
        sentences = total_founder_text.split('.')
        grammar_issues = 0
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) > 1:
                # Check for repeated consecutive words
                for i in range(len(words) - 1):
                    if words[i].lower() == words[i + 1].lower():
                        grammar_issues += 1
        
        # Vocabulary richness (unique words / total words)
        all_founder_words = total_founder_text.lower().split()
        unique_words = len(set(all_founder_words))
        vocabulary_richness = (unique_words / len(all_founder_words)) if all_founder_words else 0
        
        # 3. Interactivity (conversation turns)
        conversation_turns = 0
        last_speaker = None
        for msg in messages:
            current_speaker = "Founder" if msg.__class__.__name__ == "HumanMessage" else "Investor"
            if last_speaker and last_speaker != current_speaker:
                conversation_turns += 1
            last_speaker = current_speaker
        
        # 4. Questions Asked Analysis
        founder_questions = 0
        for msg in founder_messages:
            founder_questions += msg.content.count('?')
        
        # Calculate session duration in minutes for questions per minute
        session_duration_minutes = len(messages) * 0.5  # Rough estimate: 30 seconds per message
        questions_per_minute = (founder_questions / session_duration_minutes) if session_duration_minutes > 0 else 0
        
        return {
            "engagement": {
                "talked_count": len(founder_messages),
                "listened_count": len(investor_messages),
                "talk_percentage": round(talk_percentage, 1),
                "listen_percentage": round(listen_percentage, 1),
                "founder_word_count": founder_word_count,
                "investor_word_count": investor_word_count
            },
            "fluency": {
                "filler_count": filler_count,
                "grammar_issues": grammar_issues,
                "vocabulary_richness": round(vocabulary_richness, 3),
                "total_words": len(all_founder_words)
            },
            "interactivity": {
                "conversation_turns": conversation_turns,
                "total_messages": len(messages),
                "turn_frequency": round(conversation_turns / len(messages), 2) if messages else 0
            },
            "questions_asked": {
                "founder_questions": founder_questions,
                "session_duration_minutes": round(session_duration_minutes, 1),
                "questions_per_minute": round(questions_per_minute, 2)
            }
        }
    
    def _generate_basic_analysis(self, state: Dict, analytics: Dict) -> Dict[str, Any]:
        """Generate basic analysis when AI analysis fails"""
        
        # Count founder messages to determine conversation depth
        founder_messages = [msg for msg in state['messages'] if msg.__class__.__name__ == "HumanMessage"]
        founder_message_count = len(founder_messages)
        
        # No minimum conversation length requirement - proceed with analysis regardless of length
        # Note: We'll adjust the depth factor to scale scores appropriately for short conversations
        
        stages_completed = len(analytics['completed_stages'])
        total_insights = sum(len(insights) for insights in analytics['key_insights'].values())
        
        # Calculate communication metrics
        comm_metrics = self._calculate_communication_metrics(state['messages'])
        
        # Calculate conversation depth factor (0.1 to 1.0 based on founder message count)
        # This ensures scores scale with conversation depth but still provides analysis for very short conversations
        depth_factor = min(1.0, max(0.1, (founder_message_count + 1) / 10))
        
        # Calculate basic scores with depth factor applied
        completion_score = (stages_completed / 9) * 40 * depth_factor  # 40% for completion
        insight_score = min(total_insights * 2, 30) * depth_factor     # 30% for insights quality
        duration_score = min(analytics['duration_minutes'] / 30 * 20, 20) * depth_factor  # 20% for engagement
        response_score = min(analytics['total_questions'] * 2, 10) * depth_factor  # 10% for responsiveness
        
        overall_score = int(completion_score + insight_score + duration_score + response_score)
        
        def get_rating(score):
            if score >= 90: return "Vertx Assured"
            elif score >= 75: return "Good"
            elif score >= 60: return "Satisfactory"
            elif score >= 40: return "Below Average"
            else: return "Need to Improve"
        
        # Generate basic category scores that scale with conversation depth
        # For very short conversations, scores will be much lower
        base_category_score = min(max(10, overall_score - 20), 50) * depth_factor
        
        # Calculate video analysis scores if available
        video_enabled = state.get('video_analysis_enabled', False)
        video_insights = state.get('video_insights', [])
        gesture_feedback = state.get('gesture_feedback', [])
        posture_feedback = state.get('posture_feedback', [])
        expression_feedback = state.get('expression_feedback', [])
        
        # Check if video analysis is available from the state
        video_analysis_available = state.get('video_analysis_available', False)
        
        # Log the video analysis availability for scoring
        logger.info(f"Video Analysis Available for Scoring: {video_analysis_available}")
        
        # Video analysis scoring
        if video_analysis_available:
            # Calculate video scores based on actual feedback
            positive_feedback = len([f for f in gesture_feedback + posture_feedback + expression_feedback 
                                   if any(word in f.lower() for word in ['excellent', 'strong', 'confident', 'positive'])])
            negative_feedback = len([f for f in gesture_feedback + posture_feedback + expression_feedback 
                                   if any(word in f.lower() for word in ['poor', 'nervous', 'disengaged', 'negative'])])
            
            body_language_score = max(20, min(95, base_category_score + (positive_feedback * 10) - (negative_feedback * 15)))
            presentation_presence_score = max(20, min(95, base_category_score + (positive_feedback * 8) - (negative_feedback * 12)))
        else:
            # Provide reasonable baseline scores when video analysis not available
            # Base on communication quality and engagement
            communication_quality = min(80, 50 + (analytics['total_questions'] * 5) + (stages_completed * 3))
            body_language_score = max(45, min(70, communication_quality - 5))
            presentation_presence_score = max(45, min(70, communication_quality))
        category_scores = {
            "hooks_story": {
                "score": base_category_score,
                "rating": get_rating(base_category_score),
                "description": "Basic storytelling elements present, could be more engaging"
            },
            "problem_urgency": {
                "score": base_category_score + 5,
                "rating": get_rating(base_category_score + 5),
                "description": "Problem identification attempted, needs more urgency"
            },
            "solution_fit": {
                "score": base_category_score,
                "rating": get_rating(base_category_score),
                "description": "Solution presented, product-market fit needs validation"
            },
            "market_opportunity": {
                "score": base_category_score - 5,
                "rating": get_rating(base_category_score - 5),
                "description": "Market opportunity mentioned, needs more specific data"
            },
            "team_execution": {
                "score": base_category_score,
                "rating": get_rating(base_category_score),
                "description": "Team information provided, execution capability unclear"
            },
            "business_model": {
                "score": base_category_score - 10,
                "rating": get_rating(base_category_score - 10),
                "description": "Business model needs clearer articulation"
            },
            "competitive_edge": {
                "score": base_category_score - 5,
                "rating": get_rating(base_category_score - 5),
                "description": "Competitive advantage mentioned, needs stronger differentiation"
            },
            "traction_vision": {
                "score": base_category_score,
                "rating": get_rating(base_category_score),
                "description": "Some traction indicators, vision needs more clarity"
            },
            "funding_ask": {
                "score": base_category_score - 15,
                "rating": get_rating(base_category_score - 15),
                "description": "Funding requirements need more specific details"
            },
            "closing_impact": {
                "score": base_category_score - 10,
                "rating": get_rating(base_category_score - 10),
                "description": "Closing needs more impact and clear call to action"
            },
            "engagement": {
                "score": min(100, max(0, 100 - abs(comm_metrics['engagement']['talk_percentage'] - 75) * 2)),
                "rating": get_rating(min(100, max(0, 100 - abs(comm_metrics['engagement']['talk_percentage'] - 75) * 2))),
                "description": f"Talk/Listen ratio: {comm_metrics['engagement']['talk_percentage']:.1f}% talking, {comm_metrics['engagement']['listen_percentage']:.1f}% listening",
                "talked_count": comm_metrics['engagement']['talked_count'],
                "listened_count": comm_metrics['engagement']['listened_count'],
                "talk_percentage": comm_metrics['engagement']['talk_percentage'],
                "listen_percentage": comm_metrics['engagement']['listen_percentage']
            },
            "fluency": {
                "score": max(0, 100 - (comm_metrics['fluency']['filler_count'] * 5) - (comm_metrics['fluency']['grammar_issues'] * 10) + (comm_metrics['fluency']['vocabulary_richness'] * 50)),
                "rating": get_rating(max(0, 100 - (comm_metrics['fluency']['filler_count'] * 5) - (comm_metrics['fluency']['grammar_issues'] * 10) + (comm_metrics['fluency']['vocabulary_richness'] * 50))),
                "description": f"Fluency analysis: {comm_metrics['fluency']['filler_count']} fillers, {comm_metrics['fluency']['grammar_issues']} grammar issues, {comm_metrics['fluency']['vocabulary_richness']:.3f} vocabulary richness",
                "fillers": comm_metrics['fluency']['filler_count'],
                "grammar": comm_metrics['fluency']['grammar_issues'],
                "vocabulary": comm_metrics['fluency']['vocabulary_richness']
            },
            "interactivity": {
                "score": min(10, comm_metrics['interactivity']['turn_frequency'] * 10),
                "rating": get_rating(min(100, comm_metrics['interactivity']['turn_frequency'] * 100)),
                "description": f"Conversation flow: {comm_metrics['interactivity']['conversation_turns']} turns with {comm_metrics['interactivity']['turn_frequency']:.2f} turn frequency",
                "conversation_turns": comm_metrics['interactivity']['conversation_turns'],
                "turn_frequency": comm_metrics['interactivity']['turn_frequency']
            },
            "questions_asked": {
                "score": min(10, comm_metrics['questions_asked']['questions_per_minute'] * 5),
                "rating": get_rating(min(100, comm_metrics['questions_asked']['questions_per_minute'] * 50)),
                "description": f"Question engagement: {comm_metrics['questions_asked']['founder_questions']} questions at {comm_metrics['questions_asked']['questions_per_minute']:.2f} per minute",
                "total_questions": comm_metrics['questions_asked']['founder_questions'],
                "questions_per_minute": comm_metrics['questions_asked']['questions_per_minute']
            },
            "body_language": {
                "score": body_language_score,
                "rating": get_rating(body_language_score),
                "description": f"Body language assessment based on {'video analysis with {len(gesture_feedback)} gesture observations and {len(posture_feedback)} posture assessments' if video_analysis_available else 'professional communication patterns and vocal confidence indicators. Standard assessment mode active with focus on verbal delivery quality'}",
                "gesture_effectiveness": f"{len(gesture_feedback)} gestures analyzed" if video_analysis_available else "Professional gesture baseline",
                "posture_assessment": f"{len(posture_feedback)} posture observations" if video_analysis_available else "Standard professional posture",
                "overall_presence": "Good physical presence" if body_language_score >= 70 else "Professional presence with standard baseline assessment"
            },
            "presentation_presence": {
                "score": presentation_presence_score,
                "rating": get_rating(presentation_presence_score),
                "description": f"Presentation presence based on {'video analysis with {len(expression_feedback)} expression observations' if video_analysis_available else 'vocal delivery quality, communication clarity, and professional engagement patterns. Standard assessment mode active with focus on verbal presentation skills'}",
                "facial_expressions": f"{len(expression_feedback)} expressions analyzed" if video_analysis_available else "Professional expressions assumed",
                "eye_contact": "Good eye contact maintained" if presentation_presence_score >= 70 else "Standard eye contact assumed",
                "confidence_indicators": "Strong confidence signals" if presentation_presence_score >= 80 else "Moderate confidence" if presentation_presence_score >= 60 else "Developing confidence indicators"
            }
        }
        
        return {
            "session_id": analytics['session_id'],
            "generated_at": datetime.now().isoformat(),
            "overall_score": overall_score,
            "overall_rating": get_rating(overall_score),
            "overall_description": f"Completed {stages_completed}/9 stages with {total_insights} key insights. {'Good foundation' if overall_score >= 60 else 'Needs improvement'} for investor presentation.",
            "confidence_level": "Medium" if overall_score >= 60 else "Low",
            "pitch_readiness": "Ready" if overall_score >= 80 else "Partially Ready" if overall_score >= 60 else "Not Ready",
            "session_duration_minutes": analytics['duration_minutes'],
            "stages_completed": stages_completed,
            "total_stages": 9,
            "completion_percentage": round((stages_completed / 9) * 100, 1),
            "founder_name": state.get('founder_name', ''),
            "company_name": state.get('company_name', ''),
            "persona_used": state['persona'],
            "category_scores": category_scores,
            "strengths": [
                {"area": "Engagement", "description": f"Completed {stages_completed} stages showing good engagement", "score": min(stages_completed, 10)},
                {"area": "Detail", "description": f"Provided {total_insights} key insights across stages", "score": min(total_insights // 2, 10)}
            ] + ([{"area": "Video Analysis", "description": f"Video analysis active with {len(video_insights)} insights captured", "score": min(len(video_insights), 10)}] if video_enabled else []),
            "weaknesses": [
                {"area": "Completion", "description": f"Only completed {stages_completed}/9 stages", "improvement": "Complete all pitch stages for comprehensive feedback"}
            ] if stages_completed < 9 else [],
            "key_recommendations": [
                "Complete all 10 pitch categories for comprehensive evaluation",
                "Provide more specific metrics and data points",
                "Practice articulating value proposition clearly",
                "Strengthen competitive differentiation",
                "Develop more impactful closing statements"
            ] + (["Leverage video analysis insights to improve body language and presentation presence",
                  "Focus on confident gestures and posture during key pitch moments"] if video_enabled else 
                 ["Enable video analysis for comprehensive presentation feedback"]),
            "investor_perspective": f"Based on {stages_completed} completed stages, this pitch shows {'good potential' if overall_score >= 60 else 'needs improvement'}. Focus on completing all categories for investor readiness.",
            "next_steps": [
                "Complete remaining pitch stages" if stages_completed < 9 else "Practice with different investor personas",
                "Gather more specific metrics and data",
                "Refine value proposition and differentiation"
            ],
            "founder_performance": [
                {
                    "title": "Opener Effectiveness",
                    "description": f"The founder introduced themselves and company but {'could improve engagement' if overall_score < 60 else 'showed good initial engagement'}."
                },
                {
                    "title": "Stage Completion",
                    "description": f"Completed {stages_completed} out of 9 pitch stages, {'showing partial commitment' if stages_completed < 5 else 'demonstrating good follow-through'}."
                },
                {
                    "title": "Communication Style",
                    "description": f"Talk/listen ratio of {comm_metrics['engagement']['talk_percentage']:.1f}%/{comm_metrics['engagement']['listen_percentage']:.1f}% {'needs balancing' if abs(comm_metrics['engagement']['talk_percentage'] - 75) > 15 else 'shows good balance'}."
                },
                {
                    "title": "Question Engagement",
                    "description": f"Asked {comm_metrics['questions_asked']['founder_questions']} questions during the session, {'showing limited curiosity' if comm_metrics['questions_asked']['founder_questions'] < 2 else 'demonstrating good investor engagement'}."
                }
            ] + (self._generate_stage_video_performance_summary(state) if video_enabled else []),
            "what_worked": [
                f"Completed {stages_completed} pitch stages showing commitment to the process",
                f"Maintained {comm_metrics['interactivity']['turn_frequency']:.2f} turn frequency indicating good conversational flow" if comm_metrics['interactivity']['turn_frequency'] > 0.5 else "Engaged in structured conversation",
                f"Achieved {comm_metrics['fluency']['vocabulary_richness']:.3f} vocabulary richness showing articulate communication" if comm_metrics['fluency']['vocabulary_richness'] > 0.7 else "Communicated core business concepts clearly"
            ],
            "what_didnt_work": [
                f"Only completed {stages_completed}/9 stages, missing key pitch elements" if stages_completed < 9 else "Could benefit from more detailed responses in some areas",
                f"Used {comm_metrics['fluency']['filler_count']} filler words, affecting speech fluency" if comm_metrics['fluency']['filler_count'] > 5 else "Minor areas for communication improvement",
                f"Asked only {comm_metrics['questions_asked']['founder_questions']} questions, showing limited investor engagement" if comm_metrics['questions_asked']['founder_questions'] < 3 else "Could ask more strategic questions about investor perspective"
            ]
        }
    
    def end_session_with_analysis(self, conversation_id: str, reason: str = "user_ended") -> Dict[str, Any]:
        """End session and generate comprehensive analysis"""
        
        try:
            # Generate the analysis
            analysis = self.generate_pitch_analysis(conversation_id)
            
            if "error" not in analysis:
                # Mark session as complete
                config = {"configurable": {"thread_id": conversation_id}}
                current_state = self.workflow.get_state(config)
                
                if current_state.values:
                    current_state.values["workflow_complete"] = True
                    current_state.values["end_reason"] = reason
                    current_state.values["analysis_generated"] = True
                    
                    try:
                        self.workflow.update_state(config, current_state.values)
                    except Exception as update_error:
                        logger.warning(f"Failed to update session state: {update_error}")
                
                # Add end reason to analysis
                analysis["end_reason"] = reason
                analysis["session_ended_at"] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error ending session with analysis: {e}")
            return {"error": "Failed to generate session analysis"}
    
    def _generate_video_analysis_summary(self, state: Dict[str, Any]) -> str:
        """Generate comprehensive video analysis summary for pitch analysis"""
        
        # Debug: Log video analysis state
        video_enabled = state.get('video_analysis_enabled', False)
        video_insights = state.get('video_insights', [])
        gesture_feedback = state.get('gesture_feedback', [])
        posture_feedback = state.get('posture_feedback', [])
        expression_feedback = state.get('expression_feedback', [])
        
        logger.info(f"Video Analysis Debug - Enabled: {video_enabled}, Insights: {len(video_insights)}, Gestures: {len(gesture_feedback)}, Posture: {len(posture_feedback)}, Expressions: {len(expression_feedback)}")
        
        # FIXED: Check if we have any video analysis data, regardless of the video_enabled flag
        has_video_data = (len(video_insights) > 0 or len(gesture_feedback) > 0 or 
                         len(posture_feedback) > 0 or len(expression_feedback) > 0)
        
        # Force video_enabled to True if we have any video data
        if has_video_data and not video_enabled:
            logger.info("Video data detected despite video_enabled=False. Enabling video analysis.")
            video_enabled = True
            # Update the state to reflect that video analysis is enabled
            state['video_analysis_enabled'] = True
            
        # Store video analysis status in state for use in final analysis
        state['video_analysis_available'] = video_enabled and has_video_data
        logger.info(f"Setting video_analysis_available to {video_enabled and has_video_data}")
        
        if not video_enabled or not has_video_data:
            logger.warning("Video analysis not available for this session")
            # Set a flag in the state to indicate video analysis is not available
            state['video_analysis_available'] = False
            return """Video Analysis: Standard assessment mode active.
            
            ASSESSMENT APPROACH:
            - Focus on verbal communication quality and content
            - Evaluate engagement through conversation dynamics
            - Consider responsiveness and interaction patterns
            - Assess confidence through speech patterns and delivery
            
            SCORING GUIDANCE:
            Base presentation scores on verbal communication strength, with standard baseline (50-80 range) for professional presence."""
        
        video_insights = state.get('video_insights', [])
        gesture_feedback = state.get('gesture_feedback', [])
        posture_feedback = state.get('posture_feedback', [])
        expression_feedback = state.get('expression_feedback', [])
        
        # FIXED: Log the video analysis data for debugging
        logger.info(f"Video Analysis Data - Video Insights: {len(video_insights)}, Gesture Feedback: {len(gesture_feedback)}, Posture Feedback: {len(posture_feedback)}, Expression Feedback: {len(expression_feedback)}")
        
        # Create stage-specific video analysis summary
        stage_objectives = {
            "greeting": "Get the founder's name, company name, and a brief introduction",
            "problem_solution": "Understand the core problem they're solving and their solution approach",
            "target_market": "Identify their target customers and market opportunity",
            "business_model": "Understand how they make money and their pricing strategy",
            "competition": "Learn about their competitive landscape and differentiation",
            "traction": "Assess their current progress, metrics, and growth",
            "team": "Evaluate the team's experience and composition",
            "funding_needs": "Understand their funding requirements and use of funds",
            "future_plans": "Discuss their vision and long-term strategy"
        }
        
        current_stage = state.get('current_stage', 'unknown')
        stage_progress = state.get('stage_progress', {})
        
        summary_parts = []
        
        # Overall video analysis status
        summary_parts.append("VIDEO ANALYSIS SUMMARY:")
        summary_parts.append(f"- Video Analysis Status: ACTIVE")
        summary_parts.append(f"- Video Data Points: {len(video_insights) + len(gesture_feedback) + len(posture_feedback) + len(expression_feedback)}")
        summary_parts.append(f"- Current Stage: {current_stage}")
        summary_parts.append(f"- Stage Objective: {stage_objectives.get(current_stage, 'Unknown stage')}")
        
        # Video insights summary
        if video_insights:
            summary_parts.append(f"\nKEY VIDEO INSIGHTS ({len(video_insights)} total):")
            for insight in video_insights[-5:]:  # Last 5 insights
                summary_parts.append(f"  • {insight}")
        else:
            summary_parts.append(f"\nKEY VIDEO INSIGHTS: No significant insights detected yet")
        
        # Gesture analysis
        if gesture_feedback:
            summary_parts.append(f"\nHAND GESTURE ANALYSIS ({len(gesture_feedback)} observations):")
            for gesture in gesture_feedback[-3:]:  # Last 3 gestures
                summary_parts.append(f"  • {gesture}")
        else:
            summary_parts.append(f"\nHAND GESTURE ANALYSIS: Limited hand gesture activity detected")
        
        # Posture analysis
        if posture_feedback:
            summary_parts.append(f"\nPOSTURE & BODY LANGUAGE ({len(posture_feedback)} observations):")
            for posture in posture_feedback[-3:]:  # Last 3 posture observations
                summary_parts.append(f"  • {posture}")
        else:
            summary_parts.append(f"\nPOSTURE & BODY LANGUAGE: Neutral posture maintained")
        
        # Expression analysis
        if expression_feedback:
            summary_parts.append(f"\nFACIAL EXPRESSION ANALYSIS ({len(expression_feedback)} observations):")
            for expression in expression_feedback[-3:]:  # Last 3 expressions
                summary_parts.append(f"  • {expression}")
        else:
            summary_parts.append(f"\nFACIAL EXPRESSION ANALYSIS: Neutral expressions observed")
        
        # Stage-specific video performance summary
        summary_parts.append(f"\nSTAGE-SPECIFIC VIDEO PERFORMANCE ANALYSIS:")
        
        # Analyze video performance by stage with detailed assessment
        for stage, objective in stage_objectives.items():
            stage_info = stage_progress.get(stage, {})
            questions_asked = stage_info.get('questions_asked', 0)
            
            if questions_asked > 0:
                # This stage was covered - provide detailed analysis
                stage_video_insights = [insight for insight in video_insights if stage in insight.lower()]
                stage_gestures = [gesture for gesture in gesture_feedback if stage in gesture.lower()]
                stage_posture = [posture for posture in posture_feedback if stage in posture.lower()]
                stage_expressions = [expr for expr in expression_feedback if stage in expr.lower()]
                
                summary_parts.append(f"\n  📋 {stage.upper().replace('_', ' ')} STAGE:")
                summary_parts.append(f"     Objective: {objective}")
                summary_parts.append(f"     Questions Covered: {questions_asked}")
                
                # Video performance assessment for this stage
                stage_video_score = self._calculate_stage_video_score(
                    stage_video_insights, stage_gestures, stage_posture, stage_expressions
                )
                
                summary_parts.append(f"     Video Performance Score: {stage_video_score}/100")
                
                # Detailed stage analysis
                if stage_video_insights:
                    summary_parts.append(f"     ✅ Key Video Insights ({len(stage_video_insights)}):")
                    for insight in stage_video_insights[-2:]:
                        summary_parts.append(f"        • {insight}")
                
                if stage_gestures:
                    summary_parts.append(f"     👋 Gesture Analysis:")
                    for gesture in stage_gestures[-1:]:
                        summary_parts.append(f"        • {gesture}")
                
                if stage_posture:
                    summary_parts.append(f"     🧍 Posture Assessment:")
                    for posture in stage_posture[-1:]:
                        summary_parts.append(f"        • {posture}")
                
                if stage_expressions:
                    summary_parts.append(f"     😊 Expression Analysis:")
                    for expr in stage_expressions[-1:]:
                        summary_parts.append(f"        • {expr}")
                
                # Stage-specific recommendations
                stage_recommendations = self._generate_stage_video_recommendations(
                    stage, stage_video_insights, stage_gestures, stage_posture, stage_expressions
                )
                if stage_recommendations:
                    summary_parts.append(f"     💡 Stage Recommendations:")
                    for rec in stage_recommendations:
                        summary_parts.append(f"        • {rec}")
                
            else:
                # Stage not covered yet
                summary_parts.append(f"\n  ⏳ {stage.upper().replace('_', ' ')} STAGE:")
                summary_parts.append(f"     Objective: {objective}")
                summary_parts.append(f"     Status: Not yet covered")
                summary_parts.append(f"     Video Analysis: Pending stage completion")
        
        # Overall video analysis recommendations
        summary_parts.append(f"\nVIDEO ANALYSIS RECOMMENDATIONS:")
        
        # Generate recommendations based on collected data
        recommendations = []
        
        if len(gesture_feedback) < 3:
            recommendations.append("Increase hand gesture usage to enhance engagement and emphasis")
        
        if any("nervous" in feedback.lower() or "disengaged" in feedback.lower() 
               for feedback in posture_feedback + expression_feedback):
            recommendations.append("Focus on confident posture and positive facial expressions")
        
        if len(video_insights) < 5:
            recommendations.append("Maintain consistent eye contact and dynamic presentation style")
        
        if not recommendations:
            recommendations.append("Continue current presentation style - video analysis shows good engagement")
        
        for rec in recommendations:
            summary_parts.append(f"  • {rec}")
        
        # Add overall video performance progression summary
        summary_parts.append(f"\n" + "="*50)
        summary_parts.append("OVERALL VIDEO PERFORMANCE PROGRESSION:")
        
        # Calculate stage-by-stage performance
        stage_performances = {}
        for stage in stage_objectives.keys():
            stage_info = stage_progress.get(stage, {})
            if stage_info.get('questions_asked', 0) > 0:
                stage_video_insights = [insight for insight in video_insights if stage in insight.lower()]
                stage_gestures = [gesture for gesture in gesture_feedback if stage in gesture.lower()]
                stage_posture = [posture for posture in posture_feedback if stage in posture.lower()]
                stage_expressions = [expr for expr in expression_feedback if stage in expr.lower()]
                
                stage_score = self._calculate_stage_video_score(
                    stage_video_insights, stage_gestures, stage_posture, stage_expressions
                )
                stage_performances[stage] = stage_score
        
        # Show progression
        if stage_performances:
            summary_parts.append("Stage Performance Scores:")
            for stage, score in stage_performances.items():
                confidence_level = "🔥 Excellent" if score >= 80 else "✅ Good" if score >= 65 else "⚠️ Needs Work" if score >= 40 else "❌ Poor"
                summary_parts.append(f"  • {stage.replace('_', ' ').title()}: {score}/100 ({confidence_level})")
            
            # Performance insights
            best_stage = max(stage_performances, key=stage_performances.get)
            worst_stage = min(stage_performances, key=stage_performances.get)
            
            summary_parts.append(f"\n🏆 STRONGEST STAGE: {best_stage.replace('_', ' ').title()} ({stage_performances[best_stage]}/100)")
            summary_parts.append(f"💪 IMPROVEMENT OPPORTUNITY: {worst_stage.replace('_', ' ').title()} ({stage_performances[worst_stage]}/100)")
            
            # Overall trend
            scores = list(stage_performances.values())
            if len(scores) > 1:
                trend = "📈 Improving" if scores[-1] > scores[0] else "📉 Declining" if scores[-1] < scores[0] else "➡️ Consistent"
                summary_parts.append(f"📊 PERFORMANCE TREND: {trend} throughout the pitch")
        
        return "\n".join(summary_parts)
    
    def _generate_stage_video_performance_summary(self, state: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate stage-by-stage video performance summary for founder performance section"""
        
        video_insights = state.get('video_insights', [])
        gesture_feedback = state.get('gesture_feedback', [])
        posture_feedback = state.get('posture_feedback', [])
        expression_feedback = state.get('expression_feedback', [])
        stage_progress = state.get('stage_progress', {})
        
        stage_objectives = {
            "greeting": "Introduction and first impression",
            "problem_solution": "Problem urgency and solution clarity",
            "target_market": "Market opportunity presentation",
            "business_model": "Revenue model explanation",
            "competition": "Competitive positioning",
            "traction": "Growth and metrics presentation",
            "team": "Team strength demonstration",
            "funding_needs": "Investment ask delivery",
            "future_plans": "Vision and roadmap presentation"
        }
        
        performance_summary = []
        
        # Analyze each completed stage
        for stage, objective in stage_objectives.items():
            stage_info = stage_progress.get(stage, {})
            if stage_info.get('questions_asked', 0) > 0:
                # Stage was covered - analyze video performance
                stage_video_insights = [insight for insight in video_insights if stage in insight.lower()]
                stage_gestures = [gesture for gesture in gesture_feedback if stage in gesture.lower()]
                stage_posture = [posture for posture in posture_feedback if stage in posture.lower()]
                stage_expressions = [expr for expr in expression_feedback if stage in expr.lower()]
                
                stage_score = self._calculate_stage_video_score(
                    stage_video_insights, stage_gestures, stage_posture, stage_expressions
                )
                
                # Determine confidence level and feedback
                if stage_score >= 80:
                    confidence = "excellent confidence"
                    feedback = "Strong body language and presentation presence"
                elif stage_score >= 65:
                    confidence = "good confidence"
                    feedback = "Solid presentation skills with room for minor improvements"
                elif stage_score >= 50:
                    confidence = "moderate confidence"
                    feedback = "Adequate presentation but could benefit from more engagement"
                else:
                    confidence = "low confidence"
                    feedback = "Needs improvement in body language and presentation presence"
                
                # Create specific feedback based on stage
                stage_specific_feedback = {
                    "greeting": f"{'Warm and welcoming' if stage_score >= 70 else 'Could be more engaging'} during introductions",
                    "problem_solution": f"{'Showed urgency and passion' if stage_score >= 70 else 'Could express more conviction'} when discussing the problem",
                    "target_market": f"{'Demonstrated market excitement' if stage_score >= 70 else 'Could show more enthusiasm'} about market opportunity",
                    "business_model": f"{'Professional and confident' if stage_score >= 70 else 'Could be more assured'} during financial discussions",
                    "competition": f"{'Projected competitive strength' if stage_score >= 70 else 'Could show more determination'} when discussing competitors",
                    "traction": f"{'Displayed growth excitement' if stage_score >= 70 else 'Could be more energetic'} when sharing achievements",
                    "team": f"{'Showed team pride' if stage_score >= 70 else 'Could express more confidence'} in team capabilities",
                    "funding_needs": f"{'Clear and direct' if stage_score >= 70 else 'Appeared nervous or uncertain'} during funding ask",
                    "future_plans": f"{'Visionary and inspiring' if stage_score >= 70 else 'Could be more compelling'} when outlining future goals"
                }
                
                performance_summary.append({
                    "title": f"{stage.replace('_', ' ').title()} Stage Performance",
                    "description": f"{stage_specific_feedback.get(stage, feedback)} (Video Score: {stage_score}/100 - {confidence})"
                })
        
        return performance_summary
    
    def _calculate_stage_video_score(self, insights: List[str], gestures: List[str], 
                                   posture: List[str], expressions: List[str]) -> int:
        """Calculate video performance score for a specific stage"""
        base_score = 60  # Base score
        
        # Add points for positive indicators
        score_adjustments = 0
        
        # Insights scoring
        if insights:
            positive_insights = [i for i in insights if any(word in i.lower() 
                               for word in ['effective', 'strong', 'excellent', 'confident', 'engaged'])]
            score_adjustments += len(positive_insights) * 5
        
        # Gesture scoring
        if gestures:
            positive_gestures = [g for g in gestures if any(word in g.lower() 
                               for word in ['strong', 'effective', 'confident'])]
            score_adjustments += len(positive_gestures) * 8
        
        # Posture scoring
        if posture:
            positive_posture = [p for p in posture if any(word in p.lower() 
                              for word in ['excellent', 'engaged', 'confident'])]
            negative_posture = [p for p in posture if any(word in p.lower() 
                              for word in ['disengaged', 'poor', 'nervous'])]
            score_adjustments += len(positive_posture) * 10
            score_adjustments -= len(negative_posture) * 15
        
        # Expression scoring
        if expressions:
            positive_expressions = [e for e in expressions if any(word in e.lower() 
                                  for word in ['positive', 'confident', 'engaged'])]
            negative_expressions = [e for e in expressions if any(word in e.lower() 
                                  for word in ['nervous', 'fear', 'negative'])]
            score_adjustments += len(positive_expressions) * 8
            score_adjustments -= len(negative_expressions) * 12
        
        final_score = max(0, min(100, base_score + score_adjustments))
        return final_score
    
    def _generate_stage_video_recommendations(self, stage: str, insights: List[str], 
                                            gestures: List[str], posture: List[str], 
                                            expressions: List[str]) -> List[str]:
        """Generate stage-specific video recommendations"""
        recommendations = []
        
        # Stage-specific recommendations based on objectives
        stage_specific_advice = {
            "greeting": [
                "Maintain warm, welcoming facial expressions during introductions",
                "Use open hand gestures to appear approachable",
                "Keep confident posture to make strong first impression"
            ],
            "problem_solution": [
                "Use emphatic gestures when describing the problem urgency",
                "Lean forward slightly to show engagement with the problem",
                "Express concern through facial expressions when discussing pain points"
            ],
            "target_market": [
                "Use expansive gestures when describing market size",
                "Maintain confident posture when discussing market opportunity",
                "Show enthusiasm through facial expressions about market potential"
            ],
            "business_model": [
                "Use precise hand gestures when explaining revenue streams",
                "Maintain professional posture during financial discussions",
                "Show confidence through steady eye contact and expressions"
            ],
            "competition": [
                "Use comparative gestures when discussing competitors",
                "Stand tall to project competitive confidence",
                "Express determination through facial expressions"
            ],
            "traction": [
                "Use upward gestures when showing growth metrics",
                "Display excitement through positive facial expressions",
                "Maintain energetic posture when discussing achievements"
            ],
            "team": [
                "Use inclusive gestures when introducing team members",
                "Show pride through confident posture and expressions",
                "Maintain warm expressions when discussing team strengths"
            ],
            "funding_needs": [
                "Use clear, direct gestures when stating funding amount",
                "Maintain professional posture during financial asks",
                "Show conviction through steady expressions and eye contact"
            ],
            "future_plans": [
                "Use forward-pointing gestures when discussing vision",
                "Display visionary excitement through expressions",
                "Maintain inspiring posture when outlining future goals"
            ]
        }
        
        # Add stage-specific advice
        if stage in stage_specific_advice:
            recommendations.extend(stage_specific_advice[stage][:2])  # Top 2 recommendations
        
        # Add specific recommendations based on detected issues
        if not gestures:
            recommendations.append(f"Incorporate more hand gestures during {stage.replace('_', ' ')} discussion")
        
        if any('disengaged' in p.lower() for p in posture):
            recommendations.append(f"Improve posture engagement during {stage.replace('_', ' ')} stage")
        
        if any('nervous' in e.lower() for e in expressions):
            recommendations.append(f"Practice confident facial expressions for {stage.replace('_', ' ')} delivery")
        
        # Limit to top 3 recommendations per stage
        return recommendations[:3]

# Global workflow instance
pitch_workflow = None

def initialize_pitch_workflow(db_service: DatabaseService = None):
    """Initialize the pitch workflow"""
    global pitch_workflow
    pitch_workflow = PitchWorkflowAgent(db_service)
    return pitch_workflow

def get_pitch_workflow():
    """Get the global pitch workflow instance"""
    if pitch_workflow is None:
        initialize_pitch_workflow()
    return pitch_workflow

# Convenience functions for external use
def start_pitch_session(conversation_id: str, persona: str = "friendly") -> Dict[str, Any]:
    """Start a new pitch practice session"""
    workflow = get_pitch_workflow()
    result = workflow.start_session(conversation_id, persona)
    
    # Note: Database saving will be handled by the calling code since this function is not async
    
    return result

# Async wrapper functions for proper database integration
async def start_pitch_session_async(conversation_id: str, persona: str = "friendly") -> Dict[str, Any]:
    """Start a new pitch practice session with database integration"""
    workflow = get_pitch_workflow()
    result = workflow.start_session(conversation_id, persona)
    
    # Save session to database
    if workflow.db_service:
        try:
            logger.info(f"Attempting to save session {conversation_id} to database")
            await workflow._save_session_to_database(conversation_id, persona)
            logger.info(f"Session {conversation_id} saved to database successfully")
        except Exception as e:
            logger.error(f"Failed to save session to database: {e}", exc_info=True)
    else:
        logger.warning(f"No database service available for session {conversation_id}")
    
    return result

async def process_pitch_message_async(conversation_id: str, message: str) -> Dict[str, Any]:
    """Process a message in the pitch practice session with database integration"""
    workflow = get_pitch_workflow()
    result = workflow.process_message(conversation_id, message)
    
    # Log conversation to database
    if workflow.db_service:
        try:
            logger.debug(f"Logging conversation for session {conversation_id}")
            # Log user message
            await workflow._log_to_database(conversation_id, "user", message)
            # Log AI response
            if "message" in result:
                await workflow._log_to_database(conversation_id, "ai", result["message"])
            logger.info(f"Conversation logged successfully for session {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to log conversation to database: {e}", exc_info=True)
    else:
        logger.warning(f"No database service available for logging session {conversation_id}")
    
    return result

async def end_pitch_session_async(conversation_id: str, reason: str = "user_ended") -> Dict[str, Any]:
    """End pitch session and generate analysis report with database integration"""
    workflow = get_pitch_workflow()
    result = workflow.end_session_with_analysis(conversation_id, reason)
    
    # Save analysis and update session in database
    if workflow.db_service and "error" not in result:
        try:
            # Save analysis - the result IS the analysis
            if result and isinstance(result, dict) and "error" not in result:
                logger.info(f"Attempting to save analysis for session {conversation_id}")
                logger.debug(f"Analysis data keys: {list(result.keys())}")
                await workflow.db_service.save_analysis(result)
                logger.info(f"Analysis saved successfully for session {conversation_id}")
            else:
                logger.warning(f"Analysis not saved - result is empty or has error: {result}")
            
            # Update session status
            duration = result.get("duration_minutes", 0)
            logger.info(f"Updating session status with duration: {duration} minutes")
            await workflow.db_service.end_session(conversation_id, duration)
            logger.info(f"Session {conversation_id} marked as completed")
            
        except Exception as e:
            logger.error(f"Failed to save analysis to database: {e}", exc_info=True)
    
    return result

def start_pitch_session_with_message(conversation_id: str, persona: str, first_message: str) -> Dict[str, Any]:
    """Start a new pitch practice session with the user's first message"""
    workflow = get_pitch_workflow()
    return workflow.start_session_with_message(conversation_id, persona, first_message)

def process_pitch_message(conversation_id: str, message: str) -> Dict[str, Any]:
    """Process a message in the pitch practice session"""
    workflow = get_pitch_workflow()
    result = workflow.process_message(conversation_id, message)
    
    # Log conversation to database asynchronously if possible
    if workflow.db_service:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Log user message
                asyncio.create_task(workflow._log_to_database(conversation_id, "user", message))
                # Log AI response
                if "message" in result:
                    asyncio.create_task(workflow._log_to_database(conversation_id, "ai", result["message"]))
            else:
                # If not in async context, run it synchronously
                loop.run_until_complete(workflow._log_to_database(conversation_id, "user", message))
                if "message" in result:
                    loop.run_until_complete(workflow._log_to_database(conversation_id, "ai", result["message"]))
        except Exception as e:
            logger.error(f"Failed to log conversation to database: {e}")
    
    return result

def get_pitch_analytics(conversation_id: str) -> Dict[str, Any]:
    """Get analytics for a pitch practice session"""
    workflow = get_pitch_workflow()
    return workflow.get_session_analytics(conversation_id)

def generate_pitch_analysis_report(conversation_id: str) -> Dict[str, Any]:
    """Generate comprehensive pitch analysis report"""
    workflow = get_pitch_workflow()
    return workflow.generate_pitch_analysis(conversation_id)

def end_pitch_session_with_analysis(conversation_id: str, reason: str = "user_ended") -> Dict[str, Any]:
    """End pitch session and generate analysis report"""
    workflow = get_pitch_workflow()
    return workflow.end_session_with_analysis(conversation_id, reason)