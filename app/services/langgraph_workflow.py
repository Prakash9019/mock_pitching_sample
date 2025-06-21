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
    model="gemini-1.5-flash",
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
    
    # Analytics
    session_start: str
    stage_durations: Dict[str, float]
    questions_asked: List[str]
    key_insights: Dict[str, List[str]]

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
            
            # Add to messages
            state["messages"].append(AIMessage(content=question))
            
            logger.info(f"Generated question for {current_stage}: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            fallback_question = f"Could you tell me more about {current_stage.replace('_', ' ')}?"
            state["current_question"] = fallback_question
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

PERSONALITY: {persona_info['personality']}
QUESTIONING STYLE: {persona_info['questioning_style']}

SITUATION: This is the beginning of a pitch meeting. The founder just said: "{user_intro}"

INSTRUCTIONS:
1. Respond naturally and conversationally to what they just said
2. Acknowledge any information they've already provided (name, company, etc.)
3. If they mentioned their name, greet them by name
4. If they mentioned their company, acknowledge it and show interest
5. Ask ONLY ONE focused question - don't ask multiple questions at once
6. Keep it friendly but maintain your {persona_info['personality']} personality
7. Make it feel like a real conversation, not an interrogation
8. CRITICAL: Ask only ONE question, not 2-3-4 questions in the same response

STAGE FOCUS: This is the GREETING stage. Focus ONLY on: name, company name, and brief introduction.
Do NOT ask about revenue, metrics, traction, or detailed business questions yet - those come in later stages.

EXAMPLES OF GOOD RESPONSES (ONE QUESTION EACH):
- If they said "Hi, I'm Alex from Facebook": "Hello Alex! Great to meet you. So you're from Facebook - are you working on something new there, or is this a separate venture?"
- If they said "Hello, I'm Sarah and I run TechCorp": "Hi Sarah! Nice to meet you. TechCorp sounds interesting - can you give me a brief overview of what TechCorp does?"
- If they just said "Hello": "Hello! Great to meet you. Could you start by telling me your name and what company you're presenting?"
- If they gave name and company: Ask for a brief introduction/overview of what the company does

IMPORTANT: Your response should contain EXACTLY ONE question mark (?). Do not ask multiple questions.

Generate a natural, conversational response with ONLY ONE question:"""
        else:
            # Standard prompt for other stages or follow-up questions
            prompt = f"""You are {persona_info['name']}, {persona_info['title']}.

PERSONALITY: {persona_info['personality']}
QUESTIONING STYLE: {persona_info['questioning_style']}

CURRENT STAGE: {stage.replace('_', ' ').title()}
STAGE OBJECTIVE: {stage_objectives.get(stage, 'Gather relevant information')}

CONVERSATION CONTEXT:
{conversation_context}

INSTRUCTIONS:
1. Ask ONLY ONE focused question - never ask multiple questions at once
2. This is question #{questions_asked_count + 1} for this stage
3. Focus ONLY on the current stage objective - don't jump ahead to future stages
4. Build on previous responses but don't repeat questions
5. Keep your personality and questioning style consistent
6. If the founder seems to have answered incompletely, ask for clarification
7. If you have enough information on this topic, prepare to transition
8. CRITICAL: Your response should contain EXACTLY ONE question mark (?). Do not ask multiple questions.

STAGE FOCUS REMINDER:
- If this is GREETING: Focus on name, company, brief intro only
- If this is PROBLEM_SOLUTION: Focus on the problem they solve and their approach
- If this is TRACTION: Then you can ask about metrics, revenue, growth
- Don't mix stage objectives - stay focused on the current stage

FOUNDER INFO:
- Name: {state.get('founder_name', 'Not provided')}
- Company: {state.get('company_name', 'Not provided')}

Generate ONE specific question about {stage.replace('_', ' ')} that matches your personality (only one question mark allowed):"""
        
        return prompt
    
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
        
        # Evaluate response completeness
        evaluation_prompt = f"""
        Evaluate this founder's response to the investor question:
        
        Question: {state['current_question']}
        Response: {founder_response}
        
        1. Is the response complete and informative? (yes/no)
        2. What key insights can be extracted? (list 2-3 bullet points)
        3. Does this response warrant a follow-up question in the same topic? (yes/no)
        
        Format your response as:
        COMPLETE: yes/no
        INSIGHTS: 
        - insight 1
        - insight 2
        FOLLOW_UP_NEEDED: yes/no
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
        
        # Extract name
        name_prompt = f"Extract the founder's name from this introduction: '{response}'. Return only the name or 'Not provided'."
        try:
            name_response = llm.invoke(name_prompt)
            name = name_response.content.strip()
            if name and name != "Not provided" and len(name) < 50:
                state["founder_name"] = name
        except Exception as e:
            logger.warning(f"Error extracting name: {e}")
        
        # Extract company
        company_prompt = f"Extract the company name from this introduction: '{response}'. Return only the company name or 'Not provided'."
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
        
        # Generate final summary
        summary_prompt = f"""
        Generate a brief, encouraging summary of this pitch practice session:
        
        Founder: {state.get('founder_name', 'The founder')}
        Company: {state.get('company_name', 'Their company')}
        Stages completed: {list(state['key_insights'].keys())}
        
        Key insights gathered:
        {json.dumps(state['key_insights'], indent=2)}
        
        Provide a warm, supportive summary that:
        1. Acknowledges their participation
        2. Highlights 2-3 key strengths
        3. Offers encouragement for their pitch
        4. Keeps it concise (3-4 sentences)
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
            
            # Calculate communication metrics
            comm_metrics = self._calculate_communication_metrics(state['messages'])
            
            # Generate comprehensive analysis using AI
            analysis_prompt = f"""
            Analyze this pitch practice session and provide a comprehensive evaluation report.
            
            SESSION DATA:
            - Founder: {state.get('founder_name', 'Unknown')}
            - Company: {state.get('company_name', 'Unknown')}
            - Duration: {analytics['duration_minutes']} minutes
            - Stages Completed: {len(analytics['completed_stages'])}/9
            - Current Stage: {analytics['current_stage']}
            - Total Questions Asked: {analytics['total_questions']}
            - Session Complete: {analytics['workflow_complete']}
            
            COMMUNICATION METRICS:
            - Engagement: Talked {comm_metrics['engagement']['talked_count']} times ({comm_metrics['engagement']['talk_percentage']}%), Listened {comm_metrics['engagement']['listened_count']} times ({comm_metrics['engagement']['listen_percentage']}%)
            - Fluency: {comm_metrics['fluency']['filler_count']} fillers, {comm_metrics['fluency']['grammar_issues']} grammar issues, {comm_metrics['fluency']['vocabulary_richness']} vocabulary richness
            - Interactivity: {comm_metrics['interactivity']['conversation_turns']} turns, {comm_metrics['interactivity']['turn_frequency']} turn frequency
            - Questions: {comm_metrics['questions_asked']['founder_questions']} questions asked, {comm_metrics['questions_asked']['questions_per_minute']} per minute
            
            STAGE INSIGHTS:
            {json.dumps(analytics['key_insights'], indent=2)}
            
            CONVERSATION MESSAGES:
            {self._format_conversation_for_analysis(state['messages'])}
            
            Please provide a detailed analysis in the following JSON format:
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
                    "questions_asked": {{
                        "score": <score 0-10>,
                        "rating": "<Need to Improve/Below Average/Satisfactory/Good/Vertx Assured>",
                        "description": "<Detailed feedback on question asking frequency and quality>",
                        "total_questions": {comm_metrics['questions_asked']['founder_questions']},
                        "questions_per_minute": {comm_metrics['questions_asked']['questions_per_minute']}
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
            
            Base your analysis on these 14 key categories:
            
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
        
        stages_completed = len(analytics['completed_stages'])
        total_insights = sum(len(insights) for insights in analytics['key_insights'].values())
        
        # Calculate communication metrics
        comm_metrics = self._calculate_communication_metrics(state['messages'])
        
        # Calculate basic scores
        completion_score = (stages_completed / 9) * 40  # 40% for completion
        insight_score = min(total_insights * 2, 30)     # 30% for insights quality
        duration_score = min(analytics['duration_minutes'] / 30 * 20, 20)  # 20% for engagement
        response_score = min(analytics['total_questions'] * 2, 10)  # 10% for responsiveness
        
        overall_score = int(completion_score + insight_score + duration_score + response_score)
        
        def get_rating(score):
            if score >= 90: return "Vertx Assured"
            elif score >= 75: return "Good"
            elif score >= 60: return "Satisfactory"
            elif score >= 40: return "Below Average"
            else: return "Need to Improve"
        
        # Generate basic category scores
        base_category_score = max(30, overall_score - 20)  # Base score for each category
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
            ],
            "weaknesses": [
                {"area": "Completion", "description": f"Only completed {stages_completed}/9 stages", "improvement": "Complete all pitch stages for comprehensive feedback"}
            ] if stages_completed < 9 else [],
            "key_recommendations": [
                "Complete all 10 pitch categories for comprehensive evaluation",
                "Provide more specific metrics and data points",
                "Practice articulating value proposition clearly",
                "Strengthen competitive differentiation",
                "Develop more impactful closing statements"
            ],
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
            ],
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