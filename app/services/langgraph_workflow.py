# langgraph_workflow.py
"""
LangGraph workflow for intelligent pitch practice sessions
"""

import os
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
import json

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
    
    def __init__(self):
        """
        Initialize a new PitchWorkflowAgent instance
        
        Creates the LangGraph workflow and initializes the memory as an empty dictionary.
        Also creates a MemorySaver instance to save the state of the workflow.
        """
        # Initialize memory_saver first as it's used in _create_workflow
        self.memory_saver = MemorySaver()
        self.workflow = self._create_workflow()
        self.memory = {}  # Initialize memory as an empty dictionary
        
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
                "complete": "finalize_session"
            }
        )
        
        workflow.add_edge("transition_stage", "assess_stage")
        workflow.add_edge("finalize_session", END)
        
        return workflow.compile(checkpointer=self.memory_saver)
    
    def _initialize_session(self, state: PitchWorkflowState) -> PitchWorkflowState:
        """Initialize a new pitch practice session"""
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
        
        prompt = f"""You are {persona_info['name']}, {persona_info['title']}.

PERSONALITY: {persona_info['personality']}
QUESTIONING STYLE: {persona_info['questioning_style']}

CURRENT STAGE: {stage.replace('_', ' ').title()}
STAGE OBJECTIVE: {stage_objectives.get(stage, 'Gather relevant information')}

CONVERSATION CONTEXT:
{conversation_context}

INSTRUCTIONS:
1. Ask ONLY ONE focused question
2. This is question #{questions_asked_count + 1} for this stage
3. Build on previous responses but don't repeat questions
4. Keep your personality and questioning style consistent
5. If the founder seems to have answered incompletely, ask for clarification
6. If you have enough information on this topic, prepare to transition

FOUNDER INFO:
- Name: {state.get('founder_name', 'Not provided')}
- Company: {state.get('company_name', 'Not provided')}

Generate ONE specific question about {stage.replace('_', ' ')} that matches your personality:"""
        
        return prompt
    
    def _evaluate_founder_response(self, state: PitchWorkflowState) -> PitchWorkflowState:
        """Evaluate the founder's response and extract key insights"""
        
        # Get the latest founder message
        latest_messages = state["messages"][-2:]  # Last 2 messages (question and response)
        if len(latest_messages) < 2:
            return state
        
        founder_response = latest_messages[-1].content if latest_messages[-1].__class__.__name__ == "HumanMessage" else ""
        
        if not founder_response:
            return state
        
        current_stage = state["current_stage"]
        
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
            complete = "yes" in eval_text.split("COMPLETE:")[1].split("\n")[0].lower()
            
            # Extract insights
            if "INSIGHTS:" in eval_text:
                insights_section = eval_text.split("INSIGHTS:")[1].split("FOLLOW_UP_NEEDED:")[0]
                insights = [line.strip("- ").strip() for line in insights_section.split("\n") if line.strip().startswith("-")]
                state["key_insights"][current_stage].extend(insights)
            
            # Determine if follow-up needed
            follow_up_needed = "yes" in eval_text.split("FOLLOW_UP_NEEDED:")[1].split("\n")[0].lower()
            
            # Update state
            state["question_answered"] = complete
            state["should_transition"] = complete and not follow_up_needed
            
            # Update founder info if in greeting stage
            if current_stage == "greeting":
                self._extract_founder_info(state, founder_response)
            
            logger.info(f"Response evaluation - Complete: {complete}, Follow-up needed: {follow_up_needed}")
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            # Default to considering response complete to avoid getting stuck
            state["question_answered"] = True
            state["should_transition"] = True
        
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
            return state
        
        # Check if we have enough information for this stage
        key_insights_count = len(state["key_insights"][current_stage])
        
        if key_insights_count >= 2 and state["question_answered"]:
            state["should_transition"] = True
        else:
            state["should_transition"] = False
        
        logger.info(f"Transition decision for {current_stage}: {state['should_transition']} (questions: {questions_asked}, insights: {key_insights_count})")
        
        return state
    
    def _should_transition_or_continue(self, state: PitchWorkflowState) -> str:
        """Determine the next step in the workflow"""
        
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
    
    def process_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """Process a founder's message and return the investor's response"""
        
        config = {"configurable": {"thread_id": conversation_id}}
        
        # Add the founder's message to the state
        human_message = HumanMessage(content=message)
        
        # Get current state
        try:
            current_state = self.workflow.get_state(config)
            if current_state.values:
                # Add the human message to existing state
                current_state.values["messages"].append(human_message)
                
                # Update the workflow state
                result = self.workflow.invoke(current_state.values, config)
                
                # Return the response
                return {
                    "message": result["messages"][-1].content,
                    "stage": result["current_stage"],
                    "complete": result.get("workflow_complete", False),
                    "session_id": conversation_id,
                    "insights": result.get("key_insights", {})
                }
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

# Global workflow instance
pitch_workflow = None

def initialize_pitch_workflow():
    """Initialize the pitch workflow"""
    global pitch_workflow
    pitch_workflow = PitchWorkflowAgent()
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
    return workflow.start_session(conversation_id, persona)

def process_pitch_message(conversation_id: str, message: str) -> Dict[str, Any]:
    """Process a message in the pitch practice session"""
    workflow = get_pitch_workflow()
    return workflow.process_message(conversation_id, message)

def get_pitch_analytics(conversation_id: str) -> Dict[str, Any]:
    """Get analytics for a pitch practice session"""
    workflow = get_pitch_workflow()
    return workflow.get_session_analytics(conversation_id)