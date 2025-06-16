# intelligent_ai_agent.py
import os
import logging
import google.generativeai as genai
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum

# LangChain imports
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize LangChain LLM with API key from environment
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    convert_system_message_to_human=True
)

# Investor personas - let AI handle the complexity
INVESTOR_PERSONAS = {
    "skeptical": {
        "name": "Sarah Martinez",
        "title": "Senior Partner at Venture Capital",
        "personality": "Analytical, direct, numbers-focused. Always challenges assumptions and asks for proof. Wants to see specific metrics, revenue data, and evidence of traction. Uses a professional, no-nonsense tone.",
        "approach": "Asks tough questions about market size, competition, unit economics, and scalability. Focuses on risks and wants concrete evidence."
    },
    
    "technical": {
        "name": "Dr. Alex Chen",
        "title": "CTO-turned-Investor at TechVentures",
        "personality": "Curious about technology, detail-oriented, innovation-focused. Excited about technical solutions and wants to understand the architecture, scalability, and technical moats. Enthusiastic but thorough.",
        "approach": "Deep dives into technical details, asks about IP, development roadmap, tech stack, and how the solution scales. Interested in the technical innovation."
    },
    
    "friendly": {
        "name": "Michael Thompson",
        "title": "Angel Investor & Former Entrepreneur",
        "personality": "Supportive, empathetic, story-focused. Having been an entrepreneur, understands the journey and challenges. Encouraging tone, interested in the founder's passion and vision.",
        "approach": "Focuses on the founder's journey, team dynamics, vision, and the human side of building a business. Supportive but still thorough."
    }
}

class PitchStage(Enum):
    GREETING = "greeting"
    PROBLEM_SOLUTION = "problem_solution"
    TARGET_MARKET = "target_market"
    BUSINESS_MODEL = "business_model"
    COMPETITION = "competition"
    TRACTION = "traction"
    TEAM = "team"
    FUNDING_NEEDS = "funding_needs"
    FUTURE_PLANS = "future_plans"

PITCH_STAGES = [stage.value for stage in PitchStage]

@dataclass
class StageContext:
    """Context for each pitch stage"""
    stage: PitchStage
    is_complete: bool = False
    data: dict = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}

class StageAgent:
    """Base class for stage-specific agents"""
    
    def __init__(self, stage: PitchStage, llm):
        self.stage = stage
        self.llm = llm
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="founder_input"  # Updated to match our input variable name
        )
        self.prompt = self._create_prompt()
        self.chain = self._create_chain()
    
    def _create_prompt(self) -> PromptTemplate:
        """Create prompt template for this stage"""
        template = """You are an investor analyzing a startup pitch. Focus on the {stage} aspect.
        
        Context:
        - Current Stage: {stage}
        - Founder: {founder_name}
        - Company: {company_name}
        
        Previous conversation:
        {chat_history}
        
        Founder: {founder_input}
        
        Your response should:
        1. Acknowledge the founder's point
        2. Ask a relevant follow-up question about {stage}
        3. Be concise and professional
        
        Response:"""
        return PromptTemplate(
            input_variables=["stage", "founder_name", "company_name", "founder_input", "chat_history"],
            template=template
        )
    
    def _create_chain(self) -> LLMChain:
        """Create LangChain for this stage"""
        return LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )
    
    def respond(self, founder_input: str, context: dict) -> str:
        """Generate response for this stage"""
        try:
            # Prepare context
            chat_history = context.get("chat_history", "")
            founder_name = context.get("founder_name", "Founder")
            company_name = context.get("company_name", "Your Company")
            
            # Prepare input for the chain
            input_dict = {
                "stage": self.stage.value,
                "founder_name": founder_name,
                "company_name": company_name,
                "founder_input": founder_input,  # Changed from 'input' to 'founder_input'
                "chat_history": chat_history
            }
            
            # Run the chain
            response = self.chain.run(input_dict)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in {self.stage.value} agent response: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while processing your request. Could you please rephrase or try again later?"

class BaseStageAgent(StageAgent):
    """Base class for all stage agents with common functionality"""
    
    def __init__(self, stage: PitchStage, llm):
        super().__init__(stage, llm)
        self.next_stage = self._get_next_stage()
    
    def _get_next_stage(self) -> Optional[PitchStage]:
        """Determine the next stage in the sequence"""
        try:
            current_idx = PITCH_STAGES.index(self.stage.value)
            if current_idx + 1 < len(PITCH_STAGES):
                return PitchStage(PITCH_STAGES[current_idx + 1])
        except (ValueError, IndexError):
            pass
        return None
    
    def should_transition(self, founder_input: str, context: dict) -> bool:
        """Determine if we should transition to the next stage"""
        # Default implementation - can be overridden by specific stages
        prompt = f"""Based on the conversation, should we move from {self.stage.value} to {self.next_stage.value if self.next_stage else 'conclusion'}?
        
        Conversation so far:
        {context.get('chat_history', '')}
        
        Founder's last input: {founder_input}
        
        Respond with only 'yes' or 'no'."""
        
        try:
            response = self.llm.invoke(prompt).content.lower().strip()
            return "yes" in response
        except Exception as e:
            logger.error(f"Error in stage transition check: {e}")
            return False


class GreetingAgent(BaseStageAgent):
    """Specialized agent for greeting stage"""
    
    def _create_prompt(self) -> PromptTemplate:
        template = """You are an investor meeting a founder for the first time. 
        Your goal is to make them feel welcome and gather basic information.
        
        Important Instructions:
        1. Pay close attention to the founder's name and company when mentioned
        2. Remember and use their name naturally in conversation
        3. If they introduce themselves, acknowledge it naturally
        4. If you're unsure of their name, ask politely
        5. Once you have their name and basic info, naturally transition to discussing their business
        
        Previous conversation:
        {chat_history}
        
        Founder: {founder_input}
        
        Your response should be natural and conversational. If you know their name, use it naturally.
        
        Response:"""
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )
        
    def should_transition(self, founder_input: str, context: dict) -> bool:
        """Transition when we have basic info and the greeting feels complete"""
        prompt = f"""Has the founder shared their name and basic information, and does the conversation feel ready to move to discussing their business?
        
        Conversation so far:
        {context.get('chat_history', '')}
        
        Founder's last input: {founder_input}
        
        Respond with only 'yes' or 'no'."""
        
        try:
            response = self.llm.invoke(prompt).content.lower().strip()
            return "yes" in response
        except Exception as e:
            logger.error(f"Error in greeting transition check: {e}")
            return False


class ProblemSolutionAgent(BaseStageAgent):
    """Agent focused on understanding the problem and solution"""
    
    def _create_prompt(self) -> PromptTemplate:
        template = """You are an investor evaluating a startup's problem and solution.
        
        Your goals:
        1. Understand the core problem they're solving
        2. Evaluate how well their solution addresses the problem
        3. Ask probing questions about market need and solution effectiveness
        
        Previous conversation:
        {chat_history}
        
        Founder: {founder_input}
        
        Ask thoughtful questions to deeply understand the problem and solution.
        
        Response:"""
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )


class TargetMarketAgent(BaseStageAgent):
    """Agent focused on understanding the target market"""
    
    def _create_prompt(self) -> PromptTemplate:
        template = """You are an investor evaluating a startup's target market.
        
        Your goals:
        1. Identify and understand the target customer segments
        2. Assess market size and growth potential
        3. Evaluate market trends and timing
        
        Previous conversation:
        {chat_history}
        
        Founder: {founder_input}
        
        Ask insightful questions to understand the market opportunity.
        
        Response:"""
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )


class BusinessModelAgent(BaseStageAgent):
    """Agent focused on understanding the business model"""
    
    def _create_prompt(self) -> PromptTemplate:
        template = """You are an investor evaluating a startup's business model.
        
        Your goals:
        1. Understand how they make money
        2. Evaluate pricing strategy
        3. Assess unit economics and scalability
        
        Previous conversation:
        {chat_history}
        
        Founder: {founder_input}
        
        Ask detailed questions about their business model.
        
        Response:"""
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )


class CompetitionAgent(BaseStageAgent):
    """Agent focused on understanding the competitive landscape"""
    
    def _create_prompt(self) -> PromptTemplate:
        template = """You are an investor evaluating a startup's competitive position.
        
        Your goals:
        1. Identify key competitors
        2. Understand competitive advantages
        3. Evaluate barriers to entry
        
        Previous conversation:
        {chat_history}
        
        Founder: {founder_input}
        
        Ask probing questions about their competition and differentiation.
        
        Response:"""
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )


class TractionAgent(BaseStageAgent):
    """Agent focused on understanding traction and milestones"""
    
    def _create_prompt(self) -> PromptTemplate:
        template = """You are an investor evaluating a startup's traction.
        
        Your goals:
        1. Understand current traction and growth metrics
        2. Evaluate key milestones achieved
        3. Assess growth strategy
        
        Previous conversation:
        {chat_history}
        
        Founder: {founder_input}
        
        Ask specific questions about their traction and growth.
        
        Response:"""
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )


class TeamAgent(BaseStageAgent):
    """Agent focused on evaluating the team"""
    
    def _create_prompt(self) -> PromptTemplate:
        template = """You are an investor evaluating a startup's team.
        
        Your goals:
        1. Understand team composition and experience
        2. Evaluate relevant domain expertise
        3. Assess ability to execute
        
        Previous conversation:
        {chat_history}
        
        Founder: {founder_input}
        
        Ask insightful questions about the team and their background.
        
        Response:"""
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )


class FundingNeedsAgent(BaseStageAgent):
    """Agent focused on understanding funding needs and use of funds"""
    
    def _create_prompt(self) -> PromptTemplate:
        template = """You are an investor discussing funding with a founder.
        
        Your goals:
        1. Understand their funding requirements
        2. Evaluate proposed use of funds
        3. Assess financial projections
        
        Previous conversation:
        {chat_history}
        
        Founder: {founder_input}
        
        Ask detailed questions about their funding needs and financial plans.
        
        Response:"""
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )


class FuturePlansAgent(BaseStageAgent):
    """Agent focused on future plans and vision"""
    
    def _create_prompt(self) -> PromptTemplate:
        template = """You are an investor discussing future plans with a founder.
        
        Your goals:
        1. Understand their long-term vision
        2. Evaluate product roadmap
        3. Discuss potential exit strategies
        
        Previous conversation:
        {chat_history}
        
        Founder: {founder_input}
        
        Ask thoughtful questions about their future plans and vision.
        
        Response:"""
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )

# Agent factory to create the appropriate agent based on stage
class AgentFactory:
    """Factory to create appropriate agent for each stage"""
    
    @staticmethod
    def create_agent(stage: PitchStage, llm) -> BaseStageAgent:
        """Create an agent for the given stage"""
        agent_classes = {
            PitchStage.GREETING: GreetingAgent,
            PitchStage.PROBLEM_SOLUTION: ProblemSolutionAgent,
            PitchStage.TARGET_MARKET: TargetMarketAgent,
            PitchStage.BUSINESS_MODEL: BusinessModelAgent,
            PitchStage.COMPETITION: CompetitionAgent,
            PitchStage.TRACTION: TractionAgent,
            PitchStage.TEAM: TeamAgent,
            PitchStage.FUNDING_NEEDS: FundingNeedsAgent,
            PitchStage.FUTURE_PLANS: FuturePlansAgent
        }
        
        agent_class = agent_classes.get(stage)
        if not agent_class:
            raise ValueError(f"No agent class found for stage: {stage}")
            
        return agent_class(stage, llm)

class ConversationContext:
    """Manages conversation state and coordinates between stage agents"""
    
    def __init__(self, conversation_id: str, llm):
        from datetime import datetime
        self.conversation_id = conversation_id
        self.llm = llm
        self.current_stage = PitchStage.GREETING
        self.agent_factory = AgentFactory()
        self.current_agent = self.agent_factory.create_agent(self.current_stage, llm)
        self.chat_history = []
        self.stage_contexts = {stage: StageContext(stage) for stage in PitchStage}
        self.shared_context = {
            "founder_name": "",
            "company_name": "",
            "previous_stage": None,
            "next_stage_prompt": ""
        }
        # Track conversation timestamps
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
    def process_message(self, message: str) -> str:
        """Process a message through the current stage agent"""
        from datetime import datetime
        
        try:
            # Update last activity timestamp
        
            # Add founder's message to chat history with timestamp
            self.chat_history.append({
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'sender': 'founder'
            })
            
            # Update shared context with latest message
            self._update_shared_context(message)
            
            # Prepare context for the agent - safely handle chat history format
            chat_history_text = []
            for m in self.chat_history[-10:]:  # Last 10 messages
                if isinstance(m, dict):
                    chat_history_text.append(m['message'])
                else:
                    # Handle legacy string format if any exist
                    chat_history_text.append(str(m))
            
            context = {
                "chat_history": "\n".join(chat_history_text),
                "founder_name": self.shared_context["founder_name"],
                "company_name": self.shared_context["company_name"],
                "current_stage": self.current_stage.value,
                **self.shared_context
            }
            
            # Get response from current agent
            response = self.current_agent.respond(message, context)
            
            # Add response to chat history with timestamp
            self.chat_history.append({
                'timestamp': datetime.now().isoformat(),
                'message': response,
                'sender': 'investor'
            })
            
            # Update last activity timestamp
            self.last_activity = datetime.now()
            
            # Check if we should transition to the next stage
            if self._should_transition_stage(message, context):
                self._transition_to_next_stage()
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error processing your message. Could you please rephrase or try again later?"
    
    def _update_shared_context(self, message: str) -> None:
        """Update shared context based on conversation"""
        # This is a simplified example - you could use more sophisticated NLP here
        if not self.shared_context["founder_name"]:
            # Try to extract name from message if we don't have it yet
            name_prompt = f"""Extract the person's name from this message if present. 
            Return only the name or empty string if not found: "{message}"""
            try:
                name = self.llm.invoke(name_prompt).content.strip()
                if name and len(name) > 1 and len(name) < 50:  # Basic validation
                    self.shared_context["founder_name"] = name
            except Exception as e:
                logger.warning(f"Error extracting name: {e}")
    
    def _should_transition_stage(self, message: str, context: dict) -> bool:
        """Determine if we should transition to the next stage"""
        if not isinstance(self.current_agent, BaseStageAgent):
            return False
            
        # Check if current agent thinks we should transition
        if not self.current_agent.should_transition(message, context):
            return False
            
        # Additional checks can be added here (e.g., minimum time in stage)
        return True
    
    def _transition_to_next_stage(self) -> None:
        """Transition to the next stage in the sequence"""
        if not isinstance(self.current_agent, BaseStageAgent):
            return
            
        next_stage = self.current_agent.next_stage
        if not next_stage:
            logger.info("No more stages to transition to")
            return
            
        logger.info(f"Transitioning from {self.current_stage.value} to {next_stage.value}")
        self.shared_context["previous_stage"] = self.current_stage.value
        self.current_stage = next_stage
        self.current_agent = self.agent_factory.create_agent(next_stage, self.llm)
        
        # Add a transition message to the chat
        transition_msg = f"[Transitioning to {next_stage.value.replace('_', ' ').title()} stage]"
        self.chat_history.append({
            'timestamp': datetime.now().isoformat(),
            'message': transition_msg,
            'sender': 'system'
        })

    def get_conversation_statistics(self) -> dict:
        """Get statistics about the conversation"""
        from datetime import datetime
        
        # Calculate conversation duration in minutes
        duration_minutes = 0
        if hasattr(self, 'created_at') and hasattr(self, 'last_activity') and self.created_at and self.last_activity:
            duration_seconds = (self.last_activity - self.created_at).total_seconds()
            duration_minutes = round(duration_seconds / 60, 2)
        
        # Initialize metrics
        founder_msgs = 0
        investor_msgs = 0
        system_msgs = 0
        total_investor_length = 0
        
        # Calculate message metrics
        for msg in self.chat_history:
            if isinstance(msg, dict):
                if msg.get('sender') == 'founder':
                    founder_msgs += 1
                elif msg.get('sender') == 'investor':
                    investor_msgs += 1
                    total_investor_length += len(str(msg.get('message', '')))
                elif msg.get('sender') == 'system':
                    system_msgs += 1
            # Handle legacy string format for backward compatibility
            elif isinstance(msg, str):
                if msg.startswith("Founder:"):
                    founder_msgs += 1
                elif msg.startswith("Investor:"):
                    investor_msgs += 1
                    total_investor_length += len(msg)
                elif msg.startswith("System:"):
                    system_msgs += 1
        
        # Calculate average response length
        avg_response_length = total_investor_length / max(1, investor_msgs) if investor_msgs > 0 else 0
        
        return {
            "conversation_id": self.conversation_id,
            "current_stage": self.current_stage.value,
            "started_at": self.created_at.isoformat() if hasattr(self, 'created_at') and self.created_at else None,
            "last_activity": self.last_activity.isoformat() if hasattr(self, 'last_activity') and self.last_activity else None,
            "duration_minutes": duration_minutes,
            "completed_stages": [
                stage.value for stage, ctx in self.stage_contexts.items() 
                if ctx.is_complete
            ],
            "conversation_metrics": {
                "total_messages": len(self.chat_history),
                "founder_messages": founder_msgs,
                "investor_messages": investor_msgs,
                "system_messages": system_msgs,
                "average_response_length": round(avg_response_length, 2)
            },
            "context": {
                "founder_name": self.shared_context.get("founder_name", ""),
                "company_name": self.shared_context.get("company_name", "")
            }
        }

class IntelligentAIAgent:
    """Enhanced AI-powered investor agent with multi-stage support"""
    
    def __init__(self, llm=None):
        """Initialize the AI agent with an optional LLM instance
        
        Args:
            llm: Optional LLM instance. If not provided, will use Gemini Pro
        """
        self.llm = llm or genai.GenerativeModel('gemini-1.5-flash')
        self.conversations = {}
        self.agents = {}
    
    def _initialize_agents(self) -> Dict[PitchStage, StageAgent]:
        """Initialize specialized agents for each pitch stage"""
        return {
            PitchStage.GREETING: GreetingAgent(PitchStage.GREETING, self.llm),
            # Initialize other stage agents here
            PitchStage.PROBLEM_SOLUTION: StageAgent(PitchStage.PROBLEM_SOLUTION, self.llm),
            PitchStage.TARGET_MARKET: StageAgent(PitchStage.TARGET_MARKET, self.llm),
            PitchStage.BUSINESS_MODEL: StageAgent(PitchStage.BUSINESS_MODEL, self.llm),
            PitchStage.COMPETITION: StageAgent(PitchStage.COMPETITION, self.llm),
            PitchStage.TRACTION: StageAgent(PitchStage.TRACTION, self.llm),
            PitchStage.TEAM: StageAgent(PitchStage.TEAM, self.llm),
            PitchStage.FUNDING_NEEDS: StageAgent(PitchStage.FUNDING_NEEDS, self.llm),
            PitchStage.FUTURE_PLANS: StageAgent(PitchStage.FUTURE_PLANS, self.llm),
        }
    
    def start_conversation(self, conversation_id: str, persona: str = "friendly") -> ConversationContext:
        """Initialize a new conversation with multi-stage support"""
        context = ConversationContext(conversation_id, llm)
        self.conversations[conversation_id] = context
        
        # Get initial greeting
        initial_greeting = context.current_agent.respond("", {
            "chat_history": "",
            "founder_name": "",
            "company_name": "",
            "current_stage": context.current_stage.value
        })
        
        # Add greeting to chat history
        context.chat_history.append(f"Investor: {initial_greeting}")
        
        logger.info(f"Started new conversation {conversation_id} with {persona} persona")
        return context
    
    def _determine_next_stage(self, current_stage: PitchStage) -> PitchStage:
        """Determine the next stage in the pitch flow"""
        try:
            current_index = PITCH_STAGES.index(current_stage.value)
            if current_index + 1 < len(PITCH_STAGES):
                return PitchStage(PITCH_STAGES[current_index + 1])
        except (ValueError, IndexError):
            pass
        return current_stage  # Stay on current stage if can't determine next
    
    def generate_response(self, conversation_id: str, founder_input: str) -> str:
        """Generate investor response using the appropriate stage agent"""
        if conversation_id not in self.conversations:
            raise ValueError(f"No conversation found with ID {conversation_id}")
            
        context = self.conversations[conversation_id]
        return context.process_message(founder_input)
    
    def _generate_initial_greeting(self, persona: str) -> str:
        """Let AI generate the initial greeting"""
        persona_info = INVESTOR_PERSONAS[persona]
        
        prompt = f"""You are {persona_info['name']}, {persona_info['title']}.

Your personality: {persona_info['personality']}
Your approach: {persona_info['approach']}

Generate a natural, welcoming greeting for a founder who just joined your pitch meeting. 
- Introduce yourself
- Make them comfortable
- Ask them to introduce themselves and their company
- Keep it conversational, 2-3 sentences max
- Match your personality perfectly

Generate the greeting now:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating greeting: {str(e)}")
            return f"Hi! I'm {persona_info['name']}, {persona_info['title']}. I'm excited to hear about your business. Could you start by telling me your name and what your company does?"
    
    def get_conversation_stats(self, conversation_id: str) -> dict:
        """Get conversation statistics
        
        Args:
            conversation_id: The ID of the conversation to get stats for
            
        Returns:
            dict: Dictionary containing conversation statistics or error details
        """
        if conversation_id not in self.conversations:
            return {
                "status": "error",
                "message": f"No conversation found with ID {conversation_id}"
            }
            
        context = self.conversations[conversation_id]
        
        # Get statistics from the conversation context
        stats = context.get_conversation_statistics()
        
        # Add additional metadata
        stats.update({
            "status": "active",
            "persona": getattr(context, 'persona', 'friendly'),
            "investor_name": INVESTOR_PERSONAS.get(getattr(context, 'persona', 'friendly'), {}).get("name", "AI Investor"),
            "founder_name": context.shared_context.get("founder_name", ""),
            "company_name": getattr(context, 'company_name', ''),
            "duration_minutes": (context.last_activity - context.created_at).total_seconds() / 60
        })
        return stats

# Global agent instance (will be initialized in startup_event)
agent = None

# Convenience functions for external use
def start_new_conversation(conversation_id: str, persona: str = "friendly") -> ConversationContext:
    """Start a new conversation with the intelligent agent"""
    if agent is None:
        raise RuntimeError("AI Agent not initialized. Call startup_event() first.")
    return agent.start_conversation(conversation_id, persona)

def generate_investor_response(conversation_id: str, founder_input: str) -> str:
    """Generate investor response using the intelligent agent"""
    return agent.generate_response(conversation_id, founder_input)

def get_conversation_statistics(conversation_id: str) -> dict:
    """Get conversation statistics
    
    Args:
        conversation_id: The ID of the conversation to get stats for
        
    Returns:
        dict: Dictionary containing conversation statistics or error details
    """
    try:
        if not conversation_id:
            logger.warning("Empty conversation_id provided")
            return {"error": "conversation_id is required", "status": "error"}
        
        # Initialize conversations dictionary if it doesn't exist
        if not hasattr(agent, 'conversations'):
            agent.conversations = {}
            
        # If no conversations exist yet, return a helpful message
        if not agent.conversations:
            logger.info("No active conversations found")
            return {
                "status": "no_conversations",
                "message": "No active conversations found. Start a new conversation first.",
                "suggestion": "Call /api/conversation/start to begin a new conversation"
            }
            
        if conversation_id not in agent.conversations:
            active_conversations = list(agent.conversations.keys())
            logger.warning(f"Conversation not found: {conversation_id}. Active conversations: {active_conversations}")
            return {
                "error": f"Conversation not found: {conversation_id}",
                "status": "not_found",
                "active_conversations": active_conversations
            }
        
        context = agent.conversations[conversation_id]
        
        # Ensure context has required attributes
        required_attrs = ['current_stage', 'stages', 'conversation_history']
        missing_attrs = [attr for attr in required_attrs if not hasattr(context, attr)]
        
        if missing_attrs:
            logger.error(f"Invalid conversation context for ID: {conversation_id}. Missing attributes: {missing_attrs}")
            return {
                "error": "Invalid conversation context",
                "status": "error",
                "missing_attributes": missing_attrs
            }
        
        # Get next stage safely
        next_stage = "unknown"
        if hasattr(agent, '_determine_next_stage') and hasattr(context, 'current_stage'):
            try:
                next_stage_result = agent._determine_next_stage(context.current_stage)
                next_stage = next_stage_result.value if hasattr(next_stage_result, 'value') else str(next_stage_result)
            except Exception as e:
                logger.warning(f"Error determining next stage: {str(e)}")
        
        return {
            "status": "active",
            "conversation_id": conversation_id,
            "current_stage": context.current_stage.value if hasattr(context.current_stage, 'value') else str(context.current_stage),
            "stages_completed": [
                stage.value if hasattr(stage, 'value') else str(stage) 
                for stage, ctx in context.stages.items() 
                if hasattr(ctx, 'is_complete') and ctx.is_complete
            ],
            "next_stage": next_stage,
            "messages_exchanged": len(context.conversation_history) if hasattr(context, 'conversation_history') else 0,
            "last_activity": context.last_activity.isoformat() if hasattr(context, 'last_activity') else None
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation stats: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "status": "error",
            "type": type(e).__name__
        }