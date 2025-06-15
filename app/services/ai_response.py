#ai_response.py
import os
import logging
import google.generativeai as genai
from typing import Optional, Dict, List, TypedDict, Literal, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import re


# LangChain imports for enhanced memory
logger = logging.getLogger(__name__)
try:
    from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain successfully imported")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    logger.warning(f"LangChain not available: {e}. Using basic memory system.")

# Configure logging

# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define conversation topics to explore
CONVERSATION_TOPICS = [
    "company_overview",
    "problem_solution",
    "target_market",
    "business_model",
    "competition",
    "traction",
    "team",
    "funding_needs",
    "future_plans"
]

# Define investor personas with updated prompts for curiosity
INVESTOR_PERSONAS = {
    "skeptical": {
        "greeting": "Hi there! I'm always excited to learn about new ventures. Could you tell me about yourself and what you're working on?",
        "persona": """You are a skeptical but fair venture capital investor. You're genuinely curious about new businesses but need to be convinced.
        Focus on understanding the business fundamentals, market opportunity, and path to profitability.
        Ask direct, probing questions that get to the heart of the business model and value proposition.
        Be polite but don't shy away from challenging assumptions or asking for clarification.""",
        "style": "Ask one clear, specific question at a time based on what the founder has shared so far."
    },
    
    "technical": {
        "greeting": "Hello! I'm fascinated by technology and innovation. What's the exciting project you're working on?",
        "persona": """You are a technically-minded investor who loves diving into the details.
        You're particularly interested in the technology stack, architecture, and technical challenges.
        Ask insightful questions about the product's technical implementation, scalability, and innovation.
        Help the founder explain technical concepts clearly while digging into the technical merits.""",
        "style": "Ask one technical question at a time, following up on the founder's previous answers."
    },
    
    "friendly": {
        "greeting": "Hi! It's great to meet you. I'd love to hear about your journey and what you're building.",
        "persona": """You are a supportive and encouraging investor who wants to see founders succeed.
        You focus on understanding the founder's vision, passion, and the problem they're solving.
        Ask open-ended questions that help the founder share their story and vision.
        Be warm, empathetic, and genuinely interested in their journey.""",
        "style": "Ask one thoughtful question at a time, showing genuine interest in the founder's responses."
    }
}

# Common follow-up prompts
FOLLOW_UP_PROMPTS = {
    "default": [
        "That's interesting! Could you tell me more about {topic}?",
        "I'm curious, how does your company handle {topic}?",
        "What inspired you to focus on {topic} specifically?",
        "How is your approach to {topic} different from others in the space?"
    ],
    "technical": [
        "Could you dive deeper into the technical aspects of {topic}?",
        "What were the main technical challenges you faced with {topic}?",
        "How does your technology handle {topic} at scale?"
    ],
    "business": [
        "What's your business model for monetizing {topic}?",
        "How do you see {topic} contributing to your revenue?",
        "What's your strategy for customer acquisition in {topic}?"
    ]
}

def extract_keywords(text: str) -> List[str]:
    """Extract potential topics or keywords from the text."""
    # Simple keyword extraction - can be enhanced with NLP
    words = re.findall(r'\b\w{4,}\b', text.lower())
    # Remove common words
    common_words = {'that', 'this', 'with', 'have', 'your', 'about', 'would', 'could', 'from'}
    return [word for word in words if word not in common_words][:5]

class ConversationState:
    """Enhanced conversation state with optional LangChain memory integration."""
    
    def __init__(self, conversation_id: str, persona: str = "friendly", use_langchain: bool = True):
        self.conversation_id = conversation_id
        self.persona = persona
        self.conversation_history: List[Dict[str, str]] = []
        self.covered_topics = set()
        self.last_question = ""
        self.keywords = set()
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Initialize LangChain memory if available
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        self.langchain_memory = None
        self.llm = None
        
        if self.use_langchain:
            try:
                self._initialize_langchain_memory()
                logger.info(f"LangChain memory initialized for conversation {conversation_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize LangChain memory: {e}. Falling back to basic memory.")
                self.use_langchain = False
        
        # Add greeting as first message from investor
        greeting = INVESTOR_PERSONAS[persona]["greeting"]
        self.add_message("investor", greeting)
    
    def _initialize_langchain_memory(self):
        """Initialize LangChain memory components."""
        if not LANGCHAIN_AVAILABLE:
            return
        
        # Initialize LLM for summarization
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Initialize chat history
        self.chat_history = InMemoryChatMessageHistory()
        
        # Use ConversationSummaryBufferMemory for intelligent summarization
        self.langchain_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            chat_memory=self.chat_history,
            max_token_limit=1000,
            return_messages=True,
            memory_key="chat_history"
        )
    
    def add_message(self, role: Literal["founder", "investor"], content: str) -> None:
        """Add a message to the conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.conversation_history.append(message)
        self.last_activity = datetime.utcnow()
        
        # Add to LangChain memory if available
        if self.use_langchain and self.langchain_memory:
            try:
                if role == "founder":
                    self.langchain_memory.chat_memory.add_user_message(content)
                elif role == "investor":
                    self.langchain_memory.chat_memory.add_ai_message(content)
            except Exception as e:
                logger.warning(f"Failed to add message to LangChain memory: {e}")
        
        # Update keywords from founder's messages
        if role == "founder":
            self.keywords.update(extract_keywords(content))
    
    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation so far."""
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.conversation_history[-6:])  # Last 3 exchanges
    
    def get_langchain_context(self) -> str:
        """Get conversation context from LangChain memory with intelligent summarization."""
        if not self.use_langchain or not self.langchain_memory:
            return self.get_conversation_summary()
        
        try:
            # Get memory variables (includes summarized history if needed)
            memory_vars = self.langchain_memory.load_memory_variables({})
            messages = memory_vars.get("chat_history", [])
            
            # Format messages for context
            context_parts = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    context_parts.append(f"Founder: {msg.content}")
                elif isinstance(msg, AIMessage):
                    context_parts.append(f"Investor: {msg.content}")
                elif isinstance(msg, str):
                    # Handle string-based messages (summaries)
                    context_parts.append(msg)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"Failed to get LangChain context: {e}. Using basic summary.")
            return self.get_conversation_summary()
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics and insights."""
        stats = {
            "conversation_id": self.conversation_id,
            "persona": self.persona,
            "total_messages": len(self.conversation_history),
            "topics_discussed": list(self.keywords),
            "using_langchain": self.use_langchain,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "duration_minutes": (self.last_activity - self.created_at).total_seconds() / 60
        }
        
        if self.use_langchain and self.langchain_memory:
            try:
                # Get token count if available
                memory_vars = self.langchain_memory.load_memory_variables({})
                stats["langchain_messages"] = len(memory_vars.get("chat_history", []))
            except:
                pass
        
        return stats
    
    def get_next_question(self) -> str:
        """Generate the next question based on conversation context."""
        # Try LangChain-enhanced approach first
        if self.use_langchain and self.llm:
            return self._get_next_question_langchain()
        else:
            return self._get_next_question_basic()
    
    def _get_next_question_langchain(self) -> str:
        """Generate question using LangChain for enhanced context awareness."""
        try:
            # Get persona details
            persona = INVESTOR_PERSONAS[self.persona]
            
            # Get enhanced conversation context
            context = self.get_langchain_context()
            
            # Create enhanced prompt with LangChain
            system_prompt = f"""You are an experienced investor having a natural conversation with a startup founder. Your goal is to understand their business while making them feel comfortable and engaged.

Your persona: {persona['persona']}
Your style: {persona['style']}

KEY INSTRUCTIONS:
1. First, acknowledge or react to what the founder just said in a natural way (1 sentence)
2. Then ask ONE insightful follow-up question that builds on their last response
3. Keep it conversational - imagine you're having a friendly chat, not conducting an interview
4. If the founder mentions numbers (revenue, users, growth), ask a relevant follow-up about it
5. If you learn the founder's name, use it naturally in your responses
6. Vary your responses - don't use the same phrases repeatedly
7. Keep your total response under 3 sentences

Example responses:
- "That's impressive growth! What's driving most of that increase?"
- "I see you mentioned [topic]. Could you elaborate on how that works?"
- "Thanks for sharing that, [Name]. What's been your biggest challenge with [specific aspect]?"

Remember: Sound like a human, not a robot. Be curious and engaged."""

            human_prompt = f"""Here's our conversation so far:

{context}

Based on the most recent messages, craft a natural response that shows you're listening and asks a relevant follow-up question. Keep it conversational and under 3 sentences total."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            # Generate response using LangChain
            chain = prompt | self.llm
            response = chain.invoke({})
            
            # Extract and clean the response
            question = response.content.strip()
            question = re.sub(r'^[\s\d\."]+', '', question)
            question = question.strip('"\' ')
            
            # Validate response
            if not question or len(question.split()) < 3:
                return self._get_fallback_question()
                
            logger.info(f"Generated LangChain question: {question[:100]}...")
            return question
            
        except Exception as e:
            logger.error(f"Error generating LangChain question: {str(e)}")
            return self._get_next_question_basic()
    
    def _get_next_question_basic(self) -> str:
        """Generate question using basic Gemini approach (fallback)."""
        try:
            # Get persona details
            persona = INVESTOR_PERSONAS[self.persona]
            
            # Prepare conversation context
            context = self.get_conversation_summary()
            
            # Prepare prompt for the AI
            prompt = f"""You're an investor having a natural conversation with a startup founder. 

Your persona: {persona['persona']}
Your style: {persona['style']}

Here's our conversation so far:
{context}

Your task:
1. First, acknowledge or react to what the founder just said (1 sentence)
2. Then ask ONE natural follow-up question that shows you were listening
3. Keep it conversational and under 3 sentences total

Example responses:
- "That's interesting! How did you come up with that approach?"
- "I see you mentioned [topic]. What's been your experience with that?"
- "Thanks for sharing. What's next for your company?"

Your response (be natural and engaging):"""
            
            # Use Gemini to generate a thoughtful follow-up question
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            question = response.text.strip()
            
            # Clean up the response
            question = re.sub(r'^[\s\d\."]+', '', question)  # Remove leading numbers/quotes
            question = question.strip('"\' ')  # Remove surrounding quotes
            
            # If the response is empty or too short, use a fallback
            if not question or len(question.split()) < 3:
                return self._get_fallback_question()
                
            return question
                
        except Exception as e:
            logger.error(f"Error generating question with Gemini: {str(e)}")
            return self._get_fallback_question()
    
    def _get_fallback_question(self) -> str:
        """Get a fallback question when generation fails."""
        fallbacks = [
            "That's interesting—could you tell me more about that?",
            "I'm curious, what led you to that approach?",
            "Thanks for sharing. What's been the most surprising part of this journey for you?",
            "What's the story behind that?",
            "How did you first get started with this?"
        ]
        import random
        return random.choice(fallbacks)

def start_new_conversation(conversation_id: str = None, persona: str = "skeptical") -> ConversationState:
    """
    Initialize a new conversation with the given persona.
    
    Args:
        conversation_id: Optional conversation ID. If None, a new UUID will be generated.
        persona: The investor persona to use (skeptical, technical, friendly)
        
    Returns:
        A new ConversationState object
    """
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Validate persona
    if persona not in INVESTOR_PERSONAS:
        logger.warning(f"Unknown persona '{persona}'. Defaulting to 'skeptical'.")
        persona = "skeptical"
    
    logger.info(f"Starting new conversation {conversation_id} with {persona} persona")
    return ConversationState(conversation_id=conversation_id, persona=persona)


def generate_investor_response(conversation_state: ConversationState, founder_input: str) -> str:
    """
    Generate an AI investor response based on the conversation history and current input.
    
    Args:
        conversation_state: The current state of the conversation
        founder_input: The latest input from the founder
        
    Returns:
        The generated investor response (a question or follow-up)
    """
    logger.info(f"Generating investor response for conversation {conversation_state.conversation_id}")
    
    # Add founder's message to conversation history
    conversation_state.add_message("founder", founder_input)
    
    # Generate a thoughtful follow-up question
    question = conversation_state.get_next_question()
    
    # Add the investor's question to the conversation history
    conversation_state.add_message("investor", question)
    
    return question