# conversation_memory.py
import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

# LangChain imports
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedConversationMemory:
    """
    Enhanced conversation memory using LangChain for better context tracking.
    Combines window-based memory with summarization for longer conversations.
    """
    
    def __init__(
        self, 
        conversation_id: str, 
        persona: str = "skeptical",
        max_token_limit: int = 1000,
        window_size: int = 10,
        session_dir: Optional[str] = None
    ):
        self.conversation_id = conversation_id
        self.persona = persona
        self.session_dir = session_dir
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Initialize LangChain LLM for summarization
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Initialize memory components
        self.chat_history = InMemoryChatMessageHistory()
        
        # Use ConversationSummaryBufferMemory for automatic summarization
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            chat_memory=self.chat_history,
            max_token_limit=max_token_limit,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Additional tracking
        self.conversation_topics = set()
        self.key_insights = []
        self.question_count = 0
        self.total_exchanges = 0
        
        # Add initial greeting
        self._add_initial_greeting()
        
        logger.info(f"Initialized enhanced memory for conversation {conversation_id} with {persona} persona")
    
    def _add_initial_greeting(self):
        """Add the initial investor greeting based on persona."""
        from .ai_response import INVESTOR_PERSONAS
        
        greeting = INVESTOR_PERSONAS.get(self.persona, {}).get(
            "greeting", 
            "Hello! I'd love to hear about your business venture."
        )
        
        # Add greeting to memory
        self.memory.chat_memory.add_ai_message(greeting)
        self.last_activity = datetime.utcnow()
    
    def add_founder_message(self, message: str, metadata: Optional[Dict] = None):
        """Add a founder's message to the conversation memory."""
        self.memory.chat_memory.add_user_message(message)
        self.total_exchanges += 1
        self.last_activity = datetime.utcnow()
        
        # Extract topics and insights
        self._extract_topics(message)
        
        # Save conversation state
        if self.session_dir:
            self._save_conversation_state()
        
        logger.info(f"Added founder message to conversation {self.conversation_id}")
    
    def add_investor_message(self, message: str, metadata: Optional[Dict] = None):
        """Add an investor's message to the conversation memory."""
        self.memory.chat_memory.add_ai_message(message)
        self.question_count += 1
        self.last_activity = datetime.utcnow()
        
        # Save conversation state
        if self.session_dir:
            self._save_conversation_state()
        
        logger.info(f"Added investor message to conversation {self.conversation_id}")
    
    def get_conversation_context(self) -> str:
        """Get the current conversation context from memory."""
        try:
            # Get memory variables (includes summarized history if needed)
            memory_vars = self.memory.load_memory_variables({})
            messages = memory_vars.get("chat_history", [])
            
            # Format messages for context
            context_parts = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    context_parts.append(f"Founder: {msg.content}")
                elif isinstance(msg, AIMessage):
                    context_parts.append(f"Investor: {msg.content}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return "No conversation history available."
    
    def get_memory_summary(self) -> str:
        """Get a summary of the conversation so far."""
        try:
            if hasattr(self.memory, 'predict_new_summary'):
                # Get existing summary
                messages = self.memory.chat_memory.messages
                if len(messages) > 4:  # Only summarize if there's enough content
                    summary = self.memory.predict_new_summary(messages, "")
                    return summary
            
            # Fallback: return recent context
            return self.get_conversation_context()
            
        except Exception as e:
            logger.error(f"Error generating memory summary: {str(e)}")
            return "Unable to generate conversation summary."
    
    def _extract_topics(self, message: str):
        """Extract conversation topics from founder's message."""
        # Simple keyword extraction for topics
        import re
        
        # Business-related keywords
        business_keywords = {
            'revenue', 'customers', 'market', 'product', 'service', 'solution',
            'problem', 'team', 'funding', 'investment', 'growth', 'scale',
            'technology', 'platform', 'user', 'client', 'business model',
            'traction', 'competition', 'advantage'
        }
        
        words = set(re.findall(r'\b\w{4,}\b', message.lower()))
        found_topics = words.intersection(business_keywords)
        self.conversation_topics.update(found_topics)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the conversation."""
        return {
            "conversation_id": self.conversation_id,
            "persona": self.persona,
            "total_exchanges": self.total_exchanges,
            "question_count": self.question_count,
            "topics_discussed": list(self.conversation_topics),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "duration_minutes": (self.last_activity - self.created_at).total_seconds() / 60
        }
    
    def _save_conversation_state(self):
        """Save the current conversation state to disk."""
        if not self.session_dir:
            return
        
        try:
            conversation_data = {
                "conversation_id": self.conversation_id,
                "persona": self.persona,
                "created_at": self.created_at.isoformat(),
                "last_activity": self.last_activity.isoformat(),
                "conversation_context": self.get_conversation_context(),
                "memory_summary": self.get_memory_summary(),
                "stats": self.get_conversation_stats(),
                "messages": []
            }
            
            # Add messages from memory
            for msg in self.memory.chat_memory.messages:
                if isinstance(msg, HumanMessage):
                    conversation_data["messages"].append({
                        "role": "founder",
                        "content": msg.content,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif isinstance(msg, AIMessage):
                    conversation_data["messages"].append({
                        "role": "investor",
                        "content": msg.content,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Save to file
            file_path = os.path.join(self.session_dir, f"{self.conversation_id}_memory.json")
            with open(file_path, 'w') as f:
                json.dump(conversation_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving conversation state: {str(e)}")
    
    def load_conversation_state(self, file_path: str) -> bool:
        """Load conversation state from disk."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Restore basic properties
            self.conversation_id = data.get("conversation_id", self.conversation_id)
            self.persona = data.get("persona", self.persona)
            
            # Restore messages to memory
            messages = data.get("messages", [])
            for msg in messages:
                if msg["role"] == "founder":
                    self.memory.chat_memory.add_user_message(msg["content"])
                elif msg["role"] == "investor":
                    self.memory.chat_memory.add_ai_message(msg["content"])
            
            logger.info(f"Loaded conversation state for {self.conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading conversation state: {str(e)}")
            return False
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
        self.conversation_topics.clear()
        self.key_insights.clear()
        self.question_count = 0
        self.total_exchanges = 0
        logger.info(f"Cleared memory for conversation {self.conversation_id}")


class ConversationMemoryManager:
    """Manages multiple conversation memories."""
    
    def __init__(self, session_dir: Optional[str] = None):
        self.conversations: Dict[str, EnhancedConversationMemory] = {}
        self.session_dir = session_dir
        logger.info("Initialized ConversationMemoryManager")
    
    def get_or_create_conversation(
        self, 
        conversation_id: str, 
        persona: str = "skeptical"
    ) -> EnhancedConversationMemory:
        """Get existing conversation or create a new one."""
        
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = EnhancedConversationMemory(
                conversation_id=conversation_id,
                persona=persona,
                session_dir=self.session_dir
            )
            logger.info(f"Created new conversation memory for {conversation_id}")
        
        return self.conversations[conversation_id]
    
    def remove_conversation(self, conversation_id: str):
        """Remove a conversation from memory."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Removed conversation {conversation_id} from memory")
    
    def get_active_conversations(self) -> List[str]:
        """Get list of active conversation IDs."""
        return list(self.conversations.keys())
    
    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Remove conversations older than specified hours."""
        from datetime import timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_remove = []
        
        for conv_id, conv in self.conversations.items():
            if conv.last_activity < cutoff_time:
                to_remove.append(conv_id)
        
        for conv_id in to_remove:
            self.remove_conversation(conv_id)
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old conversations")