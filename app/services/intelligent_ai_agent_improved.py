# intelligent_ai_agent_improved.py
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

# Enhanced investor personas with specific questioning styles
INVESTOR_PERSONAS = {
    "skeptical": {
        "name": "Sarah Martinez",
        "title": "Senior Partner at Venture Capital",
        "personality": "Analytical, direct, numbers-focused. Always challenges assumptions and asks for proof. Wants to see specific metrics, revenue data, and evidence of traction. Uses a professional, no-nonsense tone.",
        "questioning_style": "Direct, challenging, evidence-based. Asks tough questions about metrics, proof points, and concrete data.",
        "sample_questions": [
            "What specific metrics do you have to prove market demand?",
            "Can you show me your actual revenue numbers from the last 6 months?",
            "How do you know customers will actually pay for this?"
        ]
    },
    
    "technical": {
        "name": "Dr. Alex Chen",
        "title": "CTO-turned-Investor at TechVentures",
        "personality": "Curious about technology, detail-oriented, innovation-focused. Excited about technical solutions and wants to understand the architecture, scalability, and technical moats. Enthusiastic but thorough.",
        "questioning_style": "Technical deep-dives, architecture-focused, innovation-oriented. Gets excited about technical breakthroughs.",
        "sample_questions": [
            "What's your technical architecture and how does it scale?",
            "What specific technology gives you a competitive advantage?",
            "How did you solve the core technical challenge in your domain?"
        ]
    },
    
    "friendly": {
        "name": "Michael Thompson",
        "title": "Angel Investor & Former Entrepreneur",
        "personality": "Supportive, empathetic, story-focused. Having been an entrepreneur, understands the journey and challenges. Encouraging tone, interested in the founder's passion and vision.",
        "questioning_style": "Story-focused, empathetic, vision-oriented. Interested in the human journey and passion behind the business.",
        "sample_questions": [
            "What inspired you to start this company?",
            "Tell me about a moment when you knew this was the right path.",
            "How has your team come together around this vision?"
        ]
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
    questions_asked: List[str] = None
    key_info_gathered: dict = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.questions_asked is None:
            self.questions_asked = []
        if self.key_info_gathered is None:
            self.key_info_gathered = {}

class ImprovedStageAgent:
    """Enhanced base class for stage-specific agents with single-question focus"""
    
    def __init__(self, stage: PitchStage, persona: str, llm):
        self.stage = stage
        self.persona = persona
        self.llm = llm
        self.persona_info = INVESTOR_PERSONAS[persona]
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="founder_input"
        )
        self.prompt = self._create_stage_specific_prompt()
        self.chain = self._create_chain()
        
        # Stage-specific question bank
        self.question_bank = self._get_stage_questions()
        self.questions_asked = []
        self.current_question_focus = None
    
    def _create_stage_specific_prompt(self) -> PromptTemplate:
        """Create highly specific prompt for this stage with single question focus"""
        # This will be overridden by specific stage agents
        return self._create_base_prompt()
    
    def _create_base_prompt(self) -> PromptTemplate:
        """Base prompt template"""
        template = f"""You are {self.persona_info['name']}, {self.persona_info['title']}.
        
        PERSONALITY: {self.persona_info['personality']}
        QUESTIONING STYLE: {self.persona_info['questioning_style']}
        
        CURRENT STAGE: {self.stage.value.replace('_', ' ').title()}
        
        CRITICAL INSTRUCTIONS:
        1. Ask ONLY ONE focused question at a time
        2. Wait for a complete answer before moving to the next question
        3. Follow up on incomplete or vague answers with clarifying questions
        4. Stay strictly within the {self.stage.value} topic area
        5. Match your personality perfectly in tone and approach
        
        CONVERSATION CONTEXT:
        {{chat_history}}
        
        FOUNDER'S RESPONSE: {{founder_input}}
        
        Based on their response, ask ONE specific follow-up question about {self.stage.value.replace('_', ' ')}.
        If they haven't fully answered your previous question, clarify that first.
        
        Your response:"""
        
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )
    
    def _get_stage_questions(self) -> List[str]:
        """Get stage-specific questions - to be overridden by specific agents"""
        return []
    
    def _create_chain(self) -> LLMChain:
        """Create LangChain for this stage"""
        return LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )
    
    def respond(self, founder_input: str, context: dict) -> str:
        """Generate response for this stage with single question focus"""
        try:
            # Prepare context
            chat_history = context.get("chat_history", "")
            
            # Prepare input for the chain
            input_dict = {
                "founder_input": founder_input,
                "chat_history": chat_history
            }
            
            # Run the chain
            response = self.chain.run(input_dict)
            
            # Track the question asked
            self.questions_asked.append(response.strip())
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in {self.stage.value} agent response: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while processing your request. Could you please rephrase or try again later?"

class GreetingAgent(ImprovedStageAgent):
    """Specialized agent for greeting stage"""
    
    def _create_stage_specific_prompt(self) -> PromptTemplate:
        template = f"""You are {self.persona_info['name']}, {self.persona_info['title']}.
        
        PERSONALITY: {self.persona_info['personality']}
        YOUR APPROACH: {self.persona_info['questioning_style']}
        
        GREETING STAGE OBJECTIVES:
        1. Make the founder feel welcome and comfortable
        2. Introduce yourself naturally
        3. Get their name and company name
        4. Ask ONE simple opening question to get them talking
        
        CRITICAL INSTRUCTIONS:
        - If this is the first interaction, introduce yourself warmly
        - If they've introduced themselves, acknowledge their name and company
        - Ask ONLY ONE question at a time
        - Keep the tone conversational and welcoming
        - Match your personality perfectly
        
        CONVERSATION SO FAR:
        {{chat_history}}
        
        FOUNDER'S MESSAGE: {{founder_input}}
        
        Respond naturally. If you need basic info (name/company), ask for ONE piece at a time.
        If you have their info, ask ONE engaging question to start the business discussion.
        
        Your response:"""
        
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )
    
    def _get_stage_questions(self) -> List[str]:
        """Questions for greeting stage"""
        return [
            "What's your name?",
            "What's your company called?",
            "What inspired you to start this business?",
            "How long have you been working on this?",
            "What's the most exciting thing about your company right now?"
        ]

class ProblemSolutionAgent(ImprovedStageAgent):
    """Agent focused on understanding the problem and solution with single questions"""
    
    def _create_stage_specific_prompt(self) -> PromptTemplate:
        template = f"""You are {self.persona_info['name']}, {self.persona_info['title']}.
        
        PERSONALITY: {self.persona_info['personality']}
        YOUR APPROACH: {self.persona_info['questioning_style']}
        
        PROBLEM & SOLUTION STAGE OBJECTIVES:
        1. Understand the core problem they're solving
        2. Evaluate how their solution addresses the problem
        3. Assess the significance and urgency of the problem
        4. Understand solution effectiveness and uniqueness
        
        CRITICAL INSTRUCTIONS:
        - Ask ONLY ONE focused question at a time
        - Don't move to next topic until current question is fully answered
        - Dig deeper on vague or incomplete answers
        - Stay focused on problem/solution fit
        - Match your personality in questioning style
        
        CONVERSATION CONTEXT:
        {{chat_history}}
        
        FOUNDER'S RESPONSE: {{founder_input}}
        
        Based on their response about the problem and solution, ask ONE specific follow-up question.
        Focus on getting clear, detailed answers about either the problem or their solution.
        
        Your response:"""
        
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )
    
    def _get_stage_questions(self) -> List[str]:
        """Questions for problem/solution stage"""
        if self.persona == "skeptical":
            return [
                "What specific problem are you solving?",
                "How do you know this problem is significant enough for people to pay to solve?",
                "What evidence do you have that your solution actually works?",
                "How many people have actually used your solution?",
                "What measurable results have you achieved?"
            ]
        elif self.persona == "technical":
            return [
                "What's the core technical problem you're solving?",
                "How is your technical approach different from existing solutions?",
                "What's the most innovative aspect of your solution?",
                "How does your technology scale to handle more users?",
                "What technical challenges did you have to overcome?"
            ]
        else:  # friendly
            return [
                "Tell me about the moment you realized this problem needed solving.",
                "How did you come up with your solution approach?",
                "What's been the most rewarding part of building this solution?",
                "How do users react when they first try your solution?",
                "What's the story behind your breakthrough moment?"
            ]

class TargetMarketAgent(ImprovedStageAgent):
    """Agent focused on understanding the target market"""
    
    def _create_stage_specific_prompt(self) -> PromptTemplate:
        template = f"""You are {self.persona_info['name']}, {self.persona_info['title']}.
        
        PERSONALITY: {self.persona_info['personality']}
        YOUR APPROACH: {self.persona_info['questioning_style']}
        
        TARGET MARKET STAGE OBJECTIVES:
        1. Identify specific target customer segments
        2. Understand market size and growth potential
        3. Assess customer acquisition strategy
        4. Evaluate market timing and trends
        
        CRITICAL INSTRUCTIONS:
        - Ask ONLY ONE focused question at a time about their target market
        - Get specific details about customer segments
        - Don't accept vague answers like "everyone needs this"
        - Push for concrete market data and customer insights
        - Match your personality in questioning approach
        
        CONVERSATION CONTEXT:
        {{chat_history}}
        
        FOUNDER'S RESPONSE: {{founder_input}}
        
        Ask ONE specific question about their target market, customer segments, or market opportunity.
        
        Your response:"""
        
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )
    
    def _get_stage_questions(self) -> List[str]:
        """Questions for target market stage"""
        if self.persona == "skeptical":
            return [
                "Who exactly is your target customer?",
                "What's the specific size of your addressable market?",
                "How much are customers currently spending to solve this problem?",
                "What data do you have on customer demand?",
                "How do you know customers will switch to your solution?"
            ]
        elif self.persona == "technical":
            return [
                "What specific market segment has the highest technical need for your solution?",
                "How do technical buyers evaluate solutions like yours?",
                "What market trends are driving demand for your technology?",
                "Which customer segment understands your technical value proposition best?",
                "How does market maturity affect adoption of your solution?"
            ]
        else:  # friendly
            return [
                "Tell me about your ideal customer.",
                "How did you discover this market opportunity?",
                "What do your potential customers say when you describe your solution?",
                "How do you connect with your target audience?",
                "What's the story of your first customer or user?"
            ]

class BusinessModelAgent(ImprovedStageAgent):
    """Agent focused on understanding the business model"""
    
    def _create_stage_specific_prompt(self) -> PromptTemplate:
        template = f"""You are {self.persona_info['name']}, {self.persona_info['title']}.
        
        PERSONALITY: {self.persona_info['personality']}
        YOUR APPROACH: {self.persona_info['questioning_style']}
        
        BUSINESS MODEL STAGE OBJECTIVES:
        1. Understand revenue generation strategy
        2. Evaluate pricing model and strategy
        3. Assess unit economics and scalability
        4. Understand cost structure and margins
        
        CRITICAL INSTRUCTIONS:
        - Ask ONLY ONE focused question at a time about their business model
        - Get specific details about how they make money
        - Push for concrete numbers and economics
        - Understand scalability and sustainability
        - Match your personality in questioning style
        
        CONVERSATION CONTEXT:
        {{chat_history}}
        
        FOUNDER'S RESPONSE: {{founder_input}}
        
        Ask ONE specific question about their business model, pricing, or revenue strategy.
        
        Your response:"""
        
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )
    
    def _get_stage_questions(self) -> List[str]:
        """Questions for business model stage"""
        if self.persona == "skeptical":
            return [
                "How exactly do you make money?",
                "What are your unit economics - cost to acquire vs lifetime value?",
                "What's your pricing strategy and how did you determine it?",
                "What are your actual revenue numbers so far?",
                "How do you know customers will pay your asking price?"
            ]
        elif self.persona == "technical":
            return [
                "How does your technical architecture support your business model?",
                "What's your cost structure for delivering your solution?",
                "How do you monetize your technical innovations?",
                "What are the technical costs of scaling your business model?",
                "How does technology give you better unit economics than competitors?"
            ]
        else:  # friendly
            return [
                "Walk me through how your business makes money.",
                "How did you decide on your pricing approach?",
                "What's been your experience with customers' willingness to pay?",
                "How do you see your business model evolving?",
                "What's the most exciting part of your revenue strategy?"
            ]

class CompetitionAgent(ImprovedStageAgent):
    """Agent focused on understanding the competitive landscape"""
    
    def _create_stage_specific_prompt(self) -> PromptTemplate:
        template = f"""You are {self.persona_info['name']}, {self.persona_info['title']}.
        
        PERSONALITY: {self.persona_info['personality']}
        YOUR APPROACH: {self.persona_info['questioning_style']}
        
        COMPETITION STAGE OBJECTIVES:
        1. Identify key direct and indirect competitors
        2. Understand competitive advantages and differentiation
        3. Assess barriers to entry and competitive moats
        4. Evaluate competitive positioning strategy
        
        CRITICAL INSTRUCTIONS:
        - Ask ONLY ONE focused question at a time about competition
        - Don't accept "we have no competition" as an answer
        - Push for honest competitive analysis
        - Understand both direct and indirect competition
        - Match your personality in questioning approach
        
        CONVERSATION CONTEXT:
        {{chat_history}}
        
        FOUNDER'S RESPONSE: {{founder_input}}
        
        Ask ONE specific question about their competition, differentiation, or competitive strategy.
        
        Your response:"""
        
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )
    
    def _get_stage_questions(self) -> List[str]:
        """Questions for competition stage"""
        if self.persona == "skeptical":
            return [
                "Who are your main competitors and how do you compare?",
                "What prevents competitors from copying your approach?",
                "How do you win deals against established competitors?",
                "What's your sustainable competitive advantage?",
                "How do customers currently solve this problem without you?"
            ]
        elif self.persona == "technical":
            return [
                "What's your technical differentiation from competitors?",
                "How deep are your technical moats?",
                "What technical barriers prevent others from replicating your solution?",
                "How does your technical architecture compare to competitors?",
                "What intellectual property gives you competitive protection?"
            ]
        else:  # friendly
            return [
                "Tell me about the competitive landscape you're operating in.",
                "What makes you unique compared to others in the space?",
                "How do you position yourself differently?",
                "What's your strategy for standing out?",
                "What do customers say about you versus alternatives?"
            ]

class TractionAgent(ImprovedStageAgent):
    """Agent focused on understanding traction and milestones"""
    
    def _create_stage_specific_prompt(self) -> PromptTemplate:
        template = f"""You are {self.persona_info['name']}, {self.persona_info['title']}.
        
        PERSONALITY: {self.persona_info['personality']}
        YOUR APPROACH: {self.persona_info['questioning_style']}
        
        TRACTION STAGE OBJECTIVES:
        1. Understand current growth metrics and traction
        2. Evaluate key milestones and achievements
        3. Assess growth trajectory and momentum
        4. Understand customer validation and retention
        
        CRITICAL INSTRUCTIONS:
        - Ask ONLY ONE focused question at a time about traction
        - Push for specific metrics and numbers
        - Don't accept vague answers about "rapid growth"
        - Get concrete evidence of market validation
        - Match your personality in questioning style
        
        CONVERSATION CONTEXT:
        {{chat_history}}
        
        FOUNDER'S RESPONSE: {{founder_input}}
        
        Ask ONE specific question about their traction, metrics, or growth milestones.
        
        Your response:"""
        
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )
    
    def _get_stage_questions(self) -> List[str]:
        """Questions for traction stage"""
        if self.persona == "skeptical":
            return [
                "What specific metrics prove you have traction?",
                "What's your month-over-month growth rate?",
                "How many customers do you have and what's your retention rate?",
                "What revenue have you generated so far?",
                "What concrete evidence shows customers love your product?"
            ]
        elif self.persona == "technical":
            return [
                "What technical metrics demonstrate your solution's performance?",
                "How has your technical performance improved over time?",
                "What usage patterns show technical product-market fit?",
                "How do technical customers validate your solution?",
                "What technical milestones have you achieved recently?"
            ]
        else:  # friendly
            return [
                "What milestones are you most proud of achieving?",
                "Tell me about your growth journey so far.",
                "What feedback do customers give you about your progress?",
                "What's been your biggest breakthrough moment?",
                "How do you measure success and momentum?"
            ]

class TeamAgent(ImprovedStageAgent):
    """Agent focused on evaluating the team"""
    
    def _create_stage_specific_prompt(self) -> PromptTemplate:
        template = f"""You are {self.persona_info['name']}, {self.persona_info['title']}.
        
        PERSONALITY: {self.persona_info['personality']}
        YOUR APPROACH: {self.persona_info['questioning_style']}
        
        TEAM STAGE OBJECTIVES:
        1. Understand team composition and experience
        2. Evaluate founder-market fit and expertise
        3. Assess team dynamics and culture
        4. Understand hiring plans and team building
        
        CRITICAL INSTRUCTIONS:
        - Ask ONLY ONE focused question at a time about the team
        - Understand both current team and future plans
        - Evaluate relevant experience and expertise
        - Assess team's ability to execute
        - Match your personality in questioning approach
        
        CONVERSATION CONTEXT:
        {{chat_history}}
        
        FOUNDER'S RESPONSE: {{founder_input}}
        
        Ask ONE specific question about their team, experience, or team building plans.
        
        Your response:"""
        
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )
    
    def _get_stage_questions(self) -> List[str]:
        """Questions for team stage"""
        if self.persona == "skeptical":
            return [
                "What specific experience qualifies your team to solve this problem?",
                "What key roles are you missing and how will you fill them?",
                "How do you know your team can execute this business plan?",
                "What's your track record of building and scaling teams?",
                "How do you retain key talent and prevent team risks?"
            ]
        elif self.persona == "technical":
            return [
                "What's the technical background and expertise of your team?",
                "How does your team's technical experience match your solution needs?",
                "What technical leadership experience does your team have?",
                "How do you plan to build your technical team?",
                "What technical skills are most critical for your success?"
            ]
        else:  # friendly
            return [
                "Tell me about your team and how you came together.",
                "What unique strengths does each team member bring?",
                "How do you work together as founders?",
                "What's your vision for building the team?",
                "What's the story of how your core team formed?"
            ]

class FundingNeedsAgent(ImprovedStageAgent):
    """Agent focused on funding needs and financial plans"""
    
    def _create_stage_specific_prompt(self) -> PromptTemplate:
        template = f"""You are {self.persona_info['name']}, {self.persona_info['title']}.
        
        PERSONALITY: {self.persona_info['personality']}
        YOUR APPROACH: {self.persona_info['questioning_style']}
        
        FUNDING NEEDS STAGE OBJECTIVES:
        1. Understand specific funding requirements
        2. Evaluate use of funds and milestones
        3. Assess financial projections and planning
        4. Understand future funding strategy
        
        CRITICAL INSTRUCTIONS:
        - Ask ONLY ONE focused question at a time about funding
        - Get specific numbers and detailed use of funds
        - Understand timeline and milestones
        - Evaluate financial planning and projections
        - Match your personality in questioning style
        
        CONVERSATION CONTEXT:
        {{chat_history}}
        
        FOUNDER'S RESPONSE: {{founder_input}}
        
        Ask ONE specific question about their funding needs, use of funds, or financial planning.
        
        Your response:"""
        
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )
    
    def _get_stage_questions(self) -> List[str]:
        """Questions for funding needs stage"""
        if self.persona == "skeptical":
            return [
                "How much funding do you need and what exactly will you use it for?",
                "What specific milestones will this funding help you achieve?",
                "How long will this funding last and what metrics will prove success?",
                "What's your plan if you can't raise the full amount?",
                "How do you justify your valuation and funding requirements?"
            ]
        elif self.persona == "technical":
            return [
                "How will you use funding to advance your technical development?",
                "What technical infrastructure investments do you need to make?",
                "How much of the funding goes to technical team expansion?",
                "What technical milestones will funding help you achieve?",
                "How do you plan to scale your technical capabilities?"
            ]
        else:  # friendly
            return [
                "What are your funding goals and how will investment help you grow?",
                "What would achieving your funding goals mean for your vision?",
                "How do you see partnership with investors beyond just funding?",
                "What's your timeline for using this investment?",
                "What support beyond capital would be most valuable?"
            ]

class FuturePlansAgent(ImprovedStageAgent):
    """Agent focused on future plans and vision"""
    
    def _create_stage_specific_prompt(self) -> PromptTemplate:
        template = f"""You are {self.persona_info['name']}, {self.persona_info['title']}.
        
        PERSONALITY: {self.persona_info['personality']}
        YOUR APPROACH: {self.persona_info['questioning_style']}
        
        FUTURE PLANS STAGE OBJECTIVES:
        1. Understand long-term vision and strategy
        2. Evaluate product roadmap and expansion plans
        3. Discuss potential exit strategies and outcomes
        4. Assess scalability and growth potential
        
        CRITICAL INSTRUCTIONS:
        - Ask ONLY ONE focused question at a time about future plans
        - Understand both short-term and long-term vision
        - Evaluate strategic thinking and planning
        - Assess growth potential and scalability
        - Match your personality in questioning approach
        
        CONVERSATION CONTEXT:
        {{chat_history}}
        
        FOUNDER'S RESPONSE: {{founder_input}}
        
        Ask ONE specific question about their future plans, vision, or growth strategy.
        
        Your response:"""
        
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )
    
    def _get_stage_questions(self) -> List[str]:
        """Questions for future plans stage"""
        if self.persona == "skeptical":
            return [
                "What's your realistic timeline for achieving profitability?",
                "What are the key risks that could derail your plans?",
                "How will you measure success over the next 2-3 years?",
                "What's your exit strategy and timeline?",
                "How do you plan to defend your market position as you grow?"
            ]
        elif self.persona == "technical":
            return [
                "What's your technical roadmap for the next 18 months?",
                "How will your technology evolve to stay competitive?",
                "What technical challenges do you anticipate as you scale?",
                "How do you plan to maintain technical innovation?",
                "What's your vision for the technical future of your industry?"
            ]
        else:  # friendly
            return [
                "Where do you see your company in 5 years?",
                "What's the impact you hope to make in your industry?",
                "How do you envision growing from where you are today?",
                "What would success look like for you personally and professionally?",
                "What legacy do you want your company to create?"
            ]

# Agent factory to create the appropriate agent based on stage and persona
class ImprovedAgentFactory:
    """Factory to create appropriate agent for each stage with persona"""
    
    @staticmethod
    def create_agent(stage: PitchStage, persona: str, llm) -> ImprovedStageAgent:
        """Create an agent for the given stage and persona"""
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
            
        return agent_class(stage, persona, llm)

class ImprovedConversationContext:
    """Enhanced conversation context with better single-question management"""
    
    def __init__(self, conversation_id: str, persona: str, llm):
        self.conversation_id = conversation_id
        self.persona = persona
        self.llm = llm
        self.current_stage = PitchStage.GREETING
        self.agent_factory = ImprovedAgentFactory()
        self.current_agent = self.agent_factory.create_agent(self.current_stage, persona, llm)
        self.chat_history = []
        self.stage_contexts = {stage: StageContext(stage) for stage in PitchStage}
        self.shared_context = {
            "founder_name": "",
            "company_name": "",
            "persona": persona,
            "current_question_topic": None,
            "awaiting_answer_for": None
        }
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def process_message(self, message: str) -> str:
        """Process a message with single-question focus"""
        try:
            # Update timestamps
            self.last_activity = datetime.now()
            
            # Add founder's message to chat history
            self.chat_history.append({
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'sender': 'founder'
            })
            
            # Update shared context
            self._update_shared_context(message)
            
            # Check if current question was adequately answered
            current_response_complete = self._assess_response_completeness(message)
            
            # Prepare context for agent
            context = {
                "chat_history": self._format_chat_history(),
                "founder_name": self.shared_context["founder_name"],
                "company_name": self.shared_context["company_name"],
                "current_stage": self.current_stage.value,
                "response_complete": current_response_complete,
                **self.shared_context
            }
            
            # Get response from current agent
            response = self.current_agent.respond(message, context)
            
            # Add response to chat history
            self.chat_history.append({
                'timestamp': datetime.now().isoformat(),
                'message': response,
                'sender': 'investor'
            })
            
            # Check for stage transition only if response was complete
            if current_response_complete and self._should_transition_stage(message, context):
                self._transition_to_next_stage()
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error. Could you please rephrase your response?"
    
    def _assess_response_completeness(self, message: str) -> bool:
        """Assess if the founder's response adequately answers the current question"""
        if len(message.strip()) < 10:  # Very short responses are likely incomplete
            return False
        
        # Use LLM to assess response completeness
        assessment_prompt = f"""
        Assess if this response adequately answers the investor's question.
        
        Last investor question: {self.chat_history[-2]['message'] if len(self.chat_history) >= 2 else 'Initial greeting'}
        Founder's response: {message}
        
        Is this response complete and informative? Respond with only 'yes' or 'no'.
        """
        
        try:
            assessment = self.llm.invoke(assessment_prompt).content.lower().strip()
            return "yes" in assessment
        except Exception as e:
            logger.warning(f"Error assessing response completeness: {e}")
            return True  # Default to complete to avoid getting stuck
    
    def _format_chat_history(self) -> str:
        """Format chat history for context"""
        formatted = []
        for msg in self.chat_history[-6:]:  # Last 6 messages for context
            sender = msg.get('sender', 'unknown')
            content = msg.get('message', '')
            formatted.append(f"{sender.title()}: {content}")
        return "\n".join(formatted)
    
    def _update_shared_context(self, message: str) -> None:
        """Enhanced context updating"""
        # Extract name if not already captured
        if not self.shared_context["founder_name"] and self.current_stage == PitchStage.GREETING:
            name_prompt = f"""Extract the person's name from this message if they're introducing themselves. 
            Return only the first name or empty string if not clear: "{message}" """
            try:
                name = self.llm.invoke(name_prompt).content.strip()
                if name and len(name) > 1 and len(name) < 30:
                    self.shared_context["founder_name"] = name
            except Exception as e:
                logger.warning(f"Error extracting name: {e}")
        
        # Extract company name
        if not self.shared_context["company_name"] and self.current_stage == PitchStage.GREETING:
            company_prompt = f"""Extract the company name from this message if mentioned. 
            Return only the company name or empty string if not clear: "{message}" """
            try:
                company = self.llm.invoke(company_prompt).content.strip()
                if company and len(company) > 1 and len(company) < 50:
                    self.shared_context["company_name"] = company
            except Exception as e:
                logger.warning(f"Error extracting company: {e}")
    
    def _should_transition_stage(self, message: str, context: dict) -> bool:
        """Enhanced stage transition logic"""
        # Don't transition too quickly (at least 2 exchanges per stage)
        stage_messages = [msg for msg in self.chat_history if msg.get('sender') == 'investor' and self.current_stage.value in str(msg)]
        if len(stage_messages) < 2:
            return False
        
        # Use agent's assessment
        if hasattr(self.current_agent, 'should_transition'):
            return self.current_agent.should_transition(message, context)
        
        return False
    
    def _transition_to_next_stage(self) -> None:
        """Transition to next stage"""
        # Mark current stage as complete
        self.stage_contexts[self.current_stage].is_complete = True
        
        # Find next stage
        try:
            current_idx = PITCH_STAGES.index(self.current_stage.value)
            if current_idx + 1 < len(PITCH_STAGES):
                next_stage = PitchStage(PITCH_STAGES[current_idx + 1])
                
                logger.info(f"Transitioning from {self.current_stage.value} to {next_stage.value}")
                
                self.current_stage = next_stage
                self.current_agent = self.agent_factory.create_agent(next_stage, self.persona, self.llm)
                
                # Add transition message
                transition_msg = f"Great! Now let's discuss {next_stage.value.replace('_', ' ')}."
                self.chat_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'message': transition_msg,
                    'sender': 'system'
                })
                
        except (ValueError, IndexError) as e:
            logger.error(f"Error transitioning stages: {e}")

class ImprovedIntelligentAIAgent:
    """Enhanced AI agent with single-question focus and better stage management"""
    
    def __init__(self, llm=None):
        self.llm = llm or llm  # Use the global llm instance
        self.conversations = {}
    
    def start_conversation(self, conversation_id: str, persona: str = "friendly") -> ImprovedConversationContext:
        """Start a new conversation with enhanced context"""
        if persona not in INVESTOR_PERSONAS:
            raise ValueError(f"Unknown persona: {persona}")
        
        context = ImprovedConversationContext(conversation_id, persona, self.llm)
        self.conversations[conversation_id] = context
        
        # Generate initial greeting based on persona
        persona_info = INVESTOR_PERSONAS[persona]
        initial_greeting = f"Hi! I'm {persona_info['name']}, {persona_info['title']}. I'm excited to learn about your business. Let's start with introductions - what's your name?"
        
        # Add greeting to chat history
        context.chat_history.append({
            'timestamp': datetime.now().isoformat(),
            'message': initial_greeting,
            'sender': 'investor'
        })
        
        logger.info(f"Started conversation {conversation_id} with {persona} persona")
        return context
    
    def generate_response(self, conversation_id: str, founder_input: str) -> str:
        """Generate response using enhanced single-question approach"""
        if conversation_id not in self.conversations:
            raise ValueError(f"No conversation found with ID {conversation_id}")
        
        context = self.conversations[conversation_id]
        return context.process_message(founder_input)
    
    def get_conversation_stats(self, conversation_id: str) -> dict:
        """Get enhanced conversation statistics"""
        if conversation_id not in self.conversations:
            return {"status": "error", "message": "Conversation not found"}
        
        context = self.conversations[conversation_id]
        
        # Calculate metrics
        duration_minutes = (context.last_activity - context.created_at).total_seconds() / 60
        completed_stages = [stage.value for stage, ctx in context.stage_contexts.items() if ctx.is_complete]
        
        return {
            "conversation_id": conversation_id,
            "persona": context.persona,
            "current_stage": context.current_stage.value,
            "completed_stages": completed_stages,
            "total_messages": len(context.chat_history),
            "duration_minutes": round(duration_minutes, 2),
            "founder_name": context.shared_context.get("founder_name", ""),
            "company_name": context.shared_context.get("company_name", ""),
            "status": "active"
        }

# Global improved agent instance
improved_agent = None

def initialize_improved_agent():
    """Initialize the improved agent"""
    global improved_agent
    improved_agent = ImprovedIntelligentAIAgent(llm)
    return improved_agent

# Convenience functions
def start_improved_conversation(conversation_id: str, persona: str = "friendly") -> ImprovedConversationContext:
    """Start conversation with improved agent"""
    if improved_agent is None:
        initialize_improved_agent()
    return improved_agent.start_conversation(conversation_id, persona)

def generate_improved_response(conversation_id: str, founder_input: str) -> str:
    """Generate response with improved agent"""
    if improved_agent is None:
        raise RuntimeError("Improved agent not initialized")
    return improved_agent.generate_response(conversation_id, founder_input)