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
    model=os.getenv("GEMINI_MODEL","gemini-2.5-flash-lite-preview-06-17"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    convert_system_message_to_human=True
)

# Enhanced investor personas with detailed psychological profiles and questioning methodologies
INVESTOR_PERSONAS = {
    "skeptical": {
        "name": "Sarah Martinez",
        "title": "Senior Partner at Venture Capital",
        "personality": "Analytical, data-driven, and methodical. Former McKinsey consultant with 15+ years in venture capital. Approaches every pitch with healthy skepticism, requiring concrete evidence for every claim. Values precision, measurable outcomes, and risk mitigation. Known for asking the tough questions that other investors avoid. Maintains professional demeanor while being direct about concerns.",
        "questioning_style": "Systematic evidence-gathering approach. Follows a logical progression from claims to proof. Uses the 'show me, don't tell me' methodology. Employs devil's advocate positioning to stress-test assumptions. Focuses on quantifiable metrics, financial projections, and market validation data.",
        "cognitive_approach": "Bottom-up analysis starting with unit economics. Seeks contradictory evidence to test thesis strength. Applies Porter's Five Forces framework mentally. Evaluates through risk-adjusted return lens.",
        "communication_patterns": "Uses precise language with financial terminology. Asks follow-up questions that drill deeper into specifics. Often requests documentation or proof points. Maintains steady pace and doesn't rush to conclusions.",
        "decision_triggers": "Concrete traction metrics, validated business model, clear path to profitability, experienced team with relevant track record, defensible competitive moats",
        "red_flags": "Vague answers about metrics, unvalidated assumptions, lack of customer evidence, unrealistic projections, weak competitive analysis",
        "sample_questions": [
            "What specific metrics do you have to prove market demand, and how did you validate these numbers?",
            "Can you walk me through your unit economics with actual data from the last 6 months?",
            "How do you know customers will actually pay for this, and what's your evidence beyond surveys?"
        ]
    },
    
    "technical": {
        "name": "Dr. Alex Chen",
        "title": "CTO-turned-Investor at TechVentures",
        "personality": "Intellectually curious with deep technical expertise in AI, distributed systems, and scalable architectures. Former Google Principal Engineer who transitioned to investing. Gets genuinely excited about elegant technical solutions and innovative approaches. Values technical depth, scalability considerations, and engineering excellence. Appreciates founders who can discuss technical trade-offs intelligently.",
        "questioning_style": "Architecture-first approach focusing on technical feasibility and scalability. Explores the 'how' behind the solution with deep technical dives. Evaluates technical moats and defensibility. Assesses team's technical competency through detailed discussions.",
        "cognitive_approach": "Systems thinking with focus on scalability bottlenecks. Evaluates technical risk vs. innovation potential. Considers technology adoption curves and market timing. Assesses technical team's ability to execute complex solutions.",
        "communication_patterns": "Uses technical terminology appropriately. Asks detailed implementation questions. Shows enthusiasm for innovative approaches. Provides technical insights and suggestions during conversation.",
        "decision_triggers": "Novel technical approach, strong technical team, scalable architecture, clear technical moats, innovative use of emerging technologies",
        "red_flags": "Weak technical foundation, non-scalable architecture, technically inexperienced team, over-engineered solutions, ignoring technical debt",
        "sample_questions": [
            "What's your technical architecture and how does it handle scale from 1K to 1M users?",
            "What specific technical innovations give you a sustainable competitive advantage?",
            "How did you solve the core technical challenges, and what trade-offs did you make?"
        ]
    },
    
    "friendly": {
        "name": "Michael Thompson",
        "title": "Angel Investor & Former Entrepreneur",
        "personality": "Warm, empathetic, and genuinely supportive. Built and sold two successful startups before becoming an angel investor. Understands the emotional rollercoaster of entrepreneurship and the personal sacrifices involved. Focuses on the human element behind the business. Values passion, resilience, and authentic storytelling. Known for providing mentorship beyond just capital.",
        "questioning_style": "Story-driven approach that explores the founder's journey and motivation. Seeks to understand the 'why' behind the business. Focuses on vision, passion, and team dynamics. Uses empathetic listening to build rapport and trust.",
        "cognitive_approach": "People-first evaluation focusing on founder-market fit. Assesses emotional intelligence and leadership potential. Evaluates team chemistry and cultural alignment. Considers long-term vision and mission alignment.",
        "communication_patterns": "Uses encouraging language and positive reinforcement. Asks open-ended questions about experiences and learnings. Shares relevant personal anecdotes. Maintains warm, conversational tone throughout.",
        "decision_triggers": "Passionate founder with clear vision, strong team dynamics, compelling origin story, mission-driven approach, coachable leadership",
        "red_flags": "Lack of passion or conviction, poor team dynamics, unclear vision, unwillingness to learn, purely profit-driven motivation",
        "sample_questions": [
            "What inspired you to start this company, and what keeps you motivated during tough times?",
            "Tell me about a pivotal moment when you knew this was the right path for you.",
            "How has your team come together around this vision, and what's the story behind your partnership?"
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
    """Base class for improved stage-specific agents with single-question focus"""
    
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
        """Enhanced base prompt template with detailed behavioral instructions"""
        template = f"""You are {self.persona_info['name']}, {self.persona_info['title']}.

CORE IDENTITY & BACKGROUND:
- Personality: {self.persona_info['personality']}
- Questioning Methodology: {self.persona_info['questioning_style']}
- Cognitive Approach: {self.persona_info.get('cognitive_approach', 'Standard analytical approach')}
- Communication Style: {self.persona_info.get('communication_patterns', 'Professional and direct')}

CURRENT PITCH STAGE: {self.stage.value.replace('_', ' ').title()}

BEHAVIORAL FRAMEWORK:
1. QUESTION DISCIPLINE: Ask EXACTLY ONE focused question per response. Multiple questions dilute focus and overwhelm founders.
2. RESPONSE COMPLETENESS: Assess if the founder's answer fully addresses your question before proceeding. Incomplete answers require clarification.
3. STAGE ADHERENCE: Stay strictly within the {self.stage.value.replace('_', ' ')} topic boundaries. Do not jump ahead to future stages or revisit past ones.
4. PERSONALITY CONSISTENCY: Every word must reflect your established personality, decision triggers, and communication patterns.
5. PROGRESSIVE DEPTH: Each follow-up question should build deeper understanding within the current stage.

CONVERSATION ANALYSIS FRAMEWORK:
- Evaluate founder's response for completeness, specificity, and credibility
- Identify gaps, vague statements, or unsupported claims that need clarification
- Assess whether the response demonstrates understanding of the topic area
- Determine if sufficient information has been gathered for this stage

RESPONSE QUALITY INDICATORS:
✓ Specific data points and metrics
✓ Clear examples and evidence
✓ Demonstrated understanding of challenges
✓ Realistic and well-reasoned explanations

⚠ Red Flags to Address:
- Vague or generic responses
- Unsupported claims or assumptions
- Lack of specific examples or data
- Evasive answers to direct questions

CONVERSATION CONTEXT:
{{chat_history}}

FOUNDER'S LATEST RESPONSE: {{founder_input}}

RESPONSE GENERATION INSTRUCTIONS:
1. First, analyze the founder's response against the quality indicators above
2. If the response is incomplete or vague, ask a clarifying question about the same topic
3. If the response is complete, ask the next logical question within the {self.stage.value.replace('_', ' ')} stage
4. Ensure your question reflects your personality's decision triggers and red flags
5. Keep the question focused, specific, and aligned with your questioning methodology

Generate your response as {self.persona_info['name']} would, maintaining perfect character consistency:"""
        
        return PromptTemplate(
            input_variables=["chat_history", "founder_input"],
            template=template
        )
    
    def _get_stage_questions(self) -> List[str]:
        """Get stage-specific questions - to be overridden by specific agents"""
        return []
    
    def _create_chain(self) -> LLMChain:
        """Create LangChain with response length enforcement"""
        return LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )
    
    def respond(self, founder_input: str, context: dict) -> str:
        """Generate response with length enforcement"""
        try:
            # Get initial response
            response = self.chain.predict(
                founder_input=founder_input,
                chat_history=context.get('chat_history', '')
            ).strip()
            
            # Enforce single question and length limit
            if '?' in response:
                # Split by question marks and take only the first question
                questions = response.split('?')
                response = questions[0].strip() + '?'
            
            # Ensure response is under 15 words
            words = response.split()
            if len(words) > 15:
                response = ' '.join(words[:15])
                if not response.endswith('?'):
                    response += '?'
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Could you elaborate on that?"

class GreetingAgent(ImprovedStageAgent):
    """Specialized agent for greeting stage"""
    
    def _create_stage_specific_prompt(self) -> PromptTemplate:
        template = f"""You are {self.persona_info['name']}, {self.persona_info['title']}.

GREETING STAGE OBJECTIVES:
Primary Goal: Establish rapport and gather foundational information (name, company, brief overview)
Secondary Goal: Set the tone for the entire pitch session based on your personality

PERSONALITY CALIBRATION:
- Core Traits: {self.persona_info['personality']}
- Communication Style: {self.persona_info.get('communication_patterns', 'Professional and engaging')}
- First Impression Strategy: {self.persona_info.get('decision_triggers', 'Build trust and assess founder confidence')}

GREETING STAGE FRAMEWORK:
1. RAPPORT BUILDING: Create immediate connection while maintaining your authentic personality
2. INFORMATION GATHERING: Systematically collect name, company, and initial context
3. TONE SETTING: Establish the interaction style that will continue throughout the session
4. ENGAGEMENT ASSESSMENT: Gauge founder's communication style and confidence level

RESPONSE CONSTRAINTS:
- Maximum 15 words per response
- EXACTLY ONE question mark (?) per response
- No compound or multi-part questions
- Direct and purposeful language only

CONVERSATION CONTEXT:
{{chat_history}}

FOUNDER'S INPUT: {{founder_input}}

RESPONSE GENERATION PROTOCOL:
1. Analyze what information you already have (name, company, context)
2. Identify the most important missing piece for this stage
3. Craft a question that reflects your personality while gathering that information
4. Ensure the question feels natural and conversational, not interrogative

INFORMATION PRIORITY ORDER:
1st Priority: Founder's name (if not provided)
2nd Priority: Company name (if not provided)  
3rd Priority: Brief company overview/what they do
4th Priority: Transition signal to next stage

Generate your response as {self.persona_info['name']}, staying true to your personality while efficiently gathering greeting-stage information:"""
        
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

PROBLEM/SOLUTION STAGE OBJECTIVES:
Primary Goal: Understand the core problem being solved and evaluate solution effectiveness
Secondary Goal: Assess problem-solution fit and validate market need

EVALUATION FRAMEWORK:
Problem Assessment Criteria:
- Problem significance and urgency
- Target audience pain points
- Current solution inadequacies
- Market validation of problem existence

Solution Assessment Criteria:
- Solution clarity and feasibility
- Problem-solution alignment
- Unique value proposition
- Implementation approach

PERSONALITY-DRIVEN QUESTIONING:
{self.persona_info['name']}'s Approach: {self.persona_info['questioning_style']}
Decision Triggers: {self.persona_info.get('decision_triggers', 'Clear problem validation and innovative solution')}
Red Flags: {self.persona_info.get('red_flags', 'Vague problem definition or unvalidated solution')}

RESPONSE CONSTRAINTS:
- Maximum 15 words per response
- EXACTLY ONE question mark (?) per response
- Focus on either problem OR solution, not both simultaneously
- Align question with your personality's analytical approach

CONVERSATION CONTEXT:
{{chat_history}}

FOUNDER'S RESPONSE: {{founder_input}}

QUESTIONING STRATEGY:
1. Analyze founder's response for problem/solution clarity and evidence
2. Identify the most critical gap in understanding
3. Formulate a question that reflects your personality's approach to validation
4. Ensure the question drives toward concrete, specific information

INFORMATION GATHERING PRIORITIES:
- Problem specificity and evidence
- Solution differentiation and effectiveness  
- Customer validation and feedback
- Implementation feasibility

Generate your response as {self.persona_info['name']}, using your characteristic questioning methodology to probe problem/solution fit:"""
        
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

TARGET MARKET ANALYSIS FRAMEWORK:

STAGE OBJECTIVES:
Primary Goal: Define and validate target customer segments and market opportunity
Secondary Goal: Assess market size, accessibility, and go-to-market strategy

MARKET EVALUATION DIMENSIONS:
1. CUSTOMER SEGMENTATION: Who exactly are the target customers, demographics, psychographics
2. MARKET SIZE & OPPORTUNITY: Total addressable market, serviceable market, realistic capture
3. CUSTOMER VALIDATION: Evidence of customer need, willingness to pay, buying behavior
4. MARKET ACCESSIBILITY: How to reach customers, distribution channels, sales strategy
5. COMPETITIVE LANDSCAPE: Market positioning, differentiation, competitive dynamics

PERSONALITY-DRIVEN ASSESSMENT:
{self.persona_info['name']}'s Approach: {self.persona_info['questioning_style']}
Market Analysis Framework: {self.persona_info.get('cognitive_approach', 'Systematic market opportunity assessment')}
Validation Priorities: {self.persona_info.get('decision_triggers', 'Clear customer definition and validated market need')}

MARKET VALIDATION HIERARCHY:
Tier 1: Paying customers with proven demand
Tier 2: Validated customer interviews and market research
Tier 3: Market analysis with customer feedback
Tier 4: Theoretical market sizing and assumptions

QUESTIONING METHODOLOGY:
1. CUSTOMER SPECIFICITY: Define exact target customer profiles
2. MARKET VALIDATION: Assess evidence of genuine market need
3. SIZE QUANTIFICATION: Understand realistic market opportunity
4. ACCESS STRATEGY: Evaluate customer acquisition approach
5. COMPETITIVE POSITIONING: Assess market positioning strategy

RESPONSE CONSTRAINTS:
- Maximum 15 words per response
- EXACTLY ONE question mark (?) per response
- Focus on customer definition and market validation
- Align with your personality's market assessment approach

CONVERSATION CONTEXT:
{{chat_history}}

FOUNDER'S RESPONSE: {{founder_input}}

RESPONSE GENERATION PROTOCOL:
1. Analyze founder's market description for specificity and validation
2. Identify the most critical gap in market understanding
3. Formulate a question that reflects your personality's approach to market assessment
4. Focus on the most important market validation element

Generate your response as {self.persona_info['name']}, applying your characteristic approach to market evaluation:"""
        
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

BUSINESS MODEL EVALUATION FRAMEWORK:

STAGE OBJECTIVES:
Primary Goal: Understand revenue generation mechanics and financial sustainability
Secondary Goal: Assess monetization strategy effectiveness and scalability

BUSINESS MODEL ANALYSIS DIMENSIONS:
1. REVENUE STREAMS: How money is generated, pricing strategy, revenue mix
2. UNIT ECONOMICS: Customer acquisition cost, lifetime value, contribution margins
3. MONETIZATION STRATEGY: Value capture mechanisms, pricing power, scalability
4. FINANCIAL SUSTAINABILITY: Path to profitability, cash flow dynamics, funding needs
5. MARKET DYNAMICS: Pricing benchmarks, customer willingness to pay, competitive pricing

PERSONALITY-DRIVEN EVALUATION:
{self.persona_info['name']}'s Lens: {self.persona_info['questioning_style']}
Financial Assessment Approach: {self.persona_info.get('cognitive_approach', 'Systematic financial model evaluation')}
Key Validation Points: {self.persona_info.get('decision_triggers', 'Proven revenue model and sustainable unit economics')}

BUSINESS MODEL VALIDATION HIERARCHY:
Tier 1: Proven revenue with positive unit economics
Tier 2: Validated pricing with early revenue
Tier 3: Tested pricing model with customer validation
Tier 4: Theoretical model with market research

QUESTIONING METHODOLOGY:
1. REVENUE CLARITY: Understand exactly how money is made
2. PRICING VALIDATION: Assess customer willingness to pay and pricing strategy
3. UNIT ECONOMICS: Evaluate the fundamental business equation
4. SCALABILITY: Determine if the model can grow efficiently
5. SUSTAINABILITY: Assess long-term financial viability

RESPONSE CONSTRAINTS:
- Maximum 15 words per response
- EXACTLY ONE question mark (?) per response
- Focus on financial mechanics and sustainability
- Align with your personality's financial evaluation approach

CONVERSATION CONTEXT:
{{chat_history}}

FOUNDER'S RESPONSE: {{founder_input}}

RESPONSE GENERATION PROTOCOL:
1. Analyze founder's business model explanation for clarity and validation
2. Identify the most critical gap in financial understanding
3. Formulate a question that reflects your personality's approach to financial assessment
4. Focus on the most important business model validation point

Generate your response as {self.persona_info['name']}, applying your characteristic approach to business model evaluation:"""
        
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

COMPETITIVE LANDSCAPE ANALYSIS FRAMEWORK:

STAGE OBJECTIVES:
Primary Goal: Map competitive landscape and assess sustainable differentiation
Secondary Goal: Evaluate competitive moats and market positioning strategy

COMPETITIVE ANALYSIS DIMENSIONS:
1. DIRECT COMPETITORS: Companies solving the same problem with similar approaches
2. INDIRECT COMPETITORS: Alternative solutions or workarounds customers currently use
3. SUBSTITUTE THREATS: Different approaches that could replace the need entirely
4. COMPETITIVE MOATS: Defensible advantages that prevent easy replication
5. MARKET POSITIONING: How the company differentiates in customer perception

PERSONALITY-DRIVEN EVALUATION:
{self.persona_info['name']}'s Lens: {self.persona_info['questioning_style']}
Analytical Framework: {self.persona_info.get('cognitive_approach', 'Systematic competitive assessment')}
Key Concerns: {self.persona_info.get('red_flags', 'Weak competitive positioning or lack of differentiation')}

COMPETITIVE INTELLIGENCE PRIORITIES:
- Competitor identification and analysis depth
- Differentiation clarity and sustainability  
- Barriers to entry and defensive strategies
- Customer switching costs and loyalty factors
- Market share dynamics and positioning

QUESTIONING METHODOLOGY:
1. Challenge "no competition" claims immediately
2. Probe for both direct and indirect competitive threats
3. Assess competitive advantage sustainability
4. Evaluate market positioning effectiveness
5. Test understanding of competitive dynamics

CONVERSATION CONTEXT:
{{chat_history}}

FOUNDER'S RESPONSE: {{founder_input}}

RESPONSE GENERATION PROTOCOL:
1. Analyze founder's competitive understanding and honesty
2. Identify gaps in competitive analysis or weak differentiation claims
3. Formulate a question that reflects your personality's approach to competitive evaluation
4. Focus on the most critical competitive insight needed

CRITICAL INSTRUCTION: Never accept "we have no competition" - every solution competes with the status quo at minimum.

Generate your response as {self.persona_info['name']}, applying your characteristic analytical rigor to competitive assessment:"""
        
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

TRACTION & GROWTH VALIDATION FRAMEWORK:

STAGE OBJECTIVES:
Primary Goal: Quantify business momentum and validate market acceptance
Secondary Goal: Assess growth sustainability and scalability indicators

TRACTION EVALUATION DIMENSIONS:
1. QUANTITATIVE METRICS: Revenue, users, growth rates, retention, conversion
2. QUALITATIVE INDICATORS: Customer feedback, market validation, product-market fit signals
3. MILESTONE ACHIEVEMENTS: Key business, product, or market milestones reached
4. GROWTH TRAJECTORY: Historical performance and future growth potential
5. MARKET VALIDATION: Evidence of genuine customer demand and willingness to pay

PERSONALITY-DRIVEN ASSESSMENT:
{self.persona_info['name']}'s Approach: {self.persona_info['questioning_style']}
Validation Framework: {self.persona_info.get('cognitive_approach', 'Evidence-based traction assessment')}
Critical Success Factors: {self.persona_info.get('decision_triggers', 'Concrete growth metrics and customer validation')}

TRACTION VALIDATION HIERARCHY:
Tier 1 Evidence: Revenue growth, paying customers, retention rates
Tier 2 Evidence: User growth, engagement metrics, product usage
Tier 3 Evidence: Market feedback, pilot programs, early indicators
Tier 4 Evidence: Interest signals, surveys, preliminary validation

QUESTIONING METHODOLOGY:
1. METRIC SPECIFICITY: Demand concrete numbers, not vague growth claims
2. VALIDATION DEPTH: Probe the quality and sustainability of traction
3. EVIDENCE QUALITY: Distinguish between vanity metrics and meaningful indicators
4. GROWTH SUSTAINABILITY: Assess whether traction is organic and repeatable
5. CUSTOMER VALIDATION: Understand genuine market demand vs. artificial interest

CONVERSATION CONTEXT:
{{chat_history}}

FOUNDER'S RESPONSE: {{founder_input}}

RESPONSE GENERATION PROTOCOL:
1. Analyze founder's traction claims for specificity and credibility
2. Identify the most critical missing validation evidence
3. Formulate a question that reflects your personality's approach to growth assessment
4. Focus on distinguishing between real traction and vanity metrics

CRITICAL INSTRUCTION: Never accept vague growth claims - always push for specific, measurable evidence.

Generate your response as {self.persona_info['name']}, applying your characteristic rigor to traction validation:"""
        
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

TEAM EVALUATION FRAMEWORK:

STAGE OBJECTIVES:
Primary Goal: Assess team composition, experience, and execution capability
Secondary Goal: Evaluate leadership, team dynamics, and scaling potential

TEAM ASSESSMENT DIMENSIONS:
1. FOUNDER QUALIFICATIONS: Relevant experience, domain expertise, track record
2. TEAM COMPOSITION: Key roles filled, skill complementarity, experience depth
3. EXECUTION CAPABILITY: Ability to deliver on business plan, past achievements
4. LEADERSHIP QUALITY: Vision communication, team building, decision making
5. SCALING READINESS: Ability to attract talent, build culture, manage growth

PERSONALITY-DRIVEN EVALUATION:
{self.persona_info['name']}'s Lens: {self.persona_info['questioning_style']}
Team Assessment Approach: {self.persona_info.get('cognitive_approach', 'Systematic team capability evaluation')}
Key Success Indicators: {self.persona_info.get('decision_triggers', 'Experienced team with relevant expertise and proven execution')}

TEAM VALIDATION HIERARCHY:
Tier 1: Proven track record with relevant experience and successful exits
Tier 2: Strong domain expertise with demonstrated execution capability
Tier 3: Relevant experience with some execution evidence
Tier 4: Promising background with limited execution proof

QUESTIONING METHODOLOGY:
1. EXPERIENCE VALIDATION: Assess relevant background and expertise
2. EXECUTION EVIDENCE: Evaluate past achievements and delivery capability
3. TEAM DYNAMICS: Understand collaboration and complementary skills
4. LEADERSHIP ASSESSMENT: Gauge vision, communication, and team building
5. SCALING CAPABILITY: Evaluate ability to attract and manage talent

RESPONSE CONSTRAINTS:
- Maximum 15 words per response
- EXACTLY ONE question mark (?) per response
- Focus on team capability and execution evidence
- Align with your personality's team evaluation approach

CONVERSATION CONTEXT:
{{chat_history}}

FOUNDER'S RESPONSE: {{founder_input}}

RESPONSE GENERATION PROTOCOL:
1. Analyze founder's team description for experience and capability evidence
2. Identify the most critical gap in team assessment
3. Formulate a question that reflects your personality's approach to team evaluation
4. Focus on the most important team validation element

Generate your response as {self.persona_info['name']}, applying your characteristic approach to team assessment:"""
        
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

FUNDING STRATEGY EVALUATION FRAMEWORK:

STAGE OBJECTIVES:
Primary Goal: Understand funding requirements, use of funds, and investment rationale
Secondary Goal: Assess financial planning, milestones, and investor value proposition

FUNDING ASSESSMENT DIMENSIONS:
1. FUNDING AMOUNT: Specific capital requirements, funding round size, timeline
2. USE OF FUNDS: Detailed allocation plan, milestone achievement, ROI expectations
3. FINANCIAL PLANNING: Runway extension, burn rate management, path to profitability
4. MILESTONE MAPPING: Key achievements enabled by funding, measurable outcomes
5. INVESTOR VALUE: Return potential, exit strategy, partnership benefits

PERSONALITY-DRIVEN EVALUATION:
{self.persona_info['name']}'s Approach: {self.persona_info['questioning_style']}
Investment Assessment Framework: {self.persona_info.get('cognitive_approach', 'Systematic funding strategy evaluation')}
Key Decision Factors: {self.persona_info.get('decision_triggers', 'Clear funding rationale with measurable milestones')}

FUNDING VALIDATION HIERARCHY:
Tier 1: Detailed financial model with proven milestones and clear ROI
Tier 2: Specific use of funds with realistic milestones and projections
Tier 3: General funding plan with some milestone definition
Tier 4: Basic funding request with limited planning detail

QUESTIONING METHODOLOGY:
1. AMOUNT JUSTIFICATION: Validate funding amount against specific needs
2. USE SPECIFICITY: Understand detailed allocation and expected outcomes
3. MILESTONE CLARITY: Assess achievable goals and measurable progress
4. FINANCIAL DISCIPLINE: Evaluate burn rate management and efficiency
5. INVESTOR RETURN: Understand value creation and exit potential

RESPONSE CONSTRAINTS:
- Maximum 15 words per response
- EXACTLY ONE question mark (?) per response
- Focus on funding strategy and financial planning
- Align with your personality's investment evaluation approach

CONVERSATION CONTEXT:
{{chat_history}}

FOUNDER'S RESPONSE: {{founder_input}}

RESPONSE GENERATION PROTOCOL:
1. Analyze founder's funding explanation for specificity and planning depth
2. Identify the most critical gap in funding strategy understanding
3. Formulate a question that reflects your personality's approach to investment evaluation
4. Focus on the most important funding validation element

Generate your response as {self.persona_info['name']}, applying your characteristic approach to funding assessment:"""
        
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

STRATEGIC VISION & FUTURE PLANNING FRAMEWORK:

STAGE OBJECTIVES:
Primary Goal: Evaluate long-term vision, strategic planning, and growth potential
Secondary Goal: Assess scalability roadmap, exit strategy, and market expansion plans

FUTURE PLANNING EVALUATION DIMENSIONS:
1. STRATEGIC VISION: Long-term company direction, market impact, industry transformation
2. GROWTH ROADMAP: Product development, market expansion, scaling strategy
3. OPERATIONAL SCALING: Team growth, infrastructure, process optimization
4. MARKET EVOLUTION: Industry trends, competitive dynamics, technology advancement
5. EXIT STRATEGY: Potential outcomes, investor returns, timeline considerations

PERSONALITY-DRIVEN ASSESSMENT:
{self.persona_info['name']}'s Lens: {self.persona_info['questioning_style']}
Strategic Evaluation Approach: {self.persona_info.get('cognitive_approach', 'Systematic strategic planning assessment')}
Vision Validation Criteria: {self.persona_info.get('decision_triggers', 'Clear strategic vision with realistic execution plan')}

STRATEGIC PLANNING VALIDATION HIERARCHY:
Tier 1: Detailed strategic roadmap with proven execution capability
Tier 2: Clear vision with realistic milestones and market understanding
Tier 3: General strategic direction with some planning detail
Tier 4: Basic vision with limited strategic planning depth

QUESTIONING METHODOLOGY:
1. VISION CLARITY: Assess strategic direction and market impact potential
2. EXECUTION REALISM: Evaluate feasibility of growth and scaling plans
3. MARKET DYNAMICS: Understand industry evolution and positioning strategy
4. SCALABILITY ASSESSMENT: Analyze growth potential and operational scaling
5. EXIT POTENTIAL: Evaluate investor return opportunities and timeline

CONVERSATION CONTEXT:
{{chat_history}}

FOUNDER'S RESPONSE: {{founder_input}}

RESPONSE GENERATION PROTOCOL:
1. Analyze founder's strategic vision for clarity and execution feasibility
2. Identify the most critical gap in strategic planning understanding
3. Formulate a question that reflects your personality's approach to strategic evaluation
4. Focus on the most important vision validation element

CRITICAL FOCUS AREAS:
- Strategic thinking depth and market understanding
- Realistic execution planning and milestone setting
- Scalability potential and competitive positioning
- Long-term value creation and exit opportunities

Generate your response as {self.persona_info['name']}, applying your characteristic approach to strategic vision assessment:"""
        
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
        
        # Enhanced response completeness assessment
        assessment_prompt = f"""
        RESPONSE COMPLETENESS EVALUATION TASK
        
        CONTEXT:
        Investor Question: {self.chat_history[-2]['message'] if len(self.chat_history) >= 2 else 'Initial greeting'}
        Founder's Response: {message}
        
        EVALUATION CRITERIA:
        
        1. DIRECT RESPONSIVENESS:
        - Does the response directly address the question asked?
        - Are the key elements of the question covered?
        
        2. INFORMATION QUALITY:
        - Are specific details, examples, or data provided?
        - Is the response substantive rather than superficial?
        - Does it demonstrate understanding of the topic?
        
        3. COMPLETENESS INDICATORS:
        ✓ Specific examples or concrete details
        ✓ Quantitative data or metrics when relevant
        ✓ Clear reasoning or explanation
        ✓ Addresses all parts of the question
        
        4. INCOMPLETENESS INDICATORS:
        ⚠ Vague or generic statements
        ⚠ Evasive or deflecting responses
        ⚠ Lack of specific examples or evidence
        ⚠ Partial answers that ignore key aspects
        
        ASSESSMENT INSTRUCTION:
        Evaluate whether this response provides sufficient information for an investor to understand the founder's position on the topic. Consider both the depth and relevance of the information provided.
        
        RESPONSE FORMAT: Answer with only 'yes' or 'no'
        - 'yes' = Response adequately addresses the question with sufficient detail
        - 'no' = Response lacks detail, specificity, or doesn't fully address the question
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
            name_prompt = f"""
            FOUNDER NAME EXTRACTION TASK
            
            Message: "{message}"
            
            EXTRACTION GUIDELINES:
            Look for self-introduction patterns:
            - "I'm [Name]" / "I am [Name]"
            - "My name is [Name]"
            - "This is [Name]"
            - "[Name] here"
            
            VALIDATION CRITERIA:
            - Must be a proper noun (person's name)
            - Typically 1-3 words
            - Not a company name or title
            - Exclude titles (Mr., Dr., CEO, etc.)
            
            OUTPUT: Return only the person's name or empty string if unclear
            """
            try:
                name = self.llm.invoke(name_prompt).content.strip()
                if name and len(name) > 1 and len(name) < 30 and not any(word in name.lower() for word in ['company', 'corp', 'inc', 'llc']):
                    self.shared_context["founder_name"] = name
            except Exception as e:
                logger.warning(f"Error extracting name: {e}")
        
        # Extract company name
        if not self.shared_context["company_name"] and self.current_stage == PitchStage.GREETING:
            company_prompt = f"""
            COMPANY NAME EXTRACTION TASK
            
            Message: "{message}"
            
            EXTRACTION GUIDELINES:
            Look for company mention patterns:
            - "I work at [Company]"
            - "I'm from [Company]"
            - "My company [Company]"
            - "We're building [Company]"
            - "founder of [Company]"
            
            VALIDATION CRITERIA:
            - Business or organization name
            - Not a person's name
            - May include entity suffixes (Inc, LLC, Corp, etc.)
            - Proper noun format
            
            OUTPUT: Return only the company name or empty string if unclear
            """
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
        initial_greeting = f"Hi! I'm {persona_info['name']}. What's your name?"
        
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