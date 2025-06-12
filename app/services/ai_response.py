import os
import logging
import google.generativeai as genai
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define investor personas
INVESTOR_PERSONAS = {
    "skeptical": """You are a skeptical venture capital investor evaluating a startup pitch. 
    You focus heavily on business metrics, market validation, and financial projections. 
    You ask tough questions about unit economics, customer acquisition costs, and the path to profitability.
    You're not easily impressed by technology alone and want to see clear evidence of product-market fit.
    Keep your responses concise, direct, and focused on business fundamentals.""",
    
    "technical": """You are a technically-focused venture capital investor evaluating a startup pitch.
    You care deeply about the technology stack, scalability, and technical differentiation.
    You ask detailed questions about the architecture, technical challenges, and IP protection.
    While you understand business metrics, you're most interested in technical innovation and execution.
    Keep your responses focused on technical aspects while being respectful but direct.""",
    
    "friendly": """You are a supportive but thorough venture capital investor evaluating a startup pitch.
    You maintain a positive and encouraging tone while still asking substantive questions.
    You focus on the founder's vision, team composition, and growth strategy.
    You're interested in understanding the founder's thought process and adaptability.
    Keep your responses constructive, thoughtful, and aimed at bringing out the best in the founder."""
}

def generate_investor_response(transcript: str, persona: str = "skeptical") -> str:
    """
    Generate an AI investor response based on the transcribed pitch.
    
    Args:
        transcript: The transcribed text from the founder's pitch
        persona: The type of investor persona to simulate (default: skeptical)
        
    Returns:
        The generated investor response
    """
    logger.info(f"Generating investor response with persona: {persona}")
    
    # Get the appropriate persona prompt
    persona_prompt = INVESTOR_PERSONAS.get(persona, INVESTOR_PERSONAS["skeptical"])
    
    try:
        # Construct the full prompt
        prompt = f"""{persona_prompt}

FOUNDER'S PITCH:
{transcript}

Based on this pitch, respond as the investor. Ask relevant questions, provide feedback, 
or request clarification on aspects of the business that need more explanation. 
Focus on business metrics and market validation.
"""

        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        investor_response = response.text
        
        logger.info(f"Generated investor response: {investor_response[:100]}...")
        return investor_response
    
    except Exception as e:
        logger.error(f"Error generating investor response: {str(e)}")
        # Return a placeholder for testing if generation fails
        return """Thank you for your pitch. I have some concerns about your business model. 
        Could you elaborate on your customer acquisition strategy and unit economics? 
        What are your CAC and LTV metrics? How do you plan to achieve profitability?
        This is a placeholder response for testing purposes."""