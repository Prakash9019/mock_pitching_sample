from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ElementAnalysis(BaseModel):
    """Schema for pitch element analysis."""
    score: float = Field(..., description="Score out of 100 for this element")
    feedback: str = Field(..., description="Detailed feedback about this element")
    strengths: List[str] = Field(..., description="List of key strengths")
    weaknesses: List[str] = Field(..., description="List of areas for improvement")

@dataclass
class PitchElement:
    """Data class for storing pitch element analysis."""
    name: str
    score: float
    feedback: str
    strengths: List[str]
    weaknesses: List[str]

@dataclass
class PitchAnalysis:
    """Data class for storing complete pitch analysis."""
    overall_score: float
    elements: dict[str, PitchElement]
    timestamp: datetime
    session_id: str
    transcript: str
    recommendations: List[str]

class PitchAnalyzer:
    """Analyzes pitch content and provides structured feedback."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize the analyzer with an LLM."""
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=ElementAnalysis)
        
        # Create a template with explicit format instructions
        self.template = PromptTemplate(
            template="""
            Analyze the following pitch transcript for the {element} element.
            Provide:
            1. A score out of 100
            2. Specific feedback
            3. Key strengths
            4. Areas for improvement

            Transcript: {transcript}

            {format_instructions}
            """,
            input_variables=["element", "transcript"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
    def analyze_pitch(self, transcript: str, session_id: str) -> PitchAnalysis:
        """Analyze a pitch transcript and return structured feedback."""
        try:
            # Define elements to analyze
            elements = [
                "problem_statement",
                "solution",
                "market_size",
                "business_model",
                "competition",
                "team",
                "traction",
                "funding_ask"
            ]
            
            # Analyze each element
            element_analyses = {}
            for element in elements:
                element_analysis = self._analyze_element(element, transcript)
                element_analyses[element] = element_analysis
            
            # Calculate overall score
            overall_score = sum(elem.score for elem in element_analyses.values()) / len(elements)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(element_analyses)
            
            return PitchAnalysis(
                overall_score=overall_score,
                elements=element_analyses,
                timestamp=datetime.utcnow(),
                session_id=session_id,
                transcript=transcript,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing pitch: {str(e)}")
            raise

    def _analyze_element(self, element: str, transcript: str) -> PitchElement:
        """Analyze a specific element of the pitch using structured output."""
        try:
            # Format the prompt with the template
            prompt = self.template.format(
                element=element.replace('_', ' '),
                transcript=transcript
            )

            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Extract content from response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Parse the content
            analysis = self.parser.parse(content)
            
            return PitchElement(
                name=element,
                score=analysis.score,
                feedback=analysis.feedback,
                strengths=analysis.strengths,
                weaknesses=analysis.weaknesses
            )

        except Exception as e:
            logger.error(f"Error analyzing element {element}: {str(e)}")
            # Return a default analysis if parsing fails
            return PitchElement(
                name=element,
                score=50.0,  # Default middle score
                feedback="Unable to analyze this element properly.",
                strengths=["Analysis not available"],
                weaknesses=["Analysis not available"]
            )

    def _generate_recommendations(self, element_analyses: dict[str, PitchElement]) -> List[str]:
        """Generate recommendations based on element analyses."""
        try:
            # Find elements with low scores
            weak_elements = [
                (name, elem) for name, elem in element_analyses.items()
                if elem.score < 60
            ]
            
            if not weak_elements:
                return ["Overall strong pitch! Consider adding more specific metrics and examples."]
            
            # Generate specific recommendations for weak elements
            recommendations = []
            for name, elem in weak_elements:
                if elem.weaknesses:
                    recommendations.append(
                        f"Improve {name.replace('_', ' ')}: {elem.weaknesses[0]}"
                    )
            
            return recommendations[:3]  # Return top 3 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate specific recommendations."]

    def get_analysis_summary(self, analysis: PitchAnalysis) -> dict:
        """Convert PitchAnalysis to a dictionary format."""
        return {
            "overall_score": analysis.overall_score,
            "elements": {
                name: {
                    "score": elem.score,
                    "feedback": elem.feedback,
                    "strengths": elem.strengths,
                    "weaknesses": elem.weaknesses
                }
                for name, elem in analysis.elements.items()
            },
            "recommendations": analysis.recommendations,
            "timestamp": analysis.timestamp.isoformat(),
            "session_id": analysis.session_id
        } 