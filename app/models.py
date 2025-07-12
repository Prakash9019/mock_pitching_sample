"""
Database Models for Pitch Simulator
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class PitchRating(str, Enum):
    VERTX_ASSURED = "Vertx Assured"
    GOOD = "Good"
    SATISFACTORY = "Satisfactory"
    BELOW_AVERAGE = "Below Average"
    NEED_TO_IMPROVE = "Need to Improve"

class CategoryScore(BaseModel):
    score: int = Field(..., ge=0, le=100)
    rating: PitchRating
    description: str

class EngagementScore(CategoryScore):
    talked_count: Optional[int] = None
    listened_count: Optional[int] = None
    talk_percentage: Optional[float] = None
    listen_percentage: Optional[float] = None

class FluencyScore(CategoryScore):
    fillers: Optional[int] = None
    grammar: Optional[int] = None
    vocabulary: Optional[float] = None

class InteractivityScore(CategoryScore):
    conversation_turns: Optional[int] = None
    turn_frequency: Optional[float] = None

class QuestionsAskedScore(CategoryScore):
    total_questions: Optional[int] = None
    questions_per_minute: Optional[float] = None

class FounderPerformance(BaseModel):
    title: str
    description: str

class Strength(BaseModel):
    area: str
    description: str
    score: Optional[int] = Field(None, ge=0, le=10)

class Weakness(BaseModel):
    area: str
    description: str
    improvement: Optional[str] = None

class PitchAnalysis(BaseModel):
    session_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Overall metrics
    overall_score: int = Field(..., ge=0, le=100)
    overall_rating: PitchRating
    overall_description: str
    confidence_level: str
    pitch_readiness: str
    session_duration_minutes: float
    completion_percentage: float
    
    # Session info
    founder_name: Optional[str] = None
    company_name: Optional[str] = None
    persona_used: Optional[str] = None
    
    # Category breakdown
    category_scores: Dict[str, CategoryScore]
    
    # Analysis insights
    strengths: List[Strength]
    weaknesses: List[Weakness]
    key_recommendations: List[str]
    investor_perspective: str
    next_steps: List[str]
    
    # New analysis sections
    founder_performance: Optional[List[FounderPerformance]] = []
    what_worked: Optional[List[str]] = []
    what_didnt_work: Optional[List[str]] = []
    
    # Metadata
    analysis_version: str = "1.0"
    created_by: str = "AI Pitch Simulator"

class PitchSession(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Session details
    founder_name: Optional[str] = None
    company_name: Optional[str] = None
    persona_used: Optional[str] = None
    status: str = "active"  # active, completed, ended
    
    # Session metrics
    duration_minutes: Optional[float] = None
    message_count: int = 0
    topics_covered: List[str] = []
    
    # Audio conversation reference
    audio_conversation_id: Optional[str] = None
    has_audio_recording: bool = False
    
    # Analysis reference
    analysis_id: Optional[str] = None
    has_analysis: bool = False

class ConversationMessage(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_type: str  # user, ai, system
    content: str
    persona: Optional[str] = None
    audio_file: Optional[str] = None
    transcription_confidence: Optional[float] = None

class AudioConversationData(BaseModel):
    """Model for storing audio conversation metadata"""
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Audio file information
    audio_file_url: Optional[str] = None
    audio_filename: Optional[str] = None
    audio_format: str = "wav"  # wav or mp3
    file_size_bytes: Optional[int] = None
    
    # Audio metrics
    total_duration_seconds: Optional[float] = None
    user_speaking_duration: Optional[float] = None
    ai_speaking_duration: Optional[float] = None
    silence_duration: Optional[float] = None
    
    # Segment counts
    user_audio_segments: int = 0
    ai_audio_segments: int = 0
    total_segments: int = 0
    
    # Quality metrics
    audio_quality_score: Optional[float] = None  # 0-1 score
    background_noise_level: Optional[float] = None
    
    # Storage information
    storage_provider: str = "google_cloud_storage"
    bucket_name: Optional[str] = None
    upload_timestamp: Optional[datetime] = None
    
    # Analysis integration
    included_in_analysis: bool = False
    analysis_notes: Optional[str] = None

class ConversationLog(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[ConversationMessage] = []



class QuickAnalytics(BaseModel):
    session_id: str
    overall_score: int
    key_insights: List[str]
    completion_percentage: float
    current_topics: List[str]
    generated_at: datetime = Field(default_factory=datetime.utcnow)