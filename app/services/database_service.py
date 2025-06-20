"""
Database Service for Pitch Simulator
Handles all database operations for sessions, analyses, and logs
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models import (
    PitchAnalysis, PitchSession, ConversationLog, 
    QuickAnalytics, CategoryScore, Strength, Weakness
)

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self, database: AsyncIOMotorDatabase):
        self.db = database
        self.sessions_collection = database.pitch_sessions
        self.analyses_collection = database.pitch_analyses
        self.logs_collection = database.conversation_logs
        self.analytics_collection = database.quick_analytics

    # Session Management
    async def create_session(self, session_data: Dict[str, Any]) -> str:
        """Create a new pitch session"""
        try:
            session = PitchSession(**session_data)
            result = await self.sessions_collection.insert_one(session.dict())
            logger.info(f"Created session: {session.session_id}")
            return session.session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        try:
            session = await self.sessions_collection.find_one({"session_id": session_id})
            if session:
                session['_id'] = str(session['_id'])  # Convert ObjectId to string
            return session
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None

    async def update_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """Update session data"""
        try:
            update_data['updated_at'] = datetime.utcnow()
            result = await self.sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {e}")
            return False

    async def end_session(self, session_id: str, duration_minutes: float) -> bool:
        """Mark session as ended"""
        try:
            update_data = {
                "status": "completed",
                "duration_minutes": duration_minutes,
                "updated_at": datetime.utcnow()
            }
            result = await self.sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}")
            return False

    # Analysis Management
    async def save_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Save pitch analysis to database"""
        try:
            # Convert category_scores to proper format
            if 'category_scores' in analysis_data:
                formatted_categories = {}
                for key, value in analysis_data['category_scores'].items():
                    if isinstance(value, dict):
                        formatted_categories[key] = CategoryScore(**value).dict()
                    else:
                        formatted_categories[key] = value
                analysis_data['category_scores'] = formatted_categories

            # Convert strengths and weaknesses to proper format
            if 'strengths' in analysis_data:
                analysis_data['strengths'] = [
                    Strength(**s).dict() if isinstance(s, dict) else s 
                    for s in analysis_data['strengths']
                ]
            
            if 'weaknesses' in analysis_data:
                analysis_data['weaknesses'] = [
                    Weakness(**w).dict() if isinstance(w, dict) else w 
                    for w in analysis_data['weaknesses']
                ]

            analysis = PitchAnalysis(**analysis_data)
            
            # Use upsert to replace existing analysis for the same session
            result = await self.analyses_collection.replace_one(
                {"session_id": analysis.session_id},
                analysis.dict(),
                upsert=True
            )
            
            # Update session to mark it has analysis
            await self.update_session(analysis.session_id, {
                "has_analysis": True,
                "analysis_id": str(result.upserted_id) if result.upserted_id else None
            })
            
            logger.info(f"Saved analysis for session: {analysis.session_id}")
            return analysis.session_id
            
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            raise

    async def get_analysis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis by session ID"""
        try:
            analysis = await self.analyses_collection.find_one({"session_id": session_id})
            if analysis:
                analysis['_id'] = str(analysis['_id'])  # Convert ObjectId to string
            return analysis
        except Exception as e:
            logger.error(f"Error getting analysis for session {session_id}: {e}")
            return None

    async def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analyses"""
        try:
            cursor = self.analyses_collection.find().sort("generated_at", -1).limit(limit)
            analyses = []
            async for analysis in cursor:
                analysis['_id'] = str(analysis['_id'])
                analyses.append(analysis)
            return analyses
        except Exception as e:
            logger.error(f"Error getting recent analyses: {e}")
            return []

    # Conversation Logs
    async def log_conversation(self, log_data: Dict[str, Any]) -> bool:
        """Log conversation message"""
        try:
            log = ConversationLog(**log_data)
            await self.logs_collection.insert_one(log.dict())
            
            # Update session message count
            await self.sessions_collection.update_one(
                {"session_id": log.session_id},
                {"$inc": {"message_count": 1}}
            )
            
            return True
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
            return False

    async def get_conversation_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation logs for a session"""
        try:
            cursor = self.logs_collection.find({"session_id": session_id}).sort("timestamp", 1)
            logs = []
            async for log in cursor:
                log['_id'] = str(log['_id'])
                logs.append(log)
            return logs
        except Exception as e:
            logger.error(f"Error getting conversation logs for session {session_id}: {e}")
            return []

    # Quick Analytics
    async def save_quick_analytics(self, analytics_data: Dict[str, Any]) -> bool:
        """Save quick analytics data"""
        try:
            analytics = QuickAnalytics(**analytics_data)
            await self.analytics_collection.replace_one(
                {"session_id": analytics.session_id},
                analytics.dict(),
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving quick analytics: {e}")
            return False

    async def get_quick_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get quick analytics for a session"""
        try:
            analytics = await self.analytics_collection.find_one({"session_id": session_id})
            if analytics:
                analytics['_id'] = str(analytics['_id'])
            return analytics
        except Exception as e:
            logger.error(f"Error getting quick analytics for session {session_id}: {e}")
            return None

    # Statistics and Reports
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get overall session statistics"""
        try:
            total_sessions = await self.sessions_collection.count_documents({})
            completed_sessions = await self.sessions_collection.count_documents({"status": "completed"})
            total_analyses = await self.analyses_collection.count_documents({})
            
            # Average scores
            pipeline = [
                {"$group": {
                    "_id": None,
                    "avg_score": {"$avg": "$overall_score"},
                    "max_score": {"$max": "$overall_score"},
                    "min_score": {"$min": "$overall_score"}
                }}
            ]
            
            score_stats = await self.analyses_collection.aggregate(pipeline).to_list(1)
            avg_score = score_stats[0]['avg_score'] if score_stats else 0
            max_score = score_stats[0]['max_score'] if score_stats else 0
            min_score = score_stats[0]['min_score'] if score_stats else 0
            
            return {
                "total_sessions": total_sessions,
                "completed_sessions": completed_sessions,
                "total_analyses": total_analyses,
                "average_score": round(avg_score, 2) if avg_score else 0,
                "highest_score": max_score,
                "lowest_score": min_score
            }
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {}

    async def search_sessions(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search sessions by founder name or company name"""
        try:
            search_filter = {
                "$or": [
                    {"founder_name": {"$regex": query, "$options": "i"}},
                    {"company_name": {"$regex": query, "$options": "i"}},
                    {"session_id": {"$regex": query, "$options": "i"}}
                ]
            }
            
            cursor = self.sessions_collection.find(search_filter).sort("created_at", -1).limit(limit)
            sessions = []
            async for session in cursor:
                session['_id'] = str(session['_id'])
                sessions.append(session)
            return sessions
        except Exception as e:
            logger.error(f"Error searching sessions: {e}")
            return []