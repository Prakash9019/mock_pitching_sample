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
    PitchAnalysis, PitchSession, ConversationLog, ConversationMessage,
    QuickAnalytics, CategoryScore, Strength, Weakness, AudioConversationData
)

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self, database: AsyncIOMotorDatabase):
        self.db = database
        self.sessions_collection = database.pitch_sessions if database is not None else None
        self.analyses_collection = database.pitch_analyses if database is not None else None
        self.logs_collection = database.conversation_logs if database is not None else None
        self.analytics_collection = database.quick_analytics if database is not None else None
        self.audio_conversations_collection = database.audio_conversations if database is not None else None
    
    def _check_connection(self):
        """Check if database connection is available"""
        if self.db is None:
            raise Exception("Database connection not available")
        return True

    # Session Management
    async def create_session(self, session_data: Dict[str, Any]) -> str:
        """Create a new pitch session"""
        try:
            logger.debug(f"Creating session with data: {session_data}")
            self._check_connection()
            session = PitchSession(**session_data)
            result = await self.sessions_collection.insert_one(session.dict())
            logger.info(f"Created session: {session.session_id}")
            return session.session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}", exc_info=True)
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
    def _transform_analysis_data(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform AI-generated analysis data to match PitchAnalysis model"""
        transformed = analysis_data.copy()
        
        # Ensure required fields have default values
        required_defaults = {
            'overall_score': 0,
            'overall_rating': 'Need to Improve',
            'overall_description': 'Analysis generated',
            'confidence_level': 'Developing',
            'pitch_readiness': 'In Progress',
            'session_duration_minutes': 0.0,
            'completion_percentage': 0.0,
            'category_scores': {},
            'strengths': [],
            'weaknesses': [],
            'key_recommendations': [],
            'investor_perspective': 'Analysis in progress',
            'next_steps': []
        }
        
        # Apply defaults for missing fields
        for key, default_value in required_defaults.items():
            if key not in transformed:
                transformed[key] = default_value
        
        # Ensure strengths and weaknesses are in correct format
        if 'strengths' in transformed and transformed['strengths']:
            formatted_strengths = []
            for strength in transformed['strengths']:
                if isinstance(strength, str):
                    formatted_strengths.append({
                        'area': 'General',
                        'description': strength,
                        'score': None
                    })
                elif isinstance(strength, dict):
                    formatted_strengths.append(strength)
            transformed['strengths'] = formatted_strengths
        
        if 'weaknesses' in transformed and transformed['weaknesses']:
            formatted_weaknesses = []
            for weakness in transformed['weaknesses']:
                if isinstance(weakness, str):
                    formatted_weaknesses.append({
                        'area': 'General',
                        'description': weakness,
                        'improvement': None
                    })
                elif isinstance(weakness, dict):
                    formatted_weaknesses.append(weakness)
            transformed['weaknesses'] = formatted_weaknesses
        
        return transformed

    async def save_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Save pitch analysis to database"""
        try:
            logger.info(f"Starting to save analysis with keys: {list(analysis_data.keys())}")
            
            # Transform the analysis data to match the model
            transformed_data = self._transform_analysis_data(analysis_data)
            
            # Convert category_scores to proper format
            if 'category_scores' in transformed_data:
                formatted_categories = {}
                for key, value in transformed_data['category_scores'].items():
                    if isinstance(value, dict):
                        formatted_categories[key] = CategoryScore(**value).dict()
                    else:
                        formatted_categories[key] = value
                transformed_data['category_scores'] = formatted_categories

            # Convert strengths and weaknesses to proper format
            if 'strengths' in transformed_data:
                transformed_data['strengths'] = [
                    Strength(**s).dict() if isinstance(s, dict) else s 
                    for s in transformed_data['strengths']
                ]
            
            if 'weaknesses' in transformed_data:
                transformed_data['weaknesses'] = [
                    Weakness(**w).dict() if isinstance(w, dict) else w 
                    for w in transformed_data['weaknesses']
                ]

            try:
                analysis = PitchAnalysis(**transformed_data)
                logger.info(f"PitchAnalysis model created successfully for session: {analysis.session_id}")
            except Exception as model_error:
                logger.error(f"Failed to create PitchAnalysis model: {model_error}")
                logger.error(f"Transformed data keys: {list(transformed_data.keys())}")
                
                # Try to save as raw document if model validation fails
                logger.warning("Attempting to save analysis as raw document")
                raw_analysis = {
                    **transformed_data,
                    "model_validation_failed": True,
                    "validation_error": str(model_error),
                    "saved_at": datetime.utcnow()
                }
                
                result = await self.analyses_collection.replace_one(
                    {"session_id": transformed_data.get("session_id")},
                    raw_analysis,
                    upsert=True
                )
                
                logger.info(f"Raw analysis saved for session: {transformed_data.get('session_id')}")
                return transformed_data.get("session_id", "unknown")
            
            # Use upsert to replace existing analysis for the same session
            result = await self.analyses_collection.replace_one(
                {"session_id": analysis.session_id},
                analysis.dict(),
                upsert=True
            )
            logger.info(f"Database operation completed. Upserted: {result.upserted_id}, Modified: {result.modified_count}")
            
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
            logger.debug(f"Logging conversation with data: {log_data}")
            session_id = log_data.get('session_id')
            if not session_id:
                raise ValueError("session_id is required")
            
            # Create a conversation message from the log data
            message_data = {k: v for k, v in log_data.items() if k != 'session_id'}
            message = ConversationMessage(**message_data)
            
            # Use upsert to add message to existing conversation log or create new one
            result = await self.logs_collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {"messages": message.dict()},
                    "$set": {"updated_at": datetime.utcnow()},
                    "$setOnInsert": {
                        "session_id": session_id,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            
            # Update session message count
            await self.sessions_collection.update_one(
                {"session_id": session_id},
                {"$inc": {"message_count": 1}}
            )
            
            return True
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
            return False

    async def get_conversation_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation messages for a session"""
        try:
            conversation_log = await self.logs_collection.find_one({"session_id": session_id})
            if conversation_log and 'messages' in conversation_log:
                # Sort messages by timestamp
                messages = sorted(conversation_log['messages'], key=lambda x: x.get('timestamp', datetime.min))
                return messages
            return []
        except Exception as e:
            logger.error(f"Error getting conversation logs for session {session_id}: {e}")
            return []

    async def get_conversation_log_document(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the full conversation log document for a session"""
        try:
            conversation_log = await self.logs_collection.find_one({"session_id": session_id})
            if conversation_log:
                conversation_log['_id'] = str(conversation_log['_id'])
            return conversation_log
        except Exception as e:
            logger.error(f"Error getting conversation log document for session {session_id}: {e}")
            return None

    async def get_conversation_stats(self, session_id: str) -> Dict[str, Any]:
        """Get conversation statistics for a session"""
        try:
            conversation_log = await self.logs_collection.find_one({"session_id": session_id})
            if not conversation_log or 'messages' not in conversation_log:
                return {"message_count": 0, "user_messages": 0, "ai_messages": 0, "system_messages": 0}
            
            messages = conversation_log['messages']
            stats = {
                "message_count": len(messages),
                "user_messages": len([m for m in messages if m.get('message_type') == 'user']),
                "ai_messages": len([m for m in messages if m.get('message_type') == 'ai']),
                "system_messages": len([m for m in messages if m.get('message_type') == 'system']),
                "first_message_time": messages[0].get('timestamp') if messages else None,
                "last_message_time": messages[-1].get('timestamp') if messages else None
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting conversation stats for session {session_id}: {e}")
            return {"message_count": 0, "user_messages": 0, "ai_messages": 0, "system_messages": 0}
            return []
        except Exception as e:
            logger.error(f"Error getting conversation logs for session {session_id}: {e}")
            return []

    async def get_conversation_log_document(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the full conversation log document for a session"""
        try:
            conversation_log = await self.logs_collection.find_one({"session_id": session_id})
            if conversation_log:
                conversation_log['_id'] = str(conversation_log['_id'])
            return conversation_log
        except Exception as e:
            logger.error(f"Error getting conversation log document for session {session_id}: {e}")
            return None

    async def get_conversation_stats(self, session_id: str) -> Dict[str, Any]:
        """Get conversation statistics for a session"""
        try:
            conversation_log = await self.logs_collection.find_one({"session_id": session_id})
            if not conversation_log or 'messages' not in conversation_log:
                return {"message_count": 0, "user_messages": 0, "ai_messages": 0, "system_messages": 0}
            
            messages = conversation_log['messages']
            stats = {
                "message_count": len(messages),
                "user_messages": len([m for m in messages if m.get('message_type') == 'user']),
                "ai_messages": len([m for m in messages if m.get('message_type') == 'ai']),
                "system_messages": len([m for m in messages if m.get('message_type') == 'system']),
                "first_message_time": messages[0].get('timestamp') if messages else None,
                "last_message_time": messages[-1].get('timestamp') if messages else None
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting conversation stats for session {session_id}: {e}")
            return {"message_count": 0, "user_messages": 0, "ai_messages": 0, "system_messages": 0}

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

    # Audio Conversation Management
    async def save_audio_conversation(self, audio_data: Dict[str, Any]) -> str:
        """Save audio conversation data"""
        try:
            self._check_connection()
            audio_conversation = AudioConversationData(**audio_data)
            result = await self.audio_conversations_collection.insert_one(audio_conversation.dict())
            
            # Update session to reference audio conversation
            await self.update_session(audio_conversation.session_id, {
                "audio_conversation_id": str(result.inserted_id),
                "has_audio_recording": True
            })
            
            logger.info(f"Saved audio conversation for session: {audio_conversation.session_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving audio conversation: {e}", exc_info=True)
            raise

    async def get_audio_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get audio conversation data by session ID"""
        try:
            audio_conversation = await self.audio_conversations_collection.find_one({"session_id": session_id})
            if audio_conversation:
                audio_conversation['_id'] = str(audio_conversation['_id'])
            return audio_conversation
        except Exception as e:
            logger.error(f"Error getting audio conversation for session {session_id}: {e}")
            return None

    async def update_audio_conversation(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """Update audio conversation data"""
        try:
            result = await self.audio_conversations_collection.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating audio conversation for session {session_id}: {e}")
            return False

    async def get_sessions_with_audio(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get sessions that have audio recordings"""
        try:
            cursor = self.sessions_collection.find({"has_audio_recording": True}).sort("created_at", -1).limit(limit)
            sessions = []
            async for session in cursor:
                session['_id'] = str(session['_id'])
                # Get audio conversation data
                audio_data = await self.get_audio_conversation(session['session_id'])
                if audio_data:
                    session['audio_conversation'] = audio_data
                sessions.append(session)
            return sessions
        except Exception as e:
            logger.error(f"Error getting sessions with audio: {e}")
            return []

    async def delete_audio_conversation(self, session_id: str) -> bool:
        """Delete audio conversation data"""
        try:
            result = await self.audio_conversations_collection.delete_one({"session_id": session_id})
            
            # Update session to remove audio reference
            if result.deleted_count > 0:
                await self.update_session(session_id, {
                    "audio_conversation_id": None,
                    "has_audio_recording": False
                })
            
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting audio conversation for session {session_id}: {e}")
            return False