"""
MongoDB Database Configuration and Connection
"""
import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.connection_string = os.getenv(
            'MONGODB_URL', 
            'mongodb://localhost:27017'
        )
        self.database_name = os.getenv('DATABASE_NAME', 'pitch_simulator')
        
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            self.database = self.client[self.database_name]
            
            # Test the connection
            await self.client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB: {self.database_name}")
            
            # Create indexes for better performance
            await self.create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Indexes for pitch_sessions collection
            await self.database.pitch_sessions.create_index("session_id", unique=True)
            await self.database.pitch_sessions.create_index("created_at")
            await self.database.pitch_sessions.create_index("founder_name")
            
            # Indexes for pitch_analyses collection
            await self.database.pitch_analyses.create_index("session_id", unique=True)
            await self.database.pitch_analyses.create_index("generated_at")
            await self.database.pitch_analyses.create_index("overall_score")
            
            # Indexes for conversation_logs collection
            await self.database.conversation_logs.create_index("session_id")
            await self.database.conversation_logs.create_index("timestamp")
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")

    def get_database(self):
        """Get database instance"""
        return self.database
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("Database connection closed")

# Global database manager instance
db_manager = DatabaseManager()

async def get_database():
    """Dependency to get database instance"""
    return db_manager.get_database()

# Database connection functions
async def connect_to_mongo():
    """Connect to MongoDB on startup"""
    await db_manager.connect()

async def close_mongo_connection():
    """Close MongoDB connection on shutdown"""
    await db_manager.disconnect()