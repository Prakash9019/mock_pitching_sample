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
        """Connect to MongoDB with graceful fallback and retry mechanism"""
        # Check if MongoDB is explicitly disabled
        if os.getenv('DISABLE_MONGODB', '').lower() == 'true':
            logger.info("🚫 MongoDB explicitly disabled via DISABLE_MONGODB environment variable")
            logger.info("🎵 Application will run with TTS functionality only")
            return
        
        # Retry connection up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to MongoDB (attempt {attempt + 1}/{max_retries}): {self.connection_string}")
                
                self.client = AsyncIOMotorClient(
                    self.connection_string,
                    serverSelectionTimeoutMS=10000,  # Increased to 10 seconds
                    connectTimeoutMS=10000,
                    socketTimeoutMS=10000,
                    maxPoolSize=10,
                    minPoolSize=1,
                    retryWrites=True,
                    w='majority'
                )
                self.database = self.client[self.database_name]
                
                # Test the connection with longer timeout
                await asyncio.wait_for(
                    self.client.admin.command('ping'), 
                    timeout=10.0
                )
                logger.info(f"✅ Successfully connected to MongoDB: {self.database_name}")
                
                # Create indexes for better performance
                await self.create_indexes()
                return  # Success - exit retry loop
                
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ MongoDB connection timed out (attempt {attempt + 1}/{max_retries})")
                self._cleanup_connection()
                if attempt < max_retries - 1:
                    logger.info("🔄 Retrying connection in 2 seconds...")
                    await asyncio.sleep(2)
                else:
                    logger.warning("⚠️ All connection attempts failed - running without database")
                    logger.info("🎵 TTS and conversation features will work normally")
                
            except Exception as e:
                logger.warning(f"⚠️ MongoDB connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                self._cleanup_connection()
                if attempt < max_retries - 1:
                    logger.info("🔄 Retrying connection in 2 seconds...")
                    await asyncio.sleep(2)
                else:
                    logger.warning("⚠️ All connection attempts failed - running without database")
                    logger.info("📝 Application will continue without database functionality")
                    logger.info("🎵 TTS and conversation features will work normally")
    
    def _cleanup_connection(self):
        """Clean up failed connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.database = None
    
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
            await self.database.conversation_logs.create_index("session_id", unique=True)
            await self.database.conversation_logs.create_index("created_at")
            await self.database.conversation_logs.create_index("updated_at")
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")

    def get_database(self):
        """Get database instance"""
        return self.database
    
    async def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            if not self.client or not self.database:
                return False
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
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

def get_database_service():
    """Get database service instance"""
    from app.services.database_service import DatabaseService
    database = db_manager.get_database()
    if database is not None:
        return DatabaseService(database)
    return None

# Database connection functions
async def connect_to_mongo():
    """Connect to MongoDB on startup"""
    await db_manager.connect()

async def close_mongo_connection():
    """Close MongoDB connection on shutdown"""
    await db_manager.disconnect()

async def test_database_connection():
    """Test database connection manually"""
    try:
        logger.info("🧪 Testing database connection...")
        
        # Create a temporary client for testing
        test_client = AsyncIOMotorClient(
            db_manager.connection_string,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
        
        # Test ping
        await asyncio.wait_for(
            test_client.admin.command('ping'),
            timeout=5.0
        )
        
        # Test database access
        test_db = test_client[db_manager.database_name]
        collections = await test_db.list_collection_names()
        
        test_client.close()
        
        logger.info(f"✅ Database connection test successful!")
        logger.info(f"📊 Database: {db_manager.database_name}")
        logger.info(f"📁 Collections: {len(collections)} found")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False