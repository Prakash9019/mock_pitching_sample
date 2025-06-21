"""
Migration script to convert conversation logs from individual documents to grouped documents
Run this script to migrate existing conversation logs to the new format
"""
import asyncio
import logging
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate_conversation_logs():
    """Migrate conversation logs from old format to new format"""
    
    # Connect to MongoDB
    connection_string = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
    database_name = os.getenv('DATABASE_NAME', 'pitch_simulator')
    
    client = AsyncIOMotorClient(connection_string)
    db = client[database_name]
    
    try:
        # Get all existing conversation logs
        old_logs = []
        async for log in db.conversation_logs.find():
            old_logs.append(log)
        
        if not old_logs:
            logger.info("No existing conversation logs found. Migration not needed.")
            return
        
        logger.info(f"Found {len(old_logs)} conversation log documents to migrate")
        
        # Group logs by session_id
        sessions_logs = {}
        for log in old_logs:
            session_id = log.get('session_id')
            if session_id:
                if session_id not in sessions_logs:
                    sessions_logs[session_id] = []
                
                # Convert to message format
                message = {
                    'timestamp': log.get('timestamp', datetime.utcnow()),
                    'message_type': log.get('message_type', 'unknown'),
                    'content': log.get('content', ''),
                    'persona': log.get('persona'),
                    'audio_file': log.get('audio_file'),
                    'transcription_confidence': log.get('transcription_confidence')
                }
                sessions_logs[session_id].append(message)
        
        logger.info(f"Grouped logs into {len(sessions_logs)} sessions")
        
        # Create backup collection
        backup_collection_name = f"conversation_logs_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Creating backup in collection: {backup_collection_name}")
        
        # Copy old data to backup
        if old_logs:
            await db[backup_collection_name].insert_many(old_logs)
            logger.info(f"Backup created with {len(old_logs)} documents")
        
        # Clear the original collection
        await db.conversation_logs.delete_many({})
        logger.info("Cleared original conversation_logs collection")
        
        # Insert new format documents
        new_documents = []
        for session_id, messages in sessions_logs.items():
            # Sort messages by timestamp
            messages.sort(key=lambda x: x.get('timestamp', datetime.min))
            
            new_doc = {
                'session_id': session_id,
                'created_at': messages[0].get('timestamp', datetime.utcnow()) if messages else datetime.utcnow(),
                'updated_at': messages[-1].get('timestamp', datetime.utcnow()) if messages else datetime.utcnow(),
                'messages': messages
            }
            new_documents.append(new_doc)
        
        if new_documents:
            await db.conversation_logs.insert_many(new_documents)
            logger.info(f"Inserted {len(new_documents)} new conversation log documents")
        
        # Create new indexes
        await db.conversation_logs.create_index("session_id", unique=True)
        await db.conversation_logs.create_index("created_at")
        await db.conversation_logs.create_index("updated_at")
        logger.info("Created new indexes")
        
        logger.info("Migration completed successfully!")
        logger.info(f"Backup available in collection: {backup_collection_name}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(migrate_conversation_logs())