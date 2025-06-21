"""
Test script to verify the new conversation logs functionality
"""
import asyncio
import logging
from datetime import datetime
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.database import connect_to_mongo, close_mongo_connection, get_database
from app.services.database_service import DatabaseService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_conversation_logs():
    """Test the new conversation logs functionality"""
    
    try:
        # Connect to database
        await connect_to_mongo()
        db = await get_database()
        db_service = DatabaseService(db)
        
        # Test session ID
        test_session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Testing with session ID: {test_session_id}")
        
        # Test 1: Log first message
        logger.info("Test 1: Logging first message")
        result1 = await db_service.log_conversation({
            'session_id': test_session_id,
            'message_type': 'user',
            'content': 'Hello, I want to pitch my startup idea.',
            'persona': None
        })
        logger.info(f"First message logged: {result1}")
        
        # Test 2: Log AI response
        logger.info("Test 2: Logging AI response")
        result2 = await db_service.log_conversation({
            'session_id': test_session_id,
            'message_type': 'ai',
            'content': 'Great! I\'d love to hear about your startup. What problem are you solving?',
            'persona': 'investor'
        })
        logger.info(f"AI response logged: {result2}")
        
        # Test 3: Log another user message
        logger.info("Test 3: Logging another user message")
        result3 = await db_service.log_conversation({
            'session_id': test_session_id,
            'message_type': 'user',
            'content': 'We are solving the problem of inefficient food delivery.',
            'persona': None
        })
        logger.info(f"Second user message logged: {result3}")
        
        # Test 4: Retrieve conversation logs
        logger.info("Test 4: Retrieving conversation logs")
        messages = await db_service.get_conversation_logs(test_session_id)
        logger.info(f"Retrieved {len(messages)} messages:")
        for i, msg in enumerate(messages, 1):
            logger.info(f"  Message {i}: [{msg['message_type']}] {msg['content'][:50]}...")
        
        # Test 5: Get full conversation document
        logger.info("Test 5: Getting full conversation document")
        full_doc = await db_service.get_conversation_log_document(test_session_id)
        if full_doc:
            logger.info(f"Full document has {len(full_doc.get('messages', []))} messages")
            logger.info(f"Created at: {full_doc.get('created_at')}")
            logger.info(f"Updated at: {full_doc.get('updated_at')}")
        
        # Test 6: Get conversation stats
        logger.info("Test 6: Getting conversation statistics")
        stats = await db_service.get_conversation_stats(test_session_id)
        logger.info(f"Conversation stats: {stats}")
        
        # Test 7: Verify only one document exists for this session
        logger.info("Test 7: Verifying document count")
        count = await db.conversation_logs.count_documents({"session_id": test_session_id})
        logger.info(f"Number of documents for session {test_session_id}: {count}")
        
        if count == 1:
            logger.info("✅ SUCCESS: Only one document exists for the session")
        else:
            logger.error(f"❌ FAILURE: Expected 1 document, found {count}")
        
        # Cleanup: Remove test data
        logger.info("Cleaning up test data")
        await db.conversation_logs.delete_one({"session_id": test_session_id})
        logger.info("Test data cleaned up")
        
        logger.info("🎉 All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(test_conversation_logs())