# MongoDB Database Integration

## Overview
Successfully integrated MongoDB database with the Moke Pitch application to store and manage pitch sessions, analyses, and conversation logs.

## Features Implemented

### 1. Database Models (`app/models.py`)
- **PitchSession**: Stores session information (founder, company, persona, timestamps)
- **PitchAnalysis**: Stores detailed pitch analysis results with scores and feedback
- **ConversationLog**: Stores conversation history between users and AI
- **CategoryScore**: Nested model for analysis category scores
- **Strength/Weakness**: Nested models for feedback items

### 2. Database Connection (`app/database.py`)
- **DatabaseManager**: Handles MongoDB connection and initialization
- **Connection Management**: Async connection with proper error handling
- **Index Creation**: Automatic creation of database indexes for performance
- **Environment Configuration**: Supports both local and cloud MongoDB instances

### 3. Database Service (`app/services/database_service.py`)
- **Session Management**: Create, retrieve, update, and search sessions
- **Analysis Storage**: Save and retrieve pitch analysis results
- **Conversation Logging**: Store and retrieve conversation history
- **Statistics**: Generate database statistics and reports
- **Search Functionality**: Search sessions by founder name, company, or session ID

### 4. API Endpoints (`main.py`)
New database management endpoints added:
- `GET /api/sessions` - List all sessions with pagination
- `GET /api/sessions/{session_id}` - Get detailed session information
- `GET /api/analyses` - Get recent pitch analyses
- `GET /api/stats` - Get database statistics
- `GET /api/search` - Search sessions

## Database Collections

### pitch_sessions
```json
{
  "_id": "ObjectId",
  "session_id": "string",
  "founder_name": "string",
  "company_name": "string",
  "persona": "string",
  "created_at": "datetime",
  "updated_at": "datetime",
  "status": "string",
  "analysis_id": "string",
  "has_analysis": "boolean"
}
```

### pitch_analyses
```json
{
  "_id": "ObjectId",
  "session_id": "string",
  "overall_score": "number",
  "confidence_level": "string",
  "pitch_readiness": "string",
  "stage_scores": "object",
  "strengths": ["object"],
  "weaknesses": ["object"],
  "key_recommendations": ["string"],
  "next_steps": ["string"],
  "generated_at": "datetime"
}
```

### conversation_logs
```json
{
  "_id": "ObjectId",
  "session_id": "string",
  "timestamp": "datetime",
  "message_type": "string",
  "content": "string",
  "persona": "string",
  "audio_file": "string",
  "transcription_confidence": "number"
}
```

## Database Indexes
- **pitch_sessions**: session_id (unique), created_at, founder_name, company_name
- **pitch_analyses**: session_id (unique), generated_at, overall_score
- **conversation_logs**: session_id, timestamp

## Configuration

### Environment Variables
- `MONGODB_URL`: MongoDB connection string (required)
- `DATABASE_NAME`: Database name (default: "moke_pitch")

### Dependencies Added
- `pymongo==4.13.2`: MongoDB Python driver
- `motor==3.7.1`: Async MongoDB driver for Python
- `dnspython==2.7.0`: DNS toolkit for Python (required for MongoDB Atlas)

## Integration Points

### 1. Session Creation
- Automatically creates database session when new pitch session starts
- Stores founder and company information
- Tracks session status and timestamps

### 2. Conversation Logging
- Real-time logging of all user-AI interactions
- Stores message content, type, and metadata
- Supports audio file references and transcription confidence

### 3. Analysis Storage
- Saves complete pitch analysis results to database
- Links analysis to session for easy retrieval
- Stores detailed scores, feedback, and recommendations

### 4. Data Retrieval
- Fast retrieval of session information
- Conversation history playback
- Analysis results with formatted reports

## Performance Optimizations
- Database indexes for fast queries
- Pagination support for large datasets
- Efficient search with text indexes
- Connection pooling with Motor async driver

## Error Handling
- Graceful fallback when database is unavailable
- Comprehensive error logging
- Connection retry logic
- Data validation with Pydantic models

## Testing
- Comprehensive test suite for all database operations
- Connection testing and validation
- CRUD operation verification
- Performance benchmarking

## Usage Examples

### Creating a Session
```python
session_data = {
    "session_id": "unique_session_id",
    "founder_name": "John Doe",
    "company_name": "TechCorp",
    "persona": "friendly"
}
session_id = await db_service.create_session(session_data)
```

### Logging Conversation
```python
log_data = {
    "session_id": session_id,
    "message_type": "user",
    "content": "Hello, I need help with my pitch",
    "persona": "friendly"
}
await db_service.log_conversation(log_data)
```

### Saving Analysis
```python
analysis_data = {
    "session_id": session_id,
    "overall_score": 85,
    "confidence_level": "High",
    "strengths": [{"category": "Problem", "description": "Clear problem statement"}],
    "weaknesses": [{"category": "Market", "description": "Need more market research"}]
}
await db_service.save_analysis(analysis_data)
```

## Future Enhancements
- Real-time analytics dashboard
- Advanced search with filters
- Data export functionality
- Backup and restore procedures
- Performance monitoring
- User authentication and authorization

## Status
✅ **COMPLETED** - MongoDB integration is fully functional and tested