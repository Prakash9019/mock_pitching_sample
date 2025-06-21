# Conversation Logs Update

## Overview
The conversation logs system has been updated to store all messages for a session in a single document instead of creating separate documents for each message. This improves performance and reduces database overhead.

## Changes Made

### 1. Model Updates (`app/models.py`)
- **New**: `ConversationMessage` model for individual messages
- **Updated**: `ConversationLog` model now contains an array of messages
- **Structure**:
  ```python
  class ConversationMessage(BaseModel):
      timestamp: datetime
      message_type: str  # user, ai, system
      content: str
      persona: Optional[str] = None
      audio_file: Optional[str] = None
      transcription_confidence: Optional[float] = None

  class ConversationLog(BaseModel):
      session_id: str
      created_at: datetime
      updated_at: datetime
      messages: List[ConversationMessage] = []
  ```

### 2. Database Service Updates (`app/services/database_service.py`)
- **Updated**: `log_conversation()` now uses upsert to add messages to existing documents
- **Updated**: `get_conversation_logs()` returns sorted messages from a single document
- **New**: `get_conversation_log_document()` returns the full conversation document
- **New**: `get_conversation_stats()` provides conversation statistics

### 3. Database Schema Changes (`app/database.py`)
- **Updated**: Indexes now include `session_id` (unique), `created_at`, and `updated_at`
- **Removed**: Individual `timestamp` index (no longer needed)

### 4. Documentation Updates
- Updated `DATABASE_INTEGRATION.md` with new schema
- Updated database indexes documentation

## Benefits

1. **Reduced Database Overhead**: One document per session instead of one per message
2. **Better Performance**: Fewer database queries and better indexing
3. **Atomic Operations**: All messages for a session are updated atomically
4. **Easier Querying**: Single query to get all conversation data for a session
5. **Better Data Locality**: Related messages are stored together

## Migration

### For Existing Data
If you have existing conversation logs, run the migration script:

```bash
python scripts/migrate_conversation_logs.py
```

This script will:
1. Backup existing data to a timestamped collection
2. Convert old format to new format
3. Create new indexes
4. Preserve all existing data

### For New Installations
No migration needed - the new format will be used automatically.

## Testing

Run the test script to verify functionality:

```bash
python scripts/test_conversation_logs.py
```

## API Changes

### Before (Old Format)
```python
# Each message was a separate document
{
  "_id": "ObjectId",
  "session_id": "session_123",
  "timestamp": "2024-01-01T10:00:00Z",
  "message_type": "user",
  "content": "Hello",
  "persona": null
}
```

### After (New Format)
```python
# All messages for a session in one document
{
  "_id": "ObjectId",
  "session_id": "session_123",
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:05:00Z",
  "messages": [
    {
      "timestamp": "2024-01-01T10:00:00Z",
      "message_type": "user",
      "content": "Hello",
      "persona": null
    },
    {
      "timestamp": "2024-01-01T10:01:00Z",
      "message_type": "ai",
      "content": "Hi there!",
      "persona": "investor"
    }
  ]
}
```

## Usage Examples

### Logging a Conversation Message
```python
await db_service.log_conversation({
    'session_id': 'session_123',
    'message_type': 'user',
    'content': 'Hello, I want to pitch my startup.',
    'persona': None
})
```

### Getting All Messages for a Session
```python
messages = await db_service.get_conversation_logs('session_123')
# Returns list of message dictionaries, sorted by timestamp
```

### Getting Conversation Statistics
```python
stats = await db_service.get_conversation_stats('session_123')
# Returns: {
#   "message_count": 10,
#   "user_messages": 5,
#   "ai_messages": 4,
#   "system_messages": 1,
#   "first_message_time": "2024-01-01T10:00:00Z",
#   "last_message_time": "2024-01-01T10:10:00Z"
# }
```

## Backward Compatibility

The API methods maintain the same signatures, so existing code should continue to work without changes. The main difference is in the underlying data structure and improved performance.

## Notes

- The `session_id` is now unique in the conversation_logs collection
- Messages within a document are automatically sorted by timestamp when retrieved
- The migration script creates a backup before making changes
- All existing functionality is preserved while improving performance# Conversation Logs Update

## Overview
The conversation logs system has been updated to store all messages for a session in a single document instead of creating separate documents for each message. This improves performance and reduces database overhead.

## Changes Made

### 1. Model Updates (`app/models.py`)
- **New**: `ConversationMessage` model for individual messages
- **Updated**: `ConversationLog` model now contains an array of messages
- **Structure**:
  ```python
  class ConversationMessage(BaseModel):
      timestamp: datetime
      message_type: str  # user, ai, system
      content: str
      persona: Optional[str] = None
      audio_file: Optional[str] = None
      transcription_confidence: Optional[float] = None

  class ConversationLog(BaseModel):
      session_id: str
      created_at: datetime
      updated_at: datetime
      messages: List[ConversationMessage] = []
  ```

### 2. Database Service Updates (`app/services/database_service.py`)
- **Updated**: `log_conversation()` now uses upsert to add messages to existing documents
- **Updated**: `get_conversation_logs()` returns sorted messages from a single document
- **New**: `get_conversation_log_document()` returns the full conversation document
- **New**: `get_conversation_stats()` provides conversation statistics

### 3. Database Schema Changes (`app/database.py`)
- **Updated**: Indexes now include `session_id` (unique), `created_at`, and `updated_at`
- **Removed**: Individual `timestamp` index (no longer needed)

### 4. Documentation Updates
- Updated `DATABASE_INTEGRATION.md` with new schema
- Updated database indexes documentation

## Benefits

1. **Reduced Database Overhead**: One document per session instead of one per message
2. **Better Performance**: Fewer database queries and better indexing
3. **Atomic Operations**: All messages for a session are updated atomically
4. **Easier Querying**: Single query to get all conversation data for a session
5. **Better Data Locality**: Related messages are stored together

## Migration

### For Existing Data
If you have existing conversation logs, run the migration script:

```bash
python scripts/migrate_conversation_logs.py
```

This script will:
1. Backup existing data to a timestamped collection
2. Convert old format to new format
3. Create new indexes
4. Preserve all existing data

### For New Installations
No migration needed - the new format will be used automatically.

## Testing

Run the test script to verify functionality:

```bash
python scripts/test_conversation_logs.py
```

## API Changes

### Before (Old Format)
```python
# Each message was a separate document
{
  "_id": "ObjectId",
  "session_id": "session_123",
  "timestamp": "2024-01-01T10:00:00Z",
  "message_type": "user",
  "content": "Hello",
  "persona": null
}
```

### After (New Format)
```python
# All messages for a session in one document
{
  "_id": "ObjectId",
  "session_id": "session_123",
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:05:00Z",
  "messages": [
    {
      "timestamp": "2024-01-01T10:00:00Z",
      "message_type": "user",
      "content": "Hello",
      "persona": null
    },
    {
      "timestamp": "2024-01-01T10:01:00Z",
      "message_type": "ai",
      "content": "Hi there!",
      "persona": "investor"
    }
  ]
}
```

## Usage Examples

### Logging a Conversation Message
```python
await db_service.log_conversation({
    'session_id': 'session_123',
    'message_type': 'user',
    'content': 'Hello, I want to pitch my startup.',
    'persona': None
})
```

### Getting All Messages for a Session
```python
messages = await db_service.get_conversation_logs('session_123')
# Returns list of message dictionaries, sorted by timestamp
```

### Getting Conversation Statistics
```python
stats = await db_service.get_conversation_stats('session_123')
# Returns: {
#   "message_count": 10,
#   "user_messages": 5,
#   "ai_messages": 4,
#   "system_messages": 1,
#   "first_message_time": "2024-01-01T10:00:00Z",
#   "last_message_time": "2024-01-01T10:10:00Z"
# }
```

## Backward Compatibility

The API methods maintain the same signatures, so existing code should continue to work without changes. The main difference is in the underlying data structure and improved performance.

## Notes

- The `session_id` is now unique in the conversation_logs collection
- Messages within a document are automatically sorted by timestamp when retrieved
- The migration script creates a backup before making changes
- All existing functionality is preserved while improving performance