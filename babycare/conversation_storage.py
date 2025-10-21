"""
Conversation Storage System for Baby Care Chatbot

This module provides NoSQL-based conversation storage using TinyDB for local persistence.
Stores conversations, messages, and metadata in a flexible JSON document format.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from tinydb import TinyDB, Query
    from tinydb.storages import JSONStorage
    from tinydb.middlewares import CachingMiddleware
except ImportError:
    raise ImportError("TinyDB not installed. Run: pip install tinydb")

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


@dataclass
class ConversationMetadata:
    """Metadata for a conversation session."""
    conversation_id: str
    user_id: Optional[str] = None
    session_start: str = None
    session_end: Optional[str] = None
    total_messages: int = 0
    topics_discussed: List[str] = None
    documents_retrieved: int = 0
    rag_usage_count: int = 0
    
    def __post_init__(self):
        if self.session_start is None:
            self.session_start = datetime.now().isoformat()
        if self.topics_discussed is None:
            self.topics_discussed = []


@dataclass
class StoredMessage:
    """A message stored in the database."""
    message_id: str
    conversation_id: str
    role: str  # 'human' or 'ai'
    content: str
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConversationStorage:
    """
    NoSQL conversation storage using TinyDB.
    
    Stores conversations as JSON documents with flexible schema for messages,
    metadata, and retrieval information.
    """
    
    def __init__(self, db_path: str = "conversations.json"):
        """
        Initialize the conversation storage.
        
        Args:
            db_path (str): Path to the TinyDB JSON file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize TinyDB with direct JSON storage (no caching to ensure data is saved)
        self.db = TinyDB(str(self.db_path))
        
        # Get table references
        self.conversations = self.db.table('conversations')
        self.messages = self.db.table('messages')
        self.metadata = self.db.table('metadata')
        
        logger.info(f"Conversation storage initialized at {self.db_path}")
    
    def start_conversation(self, 
                          conversation_id: str, 
                          user_id: Optional[str] = None,
                          initial_metadata: Optional[Dict[str, Any]] = None) -> ConversationMetadata:
        """
        Start a new conversation session.
        
        Args:
            conversation_id (str): Unique identifier for the conversation
            user_id (Optional[str]): User identifier
            initial_metadata (Optional[Dict[str, Any]]): Additional metadata
            
        Returns:
            ConversationMetadata: Created conversation metadata
        """
        metadata = ConversationMetadata(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        # Add any initial metadata
        if initial_metadata:
            for key, value in initial_metadata.items():
                setattr(metadata, key, value)
        
        # Store in database
        self.metadata.upsert(asdict(metadata), Query().conversation_id == conversation_id)
        
        logger.info(f"Started conversation {conversation_id} for user {user_id}")
        return metadata
    
    def save_message(self, 
                    conversation_id: str,
                    message: BaseMessage,
                    message_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a message to the conversation.
        
        Args:
            conversation_id (str): Conversation identifier
            message (BaseMessage): LangChain message object
            message_metadata (Optional[Dict[str, Any]]): Additional message metadata
            
        Returns:
            str: Generated message ID
        """
        # Generate unique message ID
        message_id = f"{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Determine role and content
        if isinstance(message, HumanMessage):
            role = 'human'
        elif isinstance(message, AIMessage):
            role = 'ai'
        else:
            role = 'system'
        
        # Create stored message
        stored_message = StoredMessage(
            message_id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=message.content,
            timestamp=datetime.now().isoformat(),
            metadata=message_metadata or {}
        )
        
        # Store message
        self.messages.insert(asdict(stored_message))
        
        # Update conversation metadata
        self._update_conversation_stats(conversation_id)
        
        logger.debug(f"Saved {role} message to conversation {conversation_id}")
        return message_id
    
    def save_conversation_turn(self,
                              conversation_id: str,
                              user_message: str,
                              ai_response: str,
                              retrieval_info: Optional[Dict[str, Any]] = None,
                              rag_used: bool = False) -> Tuple[str, str]:
        """
        Save a complete conversation turn (user message + AI response).
        
        Args:
            conversation_id (str): Conversation identifier
            user_message (str): User's message
            ai_response (str): AI's response
            retrieval_info (Optional[Dict[str, Any]]): RAG retrieval information
            rag_used (bool): Whether RAG was used for this response
            
        Returns:
            Tuple[str, str]: (user_message_id, ai_message_id)
        """
        # Save user message
        user_msg = HumanMessage(content=user_message)
        user_msg_id = self.save_message(conversation_id, user_msg)
        
        # Save AI response with metadata
        ai_metadata = {
            'rag_used': rag_used,
            'retrieval_info': retrieval_info or {}
        }
        ai_msg = AIMessage(content=ai_response)
        ai_msg_id = self.save_message(conversation_id, ai_msg, ai_metadata)
        
        # Update RAG usage count if RAG was used
        if rag_used:
            self._increment_rag_usage(conversation_id)
        
        return user_msg_id, ai_msg_id
    
    def get_conversation_messages(self, 
                                 conversation_id: str,
                                 limit: Optional[int] = None) -> List[StoredMessage]:
        """
        Retrieve messages from a conversation.
        
        Args:
            conversation_id (str): Conversation identifier
            limit (Optional[int]): Maximum number of messages to retrieve
            
        Returns:
            List[StoredMessage]: List of stored messages
        """
        query = Query().conversation_id == conversation_id
        messages = self.messages.search(query)
        
        # Sort by timestamp
        messages.sort(key=lambda x: x['timestamp'])
        
        if limit:
            messages = messages[-limit:]  # Get most recent messages
        
        return [StoredMessage(**msg) for msg in messages]
    
    def get_conversation_history(self, 
                               conversation_id: str,
                               limit: Optional[int] = None) -> List[BaseMessage]:
        """
        Get conversation history as LangChain messages.
        
        Args:
            conversation_id (str): Conversation identifier
            limit (Optional[int]): Maximum number of messages to retrieve
            
        Returns:
            List[BaseMessage]: LangChain message objects
        """
        stored_messages = self.get_conversation_messages(conversation_id, limit)
        
        messages = []
        for stored_msg in stored_messages:
            if stored_msg.role == 'human':
                messages.append(HumanMessage(content=stored_msg.content))
            elif stored_msg.role == 'ai':
                messages.append(AIMessage(content=stored_msg.content))
        
        return messages
    
    def list_conversations(self, 
                          user_id: Optional[str] = None,
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List conversations with optional filtering.
        
        Args:
            user_id (Optional[str]): Filter by user ID
            limit (Optional[int]): Maximum number of conversations to return
            
        Returns:
            List[Dict[str, Any]]: List of conversation metadata
        """
        if user_id:
            query = Query().user_id == user_id
            conversations = self.metadata.search(query)
        else:
            conversations = self.metadata.all()
        
        # Sort by session start time (most recent first)
        conversations.sort(key=lambda x: x.get('session_start', ''), reverse=True)
        
        if limit:
            conversations = conversations[:limit]
        
        return conversations
    
    def get_conversation_metadata(self, conversation_id: str) -> Optional[ConversationMetadata]:
        """
        Get metadata for a specific conversation.
        
        Args:
            conversation_id (str): Conversation identifier
            
        Returns:
            Optional[ConversationMetadata]: Conversation metadata or None
        """
        result = self.metadata.search(Query().conversation_id == conversation_id)
        if result:
            return ConversationMetadata(**result[0])
        return None
    
    def update_conversation_metadata(self, 
                                   conversation_id: str,
                                   updates: Dict[str, Any]) -> bool:
        """
        Update conversation metadata.
        
        Args:
            conversation_id (str): Conversation identifier
            updates (Dict[str, Any]): Fields to update
            
        Returns:
            bool: True if updated successfully
        """
        try:
            self.metadata.update(updates, Query().conversation_id == conversation_id)
            logger.info(f"Updated metadata for conversation {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update conversation metadata: {e}")
            return False
    
    def end_conversation(self, conversation_id: str) -> bool:
        """
        End a conversation session.
        
        Args:
            conversation_id (str): Conversation identifier
            
        Returns:
            bool: True if ended successfully
        """
        try:
            self.metadata.update(
                {'session_end': datetime.now().isoformat()},
                Query().conversation_id == conversation_id
            )
            logger.info(f"Ended conversation {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to end conversation: {e}")
            return False
    
    def search_conversations(self, 
                           query_text: str,
                           user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search conversations by message content.
        
        Args:
            query_text (str): Search query
            user_id (Optional[str]): Filter by user ID
            
        Returns:
            List[Dict[str, Any]]: Matching conversations with snippets
        """
        # Search in message content using case-insensitive search
        import re
        pattern = re.compile(query_text, re.IGNORECASE)
        matching_messages = []
        
        for message in self.messages.all():
            if pattern.search(message.get('content', '')):
                matching_messages.append(message)
        
        # Get unique conversation IDs
        conversation_ids = list(set(msg['conversation_id'] for msg in matching_messages))
        
        # Get conversation metadata
        results = []
        for conv_id in conversation_ids:
            metadata = self.get_conversation_metadata(conv_id)
            if metadata and (user_id is None or metadata.user_id == user_id):
                # Get matching message snippets
                matching_msgs = [msg for msg in matching_messages if msg['conversation_id'] == conv_id]
                results.append({
                    'conversation_id': conv_id,
                    'metadata': asdict(metadata),
                    'matching_messages': matching_msgs[:3]  # Limit snippets
                })
        
        return results
    
    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get statistics for a conversation.
        
        Args:
            conversation_id (str): Conversation identifier
            
        Returns:
            Dict[str, Any]: Conversation statistics
        """
        messages = self.get_conversation_messages(conversation_id)
        metadata = self.get_conversation_metadata(conversation_id)
        
        if not metadata:
            return {}
        
        # Count message types
        human_count = len([m for m in messages if m.role == 'human'])
        ai_count = len([m for m in messages if m.role == 'ai'])
        
        # Calculate session duration
        start_time = datetime.fromisoformat(metadata.session_start)
        end_time = datetime.fromisoformat(metadata.session_end) if metadata.session_end else datetime.now()
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        return {
            'conversation_id': conversation_id,
            'total_messages': len(messages),
            'human_messages': human_count,
            'ai_messages': ai_count,
            'duration_minutes': round(duration_minutes, 2),
            'rag_usage_count': metadata.rag_usage_count,
            'documents_retrieved': metadata.documents_retrieved,
            'topics_discussed': metadata.topics_discussed,
            'session_start': metadata.session_start,
            'session_end': metadata.session_end
        }
    
    def _update_conversation_stats(self, conversation_id: str):
        """Update conversation statistics."""
        try:
            # Count total messages
            message_count = len(self.messages.search(Query().conversation_id == conversation_id))
            
            # Update metadata
            self.metadata.update(
                {'total_messages': message_count},
                Query().conversation_id == conversation_id
            )
        except Exception as e:
            logger.error(f"Failed to update conversation stats: {e}")
    
    def _increment_rag_usage(self, conversation_id: str):
        """Increment RAG usage count for a conversation."""
        try:
            metadata = self.get_conversation_metadata(conversation_id)
            if metadata:
                new_count = metadata.rag_usage_count + 1
                self.metadata.update(
                    {'rag_usage_count': new_count},
                    Query().conversation_id == conversation_id
                )
        except Exception as e:
            logger.error(f"Failed to increment RAG usage: {e}")
    
    def export_conversation(self, conversation_id: str, format: str = 'json') -> str:
        """
        Export a conversation to a file.
        
        Args:
            conversation_id (str): Conversation identifier
            format (str): Export format ('json' or 'txt')
            
        Returns:
            str: Path to exported file
        """
        messages = self.get_conversation_messages(conversation_id)
        metadata = self.get_conversation_metadata(conversation_id)
        
        if format == 'json':
            export_data = {
                'metadata': asdict(metadata) if metadata else {},
                'messages': [asdict(msg) for msg in messages]
            }
            filename = f"conversation_{conversation_id}.json"
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format == 'txt':
            filename = f"conversation_{conversation_id}.txt"
            with open(filename, 'w') as f:
                f.write(f"Conversation: {conversation_id}\n")
                f.write(f"Started: {metadata.session_start if metadata else 'Unknown'}\n")
                f.write("=" * 50 + "\n\n")
                
                for msg in messages:
                    role = "User" if msg.role == 'human' else "Assistant"
                    f.write(f"{role}: {msg.content}\n\n")
        
        logger.info(f"Exported conversation {conversation_id} to {filename}")
        return filename
    
    def close(self):
        """Close the database connection."""
        self.db.close()
        logger.info("Conversation storage closed")


def main():
    """Demo the conversation storage system."""
    print("Testing Conversation Storage System...")
    
    # Initialize storage
    storage = ConversationStorage("demo_conversations.json")
    
    # Start a conversation
    conv_id = "demo_001"
    storage.start_conversation(conv_id, user_id="demo_user")
    
    # Save some messages
    storage.save_conversation_turn(
        conv_id, 
        "What should I feed my 6-month-old?",
        "For a 6-month-old, you can start introducing solid foods while continuing breast milk or formula. Begin with single-grain cereals, pureed fruits and vegetables...",
        rag_used=True
    )
    
    storage.save_conversation_turn(
        conv_id,
        "How much should they eat?",
        "Start with 1-2 tablespoons of solid food once or twice a day. Gradually increase to 3-4 tablespoons as your baby shows interest...",
        rag_used=True
    )
    
    # Get conversation history
    history = storage.get_conversation_history(conv_id)
    print(f"Retrieved {len(history)} messages from conversation")
    
    # Get stats
    stats = storage.get_conversation_stats(conv_id)
    print(f"Conversation stats: {stats}")
    
    # Export conversation
    export_file = storage.export_conversation(conv_id, 'txt')
    print(f"Exported conversation to {export_file}")
    
    # End conversation
    storage.end_conversation(conv_id)
    storage.close()
    
    print("Demo completed!")


if __name__ == "__main__":
    main()
