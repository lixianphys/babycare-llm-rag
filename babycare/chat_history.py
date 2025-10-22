"""
SQLite-based chat history management for Baby Care Assistant
"""
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from .database import ChatMessage, get_db

class ChatHistoryManager:
    """Manage chat history using SQLite database"""
    
    def __init__(self):
        pass
    
    async def save_message(self, user_id: str, conversation_id: str, message: Dict[str, Any]):
        """Save a message to chat history"""
        db = next(get_db())
        try:
            # Convert retrieved_documents to JSON string if present
            retrieved_docs_json = None
            if 'retrieved_documents' in message:
                retrieved_docs_json = json.dumps(message['retrieved_documents'])
            
            chat_message = ChatMessage(
                user_id=int(user_id),
                conversation_id=conversation_id,
                message_type=message['type'],
                content=message['content'],
                retrieved_count=message.get('retrieved_count', 0),
                retrieved_documents=retrieved_docs_json,
                timestamp=datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00'))
            )
            
            db.add(chat_message)
            db.commit()
            db.refresh(chat_message)
            return chat_message.id
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    async def get_conversation_history(self, user_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a specific conversation"""
        db = next(get_db())
        try:
            messages = db.query(ChatMessage).filter(
                ChatMessage.user_id == int(user_id),
                ChatMessage.conversation_id == conversation_id
            ).order_by(ChatMessage.timestamp.asc()).all()
            
            result = []
            for msg in messages:
                message_dict = {
                    "type": msg.message_type,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "retrieved_count": msg.retrieved_count
                }
                
                # Parse retrieved documents if present
                if msg.retrieved_documents:
                    try:
                        message_dict["retrieved_documents"] = json.loads(msg.retrieved_documents)
                    except json.JSONDecodeError:
                        message_dict["retrieved_documents"] = []
                else:
                    message_dict["retrieved_documents"] = []
                
                result.append(message_dict)
            
            return result
        finally:
            db.close()
    
    async def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all conversations for a user"""
        from sqlalchemy import func
        db = next(get_db())
        try:
            # Get unique conversation IDs with their latest message timestamp and count
            conversations = db.query(
                ChatMessage.conversation_id,
                func.max(ChatMessage.timestamp).label('last_updated'),
                func.min(ChatMessage.timestamp).label('created_at'),
                func.count(ChatMessage.id).label('message_count')
            ).filter(
                ChatMessage.user_id == int(user_id)
            ).group_by(ChatMessage.conversation_id).all()
            
            result = []
            for conv in conversations:
                # Get the last message content
                last_message = db.query(ChatMessage).filter(
                    ChatMessage.user_id == int(user_id),
                    ChatMessage.conversation_id == conv.conversation_id
                ).order_by(ChatMessage.timestamp.desc()).first()
                
                result.append({
                    "conversation_id": conv.conversation_id,
                    "last_updated": conv.last_updated.isoformat(),
                    "created_at": conv.created_at.isoformat(),
                    "message_count": conv.message_count,
                    "last_message": {
                        "content": last_message.content if last_message else "No messages",
                        "type": last_message.message_type if last_message else "user"
                    }
                })
            
            return result
        finally:
            db.close()
    
    async def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        """Delete a conversation and all its messages"""
        db = next(get_db())
        try:
            deleted_count = db.query(ChatMessage).filter(
                ChatMessage.user_id == int(user_id),
                ChatMessage.conversation_id == conversation_id
            ).delete()
            
            db.commit()
            return deleted_count > 0
        except Exception as e:
            db.rollback()
            return False
        finally:
            db.close()
    
    async def delete_all_user_conversations(self, user_id: str) -> int:
        """Delete all conversations for a user"""
        db = next(get_db())
        try:
            deleted_count = db.query(ChatMessage).filter(
                ChatMessage.user_id == int(user_id)
            ).delete()
            
            db.commit()
            return deleted_count
        except Exception as e:
            db.rollback()
            return 0
        finally:
            db.close()

# Global instance
chat_history_manager = ChatHistoryManager()
