"""
FastAPI Backend for Baby Care Chatbot
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from babycare.integrated_chatbot import IntegratedBabyCareChatbot
from babycare.database import get_db, User
from babycare.basic_auth import get_current_user, get_current_user_optional, router as auth_router
from babycare.chat_history import chat_history_manager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Baby Care Assistant API",
    description="API for Baby Care Chatbot with RAG capabilities",
    version="1.0.0"
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://localhost:3002"],  # Next.js default ports
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include authentication routes
app.include_router(auth_router)

# Global chatbot instance
chatbot: Optional[IntegratedBabyCareChatbot] = None
total_documents: int = 0

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = "default"
    user_id: Optional[str] = "user"

class ChatResponse(BaseModel):
    response: str
    retrieved_count: int
    retrieved_documents: List[Dict[str, Any]]
    conversation_id: str

class HealthResponse(BaseModel):
    status: str
    total_documents: int
    chatbot_initialized: bool

class DocumentInfo(BaseModel):
    content: str
    metadata: Dict[str, Any]

# Dependency to get chatbot instance
def get_chatbot() -> IntegratedBabyCareChatbot:
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return chatbot

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot on startup."""
    global chatbot, total_documents
    
    try:
        logger.info("Initializing Baby Care Chatbot...")
        chatbot = IntegratedBabyCareChatbot(
            enable_monitoring=True,
            enable_conversation_storage=False,  # Disable for API
            conversation_db_path=None
        )
        
        # Get total document count
        kb_info = chatbot.get_knowledge_base_info()
        total_documents = kb_info.get('document_count', 0)
        
        logger.info(f"Baby Care Chatbot initialized successfully")
        logger.info(f"Total documents in knowledge base: {total_documents}")
        
    except Exception as e:
        logger.warning(f"Chatbot initialization failed (API keys may be missing): {e}")
        logger.info("Running in limited mode - authentication and basic features available")
        chatbot = None
        total_documents = 0

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if chatbot is not None else "unhealthy",
        total_documents=total_documents,
        chatbot_initialized=chatbot is not None
    )

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    chat_message: ChatMessage,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Main chat endpoint for user messages. Works for both authenticated and guest users."""
    try:
        # Use authenticated user ID if available, otherwise use guest ID
        user_id = str(current_user.id) if current_user else f"guest_{chat_message.user_id}"
        
        # Save user message immediately if user is authenticated
        if current_user:
            try:
                await chat_history_manager.save_message(
                    user_id=str(current_user.id),
                    conversation_id=chat_message.conversation_id,
                    message={
                        "type": "user",
                        "content": chat_message.message,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to save user message: {e}")
        
        # Check if chatbot is available
        if chatbot is None:
            # Return a demo response when chatbot is not available
            demo_response = f"I'm a Baby Care Assistant, but I'm currently in demo mode. You asked: '{chat_message.message}'. To get full AI responses, please configure your API keys (OPENAI_API_KEY and PINECONE_API_KEY) in the .env file."
            
            # Save assistant response to chat history if user is authenticated
            if current_user:
                try:
                    await chat_history_manager.save_message(
                        user_id=str(current_user.id),
                        conversation_id=chat_message.conversation_id,
                        message={
                            "type": "assistant",
                            "content": demo_response,
                            "retrieved_count": 0,
                            "retrieved_documents": [],
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to save assistant response: {e}")
            
            return ChatResponse(
                response=demo_response,
                retrieved_count=0,
                retrieved_documents=[],
                conversation_id=chat_message.conversation_id
            )
        
        # Generate response with retrieval info using chatbot
        response, retrieved_count, retrieved_docs = chatbot.stream_chat_with_retrieval_info(
            chat_message.message,
            chat_message.conversation_id,
            user_id
        )
        
        # Save assistant response immediately if user is authenticated
        if current_user:
            try:
                await chat_history_manager.save_message(
                    user_id=str(current_user.id),
                    conversation_id=chat_message.conversation_id,
                    message={
                        "type": "assistant",
                        "content": response,
                        "retrieved_count": retrieved_count,
                        "retrieved_documents": retrieved_docs,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to save assistant response: {e}")
        
        # Format retrieved documents for frontend
        formatted_docs = []
        for doc in retrieved_docs:
            formatted_docs.append({
                "content": doc.get('content', ''),
                "metadata": doc.get('metadata', {})
            })
        
        return ChatResponse(
            response=response,
            retrieved_count=retrieved_count,
            retrieved_documents=formatted_docs,
            conversation_id=chat_message.conversation_id
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge base info endpoint
@app.get("/knowledge-base")
async def get_knowledge_base_info(bot: IntegratedBabyCareChatbot = Depends(get_chatbot)):
    """Get knowledge base information."""
    try:
        kb_info = bot.get_knowledge_base_info()
        return JSONResponse(content=kb_info)
    except Exception as e:
        logger.error(f"Error getting knowledge base info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat history management endpoints
@app.get("/chat/history")
async def get_chat_history(
    conversation_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get chat history for authenticated user."""
    try:
        if conversation_id:
            # Get specific conversation
            messages = await chat_history_manager.get_conversation_history(
                str(current_user.id), conversation_id
            )
            return {"messages": messages}
        else:
            # Get all conversations
            conversations = await chat_history_manager.get_user_conversations(
                str(current_user.id)
            )
            return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/history/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a specific conversation."""
    try:
        success = await chat_history_manager.delete_conversation(
            str(current_user.id), conversation_id
        )
        if success:
            return {"message": "Conversation deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/history")
async def delete_all_conversations(current_user: User = Depends(get_current_user)):
    """Delete all conversations for the user."""
    try:
        deleted_count = await chat_history_manager.delete_all_user_conversations(
            str(current_user.id)
        )
        return {"message": f"Deleted {deleted_count} conversations"}
    except Exception as e:
        logger.error(f"Error deleting all conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Example questions endpoint
@app.get("/examples")
async def get_example_questions():
    """Get example questions for the frontend."""
    examples = [
        "What should I eat during pregnancy?",
        "How to soothe a crying baby?",
        "Is it safe to take medication while breastfeeding?",
        "What are the signs of colic in babies?",
        "How to establish a sleep routine for newborns?",
        "What vaccines are safe during pregnancy?"
    ]
    return JSONResponse(content={"examples": examples})

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Baby Care Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Test endpoint for CORS
@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify CORS is working."""
    return {"message": "CORS is working!", "status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
