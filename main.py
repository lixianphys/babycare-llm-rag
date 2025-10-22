"""
FastAPI Backend for Baby Care Chatbot
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from babycare.integrated_chatbot import IntegratedBabyCareChatbot

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        logger.error(f"Error initializing chatbot: {e}")
        chatbot = None

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
    bot: IntegratedBabyCareChatbot = Depends(get_chatbot)
):
    """Main chat endpoint for user messages."""
    try:
        # Generate response with retrieval info
        response, retrieved_count, retrieved_docs = bot.stream_chat_with_retrieval_info(
            chat_message.message,
            chat_message.conversation_id,
            chat_message.user_id
        )
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
