"""
RAG (Retrieval Augmented Generation) System for Baby Care Chatbot

This module implements the RAG system that combines the vector store
with the chat flow to provide context-aware responses based on
retrieved baby care documents.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain.schema import Document
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from .vector_store import BabyCareVectorStore
from .langsmith_monitor import LangSmithMonitor

from .config import config
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGConfig:
    """
    Configuration class for RAG system parameters.
    
    This class contains all configurable parameters for the RAG system,
    allowing easy customization of retrieval and generation behavior.
    """
    def __init__(self):
        # Retrieval parameters
        self.top_k_documents = config.top_k_documents
        self.similarity_threshold = config.similarity_threshold
        self.max_context_length = config.max_context_length
        
        # Generation parameters
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        
        # RAG-specific parameters
        self.use_reranking = False
        self.include_metadata = True
        self.context_window_size = 3


class BabyCareRAGSystem:
    """
    RAG system that combines retrieval and generation for baby care knowledge.
    
    This class integrates the vector store with the chat flow to provide
    context-aware responses based on retrieved baby care documents.
    """
    
    def __init__(self, vector_store, config=None, monitor=None):
        """
        Initialize the RAG system with vector store and configuration.
        
        Args:
            vector_store (SimpleBabyCareVectorStore): Initialized vector store instance
            config (Optional[RAGConfig]): RAG configuration parameters
            monitor (Optional[LangSmithMonitor]): LangSmith monitoring instance
        """
        self.vector_store = vector_store
        self.config = config or RAGConfig()
        self.monitor = monitor or LangSmithMonitor()
        
        # Initialize LLM for generation with monitoring callbacks
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            streaming=True,
            timeout=30,
            callbacks=self.monitor.get_callbacks()
        )
        
        # Create RAG prompt template
        self.rag_prompt = self._create_rag_prompt()
        
        # Create the RAG chain
        self.rag_chain = self._create_rag_chain()
        
        logger.info("BabyCareRAGSystem initialized successfully with LangSmith monitoring")
    
    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """
        Create the RAG prompt template for context-aware generation.
        
        Returns:
            ChatPromptTemplate: Template for RAG-based responses
        """
        system_prompt = """You are a specialized AI assistant for baby care, nutrition, healthcare, and child development. 
        You have access to a knowledge base of expert information about baby care.

        Instructions:
        1. Use the provided context documents to answer questions accurately
        2. If the context doesn't contain relevant information, say so clearly
        3. Always provide evidence-based advice
        4. Include appropriate disclaimers for medical advice
        5. Be warm, supportive, and understanding
        6. Cite specific information from the context when relevant

        Context Documents:
        {context}

        Conversation History:
        {chat_history}

        Current Question: {question}

        Please provide a helpful, accurate response based on the context and conversation history."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])
    
    def _create_rag_chain(self):
        """
        Create the RAG chain that combines retrieval and generation.
        
        Returns:
            Chain: Configured RAG chain
        """
        def format_docs(docs: List[Document]) -> str:
            """Format retrieved documents for the prompt."""
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content
                metadata = doc.metadata
                
                formatted_doc = f"Document {i}:\n{content}\n"
                if self.config.include_metadata and metadata:
                    formatted_doc += f"Source: {metadata.get('source', 'Unknown')}\n"
                    formatted_doc += f"Category: {metadata.get('category', 'General')}\n"
                
                formatted_docs.append(formatted_doc)
            
            return "\n".join(formatted_docs)
        
        # Create the RAG chain
        rag_chain = (
            {
                "context": lambda x: format_docs(self._retuncate_documents(x["question"])),
                "chat_history": lambda x: x.get("chat_history", []),
                "question": RunnablePassthrough.assign(question=lambda x: x["question"])
            }
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def _retuncate_documents(self, query):
        """
        Retrieve relevant documents for the given query.
        
        Args:
            query (str): User query to retrieve documents for
            
        Returns:
            List[Document]: List of relevant documents
        """
        try:
            # Perform similarity search
            documents = self.vector_store.search_documents(
                query, 
                k=self.config.top_k_documents
            )
            
            # Filter by similarity threshold if needed
            if self.config.similarity_threshold > 0:
                # Get documents with scores
                scored_docs = self.vector_store.search_with_scores(
                    query, 
                    k=self.config.top_k_documents
                )
                # Filter by threshold
                filtered_docs = [
                    doc for doc, score in scored_docs 
                    if score >= self.config.similarity_threshold
                ]
                documents = filtered_docs
            
            # Limit context length
            total_length = sum(len(doc.page_content) for doc in documents)
            if total_length > self.config.max_context_length:
                # Truncate documents to fit context window
                documents = self._truncate_documents(documents)
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _truncate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Truncate documents to fit within the context length limit.
        
        Args:
            documents (List[Document]): List of documents to truncate
            
        Returns:
            List[Document]: Truncated documents
        """
        truncated_docs = []
        current_length = 0
        
        for doc in documents:
            doc_length = len(doc.page_content)
            if current_length + doc_length <= self.config.max_context_length:
                truncated_docs.append(doc)
                current_length += doc_length
            else:
                # Truncate the last document to fit
                remaining_length = self.config.max_context_length - current_length
                if remaining_length > 100:  # Only add if meaningful content remains
                    truncated_content = doc.page_content[:remaining_length] + "..."
                    truncated_doc = Document(
                        page_content=truncated_content,
                        metadata=doc.metadata
                    )
                    truncated_docs.append(truncated_doc)
                break
        
        return truncated_docs
    
    def generate_response(self, 
                         question: str, 
                         chat_history: Optional[List[BaseMessage]] = None) -> str:
        """
        Generate a response using RAG with retrieved context.
        
        Args:
            question (str): User's question
            chat_history (Optional[List[BaseMessage]]): Previous conversation messages
            
        Returns:
            str: Generated response
        """
        try:
            # Prepare input for RAG chain
            rag_input = {
                "question": question,
                "chat_history": chat_history or []
            }
            
            # Generate response using RAG chain
            response = self.rag_chain.invoke(rag_input)
            
            # Log to LangSmith for monitoring
            self.monitor.log_query(
                query=question,
                response=response,
                metadata={
                    "type": "rag_response",
                    "model": self.config.model_name,
                    "has_chat_history": len(chat_history or []) > 0,
                    "chat_history_length": len(chat_history or [])
                }
            )
            
            logger.info(f"Generated RAG response for question: {question[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            error_response = "I apologize, but I'm having trouble accessing my knowledge base right now. Please try again later."
            
            # Log error to LangSmith
            self.monitor.log_query(
                query=question,
                response=error_response,
                metadata={
                    "type": "rag_error",
                    "error": str(e),
                    "model": self.config.model_name
                }
            )
            
            return error_response
    
    def generate_response_with_retrieval_info(self, 
                                            question: str, 
                                            chat_history: Optional[List[BaseMessage]] = None) -> Tuple[str, int]:
        """
        Generate a response and return the number of documents actually used.
        
        Args:
            question (str): User's question
            chat_history (Optional[List[BaseMessage]]): Previous conversation messages
            
        Returns:
            Tuple[str, int]: Generated response and number of documents used
        """
        try:
            # Retrieve documents and get the actual count used
            documents = self._retuncate_documents(question)
            documents_used = len(documents)
            
            # Format documents for context
            def format_docs(docs):
                """Format retrieved documents for the prompt."""
                formatted_docs = []
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content
                    metadata = doc.metadata
                    
                    formatted_doc = f"Document {i}:\n{content}\n"
                    if self.config.include_metadata and metadata:
                        formatted_doc += f"Source: {metadata.get('source', 'Unknown')}\n"
                        formatted_doc += f"Category: {metadata.get('category', 'General')}\n"
                    
                    formatted_docs.append(formatted_doc)
                
                return "\n".join(formatted_docs)
            
            context = format_docs(documents)
            
            # Prepare input for RAG chain
            rag_input = {
                "context": context,
                "question": question,
                "chat_history": chat_history or []
            }
            
            # Generate response using RAG chain
            response = self.rag_chain.invoke(rag_input)
            
            # Log to LangSmith for monitoring
            self.monitor.log_query(
                query=question,
                response=response,
                metadata={
                    "type": "rag_response",
                    "model": self.config.model_name,
                    "documents_used": documents_used,
                    "has_chat_history": len(chat_history or []) > 0,
                    "chat_history_length": len(chat_history or [])
                }
            )
            
            logger.info(f"Generated RAG response using {documents_used} documents for question: {question[:50]}...")
            return response, documents_used
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            error_response = "I apologize, but I'm having trouble accessing my knowledge base right now. Please try again later."
            
            # Log error to LangSmith
            self.monitor.log_query(
                query=question,
                response=error_response,
                metadata={
                    "type": "rag_error",
                    "error": str(e),
                    "model": self.config.model_name,
                    "documents_used": 0
                }
            )
            
            return error_response, 0
    
    def stream_response(self, 
                       question: str, 
                       chat_history: Optional[List[BaseMessage]] = None):
        """
        Stream a response using RAG with retrieved context.
        
        Args:
            question (str): User's question
            chat_history (Optional[List[BaseMessage]]): Previous conversation messages
            
        Yields:
            str: Streaming response chunks
        """
        try:
            # Prepare input for RAG chain
            rag_input = {
                "question": question,
                "chat_history": chat_history or []
            }
            
            # Stream response using RAG chain
            for chunk in self.rag_chain.stream(rag_input):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error streaming RAG response: {e}")
            yield "I apologize, but I'm having trouble accessing my knowledge base right now. Please try again later."
    
    def get_retrieval_info(self, query: str) -> Dict[str, Any]:
        """
        Get information about document retrieval for a query.
        
        Args:
            query (str): Query to analyze
            
        Returns:
            Dict[str, Any]: Retrieval information including document count and sources
        """
        try:
            # Get documents with scores
            scored_docs = self.vector_store.search_with_scores(
                query, 
                k=self.config.top_k_documents
            )
            
            # Extract information
            sources = []
            categories = []
            for doc, score in scored_docs:
                metadata = doc.metadata
                sources.append(metadata.get('source', 'Unknown'))
                categories.append(metadata.get('category', 'General'))
            
            return {
                "query": query,
                "document_count": len(scored_docs),
                "sources": list(set(sources)),
                "categories": list(set(categories)),
                "scores": [score for _, score in scored_docs]
            }
            
        except Exception as e:
            logger.error(f"Error getting retrieval info: {e}")
            return {"error": str(e)}
    
    def update_config(self, new_config: RAGConfig) -> None:
        """
        Update the RAG system configuration.
        
        Args:
            new_config (RAGConfig): New configuration parameters
        """
        self.config = new_config
        
        # Reinitialize LLM with new config
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            streaming=True,
            timeout=30
        )
        
        # Recreate RAG chain
        self.rag_chain = self._create_rag_chain()
        
        logger.info("RAG system configuration updated")


class EnhancedBabyCareState:
    """
    Enhanced state class that includes RAG-specific information.
    
    This class extends the basic chat state to include RAG context
    and retrieval information.
    """
    
    def __init__(self, 
                 messages: List[BaseMessage],
                 user_context: Dict[str, Any],
                 conversation_id: str,
                 retrieved_documents: List[Document],
                 retrieval_info: Dict[str, Any]):
        """
        Initialize enhanced state with RAG information.
        
        Args:
            messages (List[BaseMessage]): Conversation messages
            user_context (Dict[str, Any]): User context information
            conversation_id (str): Unique conversation identifier
            retrieved_documents (List[Document]): Retrieved documents for current query
            retrieval_info (Dict[str, Any]): Information about document retrieval
        """
        self.messages = messages
        self.user_context = user_context
        self.conversation_id = conversation_id
        self.retrieved_documents = retrieved_documents
        self.retrieval_info = retrieval_info


def main():
    """
    Main function to demonstrate the RAG system functionality.
    """
    # Initialize vector store
    from .simple_vector_store import SimpleBabyCareVectorStore, create_sample_baby_care_documents
    
    print("Initializing RAG system...")
    vector_store = BabyCareVectorStore()
    
    # Add sample documents if collection is empty
    info = vector_store.get_collection_info()
    if info.get("document_count", 0) == 0:
        print("Adding sample documents...")
        sample_docs = create_sample_baby_care_documents()
        vector_store.add_custom_documents(sample_docs)
    
    # Initialize RAG system
    rag_system = BabyCareRAGSystem(vector_store)
    
    # Test queries
    test_queries = [
        "How much should a 3-month-old baby eat?",
        "What are the signs of colic in babies?",
        "When should I start baby-proofing my home?",
        "What are the normal sleep patterns for a 6-month-old?"
    ]
    
    print("\nTesting RAG system:")
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # Get retrieval info
        retrieval_info = rag_system.get_retrieval_info(query)
        print(f"Retrieved {retrieval_info['document_count']} documents")
        print(f"Sources: {retrieval_info['sources']}")
        print(f"Categories: {retrieval_info['categories']}")
        
        # Generate response
        response = rag_system.generate_response(query)
        print(f"\nResponse: {response}")


if __name__ == "__main__":
    main()
