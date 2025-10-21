"""
Integrated Baby Care Chatbot with RAG

This module combines the chat flow, vector store, and RAG system
to create a comprehensive baby care chatbot with specialized knowledge.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

from .chat_flow import BabyCareState
from ._in_memory_vector_store import SimpleBabyCareVectorStore, create_sample_baby_care_documents
from .vector_store import BabyCareVectorStore, create_sample_baby_care_documents
from .rag_system import BabyCareRAGSystem, RAGConfig
from .langsmith_monitor import LangSmithMonitor
from .conversation_storage import ConversationStorage
from .config import config
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedBabyCareChatbot:
    """
    Integrated baby care chatbot with RAG capabilities.
    
    This class combines all components to provide a comprehensive
    baby care chatbot with specialized knowledge retrieval and generation.
    """
    
    def __init__(self, 
                 rag_config: Optional[RAGConfig] = None,
                 auto_initialize_kb: bool = True,
                 enable_monitoring: bool = True,
                 enable_conversation_storage: bool = True,
                 conversation_db_path: str = "conversations.json"):
        """
        Initialize the integrated chatbot with all components.
        
        Args:
            rag_config (Optional[RAGConfig]): RAG system configuration
            auto_initialize_kb (bool): Whether to automatically initialize knowledge base
            enable_monitoring (bool): Whether to enable LangSmith monitoring
            enable_conversation_storage (bool): Whether to enable conversation storage
            conversation_db_path (str): Path to conversation database file
        """
        self.rag_config = rag_config or RAGConfig()
        self.enable_monitoring = enable_monitoring
        self.enable_conversation_storage = enable_conversation_storage
        
        # Initialize monitoring
        self.monitor = LangSmithMonitor("baby-care-chatbot") if enable_monitoring else None
        
        # Initialize conversation storage
        self.conversation_storage = ConversationStorage(conversation_db_path) if enable_conversation_storage else None
        
        # Initialize components
        self.vector_store = SimpleBabyCareVectorStore() if config.use_memory_vector_store else BabyCareVectorStore(index_name=config.index_name_vector_store)

        logger.info(f"USE_MEMORY_STORE set to {config.use_memory_vector_store} - Using vector store: {self.vector_store.__class__.__name__}")


        self.rag_system = BabyCareRAGSystem(self.vector_store, self.rag_config, self.monitor)
        
        # Initialize knowledge base if needed
        if auto_initialize_kb:
            self._initialize_knowledge_base()
        
        # Build integrated conversation graph
        self.graph = self._build_integrated_graph()
        
        logger.info("IntegratedBabyCareChatbot initialized successfully")
    
    def _initialize_knowledge_base(self) -> None:
        """
        Initialize the knowledge base with sample documents if empty.
        """
        try:
            info = self.vector_store.get_collection_info()
            if info.get("document_count", 0) == 0:
                logger.info("Initializing knowledge base with sample documents...")
                sample_docs = create_sample_baby_care_documents()
                added_count = self.vector_store.add_custom_documents(sample_docs)
                logger.info(f"Added {added_count} sample documents to knowledge base")
            else:
                logger.info(f"Knowledge base already contains {info['document_count']} documents")
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
    
    def _build_integrated_graph(self) -> StateGraph:
        """
        Build the integrated conversation graph with RAG capabilities.
        
        Returns:
            StateGraph: Compiled graph with RAG integration
        """
        # Create the state graph
        graph_builder = StateGraph(BabyCareState)
        
        # Add conversation nodes
        graph_builder.add_node("analyze_query", self._analyze_query_with_rag)
        graph_builder.add_node("retrieve_context", self._retrieve_context)
        graph_builder.add_node("generate_rag_response", self._generate_rag_response)
        graph_builder.add_node("request_clarification", self._request_clarification)
        graph_builder.add_node("fallback_response", self._fallback_response)
        
        # Define the conversation flow
        graph_builder.add_edge(START, "analyze_query")
        graph_builder.add_conditional_edges(
            "analyze_query",
            self._determine_next_step,
            {
                "clarify": "request_clarification",
                "retrieve": "retrieve_context",
                "fallback": "fallback_response"
            }
        )
        graph_builder.add_edge("retrieve_context", "generate_rag_response")
        graph_builder.add_edge("generate_rag_response", END)
        graph_builder.add_edge("request_clarification", END)
        graph_builder.add_edge("fallback_response", END)
        
        return graph_builder.compile()
    
    def _analyze_query_with_rag(self, state: BabyCareState) -> BabyCareState:
        """
        Analyze the user's query to determine the best response strategy.
        
        Args:
            state (BabyCareState): Current conversation state
            
        Returns:
            BabyCareState: Updated state with analysis results
        """
        last_message = state["messages"][-1]
        query = last_message.content
        
        # Determine if query needs clarification
        needs_clarification = False
        clarification_question = None
        
        # Check for vague queries
        if len(query.strip()) < 10:
            needs_clarification = True
            clarification_question = "Could you please provide more details about your question? For example, what is your baby's age or what specific aspect of baby care are you asking about?"
        
        # Check for queries that might benefit from RAG
        # Get dynamic keywords from knowledge base
        try:
            kb_keywords = self.vector_store.get_knowledge_base_keywords()
        except Exception as e:
            logger.warning(f"Could not get knowledge base keywords: {e}")
            kb_keywords = []
        
        # Combine with general baby care keywords
        general_keywords = [
            "nutrition", "feeding", "sleep", "development", "milestone",
            "health", "safety", "childproofing", "routine", "schedule",
            "food", "milk", "formula", "solid", "teething", "colic",
            "baby", "infant", "toddler", "newborn", "pregnancy", "pregnant"
        ]
        
        # Combine all keywords
        all_rag_keywords = general_keywords + kb_keywords
        
        # Remove duplicates and convert to lowercase for comparison
        rag_keywords = list(set([kw.lower() for kw in all_rag_keywords]))
        
        should_use_rag = any(keyword in query.lower() for keyword in rag_keywords)
        
        # Log the decision for debugging
        if should_use_rag:
            matched_keywords = [kw for kw in rag_keywords if kw in query.lower()]
            logger.info(f"RAG triggered by keywords: {matched_keywords}")
        else:
            logger.info(f"RAG not triggered. Query: '{query[:50]}...'")
        
        # Update state
        state["needs_clarification"] = needs_clarification
        state["clarification_question"] = clarification_question
        state["should_use_rag"] = should_use_rag
        
        logger.info(f"Query analyzed - needs clarification: {needs_clarification}, should use RAG: {should_use_rag}")
        return state
    
    def _determine_next_step(self, state: BabyCareState) -> str:
        """
        Determine the next step based on query analysis.
        
        Args:
            state (BabyCareState): Current conversation state
            
        Returns:
            str: Next node to execute
        """
        if state.get("needs_clarification", False):
            return "clarify"
        elif state.get("should_use_rag", False):
            return "retrieve"
        else:
            return "fallback"
    
    def _retrieve_context(self, state: BabyCareState) -> BabyCareState:
        """
        Retrieve relevant context for the user's query.
        
        Args:
            state (BabyCareState): Current conversation state
            
        Returns:
            BabyCareState: Updated state with retrieved context
        """
        last_message = state["messages"][-1]
        query = last_message.content
        
        try:
            # Retrieve documents
            documents = self.vector_store.search_documents(
                query, 
                k=self.rag_config.top_k_documents
            )
            
            # Get retrieval info
            retrieval_info = self.rag_system.get_retrieval_info(query)
            
            # Update state with retrieved context
            state["retrieved_documents"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
            state["retrieval_info"] = retrieval_info
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            state["retrieved_documents"] = []
            state["retrieval_info"] = {"error": str(e)}
        
        return state
    
    def _generate_rag_response(self, state: BabyCareState) -> BabyCareState:
        """
        Generate a response using RAG with retrieved context.
        
        Args:
            state (BabyCareState): Current conversation state
            
        Returns:
            BabyCareState: Updated state with AI response
        """
        last_message = state["messages"][-1]
        query = last_message.content
        
        try:
            # Prepare chat history
            chat_history = state["messages"][:-1]  # Exclude the current query
            
            # Generate RAG response
            response = self.rag_system.generate_response(query, chat_history)
            
            # Add response to conversation
            response_message = AIMessage(content=response)
            state["messages"].append(response_message)
            
            logger.info("Generated RAG response successfully")
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            fallback_message = AIMessage(
                content="I apologize, but I'm having trouble accessing my knowledge base right now. Please try again later."
            )
            state["messages"].append(fallback_message)
        
        return state
    
    def _request_clarification(self, state: BabyCareState) -> BabyCareState:
        """
        Request clarification from the user.
        
        Args:
            state (BabyCareState): Current conversation state
            
        Returns:
            BabyCareState: Updated state with clarification request
        """
        clarification_question = state.get("clarification_question", 
                                         "Could you please provide more details about your question?")
        
        clarification_message = AIMessage(content=clarification_question)
        state["messages"].append(clarification_message)
        
        logger.info("Clarification requested from user")
        return state
    
    def _fallback_response(self, state: BabyCareState) -> BabyCareState:
        """
        Generate a fallback response for general queries.
        
        Args:
            state (BabyCareState): Current conversation state
            
        Returns:
            BabyCareState: Updated state with fallback response
        """
        last_message = state["messages"][-1]
        query = last_message.content
        #TODO: make this more sophisticated
        # Simple fallback response
        fallback_responses = [
            "I'm here to help with baby care questions! Could you ask me something specific about nutrition, sleep, development, or safety?",
            "I specialize in baby care topics like feeding, sleep routines, developmental milestones, and safety. What would you like to know?",
            "I'd be happy to help with baby care questions! Try asking about feeding schedules, sleep patterns, or developmental milestones."
        ]
        
        # Choose response based on query content
        if "hello" in query.lower() or "hi" in query.lower():
            response = "Hello! I'm your baby care assistant. I can help with questions about nutrition, sleep, development, and safety. What would you like to know?"
        elif "help" in query.lower():
            response = "I can help with baby care topics including:\n• Feeding and nutrition\n• Sleep patterns and routines\n• Developmental milestones\n• Safety and childproofing\n• Common health concerns\n\nWhat specific topic interests you?"
        else:
            response = fallback_responses[0]
        
        fallback_message = AIMessage(content=response)
        state["messages"].append(fallback_message)
        
        logger.info("Generated fallback response")
        return state
    
    def chat(self, user_input: str, conversation_id: str = "default", user_id: Optional[str] = None) -> str:
        """
        Process a user input and return the chatbot's response.
        
        Args:
            user_input (str): The user's message
            conversation_id (str): Unique identifier for the conversation
            user_id (Optional[str]): User identifier for conversation storage
            
        Returns:
            str: The chatbot's response
        """
        # Start conversation if storage is enabled
        if self.conversation_storage:
            # Check if conversation exists, if not start a new one
            existing_metadata = self.conversation_storage.get_conversation_metadata(conversation_id)
            if not existing_metadata:
                self.conversation_storage.start_conversation(conversation_id, user_id)
        
        # Get conversation history if available
        chat_history = []
        if self.conversation_storage:
            chat_history = self.conversation_storage.get_conversation_history(conversation_id, limit=10)
        
        # Create initial state with history
        initial_state = {
            "messages": chat_history + [HumanMessage(content=user_input)],
            "user_context": {},
            "conversation_id": conversation_id,
            "session_start_time": datetime.now().isoformat(),
            "retrieved_documents": [],
            "needs_clarification": False,
            "clarification_question": None
        }
        
        # Process through the integrated graph
        result = self.graph.invoke(initial_state)
        
        # Get the AI response
        ai_message = result["messages"][-1]
        ai_response = ai_message.content
        
        # Store the conversation turn
        if self.conversation_storage:
            # Determine if RAG was used
            rag_used = result.get("should_use_rag", False)
            retrieval_info = result.get("retrieval_info", {})
            
            self.conversation_storage.save_conversation_turn(
                conversation_id=conversation_id,
                user_message=user_input,
                ai_response=ai_response,
                retrieval_info=retrieval_info,
                rag_used=rag_used
            )
        
        return ai_response
    
    def stream_chat(self, user_input: str, conversation_id: str = "default"):
        """
        Stream the chatbot's response for real-time interaction.
        
        Args:
            user_input (str): The user's message
            conversation_id (str): Unique identifier for the conversation
            
        Yields:
            str: Streaming response chunks
        """
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": {},
            "conversation_id": conversation_id,
            "session_start_time": datetime.now().isoformat(),
            "retrieved_documents": [],
            "needs_clarification": False,
            "clarification_question": None
        }
        
        # Stream through the integrated graph
        for event in self.graph.stream(initial_state):
            for node_name, node_output in event.items():
                if "messages" in node_output and node_output["messages"]:
                    last_message = node_output["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        yield last_message.content
    
    def stream_chat_with_retrieval_info(self, user_input: str, conversation_id: str = "default", user_id: Optional[str] = None):
        """
        Stream the chatbot's response and return retrieved documents with conversation storage.
        
        Args:
            user_input (str): The user's message
            conversation_id (str): Unique identifier for the conversation
            user_id (Optional[str]): User identifier for conversation storage
            
        Returns:
            Tuple[str, int, List[Dict]]: Response text, number of documents used, and document details
        """
        # Start conversation if storage is enabled
        if self.conversation_storage:
            # Check if conversation exists, if not start a new one
            existing_metadata = self.conversation_storage.get_conversation_metadata(conversation_id)
            if not existing_metadata:
                self.conversation_storage.start_conversation(conversation_id, user_id)
        
        # Get conversation history if available
        chat_history = []
        if self.conversation_storage:
            chat_history = self.conversation_storage.get_conversation_history(conversation_id, limit=10)
        
        # Use the same RAG decision logic as the chat flow
        state = {
            "messages": chat_history + [HumanMessage(content=user_input)],
            "needs_clarification": False,
            "clarification_question": "",
            "should_use_rag": False
        }
        
        # Determine if we should use RAG using the same logic as the chat flow
        state = self._analyze_query_with_rag(state)
        should_use_rag = state["should_use_rag"]
        
        if should_use_rag:
            # Use RAG system with retrieval info
            response, documents_used = self.rag_system.generate_response_with_retrieval_info(user_input)
            
            # Get the actual retrieved documents for display
            retrieved_docs = self.search_knowledge_base(user_input, k=documents_used)
            
            # Store the conversation turn
            if self.conversation_storage:
                retrieval_info = {"documents_used": documents_used, "retrieved_docs": retrieved_docs}
                self.conversation_storage.save_conversation_turn(
                    conversation_id=conversation_id,
                    user_message=user_input,
                    ai_response=response,
                    retrieval_info=retrieval_info,
                    rag_used=True
                )
            
            return response, documents_used, retrieved_docs
        else:
            # Use conversation flow for non-RAG queries
            response_chunks = []
            for chunk in self.stream_chat(user_input, conversation_id):
                response_chunks.append(chunk)
            
            response = "".join(response_chunks)
            
            # Store the conversation turn
            if self.conversation_storage:
                self.conversation_storage.save_conversation_turn(
                    conversation_id=conversation_id,
                    user_message=user_input,
                    ai_response=response,
                    retrieval_info={},
                    rag_used=False
                )
            
            return response, 0, []
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents to the knowledge base.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents to add
            
        Returns:
            int: Number of documents added
        """
        return self.vector_store.add_custom_documents(documents)
    
    def add_pdf_folder(self, folder_path: str) -> Dict[str, int]:
        """
        Add all PDF files from a folder to the knowledge base.
        
        Args:
            folder_path (str): Path to folder containing PDF files
            
        Returns:
            Dict[str, int]: Dictionary with filename as key and number of chunks added as value
        """
        logger.info(f"Adding PDF folder to knowledge base: {folder_path}")
        results = self.vector_store.add_documents_from_pdf_folder(folder_path)
        
        # Log summary
        total_chunks = sum(results.values())
        successful_files = len([f for f, chunks in results.items() if chunks > 0])
        
        logger.info(f"📚 PDF folder processing summary:")
        logger.info(f"   Files processed: {len(results)}")
        logger.info(f"   Successful: {successful_files}")
        logger.info(f"   Total chunks added: {total_chunks}")
        
        return results
    
    def add_documents_from_folder(self, folder_path: str, 
                                 file_types: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Add all supported files from a folder to the knowledge base.
        
        Args:
            folder_path (str): Path to folder containing files
            file_types (Optional[List[str]]): List of file types to process (e.g., ['pdf', 'txt'])
                                             If None, processes all supported file types
            
        Returns:
            Dict[str, int]: Dictionary with filename as key and number of chunks added as value
        """
        logger.info(f"Adding documents folder to knowledge base: {folder_path}")
        results = self.vector_store.add_documents_from_folder(folder_path, file_types)
        
        # Log summary
        total_chunks = sum(results.values())
        successful_files = len([f for f, chunks in results.items() if chunks > 0])
        
        logger.info(f"📚 Documents folder processing summary:")
        logger.info(f"   Files processed: {len(results)}")
        logger.info(f"   Successful: {successful_files}")
        logger.info(f"   Total chunks added: {total_chunks}")
        
        return results
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """
        Get information about the knowledge base.
        
        Returns:
            Dict[str, Any]: Knowledge base information
        """
        try:
            kb_info = self.vector_store.get_collection_info()
            
            # Add dynamic keywords information
            try:
                kb_keywords = self.vector_store.get_knowledge_base_keywords()
                kb_info["dynamic_keywords"] = kb_keywords
                kb_info["keyword_count"] = len(kb_keywords)
            except Exception as e:
                logger.warning(f"Could not get dynamic keywords: {e}")
                kb_info["dynamic_keywords"] = []
                kb_info["keyword_count"] = 0
            
            return kb_info
        except Exception as e:
            logger.error(f"Error getting knowledge base info: {e}")
            return {"error": str(e)}
    
    def get_rag_keywords(self) -> Dict[str, Any]:
        """
        Get all RAG keywords (both general and dynamic from knowledge base).
        
        Returns:
            Dict[str, Any]: RAG keywords information
        """
        try:
            # Get dynamic keywords from knowledge base
            kb_keywords = self.vector_store.get_knowledge_base_keywords()
            
            # General baby care keywords
            general_keywords = [
                "nutrition", "feeding", "sleep", "development", "milestone",
                "health", "safety", "childproofing", "routine", "schedule",
                "food", "milk", "formula", "solid", "teething", "colic",
                "baby", "infant", "toddler", "newborn", "pregnancy", "pregnant"
            ]
            
            # Combine and deduplicate
            all_keywords = list(set(general_keywords + kb_keywords))
            
            return {
                "general_keywords": general_keywords,
                "knowledge_base_keywords": kb_keywords,
                "all_keywords": all_keywords,
                "total_count": len(all_keywords),
                "kb_count": len(kb_keywords),
                "general_count": len(general_keywords)
            }
            
        except Exception as e:
            logger.error(f"Error getting RAG keywords: {e}")
            return {"error": str(e)}
    
    def search_knowledge_base(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        documents = self.vector_store.search_documents(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get a summary of costs and usage for the current session.
        
        Returns:
            Dict[str, Any]: Cost and usage summary
        """
        if self.monitor:
            return self.monitor.get_cost_summary()
        else:
            return {"error": "Monitoring not enabled"}
    
    def print_cost_summary(self) -> None:
        """Print a formatted cost summary to the console."""
        if self.monitor:
            self.monitor.print_cost_summary()
        else:
            print("❌ Cost monitoring is not enabled")
    
    def export_cost_data(self, filename: Optional[str] = None) -> str:
        """
        Export cost data to a JSON file.
        
        Args:
            filename (Optional[str]): Output filename
            
        Returns:
            str: Path to the exported file
        """
        if self.monitor:
            return self.monitor.export_cost_data(filename)
        else:
            logger.warning("Monitoring not enabled - cannot export cost data")
            return ""
    
    def chat_with_retrieval_info(self, message: str) -> Tuple[str, int]:
        """
        Chat with the bot and return both response and number of documents used.
        
        Args:
            message (str): User's message
            
        Returns:
            Tuple[str, int]: Response text and number of documents used
        """
        try:
            # Use the same RAG decision logic as the chat flow
            state = {
                "messages": [HumanMessage(content=message)],
                "needs_clarification": False,
                "clarification_question": "",
                "should_use_rag": False
            }
            
            # Determine if we should use RAG using the same logic as the chat flow
            state = self._analyze_query_with_rag(state)
            should_use_rag = state["should_use_rag"]
            
            if should_use_rag:
                # Use RAG system with retrieval info
                response, documents_used = self.rag_system.generate_response_with_retrieval_info(message)
                return response, documents_used
            else:
                # Use fallback response
                fallback_responses = [
                    "I'm here to help with baby care questions! Could you ask me something specific about nutrition, sleep, development, safety, or health?",
                    "I specialize in baby care topics. Please ask me about feeding, sleep patterns, developmental milestones, safety concerns, or health issues.",
                    "I'd be happy to help with baby care questions! Try asking about specific topics like nutrition, sleep routines, or developmental milestones."
                ]
                import random
                response = random.choice(fallback_responses)
                return response, 0
                
        except Exception as e:
            logger.error(f"Error in chat_with_retrieval_info: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again later.", 0
    
    def get_conversation_history(self, conversation_id: str, limit: Optional[int] = None) -> List[BaseMessage]:
        """
        Get conversation history for a specific conversation.
        
        Args:
            conversation_id (str): Conversation identifier
            limit (Optional[int]): Maximum number of messages to retrieve
            
        Returns:
            List[BaseMessage]: LangChain message objects
        """
        if self.conversation_storage:
            return self.conversation_storage.get_conversation_history(conversation_id, limit)
        return []
    
    def list_conversations(self, user_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List conversations with optional filtering.
        
        Args:
            user_id (Optional[str]): Filter by user ID
            limit (Optional[int]): Maximum number of conversations to return
            
        Returns:
            List[Dict[str, Any]]: List of conversation metadata
        """
        if self.conversation_storage:
            return self.conversation_storage.list_conversations(user_id, limit)
        return []
    
    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get statistics for a conversation.
        
        Args:
            conversation_id (str): Conversation identifier
            
        Returns:
            Dict[str, Any]: Conversation statistics
        """
        if self.conversation_storage:
            return self.conversation_storage.get_conversation_stats(conversation_id)
        return {}
    
    def search_conversations(self, query_text: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search conversations by message content.
        
        Args:
            query_text (str): Search query
            user_id (Optional[str]): Filter by user ID
            
        Returns:
            List[Dict[str, Any]]: Matching conversations with snippets
        """
        if self.conversation_storage:
            return self.conversation_storage.search_conversations(query_text, user_id)
        return []
    
    def export_conversation(self, conversation_id: str, format: str = 'json') -> str:
        """
        Export a conversation to a file.
        
        Args:
            conversation_id (str): Conversation identifier
            format (str): Export format ('json' or 'txt')
            
        Returns:
            str: Path to exported file
        """
        if self.conversation_storage:
            return self.conversation_storage.export_conversation(conversation_id, format)
        return ""
    
    def end_conversation(self, conversation_id: str) -> bool:
        """
        End a conversation session.
        
        Args:
            conversation_id (str): Conversation identifier
            
        Returns:
            bool: True if ended successfully
        """
        if self.conversation_storage:
            return self.conversation_storage.end_conversation(conversation_id)
        return False


def main():
    """
    Main function to demonstrate the integrated chatbot.
    """
    print("Initializing Integrated Baby Care Chatbot...")
    
    # Initialize the integrated chatbot with conversation storage
    chatbot = IntegratedBabyCareChatbot(
        enable_conversation_storage=True,
        conversation_db_path="babycare_conversations.json"
    )
    
    # Get knowledge base info
    kb_info = chatbot.get_knowledge_base_info()
    print(f"Knowledge base contains {kb_info.get('document_count', 0)} documents")
    
    print("\n" + "="*60)
    print("👶 INTEGRATED BABY CARE CHATBOT - Your Expert Guide 👶")
    print("="*60)
    print("Ask me anything about:")
    print("• Baby nutrition and feeding")
    print("• Healthcare and development")
    print("• Sleep patterns and routines")
    print("• Safety and childproofing")
    print("• Common concerns and troubleshooting")
    print("\nType 'quit', 'exit', or 'q' to end the conversation.")
    print("Type 'stats' to see conversation statistics.")
    print("Type 'history' to see conversation history.")
    print("="*60 + "\n")
    
    conversation_id = "integrated_session"
    user_id = "demo_user"
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["quit", "exit", "q"]:
                # End the conversation
                chatbot.end_conversation(conversation_id)
                print("\n👶 Goodbye! Take care of your little one! 👶")
                break
            
            # Handle special commands
            if user_input.lower() == "stats":
                stats = chatbot.get_conversation_stats(conversation_id)
                print(f"\n📊 Conversation Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if user_input.lower() == "history":
                history = chatbot.get_conversation_history(conversation_id, limit=5)
                print(f"\n📜 Recent Conversation History ({len(history)} messages):")
                for i, msg in enumerate(history[-5:], 1):
                    role = "👤 User" if msg.__class__.__name__ == "HumanMessage" else "🤖 Assistant"
                    print(f"  {i}. {role}: {msg.content[:100]}...")
                continue
            
            print("Assistant: ", end="", flush=True)
            
            # Get AI response with conversation storage
            response = chatbot.chat(
                user_input=user_input,
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            print(response)
            print()  # New line after response
            
        except KeyboardInterrupt:
            chatbot.end_conversation(conversation_id)
            print("\n\n👶 Goodbye! Take care of your little one! 👶")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or contact support if the issue persists.")


if __name__ == "__main__":
    main()
