"""
Enhanced Chat Flow Module for Baby Care Chatbot

This module implements the core conversation flow using LangGraph,
providing structured state management and conversation handling
for baby care related queries.
"""

from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BabyCareState(TypedDict):
    """
    State management for the baby care chatbot conversation.
    
    This TypedDict defines the structure of the conversation state,
    including messages, user context, and conversation metadata.
    """
    # Messages in the conversation - uses add_messages for proper message handling
    messages: Annotated[List[BaseMessage], add_messages]
    
    # User context information (baby's age, specific concerns, etc.)
    user_context: Dict[str, Any]
    
    # Conversation metadata
    conversation_id: str
    session_start_time: str
    
    # RAG context (will be populated when RAG is integrated)
    retrieved_documents: List[Dict[str, Any]]
    
    # Conversation flow control
    needs_clarification: bool
    clarification_question: Optional[str]


class BabyCareChatFlow:
    """
    Enhanced chat flow manager for baby care conversations.
    
    This class manages the conversation flow, state transitions,
    and provides specialized handling for baby care related queries.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """
        Initialize the chat flow with specified LLM configuration.
        
        Args:
            model_name (str): The OpenAI model to use for chat completion
            temperature (float): Temperature setting for response generation (0.0-1.0)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            max_tokens=1024,
            temperature=temperature,
            streaming=True,
            timeout=30
        )
        
        # System prompt for baby care specialization
        self.system_prompt = self._create_system_prompt()
        
        # Build the conversation graph
        self.graph = self._build_conversation_graph()
        
        logger.info(f"BabyCareChatFlow initialized with model: {model_name}")
    
    def _create_system_prompt(self) -> str:
        """
        Create a specialized system prompt for baby care conversations.
        
        Returns:
            str: System prompt that defines the chatbot's role and capabilities
        """
        return """You are a specialized AI assistant focused on providing expert guidance on baby care, 
        nutrition, healthcare, and child development. Your expertise includes:

        - Baby nutrition and feeding schedules
        - Healthcare and medical guidance (with appropriate disclaimers)
        - Developmental milestones and activities
        - Sleep patterns and routines
        - Safety and childproofing
        - Common concerns and troubleshooting

        Always provide:
        1. Evidence-based information
        2. Age-appropriate advice
        3. Clear disclaimers for medical advice
        4. Encouragement to consult healthcare professionals when needed
        5. Practical, actionable guidance

        Be warm, supportive, and understanding of parents' concerns while maintaining professional accuracy."""
    
    def _build_conversation_graph(self) -> StateGraph:
        """
        Build the LangGraph conversation flow with specialized nodes.
        
        Returns:
            StateGraph: Compiled graph for conversation management
        """
        # Create the state graph
        graph_builder = StateGraph(BabyCareState)
        
        # Add conversation nodes
        graph_builder.add_node("analyze_query", self._analyze_query)
        graph_builder.add_node("generate_response", self._generate_response)
        graph_builder.add_node("request_clarification", self._request_clarification)
        
        # Define the conversation flow
        graph_builder.add_edge(START, "analyze_query")
        graph_builder.add_conditional_edges(
            "analyze_query",
            self._should_clarify,
            {
                "clarify": "request_clarification",
                "respond": "generate_response"
            }
        )
        graph_builder.add_edge("request_clarification", END)
        graph_builder.add_edge("generate_response", END)
        
        return graph_builder.compile()
    
    def _analyze_query(self, state: BabyCareState) -> BabyCareState:
        """
        Analyze the user's query to determine if clarification is needed.
        
        Args:
            state (BabyCareState): Current conversation state
            
        Returns:
            BabyCareState: Updated state with analysis results
        """
        last_message = state["messages"][-1]
        
        # Simple analysis - in a real implementation, this could be more sophisticated
        needs_clarification = False
        clarification_question = None
        
        # Check if the query is too vague or needs more context
        if len(last_message.content) < 10:
            needs_clarification = True
            clarification_question = "Could you please provide more details about your question? For example, what is your baby's age or what specific aspect of baby care are you asking about?"
        
        # Update state
        state["needs_clarification"] = needs_clarification
        state["clarification_question"] = clarification_question
        
        logger.info(f"Query analyzed - needs clarification: {needs_clarification}")
        return state
    
    def _should_clarify(self, state: BabyCareState) -> str:
        """
        Determine if clarification is needed based on query analysis.
        
        Args:
            state (BabyCareState): Current conversation state
            
        Returns:
            str: Next node to execute ("clarify" or "respond")
        """
        return "clarify" if state.get("needs_clarification", False) else "respond"
    
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
    
    def _generate_response(self, state: BabyCareState) -> BabyCareState:
        """
        Generate a response using the LLM with baby care specialization.
        
        Args:
            state (BabyCareState): Current conversation state
            
        Returns:
            BabyCareState: Updated state with AI response
        """
        # Prepare messages with system prompt
        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
        
        # Generate response
        response = self.llm.invoke(messages)
        
        # Add response to conversation
        state["messages"].append(response)
        
        logger.info("Response generated successfully")
        return state
    
    def chat(self, user_input: str, conversation_id: str = "default") -> str:
        """
        Process a user input and return the chatbot's response.
        
        Args:
            user_input (str): The user's message
            conversation_id (str): Unique identifier for the conversation
            
        Returns:
            str: The chatbot's response
        """
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": {},
            "conversation_id": conversation_id,
            "session_start_time": "",
            "retrieved_documents": [],
            "needs_clarification": False,
            "clarification_question": None
        }
        
        # Process through the graph
        result = self.graph.invoke(initial_state)
        
        # Return the last AI message
        last_message = result["messages"][-1]
        return last_message.content
    
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
            "session_start_time": "",
            "retrieved_documents": [],
            "needs_clarification": False,
            "clarification_question": None
        }
        
        # Stream through the graph
        for event in self.graph.stream(initial_state):
            for node_name, node_output in event.items():
                if node_name == "generate_response" and "messages" in node_output:
                    last_message = node_output["messages"][-1]
                    yield last_message.content


def main():
    """
    Main function to demonstrate the enhanced chat flow.
    """
    # Initialize the chat flow
    chat_flow = BabyCareChatFlow()
    
    print("Baby Care Chatbot - Enhanced Version")
    print("Ask me anything about baby care, nutrition, healthcare, or development!")
    print("Type 'quit', 'exit', or 'q' to end the conversation.\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye! Take care of your little one! 👶")
                break
            
            print("Assistant: ", end="", flush=True)
            for chunk in chat_flow.stream_chat(user_input):
                print(chunk, end="", flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye! Take care of your little one! 👶")
            break
        except Exception as e:
            print(f"Error: {e}")
            logger.error(f"Error in main chat loop: {e}")


if __name__ == "__main__":
    main()
