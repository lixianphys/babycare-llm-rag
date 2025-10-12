"""
LangSmith Monitoring Module for Baby Care Chatbot

This module provides comprehensive monitoring, cost tracking, and performance
analytics for the baby care chatbot using LangSmith.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CostTrackingCallback(BaseCallbackHandler):
    """
    Custom callback handler for tracking costs and performance metrics.
    
    This handler captures detailed information about each LLM call including
    token usage, costs, and timing information.
    """
    
    def __init__(self):
        """Initialize the cost tracking callback."""
        self.total_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self.call_details = []
        
        # OpenAI pricing (as of 2024 - update as needed)
        self.pricing = {
            "gpt-4o-mini": {
                "input": 0.00015 / 1000,  # $0.15 per 1K tokens
                "output": 0.0006 / 1000   # $0.60 per 1K tokens
            },
            "text-embedding-3-small": {
                "input": 0.00002 / 1000   # $0.02 per 1K tokens
            }
        }
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts processing."""
        self.call_count += 1
        logger.info(f"LLM call #{self.call_count} started")
    
    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Called when LLM finishes processing."""
        try:
            # Extract token usage information from different possible locations
            token_usage = None
            model_name = 'gpt-4o-mini'
            
            # Try to get token usage from response.llm_output
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
            
            # Try to get token usage from response.usage_metadata
            elif hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_usage = {
                    'prompt_tokens': response.usage_metadata.input_tokens,
                    'completion_tokens': response.usage_metadata.output_tokens,
                    'total_tokens': response.usage_metadata.total_tokens
                }
            
            # Try to get model name
            if hasattr(response, 'model_name'):
                model_name = response.model_name
            elif hasattr(response, 'model'):
                model_name = response.model
            
            if token_usage and any(token_usage.values()):
                prompt_tokens = token_usage.get('prompt_tokens', 0)
                completion_tokens = token_usage.get('completion_tokens', 0)
                total_tokens = token_usage.get('total_tokens', 0)
                
                # Calculate cost based on model
                cost = self._calculate_cost(model_name, prompt_tokens, completion_tokens)
                
                # Update totals
                self.total_tokens += total_tokens
                self.total_cost += cost
                
                # Store call details
                call_detail = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost,
                    "type": "chat_completion"
                }
                self.call_details.append(call_detail)
                
                logger.info(f"LLM call completed - Model: {model_name}, Tokens: {total_tokens}, Cost: ${cost:.4f}")
            else:
                logger.warning("No token usage information found in LLM response")
        
        except Exception as e:
            logger.error(f"Error processing LLM end callback: {e}")
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM encounters an error."""
        logger.error(f"LLM error: {error}")
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the cost for a given model and token usage.
        
        Args:
            model (str): Model name
            prompt_tokens (int): Number of prompt tokens
            completion_tokens (int): Number of completion tokens
            
        Returns:
            float: Calculated cost in USD
        """
        try:
            if model in self.pricing:
                pricing = self.pricing[model]
                input_cost = prompt_tokens * pricing.get("input", 0)
                output_cost = completion_tokens * pricing.get("output", 0)
                return input_cost + output_cost
            else:
                logger.warning(f"Unknown model pricing: {model}")
                return 0.0
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tracked metrics.
        
        Returns:
            Dict[str, Any]: Summary of costs and usage
        """
        return {
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "average_cost_per_call": round(self.total_cost / max(self.call_count, 1), 4),
            "call_details": self.call_details
        }


class LangSmithMonitor:
    """
    LangSmith monitoring manager for the baby care chatbot.
    
    This class provides comprehensive monitoring capabilities including
    cost tracking, performance analytics, and debugging support.
    """
    
    def __init__(self, project_name: str = "baby-care-chatbot"):
        """
        Initialize the LangSmith monitor.
        
        Args:
            project_name (str): Name of the LangSmith project
        """
        self.project_name = project_name
        self.client = None
        self.tracer = None
        self.cost_tracker = CostTrackingCallback()
        
        # Initialize LangSmith client if API key is available
        if os.getenv("LANGCHAIN_API_KEY"):
            try:
                self.client = Client()
                self.tracer = LangChainTracer(project_name=project_name)
                logger.info(f"LangSmith monitoring initialized for project: {project_name}")
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith client: {e}")
        else:
            logger.warning("LANGCHAIN_API_KEY not found. LangSmith monitoring disabled.")
    
    def get_callbacks(self) -> List[BaseCallbackHandler]:
        """
        Get the list of callback handlers for monitoring.
        
        Returns:
            List[BaseCallbackHandler]: List of callback handlers
        """
        callbacks = [self.cost_tracker]
        
        if self.tracer:
            callbacks.append(self.tracer)
        
        return callbacks
    
    def log_query(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a query and response to LangSmith.
        
        Args:
            query (str): User query
            response (str): Bot response
            metadata (Optional[Dict[str, Any]]): Additional metadata
        """
        if not self.client:
            return
        
        try:
            # Create a run for this query
            run_data = {
                "name": "baby_care_query",
                "inputs": {"query": query},
                "outputs": {"response": response},
                "metadata": metadata or {},
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "run_type": "chain"
            }
            
            # Log to LangSmith
            self.client.create_run(**run_data)
            logger.info(f"Query logged to LangSmith: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error logging query to LangSmith: {e}")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get a summary of costs and usage.
        
        Returns:
            Dict[str, Any]: Cost and usage summary
        """
        return self.cost_tracker.get_summary()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current session.
        
        Returns:
            Dict[str, Any]: Session statistics
        """
        summary = self.get_cost_summary()
        
        return {
            "session_start": datetime.now().isoformat(),
            "total_queries": summary["total_calls"],
            "total_tokens": summary["total_tokens"],
            "total_cost_usd": summary["total_cost"],
            "average_cost_per_query": summary["average_cost_per_call"],
            "langsmith_enabled": self.client is not None,
            "project_name": self.project_name
        }
    
    def export_cost_data(self, filename: Optional[str] = None) -> str:
        """
        Export cost data to a JSON file.
        
        Args:
            filename (Optional[str]): Output filename
            
        Returns:
            str: Path to the exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"baby_care_chatbot_costs_{timestamp}.json"
        
        try:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "session_stats": self.get_session_stats(),
                "detailed_calls": self.cost_tracker.call_details
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Cost data exported to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting cost data: {e}")
            return ""
    
    def print_cost_summary(self) -> None:
        """Print a formatted cost summary to the console."""
        stats = self.get_session_stats()
        
        print("\n" + "="*60)
        print("💰 BABY CARE CHATBOT - COST SUMMARY")
        print("="*60)
        print(f"📊 Total Queries: {stats['total_queries']}")
        print(f"🔤 Total Tokens: {stats['total_tokens']:,}")
        print(f"💵 Total Cost: ${stats['total_cost_usd']:.4f}")
        print(f"📈 Avg Cost/Query: ${stats['average_cost_per_query']:.4f}")
        print(f"🔍 LangSmith: {'✅ Enabled' if stats['langsmith_enabled'] else '❌ Disabled'}")
        print(f"📁 Project: {stats['project_name']}")
        print("="*60)
        
        if stats['total_queries'] > 0:
            print("\n📋 Recent Calls:")
            for i, call in enumerate(self.cost_tracker.call_details[-5:], 1):
                print(f"  {i}. {call['model']} - {call['total_tokens']} tokens - ${call['cost']:.4f}")
        
        print()


def create_monitoring_decorator(monitor: LangSmithMonitor):
    """
    Create a decorator for monitoring function calls.
    
    Args:
        monitor (LangSmithMonitor): The monitoring instance
        
    Returns:
        function: Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful execution
                execution_time = (datetime.now() - start_time).total_seconds()
                monitor.log_query(
                    query=f"Function call: {func.__name__}",
                    response=f"Success - {execution_time:.2f}s",
                    metadata={
                        "function": func.__name__,
                        "execution_time": execution_time,
                        "status": "success"
                    }
                )
                
                return result
                
            except Exception as e:
                # Log error
                execution_time = (datetime.now() - start_time).total_seconds()
                monitor.log_query(
                    query=f"Function call: {func.__name__}",
                    response=f"Error: {str(e)}",
                    metadata={
                        "function": func.__name__,
                        "execution_time": execution_time,
                        "status": "error",
                        "error": str(e)
                    }
                )
                raise
        
        return wrapper
    return decorator


def main():
    """
    Main function to demonstrate LangSmith monitoring.
    """
    print("🔍 LangSmith Monitoring Demo")
    print("="*40)
    
    # Initialize monitor
    monitor = LangSmithMonitor()
    
    # Get callbacks for use with LangChain
    callbacks = monitor.get_callbacks()
    print(f"✅ Initialized with {len(callbacks)} callback handlers")
    
    # Print initial stats
    monitor.print_cost_summary()
    
    # Simulate some usage
    print("📝 Simulating query logging...")
    monitor.log_query(
        query="How much should a 3-month-old baby eat?",
        response="For a 3-month-old baby, they should typically eat 4-6 ounces of formula or breast milk every 3-4 hours...",
        metadata={"category": "nutrition", "age_range": "3_months"}
    )
    
    # Print updated stats
    monitor.print_cost_summary()
    
    # Export data
    export_file = monitor.export_cost_data()
    if export_file:
        print(f"📁 Cost data exported to: {export_file}")


if __name__ == "__main__":
    main()
