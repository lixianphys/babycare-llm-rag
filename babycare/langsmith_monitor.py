"""
LangSmith Monitoring Module

This module provides comprehensive monitoring, cost tracking, and performance analytics using LangSmith.
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


class LangSmithCostTracker:
    """
    Cost tracker that uses LangSmith Client to query token usage information.
    """
    def __init__(self, client: Client, project_name: str):
        """
        Initialize the cost tracker.
        
        Args:
            client (Client): LangSmith client instance
            project_name (str): Name of the LangSmith project
        """
        self.client = client
        self.project_name = project_name
        self.cached_runs = []
        self.last_query_time = None
        
        # OpenAI pricing (as of 2025 - update as needed)
        self.pricing = {
            "gpt-4o-mini": {
                "input": 0.00015 / 1000,  # $0.15 per 1K tokens
                "output": 0.0006 / 1000   # $0.60 per 1K tokens
            },
            "gpt-4o": {
                "input": 0.0025 / 1000,   # $2.50 per 1K tokens
                "output": 0.01 / 1000     # $10.00 per 1K tokens
            },
            "text-embedding-3-small": {
                "input": 0.00002 / 1000   # $0.02 per 1K tokens
            },
            "text-embedding-3-large": {
                "input": 0.00013 / 1000   # $0.13 per 1K tokens
            }
        }
    
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
            # Normalize model name to match pricing keys
            model_key = model.lower().replace("-", "_")
            if model_key in self.pricing:
                pricing = self.pricing[model_key]
                input_cost = prompt_tokens * pricing.get("input", 0)
                output_cost = completion_tokens * pricing.get("output", 0)
                return input_cost + output_cost
            else:
                logger.warning(f"Unknown model pricing: {model}")
                return 0.0
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return 0.0
    
    def _extract_token_usage_from_run(self, run) -> Optional[Dict[str, Any]]:
        """
        Extract token usage information from a LangSmith run.
        
        Args:
            run: LangSmith run object
            
        Returns:
            Optional[Dict[str, Any]]: Token usage data or None if not found
        """
        try:
            
            token_usage = {
            'prompt_tokens': run.prompt_tokens,
            'completion_tokens': run.completion_tokens,
            'total_tokens': run.total_tokens
            }
            return token_usage
            
        except Exception as e:
            logger.error(f"Error extracting token usage from run: {e}")
            return None
    
    def get_recent_runs_with_tokens(self, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Get recent runs from LangSmith that have token usage information.
        
        Args:
            limit (int): Maximum number of runs to fetch
            
        Returns:
            List[Dict[str, Any]]: List of runs with token usage data
        """
        try:
            # Query recent runs from the project
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                limit=limit,
                order_by="start_time",
                order="desc"
            ))
            logger.info(f"Retrieved {len(runs)} runs from LangSmith")
            runs_with_tokens = []
            for run in runs:
                token_usage = self._extract_token_usage_from_run(run)
                if token_usage:
                    run_data = {
                        'run_id': run.id,
                        'name': run.name,
                        'start_time': run.start_time.isoformat() if run.start_time else None,
                        'end_time': run.end_time.isoformat() if run.end_time else None,
                        'model': self._extract_model_name(run),
                        'token_usage': token_usage,
                        'cost': self._calculate_cost(
                            self._extract_model_name(run),
                            token_usage.get('prompt_tokens', 0),
                            token_usage.get('completion_tokens', 0)
                        )
                    }
                    runs_with_tokens.append(run_data)
            
            self.cached_runs = runs_with_tokens
            self.last_query_time = datetime.now()
            
            logger.info(f"Retrieved {len(runs_with_tokens)} runs with token usage from LangSmith")
            return runs_with_tokens
            
        except Exception as e:
            logger.error(f"Error fetching runs from LangSmith: {e}")
            return []
    
    def _extract_model_name(self, run) -> str:
        """
        Extract model name from a LangSmith run.
        
        Args:
            run: LangSmith run object
            
        Returns:
            str: Model name or default
        """
        try:
            # Check various possible locations for model name
            extra = getattr(run, 'extra', None)
            if extra:
                if isinstance(extra, dict):
                    # Check for model in extra metadata
                    if 'model' in extra:
                        return extra['model']
                    if 'model_name' in extra:
                        return extra['model_name']
                else:
                    # If extra is an object, try attribute access
                    if hasattr(extra, 'model'):
                        return extra.model
                    if hasattr(extra, 'model_name'):
                        return extra.model_name
            
            # Check inputs for model information
            inputs = getattr(run, 'inputs', None)
            if inputs:
                if isinstance(inputs, dict):
                    if 'model' in inputs:
                        return inputs['model']
                    if 'model_name' in inputs:
                        return inputs['model_name']
                else:
                    # If inputs is an object, try attribute access
                    if hasattr(inputs, 'model'):
                        return inputs.model
                    if hasattr(inputs, 'model_name'):
                        return inputs.model_name
            
            # Check outputs for model information
            outputs = getattr(run, 'outputs', None)
            if outputs:
                if isinstance(outputs, dict):
                    if 'model' in outputs:
                        return outputs['model']
                    if 'model_name' in outputs:
                        return outputs['model_name']
                else:
                    # If outputs is an object, try attribute access
                    if hasattr(outputs, 'model'):
                        return outputs.model
                    if hasattr(outputs, 'model_name'):
                        return outputs.model_name
            
            # Default fallback
            return 'gpt-4o-mini'
            
        except Exception as e:
            logger.error(f"Error extracting model name: {e}")
            return 'gpt-4o-mini'
    
    def get_cost_summary(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get a comprehensive cost summary from LangSmith data.
        
        Args:
            limit (int): Maximum number of runs to analyze
            
        Returns:
            Dict[str, Any]: Cost and usage summary
        """
        try:
            # Get recent runs with token usage
            runs = self.get_recent_runs_with_tokens(limit)
            
            if not runs:
                return {
                    "total_calls": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "average_cost_per_call": 0.0,
                    "call_details": [],
                    "model_breakdown": {},
                    "last_updated": datetime.now().isoformat()
                }
            
            # Calculate totals
            total_calls = len(runs)
            total_tokens = sum(run['token_usage'].get('total_tokens', 0) for run in runs)
            total_cost = sum(run['cost'] for run in runs)
            
            # Model breakdown
            model_breakdown = {}
            for run in runs:
                model = run['model']
                if model not in model_breakdown:
                    model_breakdown[model] = {
                        'calls': 0,
                        'tokens': 0,
                        'cost': 0.0
                    }
                model_breakdown[model]['calls'] += 1
                model_breakdown[model]['tokens'] += run['token_usage'].get('total_tokens', 0)
                model_breakdown[model]['cost'] += run['cost']
            
            return {
                "total_calls": total_calls,
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 4),
                "average_cost_per_call": round(total_cost / max(total_calls, 1), 4),
                "call_details": runs,
                "model_breakdown": model_breakdown,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating cost summary: {e}")
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "average_cost_per_call": 0.0,
                "call_details": [],
                "model_breakdown": {},
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }


class LangSmithMonitor:
    """
    LangSmith monitoring manager for the baby care chatbot.
    
    This class provides comprehensive monitoring capabilities including
    cost tracking, performance analytics, and debugging support.
    """
    
    def __init__(self, project_name: str):
        """
        Initialize the LangSmith monitor.
        
        Args:
            project_name (str): Name of the LangSmith project
        """
        self.project_name = project_name
        self.client = None
        self.tracer = None
        self.cost_tracker = None
        
        # Initialize LangSmith client if API key is available
        if os.getenv("LANGCHAIN_API_KEY"):
            try:
                self.client = Client()
                self.tracer = LangChainTracer(project_name=project_name)
                self.cost_tracker = LangSmithCostTracker(self.client, project_name)
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
        callbacks = []
        
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
                "name": self.project_name + "_query",
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
    
    def get_cost_summary(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get a summary of costs and usage from LangSmith data.
        
        Args:
            limit (int): Maximum number of runs to analyze
            
        Returns:
            Dict[str, Any]: Cost and usage summary
        """
        if self.cost_tracker:
            return self.cost_tracker.get_cost_summary(limit)
        else:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "average_cost_per_call": 0.0,
                "call_details": [],
                "model_breakdown": {},
                "error": "LangSmith client not initialized",
                "last_updated": datetime.now().isoformat()
            }
    
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
            "project_name": self.project_name,
            "model_breakdown": summary.get("model_breakdown", {}),
            "last_updated": summary.get("last_updated", datetime.now().isoformat())
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
            filename = f"{self.project_name}_costs_{timestamp}.json"
        
        try:
            cost_summary = self.get_cost_summary()
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "session_stats": self.get_session_stats(),
                "detailed_calls": cost_summary.get("call_details", []),
                "model_breakdown": cost_summary.get("model_breakdown", {})
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Cost data exported to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting cost data: {e}")
            return ""
    
    def _print_cost_summary(self) -> None:
        """Print a formatted cost summary to the console (For debugging purposes)."""
        stats = self.get_session_stats()
        cost_summary = self.get_cost_summary()
        
        print("\n" + "="*60)
        print("Cost Summary")
        print("="*60)
        print(f" Total Queries: {stats['total_queries']}")
        print(f" Total Tokens: {stats['total_tokens']:,}")
        print(f" Total Cost: ${stats['total_cost_usd']:.4f}")
        print(f" Avg Cost/Query: ${stats['average_cost_per_query']:.4f}")
        print(f" LangSmith: {'Enabled' if stats['langsmith_enabled'] else 'Disabled'}")
        print(f" Project: {stats['project_name']}")
        print(f" Last Updated: {stats.get('last_updated', 'Unknown')}")
        print("="*60)
        
        # Print model breakdown
        model_breakdown = stats.get('model_breakdown', {})
        if model_breakdown:
            print("\n Model Breakdown:")
            for model, data in model_breakdown.items():
                print(f"  {model}: {data['calls']} calls, {data['tokens']:,} tokens, ${data['cost']:.4f}")
        
        # Print recent calls
        call_details = cost_summary.get('call_details', [])
        if call_details:
            print("\n Recent Calls:")
            for i, call in enumerate(call_details[-5:], 1):
                print(f"  {i}. {call['model']} - {call['token_usage'].get('total_tokens', 0)} tokens - ${call['cost']:.4f}")
        
        print()
    
    def refresh_cost_data(self, limit: int = 100) -> Dict[str, Any]:
        """
        Refresh cost data from LangSmith by querying recent runs.
        
        Args:
            limit (int): Maximum number of runs to fetch
            
        Returns:
            Dict[str, Any]: Updated cost summary
        """
        if self.cost_tracker:
            logger.info(f"Refreshing cost data from LangSmith (limit: {limit})")
            return self.cost_tracker.get_cost_summary(limit)
        else:
            logger.warning("Cannot refresh cost data: LangSmith client not initialized")
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "average_cost_per_call": 0.0,
                "call_details": [],
                "model_breakdown": {},
                "error": "LangSmith client not initialized",
                "last_updated": datetime.now().isoformat()
            }


def create_monitoring_decorator(monitor: LangSmithMonitor):
    """
    Create a decorator for monitoring function calls.
    
    Args:
        monitor (LangSmithMonitor): The monitoring instance
        
    Returns:
        function: Decorator function

    Usage:
    @create_monitoring_decorator(monitor)
    def my_function(*args, **kwargs):
        ...
    """
    from functools import wraps
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function for the decorated function."""
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
    For debugging purposes.
    """
    print(" LangSmith Monitoring Demo")
    print("="*40)
    
    # Initialize monitor
    monitor = LangSmithMonitor("baby-care-chatbot")
    
    # Get callbacks for use with LangChain
    callbacks = monitor.get_callbacks()
    print(f"Initialized with {len(callbacks)} callback handlers")
    
    # Print initial stats
    monitor._print_cost_summary()
    
    # Simulate some usage
    print("Simulating query logging...")
    monitor.log_query(
        query="How much should a 3-month-old baby eat?",
        response="For a 3-month-old baby, they should typically eat 4-6 ounces of formula or breast milk every 3-4 hours...",
        metadata={"category": "nutrition", "age_range": "3_months"}
    )
    
    # Print updated stats
    monitor._print_cost_summary()
    
    # Export data
    export_file = monitor.export_cost_data("test_cost_data.json")
    if export_file:
        print(f" Cost data exported to: {export_file}")


if __name__ == "__main__":
    main()
