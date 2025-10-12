"""
Simple Vector Store Module for Baby Care Knowledge Base

This module implements a simple in-memory vector storage system for storing
and retrieving baby care documents with semantic search capabilities.
Uses only packages available in the climate-question-answering environment.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader, 
    DirectoryLoader
)
from dotenv import load_dotenv

from .embeddings import get_embeddings_function
from .config import config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleBabyCareVectorStore:
    """
    Simple in-memory vector store manager for baby care knowledge base.
    
    This class handles the creation, management, and querying of a simple
    in-memory vector database containing baby care documents with semantic search.
    """
    
    def __init__(self):
        """
        Initialize the simple vector store with specified configuration.
        
        Args:
            embedding_model (str): OpenAI embedding model to use
        """
        
        # Initialize OpenAI embeddings
        self.embeddings, self.embeddings_model_spec = get_embeddings_function(local=config.local_embedding_model)
        
        # In-memory storage
        self.documents = []  # List of Document objects
        self.embeddings_cache = []  # List of embedding vectors
        self.metadata_cache = []  # List of metadata dictionaries
        
        # Text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"SimpleBabyCareVectorStore initialized with model: {self.embeddings_model_spec['model_name']}")
    
    def add_documents_from_directory(self, directory_path: str, 
                                   file_pattern: str = "**/*.txt") -> int:
        """
        Add documents from a directory to the vector store.
        
        Args:
            directory_path (str): Path to directory containing documents
            file_pattern (str): File pattern to match (e.g., "**/*.txt")
            
        Returns:
            int: Number of documents added
        """
        try:
            # Load documents from directory
            loader = DirectoryLoader(
                directory_path,
                glob=file_pattern,
                loader_cls=TextLoader,
                show_progress=True
            )
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No documents found in {directory_path} with pattern {file_pattern}")
                return 0
            
            # Split documents into chunks
            split_documents = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            return self._add_documents(split_documents)
            
        except Exception as e:
            logger.error(f"Error adding documents from directory: {e}")
            return 0
    
    def add_documents_from_web(self, urls: List[str]) -> int:
        """
        Add documents from web URLs to the vector store.
        Note: This requires beautifulsoup4 which may not be available.
        
        Args:
            urls (List[str]): List of URLs to scrape and add
            
        Returns:
            int: Number of document chunks added
        """
        logger.warning("Web document loading requires beautifulsoup4 which is not available in this environment")
        return 0
    
    def add_custom_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add custom documents to the vector store.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents with 'content' and 'metadata' keys
            
        Returns:
            int: Number of document chunks added
        """
        try:
            # Convert to LangChain Document format
            langchain_docs = []
            for doc in documents:
                langchain_doc = Document(
                    page_content=doc.get('content', ''),
                    metadata=doc.get('metadata', {})
                )
                langchain_docs.append(langchain_doc)
            
            # Split documents into chunks
            split_documents = self.text_splitter.split_documents(langchain_docs)
            
            # Add to vector store
            return self._add_documents(split_documents)
            
        except Exception as e:
            logger.error(f"Error adding custom documents: {e}")
            raise
            return 0
    
    def _add_documents(self, documents: List[Document]) -> int:
        """
        Internal method to add documents and compute embeddings.
        
        Args:
            documents (List[Document]): List of documents to add
            
        Returns:
            int: Number of documents added
        """
        try:
            added_count = 0
            
            for doc in documents:
                # Compute embedding
                embedding = self.embeddings.embed_query(doc.page_content)
                
                # Store document and embedding
                self.documents.append(doc)
                self.embeddings_cache.append(embedding)
                self.metadata_cache.append(doc.metadata)
                
                added_count += 1
            
            logger.info(f"Added {added_count} documents to vector store")
            return added_count
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return 0
    
    def search_documents(self, query: str, k: int = 5, 
                        filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Search for relevant documents using semantic similarity.
        
        Args:
            query (str): Search query
            k (int): Number of documents to return
            filter_metadata (Optional[Dict[str, Any]]): Metadata filters
            
        Returns:
            List[Document]: List of relevant documents
        """
        try:
            if not self.documents:
                logger.warning("No documents in vector store")
                return []
            
            # Compute query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings_cache):
                # Apply metadata filter if provided
                if filter_metadata:
                    doc_metadata = self.metadata_cache[i]
                    if not self._matches_filter(doc_metadata, filter_metadata):
                        continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in similarities[:k]]
            
            # Return documents
            results = [self.documents[idx] for idx in top_indices]
            
            logger.info(f"Found {len(results)} relevant documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents with similarity scores.
        
        Args:
            query (str): Search query
            k (int): Number of documents to return
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        try:
            if not self.documents:
                logger.warning("No documents in vector store")
                return []
            
            # Compute query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings_cache):
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = [(self.documents[idx], score) for idx, score in similarities[:k]]
            
            logger.info(f"Found {len(top_results)} relevant documents with scores for query: {query[:50]}...")
            return top_results
            
        except Exception as e:
            logger.error(f"Error searching documents with scores: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1 (List[float]): First vector
            vec2 (List[float]): Second vector
            
        Returns:
            float: Cosine similarity score
        """
        try:
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """
        Check if document metadata matches the filter criteria.
        
        Args:
            metadata (Dict[str, Any]): Document metadata
            filter_dict (Dict[str, Any]): Filter criteria
            
        Returns:
            bool: True if metadata matches filter
        """
        try:
            for key, value in filter_dict.items():
                if key not in metadata or metadata[key] != value:
                    return False
            return True
        except Exception as e:
            logger.error(f"Error checking metadata filter: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dict[str, Any]: Collection information including document count
        """
        try:
            return {
                "collection_name": "simple_baby_care_knowledge",
                "document_count": len(self.documents),
                "embedding_model": self.embeddings_model_spec['model_name'],
                "storage_type": "in_memory"
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.documents.clear()
            self.embeddings_cache.clear()
            self.metadata_cache.clear()
            
            logger.info("Collection cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False


def create_sample_baby_care_documents() -> List[Dict[str, Any]]:
    """
    Create sample baby care documents for initial knowledge base.
    
    Returns:
        List[Dict[str, Any]]: List of sample documents
    """
    sample_docs = [
        {
            "content": """
            Baby Nutrition Guidelines (0-6 months):
            
            For the first 6 months, breast milk or formula should be the only source of nutrition.
            Breast milk provides all necessary nutrients and antibodies for healthy development.
            Formula-fed babies should be fed every 2-3 hours, about 2-3 ounces per feeding.
            Signs of hunger include rooting, sucking motions, and crying.
            Never give honey to babies under 12 months due to botulism risk.
            """,
            "metadata": {
                "category": "nutrition",
                "age_range": "0-6_months",
                "topic": "feeding_basics",
                "source": "pediatric_guidelines"
            }
        },
        {
            "content": """
            Baby Sleep Patterns and Routines:
            
            Newborns sleep 14-17 hours per day in 2-4 hour intervals.
            By 3-4 months, babies may sleep 5-6 hours at night.
            Establish a consistent bedtime routine: bath, feeding, story, sleep.
            Place baby on back to sleep to reduce SIDS risk.
            Keep room temperature between 68-72°F (20-22°C).
            Avoid overstimulation before bedtime.
            """,
            "metadata": {
                "category": "sleep",
                "age_range": "0-12_months",
                "topic": "sleep_routines",
                "source": "sleep_guidelines"
            }
        },
        {
            "content": """
            Developmental Milestones (0-12 months):
            
            0-3 months: Lifts head, follows objects with eyes, responds to sounds
            3-6 months: Rolls over, sits with support, reaches for objects
            6-9 months: Sits without support, crawls, says "mama" or "dada"
            9-12 months: Stands with support, walks with assistance, understands simple commands
            Every baby develops at their own pace - these are general guidelines.
            Consult your pediatrician if you have concerns about development.
            """,
            "metadata": {
                "category": "development",
                "age_range": "0-12_months",
                "topic": "milestones",
                "source": "developmental_guidelines"
            }
        },
        {
            "content": """
            Baby Safety and Childproofing:
            
            Install safety gates at top and bottom of stairs.
            Cover electrical outlets with safety covers.
            Keep small objects, coins, and batteries out of reach.
            Use corner guards on sharp furniture edges.
            Secure heavy furniture to walls to prevent tipping.
            Keep cleaning products and medications locked away.
            Never leave baby unattended on elevated surfaces.
            """,
            "metadata": {
                "category": "safety",
                "age_range": "0-24_months",
                "topic": "childproofing",
                "source": "safety_guidelines"
            }
        },
        {
            "content": """
            Common Baby Health Concerns:
            
            Fever: Contact doctor if temperature is above 100.4°F (38°C) for babies under 3 months.
            Diaper rash: Change diapers frequently, use barrier cream, let skin air dry.
            Colic: Excessive crying for 3+ hours, 3+ days per week - usually resolves by 4 months.
            Teething: Begins around 6 months, symptoms include drooling, irritability, gum swelling.
            Colds: Common in first year, watch for breathing difficulties or high fever.
            Always consult your pediatrician for health concerns.
            """,
            "metadata": {
                "category": "health",
                "age_range": "0-12_months",
                "topic": "common_concerns",
                "source": "health_guidelines"
            }
        }
    ]
    
    return sample_docs


def main():
    """
    Main function to demonstrate the simple vector store functionality.
    """
    # Initialize vector store
    vector_store = SimpleBabyCareVectorStore()
    
    # Add sample documents
    print("Adding sample baby care documents...")
    sample_docs = create_sample_baby_care_documents()
    added_count = vector_store.add_custom_documents(sample_docs)
    print(f"Added {added_count} document chunks")
    
    # Get collection info
    info = vector_store.get_collection_info()
    print(f"Collection info: {info}")
    
    # Test search functionality
    test_queries = [
        "How much should a newborn eat?",
        "When do babies start sleeping through the night?",
        "What are the safety concerns for babies?",
        "When should I be concerned about my baby's development?"
    ]
    
    print("\nTesting search functionality:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.search_documents(query, k=2)
        for i, doc in enumerate(results, 1):
            print(f"Result {i}: {doc.page_content[:100]}...")
            print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    main()
