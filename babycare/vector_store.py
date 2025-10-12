"""
Vector Store Module for Baby Care Knowledge Base

Implements a vector storage system using Pinecone for storing
and retrieving baby care documents with semantic search capabilities.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

from .embeddings import get_embeddings_function
from .utils import (
    get_supported_file_types,
    get_supported_extensions,
    load_documents_from_file,
    load_documents_from_file_with_metadata,
    load_metadata_file,
    find_supported_files
)

from .config import config



# Load environment variables
load_dotenv()



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class BabyCareVectorStore:
    """
    Vector store manager for baby care knowledge base.
    
    This class handles the creation, management, and querying of a vector database
    containing baby care documents with semantic search capabilities using Pinecone.
    """
    
    def __init__(self, 
                 index_name: str = "baby-care-knowledge",
                 environment: str = "us-east-1"):
        """
        Initialize the vector store with specified configuration.
        
        Args:
            index_name (str): Name of the Pinecone index
            embedding_model (str): OpenAI embedding model to use
            environment (str): Pinecone environment/region
        """
        self.index_name = index_name
        self.environment = environment
        
        # Initialize OpenAI embeddings
        self.embeddings, self.embeddings_model_spec = get_embeddings_function(local=config.local_embedding_model)
        
        # Initialize Pinecone client
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in your .env file.")
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        # Text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize cached keywords
        self._cached_keywords = None
        self._keywords_initialized = False
        
        logger.info(f"BabyCareVectorStore initialized with index: {index_name}")
    
    def _initialize_vector_store(self) -> PineconeVectorStore:
        """
        Initialize or load the Pinecone vector store.
        
        Returns:
            PineconeVectorStore: Initialized Pinecone vector store
        """
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes] if hasattr(existing_indexes, '__iter__') else existing_indexes.names()
            if self.index_name not in index_names:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embeddings_model_spec["dimension"],  # OpenAI embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
                
                # Wait for index to be ready
                import time
                while not self.pc.describe_index(self.index_name).status['ready']:
                    logger.info("Waiting for index to be ready...")
                    time.sleep(1)
            
            # Get the index
            index = self.pc.Index(self.index_name)
            
            # Initialize LangChain Pinecone vector store
            vector_store = PineconeVectorStore(
                index=index,
                embedding=self.embeddings,
                text_key="text"
            )
            
            # Check if index has documents
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
            if total_vectors > 0:
                logger.info(f"Loaded existing index with {total_vectors} vectors")
            else:
                logger.info("Index exists but is empty")
                
        except Exception as e:
            logger.error(f"Error initializing Pinecone vector store: {e}")
            raise
        
        return vector_store
    
    def _process_documents(self, documents: List[Document]) -> int:
        """
        Process and add documents to the vector store.
        
        Args:
            documents (List[Document]): Documents to process
            
        Returns:
            int: Number of document chunks added
        """
        if not documents:
            return 0
        
        # Split documents into chunks
        split_documents = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(split_documents)
        
        # Invalidate keywords cache since we added new documents
        self._keywords_initialized = False
        
        return len(split_documents)
    
    def add_documents_from_directory(self, directory_path: str, 
                                   file_types: Optional[List[str]] = None) -> int:
        """
        Add documents from a directory to the vector store.
        
        Args:
            directory_path (str): Path to directory containing documents
            file_types (Optional[List[str]]): List of file types to process (e.g., ['pdf', 'txt'])
                                             If None, processes all supported file types
            
        Returns:
            int: Number of documents added
        """
        try:
            # Find all supported files in the directory
            files = find_supported_files(directory_path, file_types)
            
            if not files:
                logger.warning(f"No supported files found in directory: {directory_path}")
                return 0
            
            # Load metadata if available
            metadata = load_metadata_file(directory_path)
            if metadata:
                logger.info(f"📋 Found metadata.yml in {directory_path}")
            
            total_chunks = 0
            
            # Process each file
            for file_path in files:
                try:
                    if metadata:
                        documents = load_documents_from_file_with_metadata(str(file_path), metadata)
                    else:
                        documents = load_documents_from_file(str(file_path))
                    
                    chunks_added = self._process_documents(documents)
                    total_chunks += chunks_added
                    
                    if chunks_added > 0:
                        logger.info(f"✅ Added {chunks_added} chunks from {file_path.name}")
                    else:
                        logger.warning(f"⚠️ No chunks added from {file_path.name}")
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
            
            logger.info(f"📚 Directory processing complete! Total chunks added: {total_chunks}")
            return total_chunks
            
        except Exception as e:
            logger.error(f"Error adding documents from directory: {e}")
            return 0
    
    def add_documents_from_file(self, file_path: str) -> int:
        """
        Add documents from a single file to the vector store.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            int: Number of document chunks added
        """
        try:
            documents = load_documents_from_file(file_path)
            chunks_added = self._process_documents(documents)
            
            if chunks_added > 0:
                logger.info(f"✅ Added {chunks_added} chunks from {Path(file_path).name}")
            else:
                logger.warning(f"⚠️ No chunks added from {Path(file_path).name}")
            
            return chunks_added
            
        except Exception as e:
            logger.error(f"Error adding document from file {file_path}: {e}")
            return 0
    
    def add_documents_from_pdf(self, pdf_path: str) -> int:
        """
        Add documents from a PDF file to the vector store.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            int: Number of document chunks added
        """
        return self.add_documents_from_file(pdf_path)
    
    def add_documents_from_txt(self, txt_path: str) -> int:
        """
        Add documents from a text file to the vector store.
        
        Args:
            txt_path (str): Path to text file
            
        Returns:
            int: Number of document chunks added
        """
        return self.add_documents_from_file(txt_path)
    
    def add_documents_from_folder(self, folder_path: str, 
                                 file_types: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Add documents from all supported files in a folder to the vector store.
        
        Args:
            folder_path (str): Path to folder containing files
            file_types (Optional[List[str]]): List of file types to process (e.g., ['pdf', 'txt'])
                                             If None, processes all supported file types
            
        Returns:
            Dict[str, int]: Dictionary with filename as key and number of chunks added as value
        """
        results = {}
        total_chunks = 0
        
        try:
            # Find all supported files in the folder
            files = find_supported_files(folder_path, file_types)
            
            if not files:
                logger.warning(f"No supported files found in folder: {folder_path}")
                return results
            
            # Load metadata if available
            metadata = load_metadata_file(folder_path)
            if metadata:
                logger.info(f"📋 Found metadata.yml in {folder_path}")
            
            logger.info(f"Found {len(files)} supported files in {folder_path}")
            
            # Process each file
            for file_path in files:
                try:
                    logger.info(f"Processing file: {file_path.name}")
                    
                    if metadata:
                        documents = load_documents_from_file_with_metadata(str(file_path), metadata)
                    else:
                        documents = load_documents_from_file(str(file_path))
                    
                    chunks_added = self._process_documents(documents)
                    results[file_path.name] = chunks_added
                    total_chunks += chunks_added
                    
                    if chunks_added > 0:
                        logger.info(f"✅ Successfully added {chunks_added} chunks from {file_path.name}")
                    else:
                        logger.warning(f"⚠️ No chunks added from {file_path.name}")
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    results[file_path.name] = 0
            
            logger.info(f"📚 Folder processing complete! Total chunks added: {total_chunks}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing folder: {e}")
            return results
    
    
    # def add_documents_from_web(self, urls: List[str]) -> int:
    #     """
    #     Add documents from web URLs to the vector store.
        
    #     Args:
    #         urls (List[str]): List of URLs to scrape and add
            
    #     Returns:
    #         int: Number of document chunks added
    #     """
    #     try:
    #         total_chunks = 0
            
    #         for url in urls:
    #             try:
    #                 # Load web content
    #                 loader = WebBaseLoader(url)
    #                 documents = loader.load()
                    
    #                 chunks_added = self._process_documents(documents)
    #                 total_chunks += chunks_added
                    
    #                 if chunks_added > 0:
    #                     logger.info(f"✅ Added {chunks_added} chunks from URL: {url}")
    #                 else:
    #                     logger.warning(f"⚠️ No chunks added from URL: {url}")
                    
    #             except Exception as e:
    #                 logger.error(f"Error processing URL {url}: {e}")
    #                 continue
            
    #         return total_chunks
            
    #     except Exception as e:
    #         logger.error(f"Error adding web documents: {e}")
    #         return 0
    
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
            
            chunks_added = self._process_documents(langchain_docs)
            
            if chunks_added > 0:
                logger.info(f"✅ Added {chunks_added} custom document chunks")
            else:
                logger.warning("⚠️ No chunks added from custom documents")
            
            return chunks_added
            
        except Exception as e:
            logger.error(f"Error adding custom documents: {e}")
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
            # Perform similarity search
            if filter_metadata:
                documents = self.vector_store.similarity_search(
                    query, 
                    k=k, 
                    filter=filter_metadata
                )
            else:
                documents = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Found {len(documents)} relevant documents for query: {query[:50]}...")
            return documents
            
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
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            logger.info(f"Found {len(results)} relevant documents with scores for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents with scores: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dict[str, Any]: Collection information including document count
        """
        try:
            # Get the index
            index = self.pc.Index(self.index_name)
            
            # Get index stats
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
            
            return {
                "collection_name": self.index_name,
                "document_count": total_vectors,
                "embedding_model": self.embeddings_model_spec['model_name'],
                "environment": self.environment,
                "supported_file_types": list(get_supported_file_types().keys()),
                "supported_extensions": get_supported_extensions()
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def get_supported_file_types(self) -> Dict[str, List[str]]:
        """
        Get information about supported file types and their extensions.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping file types to their extensions
        """
        return get_supported_file_types()
    
    def _fetch_all_metadata_from_database(self) -> Dict[str, set]:
        """
        Fetch all metadata directly from the Pinecone database.
        
        Returns:
            Dict[str, set]: Dictionary with metadata keys as keys and sets of unique values as values
        """
        try:
            # Get the index
            index = self.pc.Index(self.index_name)
            
            # Get index stats to check if there are any vectors
            stats = index.describe_index_stats()
            if stats.total_vector_count == 0:
                logger.warning("No documents in the knowledge base")
                return {}
            
            # Initialize metadata collection
            metadata_collection = {}
            
            # Fetch all vectors from the index
            # We'll use the query method with an empty vector to get all documents
            # This is more efficient than multiple similarity searches
            try:
                # Get all vectors by querying with a zero vector
                # This will return all documents in the index
                query_response = index.query(
                    vector=[0.0] * self.embeddings_model_spec["dimension"],
                    top_k=stats.total_vector_count,
                    include_metadata=True
                )
                
                # Extract metadata from all results
                for match in query_response.matches:
                    if match.metadata:
                        for key, value in match.metadata.items():
                            if key not in metadata_collection:
                                metadata_collection[key] = set()
                            
                            # Handle both string and list values
                            if isinstance(value, list):
                                metadata_collection[key].update([str(v) for v in value])
                            else:
                                metadata_collection[key].add(str(value))
                
                logger.info(f"Fetched metadata from {len(query_response.matches)} documents")
                
            except Exception as e:
                logger.warning(f"Could not fetch all metadata directly: {e}")
            # Convert sets to lists for easier handling
            result = {key: list(values) for key, values in metadata_collection.items()}
            
            logger.info(f"Extracted metadata keys: {list(result.keys())}")
            for key, values in result.items():
                logger.debug(f"  {key}: {len(values)} unique values")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching metadata from database: {e}")
            return {}
    
    def _initialize_keywords(self) -> None:
        """
        Initialize keywords by fetching all metadata from the database.
        This is called once during initialization or when documents are added.
        """
        try:
            logger.info("Initializing keywords from knowledge base metadata...")
            
            # Fetch all metadata from database
            all_metadata = self._fetch_all_metadata_from_database()
            
            if not all_metadata:
                logger.warning("No metadata found in knowledge base")
                self._cached_keywords = []
                self._keywords_initialized = True
                return
            
            # Extract keywords from relevant metadata fields
            keywords = set()
            
            # Get topics
            topics = all_metadata.get('topic', [])
            keywords.update(topics)
            
            # Get categories
            categories = all_metadata.get('category', [])
            keywords.update(categories)
            
            # Get age ranges
            age_ranges = all_metadata.get('age_range', [])
            keywords.update(age_ranges)
            
            # Process keywords (split compound words and clean up)
            keyword_list = []
            for keyword in keywords:
                if keyword and keyword.lower() not in ['not specified', 'unknown', 'general', '']:
                    # Split compound keywords (e.g., "ACE Inhibitors" -> ["ACE", "Inhibitors"])
                    words = keyword.replace('-', ' ').replace('_', ' ').split()
                    keyword_list.extend([word.lower() for word in words if len(word) > 2])
            
            # Remove duplicates and sort
            self._cached_keywords = sorted(list(set(keyword_list)))
            self._keywords_initialized = True
            
            logger.info(f"Initialized {len(self._cached_keywords)} keywords")
            
        except Exception as e:
            logger.error(f"Error initializing keywords: {e}")
            self._cached_keywords = []
            self._keywords_initialized = True
    
    def get_knowledge_base_keywords(self) -> List[str]:
        """
        Get all unique topics and categories from the knowledge base to use as RAG keywords.
        Uses cached keywords for efficiency.
        
        Returns:
            List[str]: List of unique keywords from topics and categories
        """
        try:
            # Initialize keywords if not already done
            if not self._keywords_initialized:
                self._initialize_keywords()
            
            return self._cached_keywords.copy() if self._cached_keywords else []
            
        except Exception as e:
            logger.error(f"Error getting knowledge base keywords: {e}")
            return []
    
    def refresh_keywords_cache(self) -> None:
        """
        Manually refresh the keywords cache by re-fetching all metadata from the database.
        Useful when you know the knowledge base has been updated externally.
        """
        logger.info("Manually refreshing keywords cache...")
        self._keywords_initialized = False
        self._initialize_keywords()
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete the index
            self.pc.delete_index(self.index_name)
            
            # Reinitialize the vector store
            self.vector_store = self._initialize_vector_store()
            
            # Reset keywords cache
            self._cached_keywords = None
            self._keywords_initialized = False
            
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
    Main function to demonstrate the vector store functionality.
    """
    # Initialize vector store
    vector_store = BabyCareVectorStore()
    
    # Display supported file types
    print("Supported file types:")
    supported_types = vector_store.get_supported_file_types()
    for file_type, extensions in supported_types.items():
        print(f"  {file_type}: {', '.join(extensions)}")
    
    # Add sample documents
    print("\nAdding sample baby care documents...")
    sample_docs = create_sample_baby_care_documents()
    added_count = vector_store.add_custom_documents(sample_docs)
    print(f"Added {added_count} document chunks")
    
    # Get collection info
    info = vector_store.get_collection_info()
    print(f"\nCollection info: {info}")
    
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
    
    # Demonstrate file type extension
    print(f"\nTo add support for new file types, you can:")
    print("1. Add the file type to SUPPORTED_FILE_TYPES constant in utils.py")
    print("2. Create a new DocumentLoader class in utils.py")
    print("3. Register it with DocumentLoaderFactory.register_loader()")
    print("4. Example: DocumentLoaderFactory.register_loader('docx', DocxDocumentLoader())")


if __name__ == "__main__":
    main()
