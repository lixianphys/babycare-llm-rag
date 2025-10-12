"""
Document Loading Utilities for Baby Care Knowledge Base

This module provides document loading functionality with support for multiple file types
following SOLID principles and the Strategy pattern for easy extensibility.
"""

import logging
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from langchain.schema import Document
from langchain_community.document_loaders import TextLoader

# Configure logging
logger = logging.getLogger(__name__)

# Supported file types constants
SUPPORTED_FILE_TYPES = {
    'pdf': ['.pdf'],
    'txt': ['.txt'],
    'text': ['.text']
}

# File type mappings for easy extension
FILE_TYPE_EXTENSIONS = {
    extension: file_type 
    for file_type, extensions in SUPPORTED_FILE_TYPES.items() 
    for extension in extensions
}


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders following the Strategy pattern.
    This allows for easy extension to support new file types.
    """
    
    @abstractmethod
    def load_documents(self, file_path: str) -> List[Document]:
        """
        Load documents from a file.
        
        Args:
            file_path (str): Path to the file to load
            
        Returns:
            List[Document]: List of loaded documents
        """
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported file extensions
        """
        pass


class PDFDocumentLoader(DocumentLoader):
    """Concrete implementation for loading PDF documents."""
    
    def load_documents(self, file_path: str) -> List[Document]:
        """Load documents from a PDF file."""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            return loader.load()
        except ImportError:
            logger.error("PyPDFLoader not available. Please install: pip install pypdf")
            return []
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return []
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported PDF extensions."""
        return SUPPORTED_FILE_TYPES['pdf']


class TextDocumentLoader(DocumentLoader):
    """Concrete implementation for loading text documents."""
    
    def load_documents(self, file_path: str) -> List[Document]:
        """Load documents from a text file."""
        try:
            loader = TextLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return []
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported text extensions."""
        return SUPPORTED_FILE_TYPES['txt'] + SUPPORTED_FILE_TYPES['text']


class DocumentLoaderFactory:
    """
    Factory class for creating document loaders based on file extension.
    Follows the Factory pattern for easy extension.
    """
    
    _loaders = {
        'pdf': PDFDocumentLoader(),
        'txt': TextDocumentLoader(),
        'text': TextDocumentLoader()
    }
    
    @classmethod
    def get_loader(cls, file_path: str) -> Optional[DocumentLoader]:
        """
        Get appropriate loader for the given file path.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            Optional[DocumentLoader]: Appropriate loader or None if not supported
        """
        file_extension = Path(file_path).suffix.lower()
        file_type = FILE_TYPE_EXTENSIONS.get(file_extension)
        
        if file_type and file_type in cls._loaders:
            return cls._loaders[file_type]
        
        logger.warning(f"Unsupported file type: {file_extension}")
        return None
    
    @classmethod
    def register_loader(cls, file_type: str, loader: DocumentLoader):
        """
        Register a new loader for a file type.
        
        Args:
            file_type (str): File type identifier
            loader (DocumentLoader): Loader instance
        """
        cls._loaders[file_type] = loader
        logger.info(f"Registered loader for file type: {file_type}")
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        Get all supported file extensions.
        
        Returns:
            List[str]: List of all supported extensions
        """
        extensions = []
        for loader in cls._loaders.values():
            extensions.extend(loader.get_supported_extensions())
        return list(set(extensions))
    
    @classmethod
    def get_supported_file_types(cls) -> Dict[str, List[str]]:
        """
        Get information about supported file types and their extensions.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping file types to their extensions
        """
        return SUPPORTED_FILE_TYPES.copy()


def get_supported_file_types() -> Dict[str, List[str]]:
    """
    Get information about supported file types and their extensions.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping file types to their extensions
    """
    return SUPPORTED_FILE_TYPES.copy()


def get_supported_extensions() -> List[str]:
    """
    Get all supported file extensions.
    
    Returns:
        List[str]: List of all supported extensions
    """
    return DocumentLoaderFactory.get_supported_extensions()


def is_supported_file_type(file_path: str) -> bool:
    """
    Check if a file type is supported.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        bool: True if file type is supported, False otherwise
    """
    file_extension = Path(file_path).suffix.lower()
    return file_extension in FILE_TYPE_EXTENSIONS


def get_file_type(file_path: str) -> Optional[str]:
    """
    Get the file type for a given file path.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        Optional[str]: File type or None if not supported
    """
    file_extension = Path(file_path).suffix.lower()
    return FILE_TYPE_EXTENSIONS.get(file_extension)


def load_documents_from_file(file_path: str) -> List[Document]:
    """
    Load documents from a single file using the appropriate loader.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        List[Document]: List of loaded documents
    """
    loader = DocumentLoaderFactory.get_loader(file_path)
    if not loader:
        logger.error(f"No loader available for file: {file_path}")
        return []
    
    return loader.load_documents(file_path)


def find_supported_files(directory_path: str, 
                        file_types: Optional[List[str]] = None) -> List[Path]:
    """
    Find all supported files in a directory.
    
    Args:
        directory_path (str): Path to directory to search
        file_types (Optional[List[str]]): List of file types to find (e.g., ['pdf', 'txt'])
                                         If None, finds all supported file types
        
    Returns:
        List[Path]: List of file paths
    """
    directory = Path(directory_path)
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory_path}")
        return []
    
    # Determine file types to find
    if file_types is None:
        file_types = list(SUPPORTED_FILE_TYPES.keys())
    
    all_files = []
    for file_type in file_types:
        if file_type in SUPPORTED_FILE_TYPES:
            extensions = SUPPORTED_FILE_TYPES[file_type]
            for extension in extensions:
                files = list(directory.glob(f"**/*{extension}"))
                all_files.extend(files)
    
    return all_files


def load_metadata_file(directory_path: str) -> Optional[Dict[str, Any]]:
    """
    Load metadata from metadata.yml file in a directory.
    
    Args:
        directory_path (str): Path to directory containing metadata.yml
        
    Returns:
        Optional[Dict[str, Any]]: Metadata dictionary or None if not found
    """
    metadata_path = Path(directory_path) / "metadata.yml"
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata from {metadata_path}: {e}")
        return None


def validate_metadata_for_pinecone(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean metadata to ensure compatibility with Pinecone.
    
    Args:
        metadata (Dict[str, Any]): Metadata to validate
        
    Returns:
        Dict[str, Any]: Validated metadata
    """
    validated_metadata = {}
    
    for key, value in metadata.items():
        # Pinecone supports: string, number, boolean, list of strings
        if isinstance(value, (str, int, float, bool)):
            validated_metadata[key] = value
        elif isinstance(value, list):
            # Ensure all items in list are strings
            if all(isinstance(item, str) for item in value):
                validated_metadata[key] = value
            else:
                logger.warning(f"Metadata key '{key}' has non-string items in list, converting to strings")
                validated_metadata[key] = [str(item) for item in value]
        else:
            # Convert other types to string
            logger.warning(f"Metadata key '{key}' has unsupported type {type(value)}, converting to string")
            validated_metadata[key] = str(value)
    
    return validated_metadata


def process_metadata_template(metadata: Dict[str, Any], file_title: str) -> Dict[str, Any]:
    """
    Process metadata template by replacing placeholders with actual values.
    
    Args:
        metadata (Dict[str, Any]): Metadata template
        file_title (str): File title (filename without extension)
        
    Returns:
        Dict[str, Any]: Processed metadata with placeholders replaced
    """
    processed_metadata = {}
    
    for key, value in metadata.items():
        if isinstance(value, str):
            # Replace {FILE_TITLE} placeholder with actual file title
            processed_value = value.replace("{FILE_TITLE}", file_title)
            processed_metadata[key] = processed_value
            logger.debug(f"Processed metadata {key}: '{value}' -> '{processed_value}'")
        else:
            processed_metadata[key] = value
            logger.debug(f"Metadata {key}: {value} (not a string, keeping as-is)")
    
    # Validate metadata for Pinecone compatibility
    validated_metadata = validate_metadata_for_pinecone(processed_metadata)
    
    logger.info(f"Processed metadata for file '{file_title}': {validated_metadata}")
    return validated_metadata


def get_file_title(file_path: Path) -> str:
    """
    Get file title (filename without extension).
    
    Args:
        file_path (Path): Path to the file
        
    Returns:
        str: File title
    """
    return file_path.stem


def load_documents_from_file_with_metadata(file_path: str, 
                                         metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
    """
    Load documents from a single file with optional metadata.
    
    Args:
        file_path (str): Path to the file
        metadata (Optional[Dict[str, Any]]): Metadata to add to documents
        
    Returns:
        List[Document]: List of loaded documents with metadata
    """
    documents = load_documents_from_file(file_path)
    
    if metadata and documents:
        file_path_obj = Path(file_path)
        file_title = get_file_title(file_path_obj)
        processed_metadata = process_metadata_template(metadata, file_title)
        
        # Add metadata to all documents
        for doc in documents:
            doc.metadata.update(processed_metadata)
    
    return documents


def load_documents_from_directory(directory_path: str, 
                                 file_types: Optional[List[str]] = None) -> List[Document]:
    """
    Load documents from all supported files in a directory.
    
    Args:
        directory_path (str): Path to directory containing files
        file_types (Optional[List[str]]): List of file types to process (e.g., ['pdf', 'txt'])
                                         If None, processes all supported file types
        
    Returns:
        List[Document]: List of all loaded documents
    """
    files = find_supported_files(directory_path, file_types)
    all_documents = []
    
    # Load metadata if available
    metadata = load_metadata_file(directory_path)
    
    for file_path in files:
        try:
            if metadata:
                documents = load_documents_from_file_with_metadata(str(file_path), metadata)
            else:
                documents = load_documents_from_file(str(file_path))
            
            all_documents.extend(documents)
            logger.info(f"Loaded {len(documents)} documents from {file_path.name}")
        except Exception as e:
            logger.error(f"Error loading documents from {file_path.name}: {e}")
    
    return all_documents
