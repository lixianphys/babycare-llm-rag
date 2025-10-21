
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import TypedDict
import os
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModelSpec(TypedDict):
    model_name: str
    dimension: int
    query_instruction: str | None = None
    encode_kwargs: dict | None = None


OPENAI_EMBEDDING_MODEL_SPEC: EmbeddingModelSpec = {
    "model_name": "text-embedding-3-small",
    "dimension": 768,
}

LOCAL_EMBEDDING_MODEL_SPEC: EmbeddingModelSpec = {
    "model_name": "BAAI/bge-base-en-v1.5",
    "dimension": 768,
    "query_instruction": "Represent this sentence for searching relevant passages: ",
    "encode_kwargs": {'normalize_embeddings': True,"show_progress_bar":False}
}

def get_embeddings_function(local:bool = True):
    """
    Get embeddings function with Hugging Face Spaces compatibility.
    """
    # https://huggingface.co/BAAI/bge-base-en-v1.5
    # Best embedding model at a reasonable size at the moment (2023-11-22)
    if local: 
        try:
            logger.info(f"Loading embeddings model: {LOCAL_EMBEDDING_MODEL_SPEC['model_name']}")
            # Add trust_remote_code=True for Hugging Face Spaces compatibility
            embeddings_function = HuggingFaceBgeEmbeddings(
                model_name=LOCAL_EMBEDDING_MODEL_SPEC["model_name"],
                encode_kwargs=LOCAL_EMBEDDING_MODEL_SPEC["encode_kwargs"],
                query_instruction=LOCAL_EMBEDDING_MODEL_SPEC["query_instruction"],
                model_kwargs={'trust_remote_code': True}
            )
            embeddings_model_spec = LOCAL_EMBEDDING_MODEL_SPEC
            return embeddings_function, embeddings_model_spec
        except Exception as e:
            logger.warning(f"Failed to load BAAI/bge-base-en-v1.5: {e}")
            logger.warning("Falling back to OpenAI embeddings for Hugging Face Spaces...")
            # Fall back to OpenAI embeddings for Hugging Face Spaces
            pass

    # Use OpenAI embeddings (either requested or as fallback)
    logger.info(f"Loading embeddings model: {OPENAI_EMBEDDING_MODEL_SPEC['model_name']}")
    embeddings_function = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL_SPEC["model_name"],
        dimensions=OPENAI_EMBEDDING_MODEL_SPEC["dimension"],
    )
    embeddings_model_spec = OPENAI_EMBEDDING_MODEL_SPEC

    return embeddings_function, embeddings_model_spec



