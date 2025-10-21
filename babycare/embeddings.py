
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import TypedDict
import os

class EmbeddingModelSpec(TypedDict):
    model_name: str
    dimension: int|None
    query_instruction: str
    encode_kwargs: dict


OPENAI_EMBEDDING_MODEL_SPEC: EmbeddingModelSpec = {
    "model_name": "text-embedding-3-small",
    "dimension": 768,
    "query_instruction": "",
    "encode_kwargs": {}
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
    # Check if we should force OpenAI embeddings (for Hugging Face Spaces)
    force_openai = os.getenv("FORCE_OPENAI_EMBEDDINGS", "false").lower() == "true"
    
    # https://huggingface.co/BAAI/bge-base-en-v1.5
    # Best embedding model at a reasonable size at the moment (2023-11-22)
    if local and not force_openai: 
        try:
            print("Loading embeddings model: ", LOCAL_EMBEDDING_MODEL_SPEC["model_name"])
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
            print(f"Failed to load BAAI/bge-base-en-v1.5: {e}")
            print("Falling back to OpenAI embeddings for Hugging Face Spaces...")
            # Fall back to OpenAI embeddings for Hugging Face Spaces
            pass

    # Use OpenAI embeddings (either requested or as fallback)
    print("Loading embeddings model: ", OPENAI_EMBEDDING_MODEL_SPEC["model_name"])
    embeddings_function = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_SPEC["model_name"])
    embeddings_model_spec = OPENAI_EMBEDDING_MODEL_SPEC

    return embeddings_function, embeddings_model_spec



