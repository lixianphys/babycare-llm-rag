
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import TypedDict

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

        # https://huggingface.co/BAAI/bge-base-en-v1.5
        # Best embedding model at a reasonable size at the moment (2023-11-22)
    if local: 
        print("Loading embeddings model: ", LOCAL_EMBEDDING_MODEL_SPEC["model_name"])
        embeddings_function = HuggingFaceBgeEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL_SPEC["model_name"],
            encode_kwargs=LOCAL_EMBEDDING_MODEL_SPEC["encode_kwargs"],
            query_instruction=LOCAL_EMBEDDING_MODEL_SPEC["query_instruction"],
        )
        embeddings_model_spec = LOCAL_EMBEDDING_MODEL_SPEC

    else:
        print("Loading embeddings model: ", OPENAI_EMBEDDING_MODEL_SPEC["model_name"])
        embeddings_function = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_SPEC["model_name"])
        embeddings_model_spec = OPENAI_EMBEDDING_MODEL_SPEC

    return embeddings_function, embeddings_model_spec



