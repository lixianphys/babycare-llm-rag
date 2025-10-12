from pydantic_settings import BaseSettings
from pydantic import Field

class Config(BaseSettings):
    
    # Embedding model
    local_embedding_model: bool = Field(default=True, alias="LOCAL_EMBEDDING_MODEL")
    
    # Vector store
    use_memory_vector_store: bool = Field(default=False, alias="USE_MEMORY_STORE")
    index_name_vector_store: str = Field(default="baby-care-knowledge", alias="INDEX_NAME_VECTOR_STORE")

    # RAG system
    top_k_documents: int = Field(default=10, alias="TOP_K_DOCUMENTS")
    similarity_threshold: float = Field(default=0.6, alias="SIMILARITY_THRESHOLD")
    max_context_length: int = Field(default=4000, alias="MAX_CONTEXT_LENGTH")
    model_name: str = Field(default="gpt-4o-mini", alias="MODEL_NAME")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")
    max_tokens: int = Field(default=1024, alias="MAX_TOKENS")

config = Config()

if __name__ == "__main__":
    print(config.model_dump())