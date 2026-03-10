"""Configuration for the RAG system."""


class Config:
    """Default configuration values for the RAG pipeline."""

    # Ollama model used for text generation
    LLM_MODEL: str = "llama3.2"

    # Ollama model used for generating embeddings
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # Number of characters per document chunk
    CHUNK_SIZE: int = 500

    # Number of overlapping characters between consecutive chunks
    CHUNK_OVERLAP: int = 50

    # Number of context documents to retrieve per query
    N_RESULTS: int = 3

    # ChromaDB collection name
    COLLECTION_NAME: str = "virtualbot_rag"
