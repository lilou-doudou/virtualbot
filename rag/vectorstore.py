"""ChromaDB-backed vector store with Ollama embeddings."""

from typing import Any, Dict, List, Optional

import chromadb
import ollama

from .config import Config


class VectorStore:
    """Wrapper around a ChromaDB collection that uses Ollama for embeddings.

    Args:
        collection_name:    Name of the ChromaDB collection.
        persist_directory:  Directory for persistent storage.
                            If ``None`` an in-memory client is used.
        embedding_model:    Ollama model name used to generate embeddings.
    """

    def __init__(
        self,
        collection_name: str = Config.COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        embedding_model: str = Config.EMBEDDING_MODEL,
    ) -> None:
        if persist_directory:
            self.client: chromadb.ClientAPI = chromadb.PersistentClient(
                path=persist_directory
            )
        else:
            self.client = chromadb.Client()

        self.embedding_model = embedding_model
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_embedding(self, text: str) -> List[float]:
        """Return the Ollama embedding vector for *text*."""
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Embed and insert *documents* into the collection.

        Each document dict must contain ``content``, ``source``, and
        optionally ``chunk_id`` keys (as produced by
        :func:`~rag.document_loader.prepare_documents`).

        Args:
            documents: List of document dicts to index.
        """
        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        contents: List[str] = []

        for i, doc in enumerate(documents):
            chunk_id = doc.get("chunk_id", i)
            doc_id = f"{doc['source']}_chunk_{chunk_id}"
            embedding = self._get_embedding(doc["content"])

            ids.append(doc_id)
            embeddings.append(embedding)
            metadatas.append({"source": doc["source"], "chunk_id": chunk_id})
            contents.append(doc["content"])

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=contents,
        )

    def query(
        self, query_text: str, n_results: int = Config.N_RESULTS
    ) -> List[Dict[str, Any]]:
        """Retrieve the *n_results* most relevant documents for *query_text*.

        Args:
            query_text: Natural-language query.
            n_results:  Number of documents to return.

        Returns:
            A list of dicts with ``content``, ``source``, and ``distance`` keys.
        """
        query_embedding = self._get_embedding(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        documents: List[Dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            documents.append(
                {
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "distance": (
                        results["distances"][0][i]
                        if "distances" in results
                        else None
                    ),
                }
            )
        return documents

    def reset(self) -> None:
        """Delete all documents from the collection."""
        self.collection.delete(where={"chunk_id": {"$gte": 0}})
