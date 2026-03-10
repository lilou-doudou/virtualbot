"""High-level RAG pipeline: ingest documents then answer questions."""

from typing import Any, Dict, List, Optional

import ollama

from .config import Config
from .document_loader import load_documents_from_directory, prepare_documents
from .vectorstore import VectorStore


class RAGChain:
    """End-to-end Retrieval-Augmented Generation pipeline backed by Ollama.

    Typical usage::

        rag = RAGChain()
        rag.ingest("data/")
        result = rag.query("What is Ollama?")
        print(result["answer"])

    Args:
        llm_model:         Ollama model name used for answer generation.
        embedding_model:   Ollama model name used for embeddings.
        collection_name:   ChromaDB collection name.
        persist_directory: Directory for persistent ChromaDB storage.
                           Pass ``None`` to use an in-memory store.
        n_results:         Number of context chunks retrieved per query.
        chunk_size:        Characters per document chunk.
        chunk_overlap:     Overlapping characters between consecutive chunks.
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        n_results: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self.llm_model: str = llm_model or Config.LLM_MODEL
        self.n_results: int = n_results or Config.N_RESULTS
        self.chunk_size: int = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap: int = chunk_overlap or Config.CHUNK_OVERLAP

        self.vectorstore = VectorStore(
            collection_name=collection_name or Config.COLLECTION_NAME,
            persist_directory=persist_directory,
            embedding_model=embedding_model or Config.EMBEDDING_MODEL,
        )

    def ingest(self, data_directory: str) -> int:
        """Load documents from *data_directory* and index them.

        Args:
            data_directory: Path to a folder containing ``.txt`` / ``.md`` files.

        Returns:
            The number of chunks that were indexed.
        """
        documents = load_documents_from_directory(data_directory)
        prepared = prepare_documents(documents, self.chunk_size, self.chunk_overlap)
        self.vectorstore.add_documents(prepared)
        return len(prepared)

    def _build_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Construct the prompt sent to the LLM.

        Args:
            query:        The user's question.
            context_docs: Retrieved context chunks from the vector store.

        Returns:
            A formatted prompt string.
        """
        context = "\n\n---\n\n".join(
            f"Source: {doc['source']}\n{doc['content']}" for doc in context_docs
        )
        return (
            "You are a helpful assistant. "
            "Use the following context to answer the question.\n"
            "If the answer is not in the context, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Answer *question* using retrieved context.

        Args:
            question: Natural-language question to answer.

        Returns:
            A dict with keys:

            - ``answer``  – the generated text response.
            - ``sources`` – list of source file names used as context.
            - ``context`` – the raw list of retrieved document dicts.
        """
        context_docs = self.vectorstore.query(question, n_results=self.n_results)
        prompt = self._build_prompt(question, context_docs)

        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        answer: str = response["message"]["content"]

        return {
            "answer": answer,
            "sources": [doc["source"] for doc in context_docs],
            "context": context_docs,
        }
