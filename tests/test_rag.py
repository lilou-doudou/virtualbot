"""Unit tests for the RAG components.

All tests mock external services (Ollama, ChromaDB) so that the test suite
runs without a live Ollama instance or any network connection.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from rag.config import Config
from rag.document_loader import (
    chunk_text,
    load_document,
    load_documents_from_directory,
    prepare_documents,
)


# ---------------------------------------------------------------------------
# document_loader tests
# ---------------------------------------------------------------------------


class TestLoadDocument:
    def test_reads_file_content(self, tmp_path):
        doc = tmp_path / "hello.txt"
        doc.write_text("Hello, world!", encoding="utf-8")
        assert load_document(str(doc)) == "Hello, world!"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_document(str(tmp_path / "nonexistent.txt"))


class TestLoadDocumentsFromDirectory:
    def test_loads_txt_and_md_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
        (tmp_path / "b.md").write_text("beta", encoding="utf-8")
        (tmp_path / "c.pdf").write_text("should be ignored", encoding="utf-8")

        docs = load_documents_from_directory(str(tmp_path))
        sources = {d["source"] for d in docs}

        assert "a.txt" in sources
        assert "b.md" in sources
        assert "c.pdf" not in sources

    def test_returns_content_and_source_keys(self, tmp_path):
        (tmp_path / "doc.txt").write_text("content here", encoding="utf-8")
        docs = load_documents_from_directory(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["content"] == "content here"
        assert docs[0]["source"] == "doc.txt"

    def test_empty_directory(self, tmp_path):
        docs = load_documents_from_directory(str(tmp_path))
        assert docs == []


class TestChunkText:
    def test_short_text_is_single_chunk(self):
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert chunks == ["Short text"]

    def test_long_text_is_split(self):
        text = "A" * 200
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_chunks_respect_chunk_size(self):
        text = "B" * 500
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_overlap_creates_shared_content(self):
        # With overlap=10 consecutive chunks share the last 10 chars of the
        # previous chunk as the first 10 chars of the next chunk.
        text = "ABCDEFGHIJ" * 30  # 300 chars
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        assert chunks[0][-10:] == chunks[1][:10]

    def test_whitespace_only_chunks_are_skipped(self):
        text = "Hello" + "   " * 100 + "World"
        chunks = chunk_text(text, chunk_size=10, overlap=2)
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_raises_on_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("text", chunk_size=0)

    def test_raises_on_negative_overlap(self):
        with pytest.raises(ValueError):
            chunk_text("text", chunk_size=10, overlap=-1)

    def test_raises_when_overlap_gte_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("text", chunk_size=10, overlap=10)

    def test_empty_string_returns_empty_list(self):
        assert chunk_text("", chunk_size=100, overlap=10) == []


class TestPrepareDocuments:
    def test_attaches_metadata(self):
        docs = [{"content": "A" * 600, "source": "test.txt"}]
        prepared = prepare_documents(docs, chunk_size=200, overlap=20)
        assert all("source" in d for d in prepared)
        assert all("chunk_id" in d for d in prepared)
        assert all(d["source"] == "test.txt" for d in prepared)

    def test_chunk_ids_are_sequential(self):
        docs = [{"content": "X" * 400, "source": "file.txt"}]
        prepared = prepare_documents(docs, chunk_size=100, overlap=10)
        ids = [d["chunk_id"] for d in prepared]
        assert ids == list(range(len(ids)))

    def test_multiple_documents(self):
        docs = [
            {"content": "Doc1 content", "source": "d1.txt"},
            {"content": "Doc2 content", "source": "d2.txt"},
        ]
        prepared = prepare_documents(docs)
        sources = {d["source"] for d in prepared}
        assert sources == {"d1.txt", "d2.txt"}


# ---------------------------------------------------------------------------
# VectorStore tests  (Ollama and ChromaDB are mocked)
# ---------------------------------------------------------------------------


class TestVectorStore:
    """Tests for rag.vectorstore.VectorStore with mocked dependencies."""

    def _make_fake_embedding(self, dim: int = 4) -> list:
        return [0.1] * dim

    @patch("rag.vectorstore.ollama.embeddings")
    @patch("rag.vectorstore.chromadb.Client")
    def test_add_and_query_documents(self, mock_chroma_client, mock_embed):
        """add_documents feeds embeddings to the collection; query returns results."""
        from rag.vectorstore import VectorStore

        # --- Set up ChromaDB mock ---
        mock_collection = MagicMock()
        mock_chroma_client.return_value.get_or_create_collection.return_value = (
            mock_collection
        )

        # query() returns ChromaDB-style result structure
        mock_collection.query.return_value = {
            "ids": [["src_chunk_0"]],
            "documents": [["relevant content"]],
            "metadatas": [[{"source": "doc.txt", "chunk_id": 0}]],
            "distances": [[0.1]],
        }

        # --- Set up Ollama mock ---
        mock_embed.return_value = {"embedding": self._make_fake_embedding()}

        store = VectorStore()
        docs = [{"content": "relevant content", "source": "doc.txt", "chunk_id": 0}]
        store.add_documents(docs)

        # Embedding was requested for the document content
        mock_embed.assert_called()

        results = store.query("test query", n_results=1)
        assert len(results) == 1
        assert results[0]["content"] == "relevant content"
        assert results[0]["source"] == "doc.txt"
        assert results[0]["distance"] == 0.1

    @patch("rag.vectorstore.ollama.embeddings")
    @patch("rag.vectorstore.chromadb.PersistentClient")
    def test_persistent_client_used_when_directory_given(
        self, mock_persist, mock_embed
    ):
        from rag.vectorstore import VectorStore

        mock_collection = MagicMock()
        mock_persist.return_value.get_or_create_collection.return_value = (
            mock_collection
        )
        mock_embed.return_value = {"embedding": self._make_fake_embedding()}

        VectorStore(persist_directory="/tmp/chroma_test")
        mock_persist.assert_called_once_with(path="/tmp/chroma_test")

    @patch("rag.vectorstore.ollama.embeddings")
    @patch("rag.vectorstore.chromadb.Client")
    def test_reset_calls_collection_delete(self, mock_chroma_client, mock_embed):
        from rag.vectorstore import VectorStore

        mock_collection = MagicMock()
        mock_chroma_client.return_value.get_or_create_collection.return_value = (
            mock_collection
        )

        store = VectorStore()
        store.reset()
        mock_collection.delete.assert_called_once()


# ---------------------------------------------------------------------------
# RAGChain tests  (Ollama and ChromaDB are mocked via VectorStore patch)
# ---------------------------------------------------------------------------


class TestRAGChain:
    """Tests for rag.rag_chain.RAGChain with mocked VectorStore and Ollama."""

    def _make_context_docs(self):
        return [
            {"content": "Ollama runs LLMs locally.", "source": "doc.txt", "distance": 0.05},
            {"content": "ChromaDB stores embeddings.", "source": "doc.txt", "distance": 0.12},
        ]

    @patch("rag.rag_chain.ollama.chat")
    @patch("rag.rag_chain.VectorStore")
    def test_query_returns_answer_and_sources(self, mock_vs_cls, mock_chat):
        from rag.rag_chain import RAGChain

        mock_vs = mock_vs_cls.return_value
        mock_vs.query.return_value = self._make_context_docs()
        mock_chat.return_value = {
            "message": {"content": "Ollama is a local LLM runner."}
        }

        rag = RAGChain()
        result = rag.query("What is Ollama?")

        assert result["answer"] == "Ollama is a local LLM runner."
        assert "doc.txt" in result["sources"]
        assert len(result["context"]) == 2

    @patch("rag.rag_chain.ollama.chat")
    @patch("rag.rag_chain.VectorStore")
    def test_build_prompt_includes_context_and_question(self, mock_vs_cls, mock_chat):
        from rag.rag_chain import RAGChain

        mock_chat.return_value = {"message": {"content": "answer"}}
        rag = RAGChain()
        prompt = rag._build_prompt("My question?", self._make_context_docs())

        assert "My question?" in prompt
        assert "Ollama runs LLMs locally." in prompt
        assert "ChromaDB stores embeddings." in prompt
        assert "Source: doc.txt" in prompt

    @patch("rag.rag_chain.load_documents_from_directory")
    @patch("rag.rag_chain.prepare_documents")
    @patch("rag.rag_chain.VectorStore")
    def test_ingest_returns_chunk_count(
        self, mock_vs_cls, mock_prepare, mock_load
    ):
        from rag.rag_chain import RAGChain

        mock_load.return_value = [{"content": "text", "source": "f.txt"}]
        mock_prepare.return_value = [
            {"content": "chunk1", "source": "f.txt", "chunk_id": 0},
            {"content": "chunk2", "source": "f.txt", "chunk_id": 1},
        ]

        rag = RAGChain()
        count = rag.ingest("/some/dir")

        assert count == 2
        mock_vs_cls.return_value.add_documents.assert_called_once()

    def test_default_config_values(self):
        from rag.rag_chain import RAGChain

        with patch("rag.rag_chain.VectorStore"):
            rag = RAGChain()

        assert rag.llm_model == Config.LLM_MODEL
        assert rag.n_results == Config.N_RESULTS
        assert rag.chunk_size == Config.CHUNK_SIZE
        assert rag.chunk_overlap == Config.CHUNK_OVERLAP

    def test_custom_config_values(self):
        from rag.rag_chain import RAGChain

        with patch("rag.rag_chain.VectorStore"):
            rag = RAGChain(
                llm_model="mistral",
                n_results=5,
                chunk_size=300,
                chunk_overlap=30,
            )

        assert rag.llm_model == "mistral"
        assert rag.n_results == 5
        assert rag.chunk_size == 300
        assert rag.chunk_overlap == 30


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_values_are_sensible(self):
        assert Config.LLM_MODEL
        assert Config.EMBEDDING_MODEL
        assert Config.CHUNK_SIZE > 0
        assert Config.CHUNK_OVERLAP >= 0
        assert Config.CHUNK_OVERLAP < Config.CHUNK_SIZE
        assert Config.N_RESULTS > 0
        assert Config.COLLECTION_NAME
