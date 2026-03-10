"""Interactive demo for the VirtualBot RAG system."""

import os
import sys

from rag import RAGChain, Config


def main() -> None:
    print("=== VirtualBot RAG System ===")
    print(f"LLM model      : {Config.LLM_MODEL}")
    print(f"Embedding model: {Config.EMBEDDING_MODEL}")
    print()

    rag = RAGChain()

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        print(f"Error: data directory '{data_dir}' not found.")
        sys.exit(1)

    print(f"Loading documents from '{data_dir}' …")
    n_chunks = rag.ingest(data_dir)
    print(f"Indexed {n_chunks} chunk(s).")
    print()

    print("RAG system ready. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not question:
            continue

        print("Searching and generating answer …")
        result = rag.query(question)

        print(f"\nAnswer  : {result['answer']}")
        print(f"Sources : {', '.join(sorted(set(result['sources'])))}")
        print()


if __name__ == "__main__":
    main()
