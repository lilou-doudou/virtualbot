# virtualbot

Projet de test pour un système RAG (*Retrieval-Augmented Generation*) utilisant [Ollama](https://ollama.com) comme LLM local.

## Architecture

```
virtualbot/
├── rag/
│   ├── config.py           # Paramètres par défaut (modèles, taille des chunks…)
│   ├── document_loader.py  # Chargement et découpage des documents
│   ├── vectorstore.py      # Stockage vectoriel ChromaDB + embeddings Ollama
│   └── rag_chain.py        # Pipeline RAG complet (ingestion + requête)
├── data/
│   └── example.txt         # Base de connaissances d'exemple
├── tests/
│   └── test_rag.py         # Tests unitaires (sans Ollama requis)
├── main.py                 # Script interactif de démonstration
└── requirements.txt
```

## Prérequis

| Logiciel | Version minimale |
|----------|-----------------|
| Python   | 3.10+           |
| [Ollama](https://ollama.com) | dernière version |

### Installer Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
```

### Télécharger les modèles nécessaires

```bash
ollama pull llama3.2          # modèle de génération de texte
ollama pull nomic-embed-text  # modèle d'embeddings
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation rapide

### Script interactif

```bash
python main.py
```

Le script :
1. Charge tous les fichiers `.txt` / `.md` du répertoire `data/`.
2. Découpe les documents en chunks et les indexe dans ChromaDB.
3. Lance une boucle de questions/réponses interactive.

```
=== VirtualBot RAG System ===
LLM model      : llama3.2
Embedding model: nomic-embed-text

Loading documents from 'data/' …
Indexed 12 chunk(s).

RAG system ready. Type 'quit' to exit.

Your question: Qu'est-ce qu'Ollama ?
Searching and generating answer …

Answer  : Ollama est un outil qui permet d'exécuter des grands modèles de langage open-source localement…
Sources : example.txt
```

### API Python

```python
from rag import RAGChain

# Initialiser le pipeline (stockage en mémoire par défaut)
rag = RAGChain()

# Indexer des documents
rag.ingest("data/")

# Poser une question
result = rag.query("Qu'est-ce que le RAG ?")
print(result["answer"])
print("Sources :", result["sources"])
```

#### Stockage persistant

```python
rag = RAGChain(persist_directory="chroma_db/")
```

#### Changer de modèle

```python
rag = RAGChain(llm_model="mistral", embedding_model="nomic-embed-text")
```

### Ajouter vos propres documents

Déposez vos fichiers `.txt` ou `.md` dans le répertoire `data/` puis relancez `main.py`.

## Configuration

Tous les paramètres par défaut se trouvent dans `rag/config.py` :

| Paramètre         | Valeur par défaut    | Description                              |
|-------------------|----------------------|------------------------------------------|
| `LLM_MODEL`       | `llama3.2`           | Modèle Ollama pour la génération         |
| `EMBEDDING_MODEL` | `nomic-embed-text`   | Modèle Ollama pour les embeddings        |
| `CHUNK_SIZE`      | `500`                | Caractères par chunk                     |
| `CHUNK_OVERLAP`   | `50`                 | Chevauchement entre chunks consécutifs   |
| `N_RESULTS`       | `3`                  | Nombre de chunks récupérés par requête   |
| `COLLECTION_NAME` | `virtualbot_rag`     | Nom de la collection ChromaDB            |

## Tests

Les tests n'ont pas besoin d'Ollama — toutes les dépendances externes sont mockées.

```bash
pytest tests/ -v
```

## Fonctionnement

```
Documents (.txt/.md)
        │
        ▼
  Document Loader  ──► Chunking (chunk_text)
        │
        ▼
  Ollama Embeddings (nomic-embed-text)
        │
        ▼
  ChromaDB (stockage vectoriel)

        ▲
        │  requête utilisateur
        │
  Similarity Search  ──► chunks les plus proches
        │
        ▼
  Prompt = contexte + question
        │
        ▼
  Ollama LLM (llama3.2) ──► réponse
```