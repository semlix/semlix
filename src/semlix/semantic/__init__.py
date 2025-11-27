"""Semantic search extension for semlix.

This module adds semantic (vector-based) search capabilities to semlix,
enabling hybrid search that combines traditional lexical matching with
modern embedding-based semantic similarity.

Quick Start:
    >>> from semlix.index import create_in
    >>> from semlix.fields import Schema, TEXT, ID
    >>> from semlix.semantic import (
    ...     HybridSearcher,
    ...     HybridIndexWriter,
    ...     SentenceTransformerProvider,
    ... )
    >>> from semlix.semantic.stores import NumpyVectorStore
    >>> 
    >>> # Create schema and index
    >>> schema = Schema(id=ID(stored=True, unique=True), content=TEXT(stored=True))
    >>> ix = create_in("my_index", schema)
    >>> 
    >>> # Create semantic components
    >>> embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")
    >>> vector_store = NumpyVectorStore(dimension=embedder.dimension)
    >>> 
    >>> # Index documents
    >>> with HybridIndexWriter(ix, vector_store, embedder) as writer:
    ...     writer.add_document(id="1", content="Python programming basics")
    ...     writer.add_document(id="2", content="How to fix login issues")
    >>> 
    >>> # Save vector store
    >>> vector_store.save("vectors.pkl")
    >>> 
    >>> # Search
    >>> searcher = HybridSearcher(ix, vector_store, embedder)
    >>> results = searcher.search("authentication problems")  # Matches "login issues"!

Components:
    - HybridSearcher: Main search interface combining lexical + semantic
    - HybridIndexWriter: Index writer maintaining both stores
    - EmbeddingProvider: Protocol for embedding generation
    - VectorStore: Protocol for vector storage backends

Embedding Providers:
    - SentenceTransformerProvider: Local models via sentence-transformers
    - OpenAIProvider: OpenAI embedding API
    - CohereProvider: Cohere embedding API
    - HuggingFaceInferenceProvider: HuggingFace Inference API

Vector Stores:
    - NumpyVectorStore: Pure-Python, no dependencies (< 100k vectors)
    - FaissVectorStore: High-performance FAISS backend (millions of vectors)

Fusion Methods:
    - RRF (Reciprocal Rank Fusion): Robust, rank-based (recommended)
    - LINEAR: Simple weighted score combination
    - DBSF: Distribution-based score fusion
    - RELATIVE_SCORE: Percentile-based normalization
"""

__version__ = "0.1.0"

# Core classes
from .hybrid import HybridSearcher, HybridSearchResult
from .indexing import HybridIndexWriter, build_vector_store_from_index

# Embedding providers
from .embeddings import (
    EmbeddingProvider,
    SentenceTransformerProvider,
    OpenAIProvider,
    CohereProvider,
    HuggingFaceInferenceProvider,
)

# Fusion
from .fusion import FusionMethod, FusedResult

# Re-export stores at top level for convenience
from .stores import NumpyVectorStore, VectorStore, VectorSearchResult

__all__ = [
    # Version
    "__version__",
    # Core
    "HybridSearcher",
    "HybridSearchResult",
    "HybridIndexWriter",
    "build_vector_store_from_index",
    # Embeddings
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "OpenAIProvider",
    "CohereProvider",
    "HuggingFaceInferenceProvider",
    # Stores
    "VectorStore",
    "VectorSearchResult",
    "NumpyVectorStore",
    # Fusion
    "FusionMethod",
    "FusedResult",
]


# Lazy import for optional FAISS backend
def __getattr__(name: str):
    if name == "FaissVectorStore":
        from .stores.faiss_store import FaissVectorStore
        return FaissVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
