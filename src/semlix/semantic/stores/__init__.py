"""Vector store backends for semantic search.

This module provides various vector store implementations:

- NumpyVectorStore: Pure-Python implementation for small/medium datasets
- FaissVectorStore: High-performance FAISS backend for large datasets

Example:
    >>> from semlix.semantic.stores import NumpyVectorStore
    >>> 
    >>> store = NumpyVectorStore(dimension=384)
    >>> store.add(["doc1", "doc2"], embeddings)
    >>> results = store.search(query_embedding, k=5)
"""

from .base import VectorSearchResult, VectorStore
from .numpy_store import NumpyVectorStore

__all__ = [
    "VectorSearchResult",
    "VectorStore",
    "NumpyVectorStore",
]

# Lazy import for optional FAISS backend
def __getattr__(name: str):
    if name == "FaissVectorStore":
        from .faiss_store import FaissVectorStore
        return FaissVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
