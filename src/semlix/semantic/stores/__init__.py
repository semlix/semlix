"""Vector store backends for semantic search.

This module provides various vector store implementations:

- NumpyVectorStore: Pure-Python implementation for small/medium datasets
- FaissVectorStore: High-performance FAISS backend for large datasets
- PgVectorStore: PostgreSQL + pgvector backend with ACID guarantees

Example:
    >>> from semlix.semantic.stores import NumpyVectorStore
    >>>
    >>> store = NumpyVectorStore(dimension=384)
    >>> store.add(["doc1", "doc2"], embeddings)
    >>> results = store.search(query_embedding, k=5)

    >>> # Or use PostgreSQL backend
    >>> from semlix.semantic.stores import PgVectorStore
    >>>
    >>> store = PgVectorStore(
    ...     connection_string="postgresql://localhost/mydb",
    ...     dimension=384
    ... )
    >>> store.create_index(index_type="hnsw")
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

# Lazy import for optional backends
def __getattr__(name: str):
    if name == "FaissVectorStore":
        from .faiss_store import FaissVectorStore
        return FaissVectorStore
    elif name == "PgVectorStore":
        from .pgvector_store import PgVectorStore
        return PgVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
