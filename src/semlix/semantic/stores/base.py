"""Base interfaces for vector storage backends.

This module defines the protocol that all vector store implementations
must follow, ensuring consistent behavior across different backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass
class VectorSearchResult:
    """Result from vector similarity search.
    
    Attributes:
        doc_id: Unique identifier for the document
        score: Similarity score (higher is more similar)
        metadata: Optional metadata associated with the document
    """
    doc_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"VectorSearchResult(doc_id={self.doc_id!r}, score={self.score:.4f})"


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage and retrieval backends.
    
    All vector store implementations must conform to this protocol
    to be usable with the HybridSearcher.
    
    Example:
        >>> class MyVectorStore:
        ...     @property
        ...     def dimension(self) -> int: ...
        ...     @property
        ...     def count(self) -> int: ...
        ...     def add(self, doc_ids, embeddings, metadata=None): ...
        ...     def search(self, query_embedding, k=10, filter_ids=None): ...
        ...     def delete(self, doc_ids): ...
        ...     def save(self, path): ...
        ...     @classmethod
        ...     def load(cls, path): ...
    """
    
    @property
    def dimension(self) -> int:
        """Return the dimensionality of stored vectors."""
        ...
    
    @property
    def count(self) -> int:
        """Return number of vectors in the store."""
        ...
    
    def add(
        self,
        doc_ids: List[str],
        embeddings: NDArray[np.float32],
        metadata: List[Dict[str, Any]] | None = None
    ) -> None:
        """Add vectors to the store.
        
        Args:
            doc_ids: Unique identifiers for each document
            embeddings: Array of shape (n, dimension) containing vectors
            metadata: Optional metadata for each document
            
        Raises:
            ValueError: If embedding dimensions don't match store dimension
        """
        ...
    
    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int = 10,
        filter_ids: List[str] | None = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors.
        
        Args:
            query_embedding: Query vector of shape (dimension,)
            k: Number of results to return
            filter_ids: Optional list of doc_ids to restrict search to
            
        Returns:
            List of VectorSearchResult sorted by descending similarity
        """
        ...
    
    def delete(self, doc_ids: List[str]) -> int:
        """Delete vectors by document ID.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of vectors deleted
            
        Raises:
            NotImplementedError: If backend doesn't support deletion
        """
        ...
    
    def save(self, path: Path | str) -> None:
        """Persist the vector store to disk.
        
        Args:
            path: Path to save the store to
        """
        ...
    
    @classmethod
    def load(cls, path: Path | str) -> "VectorStore":
        """Load a vector store from disk.
        
        Args:
            path: Path to load the store from
            
        Returns:
            Loaded vector store instance
        """
        ...
