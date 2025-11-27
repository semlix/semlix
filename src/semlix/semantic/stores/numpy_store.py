"""Pure-Python vector store using NumPy.

This module provides a simple, dependency-free vector store implementation
suitable for small to medium datasets (< 100k vectors).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray

from .base import VectorSearchResult


class NumpyVectorStore:
    """Pure-Python vector store using NumPy.
    
    Uses brute-force cosine similarity search. Suitable for small to medium 
    datasets (< 100k vectors). For larger datasets, consider using 
    FaissVectorStore.
    
    Example:
        >>> store = NumpyVectorStore(dimension=384)
        >>> store.add(
        ...     doc_ids=["doc1", "doc2"],
        ...     embeddings=np.random.randn(2, 384).astype(np.float32)
        ... )
        >>> results = store.search(query_embedding, k=5)
        >>> for r in results:
        ...     print(f"{r.doc_id}: {r.score:.4f}")
    
    Attributes:
        dimension: Dimensionality of stored vectors
        count: Number of vectors in the store
    """
    
    def __init__(self, dimension: int):
        """Initialize the vector store.
        
        Args:
            dimension: Dimensionality of vectors to store
        """
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        
        self._dimension = dimension
        self._embeddings: NDArray[np.float32] | None = None
        self._doc_ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}
    
    @property
    def dimension(self) -> int:
        """Return the dimensionality of stored vectors."""
        return self._dimension
    
    @property
    def count(self) -> int:
        """Return number of vectors in the store."""
        return len(self._doc_ids)
    
    def add(
        self,
        doc_ids: List[str],
        embeddings: NDArray[np.float32],
        metadata: List[Dict[str, Any]] | None = None
    ) -> None:
        """Add vectors to the store.
        
        Vectors are L2-normalized before storage to enable efficient
        cosine similarity computation via dot product.
        
        Args:
            doc_ids: Unique identifiers for each document
            embeddings: Array of shape (n, dimension) containing vectors
            metadata: Optional metadata for each document
            
        Raises:
            ValueError: If embedding dimensions don't match store dimension
            ValueError: If number of doc_ids doesn't match number of embeddings
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        if embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"store dimension {self._dimension}"
            )
        
        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Number of doc_ids ({len(doc_ids)}) doesn't match "
                f"number of embeddings ({embeddings.shape[0]})"
            )
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.maximum(norms, 1e-9)
        
        # Handle metadata
        if metadata is None:
            metadata = [{} for _ in doc_ids]
        elif len(metadata) != len(doc_ids):
            raise ValueError(
                f"Number of metadata entries ({len(metadata)}) doesn't match "
                f"number of doc_ids ({len(doc_ids)})"
            )
        
        # Separate new docs from updates
        new_ids = []
        new_embeddings = []
        new_metadata = []
        
        for i, doc_id in enumerate(doc_ids):
            if doc_id in self._id_to_idx:
                # Update existing document
                idx = self._id_to_idx[doc_id]
                self._embeddings[idx] = normalized[i]
                self._metadata[idx] = metadata[i]
            else:
                # Queue new document
                new_ids.append(doc_id)
                new_embeddings.append(normalized[i])
                new_metadata.append(metadata[i])
        
        # Add new documents
        if new_ids:
            start_idx = len(self._doc_ids)
            for i, doc_id in enumerate(new_ids):
                self._id_to_idx[doc_id] = start_idx + i
            
            self._doc_ids.extend(new_ids)
            self._metadata.extend(new_metadata)
            
            new_embeddings_array = np.array(new_embeddings, dtype=np.float32)
            if self._embeddings is None:
                self._embeddings = new_embeddings_array
            else:
                self._embeddings = np.vstack([self._embeddings, new_embeddings_array])
    
    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int = 10,
        filter_ids: List[str] | None = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors using cosine similarity.
        
        Args:
            query_embedding: Query vector of shape (dimension,)
            k: Number of results to return
            filter_ids: Optional list of doc_ids to restrict search to
            
        Returns:
            List of VectorSearchResult sorted by descending similarity
        """
        if self._embeddings is None or len(self._doc_ids) == 0:
            return []
        
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()
        
        if len(query_embedding) != self._dimension:
            raise ValueError(
                f"Query dimension {len(query_embedding)} doesn't match "
                f"store dimension {self._dimension}"
            )
        
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            query_normalized = query_embedding
        
        # Compute cosine similarities (dot product of normalized vectors)
        similarities = np.dot(self._embeddings, query_normalized)
        
        # Apply filter if provided
        if filter_ids is not None:
            filter_set = set(filter_ids)
            mask = np.array([doc_id in filter_set for doc_id in self._doc_ids])
            similarities = np.where(mask, similarities, -np.inf)
        
        # Get top-k indices
        k = min(k, len(self._doc_ids))
        if k <= 0:
            return []
        
        # Use argpartition for efficiency when k << n
        if k < len(similarities) // 2:
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1][:k]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > -np.inf:
                results.append(VectorSearchResult(
                    doc_id=self._doc_ids[idx],
                    score=score,
                    metadata=self._metadata[idx].copy()
                ))
        
        return results
    
    def delete(self, doc_ids: List[str]) -> int:
        """Delete vectors by document ID.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of vectors deleted
        """
        indices_to_remove = []
        
        for doc_id in doc_ids:
            if doc_id in self._id_to_idx:
                indices_to_remove.append(self._id_to_idx[doc_id])
        
        if not indices_to_remove:
            return 0
        
        deleted = len(indices_to_remove)
        
        # Remove from arrays
        mask = np.ones(len(self._doc_ids), dtype=bool)
        mask[indices_to_remove] = False
        
        self._embeddings = self._embeddings[mask]
        self._doc_ids = [d for i, d in enumerate(self._doc_ids) if mask[i]]
        self._metadata = [m for i, m in enumerate(self._metadata) if mask[i]]
        
        # Rebuild index
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self._doc_ids)}
        
        return deleted
    
    def get(self, doc_id: str) -> VectorSearchResult | None:
        """Get a single document by ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            VectorSearchResult if found, None otherwise
        """
        if doc_id not in self._id_to_idx:
            return None
        
        idx = self._id_to_idx[doc_id]
        return VectorSearchResult(
            doc_id=doc_id,
            score=1.0,  # Self-similarity
            metadata=self._metadata[idx].copy()
        )
    
    def clear(self) -> None:
        """Remove all vectors from the store."""
        self._embeddings = None
        self._doc_ids = []
        self._metadata = []
        self._id_to_idx = {}
    
    def save(self, path: Path | str) -> None:
        """Persist the vector store to disk.
        
        Args:
            path: Path to save the store to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": 1,
            "dimension": self._dimension,
            "embeddings": self._embeddings,
            "doc_ids": self._doc_ids,
            "metadata": self._metadata,
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, path: Path | str) -> "NumpyVectorStore":
        """Load a vector store from disk.
        
        Args:
            path: Path to load the store from
            
        Returns:
            Loaded NumpyVectorStore instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        store = cls(dimension=data["dimension"])
        store._embeddings = data["embeddings"]
        store._doc_ids = data["doc_ids"]
        store._metadata = data["metadata"]
        store._id_to_idx = {doc_id: i for i, doc_id in enumerate(store._doc_ids)}
        
        return store
    
    def __len__(self) -> int:
        """Return number of vectors in the store."""
        return self.count
    
    def __contains__(self, doc_id: str) -> bool:
        """Check if a document ID exists in the store."""
        return doc_id in self._id_to_idx
    
    def __repr__(self) -> str:
        return f"NumpyVectorStore(dimension={self._dimension}, count={self.count})"
