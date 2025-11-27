"""High-performance vector store using FAISS.

This module provides a FAISS-backed vector store implementation
suitable for large datasets (millions of vectors).

Requires: pip install faiss-cpu (or faiss-gpu for GPU support)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
from numpy.typing import NDArray

from .base import VectorSearchResult


IndexType = Literal["Flat", "IVF", "HNSW"]


class FaissVectorStore:
    """High-performance vector store using FAISS.
    
    Supports various index types for different performance/accuracy tradeoffs:
    
    - "Flat": Exact search, best accuracy, O(n) search time
    - "IVF": Approximate search, good accuracy, requires training
    - "HNSW": Approximate search, excellent speed, higher memory
    
    Example:
        >>> store = FaissVectorStore(dimension=384, index_type="IVF")
        >>> # For IVF, train on representative sample first
        >>> store.train(sample_embeddings)
        >>> store.add(doc_ids, embeddings)
        >>> results = store.search(query, k=10)
    
    Attributes:
        dimension: Dimensionality of stored vectors
        count: Number of vectors in the store
        is_trained: Whether the index has been trained (always True for Flat/HNSW)
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: IndexType = "Flat",
        nlist: int = 100,
        nprobe: int = 10,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128
    ):
        """Initialize the FAISS vector store.
        
        Args:
            dimension: Dimensionality of vectors to store
            index_type: Type of FAISS index ("Flat", "IVF", or "HNSW")
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search for IVF index
            m: Number of connections per layer for HNSW
            ef_construction: Size of dynamic candidate list for HNSW construction
            ef_search: Size of dynamic candidate list for HNSW search
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required for FaissVectorStore. "
                "Install with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
        
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        
        self._faiss = faiss
        self._dimension = dimension
        self._index_type = index_type
        self._nlist = nlist
        self._nprobe = nprobe
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        
        # Create index based on type
        self._base_index = self._create_base_index()
        
        # Wrap with IDMap for document ID support
        self._index = faiss.IndexIDMap(self._base_index)
        
        # Metadata storage
        self._doc_ids: List[str] = []
        self._id_to_faiss_id: Dict[str, int] = {}
        self._faiss_id_to_doc_id: Dict[int, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._next_id: int = 0
        self._is_trained: bool = index_type in ("Flat", "HNSW")
    
    def _create_base_index(self):
        """Create the underlying FAISS index."""
        faiss = self._faiss
        
        if self._index_type == "Flat":
            return faiss.IndexFlatIP(self._dimension)
        
        elif self._index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self._dimension)
            index = faiss.IndexIVFFlat(
                quantizer, 
                self._dimension, 
                self._nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            index.nprobe = self._nprobe
            return index
        
        elif self._index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self._dimension, self._m)
            index.hnsw.efConstruction = self._ef_construction
            index.hnsw.efSearch = self._ef_search
            return index
        
        else:
            raise ValueError(f"Unknown index type: {self._index_type}")
    
    @property
    def dimension(self) -> int:
        """Return the dimensionality of stored vectors."""
        return self._dimension
    
    @property
    def count(self) -> int:
        """Return number of vectors in the store."""
        return self._index.ntotal
    
    @property
    def is_trained(self) -> bool:
        """Return whether the index has been trained."""
        return self._is_trained
    
    @property
    def index_type(self) -> str:
        """Return the type of FAISS index."""
        return self._index_type
    
    def train(self, embeddings: NDArray[np.float32]) -> None:
        """Train the index on representative vectors.
        
        Required for IVF-based indexes before adding vectors.
        For Flat and HNSW indexes, this is a no-op.
        
        Args:
            embeddings: Training vectors of shape (n, dimension)
        """
        if self._index_type not in ("IVF",):
            return  # No training needed
        
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Normalize for cosine similarity
        embeddings = embeddings.copy()
        self._faiss.normalize_L2(embeddings)
        
        # Train the underlying index (not the IDMap wrapper)
        self._base_index.train(embeddings)
        self._is_trained = True
    
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
            RuntimeError: If index requires training but hasn't been trained
            ValueError: If dimensions don't match
        """
        if not self._is_trained:
            raise RuntimeError(
                "Index must be trained before adding vectors. "
                "Call train() with representative vectors first."
            )
        
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
        
        # Normalize for cosine similarity (using inner product)
        embeddings = embeddings.copy()
        self._faiss.normalize_L2(embeddings)
        
        # Assign FAISS IDs
        faiss_ids = np.arange(
            self._next_id, 
            self._next_id + len(doc_ids), 
            dtype=np.int64
        )
        self._next_id += len(doc_ids)
        
        # Update mappings
        for doc_id, faiss_id in zip(doc_ids, faiss_ids):
            faiss_id_int = int(faiss_id)
            self._id_to_faiss_id[doc_id] = faiss_id_int
            self._faiss_id_to_doc_id[faiss_id_int] = doc_id
            self._doc_ids.append(doc_id)
        
        # Store metadata
        if metadata:
            for doc_id, meta in zip(doc_ids, metadata):
                self._metadata[doc_id] = meta
        
        # Add to index
        self._index.add_with_ids(embeddings, faiss_ids)
    
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
        if self.count == 0:
            return []
        
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        query = query_embedding.copy()
        self._faiss.normalize_L2(query)
        
        # Search (get more results if filtering)
        search_k = min(k * 3 if filter_ids else k, self.count)
        scores, faiss_ids = self._index.search(query, search_k)
        
        # Build results
        filter_set = set(filter_ids) if filter_ids else None
        results = []
        
        for faiss_id, score in zip(faiss_ids[0], scores[0]):
            if faiss_id == -1:
                continue
            
            doc_id = self._faiss_id_to_doc_id.get(int(faiss_id))
            if doc_id is None:
                continue
            
            if filter_set and doc_id not in filter_set:
                continue
            
            results.append(VectorSearchResult(
                doc_id=doc_id,
                score=float(score),
                metadata=self._metadata.get(doc_id, {}).copy()
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def delete(self, doc_ids: List[str]) -> int:
        """Delete vectors by document ID.
        
        Note: FAISS does not support efficient deletion for most index types.
        For workloads with frequent deletions, consider using NumpyVectorStore
        or rebuilding the index periodically.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Raises:
            NotImplementedError: FAISS indexes don't support deletion
        """
        raise NotImplementedError(
            "FAISS index does not support efficient deletion. "
            "Consider using NumpyVectorStore for workloads with frequent "
            "deletions, or rebuild the index periodically."
        )
    
    def save(self, path: Path | str) -> None:
        """Persist the vector store to disk.
        
        Args:
            path: Base path for saving (will create .faiss and .meta files)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_path = path.with_suffix(".faiss")
        self._faiss.write_index(self._index, str(faiss_path))
        
        # Save metadata and mappings
        meta = {
            "version": 1,
            "dimension": self._dimension,
            "index_type": self._index_type,
            "nlist": self._nlist,
            "nprobe": self._nprobe,
            "m": self._m,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_search,
            "doc_ids": self._doc_ids,
            "id_to_faiss_id": self._id_to_faiss_id,
            "faiss_id_to_doc_id": self._faiss_id_to_doc_id,
            "metadata": self._metadata,
            "next_id": self._next_id,
            "is_trained": self._is_trained,
        }
        
        meta_path = path.with_suffix(".meta")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, path: Path | str) -> "FaissVectorStore":
        """Load a vector store from disk.
        
        Args:
            path: Base path to load from (expects .faiss and .meta files)
            
        Returns:
            Loaded FaissVectorStore instance
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required for FaissVectorStore. "
                "Install with: pip install faiss-cpu"
            )
        
        path = Path(path)
        
        # Load metadata first to get parameters
        meta_path = path.with_suffix(".meta")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        
        # Create store with same parameters
        store = cls(
            dimension=meta["dimension"],
            index_type=meta["index_type"],
            nlist=meta.get("nlist", 100),
            nprobe=meta.get("nprobe", 10),
            m=meta.get("m", 32),
            ef_construction=meta.get("ef_construction", 200),
            ef_search=meta.get("ef_search", 128)
        )
        
        # Load FAISS index
        faiss_path = path.with_suffix(".faiss")
        store._index = faiss.read_index(str(faiss_path))
        
        # Restore state
        store._doc_ids = meta["doc_ids"]
        store._id_to_faiss_id = meta["id_to_faiss_id"]
        store._faiss_id_to_doc_id = {
            int(k): v for k, v in meta["faiss_id_to_doc_id"].items()
        }
        store._metadata = meta["metadata"]
        store._next_id = meta["next_id"]
        store._is_trained = meta["is_trained"]
        
        return store
    
    def __len__(self) -> int:
        """Return number of vectors in the store."""
        return self.count
    
    def __contains__(self, doc_id: str) -> bool:
        """Check if a document ID exists in the store."""
        return doc_id in self._id_to_faiss_id
    
    def __repr__(self) -> str:
        return (
            f"FaissVectorStore(dimension={self._dimension}, "
            f"index_type={self._index_type!r}, count={self.count})"
        )
