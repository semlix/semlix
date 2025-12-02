# Copyright 2025 Semlix Contributors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.

"""BM25 storage using the bm25s library for fast lexical search.

This module provides BM25sStore, a high-performance storage backend for
BM25-based lexical search using the bm25s library. It can achieve 1000+
queries per second and integrates seamlessly with Semlix analyzers.
"""

from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

try:
    import bm25s
    import Stemmer  # PyStemmer for bm25s
except ImportError:
    raise ImportError(
        "bm25s and PyStemmer are required for BM25sStore. "
        "Install with: pip install bm25s PyStemmer"
    )


@dataclass
class SearchResult:
    """Result from BM25 search.

    Attributes:
        doc_id: Document identifier
        score: BM25 relevance score
        rank: Ranking position (0-indexed)
        stored_fields: Fields stored for this document
    """
    doc_id: str
    score: float
    rank: int
    stored_fields: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SearchResult(doc_id={self.doc_id!r}, score={self.score:.4f}, rank={self.rank})"

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to stored fields."""
        return self.stored_fields[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get a stored field with default."""
        return self.stored_fields.get(key, default)


class BM25sStore:
    """High-performance BM25 storage using bm25s library.

    This store provides fast BM25-based lexical search with support for
    Whoosh analyzers. It can handle 1000+ queries per second and uses
    memory-mapped files for efficient large dataset handling.

    Example:
        >>> from semlix.stores import BM25sStore
        >>> from semlix.analysis import StandardAnalyzer
        >>>
        >>> # Create store
        >>> store = BM25sStore.create(
        ...     index_dir="my_index",
        ...     analyzer=StandardAnalyzer(),
        ...     fields=["title", "content"]
        ... )
        >>>
        >>> # Add documents
        >>> docs = [
        ...     {"id": "1", "title": "Python", "content": "Python programming..."},
        ...     {"id": "2", "title": "SQL", "content": "Database queries..."}
        ... ]
        >>> store.add_documents(docs, id_field="id")
        >>>
        >>> # Search
        >>> results = store.search("python tutorial", k=10)
        >>> for r in results:
        ...     print(f"{r.doc_id}: {r.score:.3f}")

    Attributes:
        index_dir: Directory containing the index
        analyzer: Analyzer for text processing
        fields: List of searchable field names
        id_field: Field name containing document IDs
        retriever: bm25s retriever instance
        corpus_texts: Original corpus texts
        doc_ids: List of document IDs
        stored_fields_map: Mapping from doc_id to stored fields
    """

    def __init__(
        self,
        index_dir: Union[str, Path],
        analyzer: Any = None,
        fields: Optional[List[str]] = None,
        id_field: str = "id",
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5
    ):
        """Initialize BM25sStore.

        Args:
            index_dir: Directory to store index files
            analyzer: Whoosh analyzer for tokenization (default: StandardAnalyzer)
            fields: List of fields to index and search
            id_field: Field name containing document IDs
            method: BM25 variant (lucene, robertson, atire, bm25l, bm25+)
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
            delta: BM25+ delta parameter (only for bm25+ method)
        """
        self.index_dir = Path(index_dir)
        self.id_field = id_field
        self.fields = fields or ["content"]
        self.method = method
        self.k1 = k1
        self.b = b
        self.delta = delta

        # Setup analyzer
        if analyzer is None:
            from semlix.analysis import StandardAnalyzer
            analyzer = StandardAnalyzer()
        self.analyzer = analyzer

        # Initialize storage
        self.retriever = None
        self.corpus_texts = []
        self.doc_ids = []
        self.stored_fields_map = {}

        # Ensure index directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        index_dir: Union[str, Path],
        analyzer: Any = None,
        fields: Optional[List[str]] = None,
        id_field: str = "id",
        **kwargs
    ) -> "BM25sStore":
        """Create a new BM25sStore.

        Args:
            index_dir: Directory to store index files
            analyzer: Whoosh analyzer for tokenization
            fields: List of fields to index and search
            id_field: Field name containing document IDs
            **kwargs: Additional parameters passed to __init__

        Returns:
            New BM25sStore instance
        """
        store = cls(index_dir, analyzer, fields, id_field, **kwargs)
        return store

    @classmethod
    def load(cls, index_dir: Union[str, Path], mmap: bool = True) -> "BM25sStore":
        """Load an existing BM25sStore from disk.

        Args:
            index_dir: Directory containing the index
            mmap: Use memory-mapped files (recommended for large datasets)

        Returns:
            Loaded BM25sStore instance

        Raises:
            FileNotFoundError: If index doesn't exist
        """
        index_path = Path(index_dir)

        if not index_path.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")

        # Load metadata
        meta_path = index_path / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Index metadata not found: {meta_path}")

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Create store instance
        store = cls(
            index_dir=index_dir,
            analyzer=None,  # Will load from metadata
            fields=metadata["fields"],
            id_field=metadata["id_field"],
            method=metadata.get("method", "lucene"),
            k1=metadata.get("k1", 1.5),
            b=metadata.get("b", 0.75),
            delta=metadata.get("delta", 0.5)
        )

        # Load analyzer
        analyzer_path = index_path / "analyzer.pkl"
        if analyzer_path.exists():
            with open(analyzer_path, "rb") as f:
                store.analyzer = pickle.load(f)

        # Load retriever
        retriever_path = index_path / "retriever"
        if retriever_path.exists():
            store.retriever = bm25s.BM25.load(str(retriever_path), mmap=mmap)

        # Load corpus texts
        corpus_path = index_path / "corpus.json"
        if corpus_path.exists():
            with open(corpus_path, "r") as f:
                store.corpus_texts = json.load(f)

        # Load doc IDs
        doc_ids_path = index_path / "doc_ids.json"
        if doc_ids_path.exists():
            with open(doc_ids_path, "r") as f:
                store.doc_ids = json.load(f)

        # Load stored fields
        stored_path = index_path / "stored_fields.json"
        if stored_path.exists():
            with open(stored_path, "r") as f:
                store.stored_fields_map = json.load(f)

        return store

    def save(self) -> None:
        """Save the index to disk.

        Saves the retriever, corpus, doc IDs, stored fields, and metadata.
        """
        # Save retriever
        if self.retriever is not None:
            retriever_path = self.index_dir / "retriever"
            self.retriever.save(str(retriever_path))

        # Save corpus texts
        corpus_path = self.index_dir / "corpus.json"
        with open(corpus_path, "w") as f:
            json.dump(self.corpus_texts, f)

        # Save doc IDs
        doc_ids_path = self.index_dir / "doc_ids.json"
        with open(doc_ids_path, "w") as f:
            json.dump(self.doc_ids, f)

        # Save stored fields
        stored_path = self.index_dir / "stored_fields.json"
        with open(stored_path, "w") as f:
            json.dump(self.stored_fields_map, f)

        # Save analyzer
        analyzer_path = self.index_dir / "analyzer.pkl"
        with open(analyzer_path, "wb") as f:
            pickle.dump(self.analyzer, f)

        # Save metadata
        metadata = {
            "fields": self.fields,
            "id_field": self.id_field,
            "method": self.method,
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta,
            "doc_count": len(self.doc_ids)
        }
        meta_path = self.index_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _tokenize_with_analyzer(self, text: str) -> List[str]:
        """Tokenize text using Whoosh analyzer.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        tokens = []
        for token in self.analyzer(text):
            tokens.append(token.text)
        return tokens

    def _extract_field_text(self, doc: Dict[str, Any]) -> str:
        """Extract searchable text from document fields.

        Args:
            doc: Document dictionary

        Returns:
            Combined text from all searchable fields
        """
        texts = []
        for field in self.fields:
            if field in doc:
                value = doc[field]
                if isinstance(value, str):
                    texts.append(value)
                else:
                    texts.append(str(value))
        return " ".join(texts)

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        id_field: Optional[str] = None,
        batch_size: int = 1000
    ) -> None:
        """Add documents to the index.

        Args:
            documents: List of document dictionaries
            id_field: Field containing document ID (uses self.id_field if None)
            batch_size: Number of documents to process in one batch

        Raises:
            ValueError: If documents don't contain required ID field
        """
        if not documents:
            return

        id_field = id_field or self.id_field

        # Extract texts and IDs
        new_texts = []
        new_ids = []

        for doc in documents:
            if id_field not in doc:
                raise ValueError(f"Document missing required field: {id_field}")

            doc_id = str(doc[id_field])
            text = self._extract_field_text(doc)

            new_texts.append(text)
            new_ids.append(doc_id)
            self.stored_fields_map[doc_id] = doc

        # Update corpus and doc IDs
        self.corpus_texts.extend(new_texts)
        self.doc_ids.extend(new_ids)

        # Tokenize corpus using Whoosh analyzer
        corpus_tokens = [
            self._tokenize_with_analyzer(text)
            for text in self.corpus_texts
        ]

        # Create or update retriever
        self.retriever = bm25s.BM25(method=self.method, k1=self.k1, b=self.b, delta=self.delta)
        self.retriever.index(corpus_tokens)

    def search(
        self,
        query: str,
        k: int = 10,
        return_docs: bool = True
    ) -> List[SearchResult]:
        """Search the index using BM25.

        Args:
            query: Search query string
            k: Number of results to return
            return_docs: Whether to include stored fields in results

        Returns:
            List of SearchResult objects sorted by relevance

        Raises:
            RuntimeError: If index is empty or not initialized
        """
        if self.retriever is None:
            raise RuntimeError("Index not initialized. Add documents first.")

        if not self.doc_ids:
            return []

        # Tokenize query
        query_tokens = self._tokenize_with_analyzer(query)

        # Search
        results, scores = self.retriever.retrieve(
            bm25s.tokenize([query_tokens], update_vocab=False),
            k=k
        )

        # Build result objects
        search_results = []
        for rank, (doc_indices, doc_scores) in enumerate(zip(results[0], scores[0])):
            if doc_indices < len(self.doc_ids):
                doc_id = self.doc_ids[doc_indices]
                stored = self.stored_fields_map.get(doc_id, {}) if return_docs else {}

                search_results.append(SearchResult(
                    doc_id=doc_id,
                    score=float(doc_scores),
                    rank=rank,
                    stored_fields=stored
                ))

        return search_results

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get stored fields for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Document fields dict or None if not found
        """
        return self.stored_fields_map.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the index.

        Note: This requires rebuilding the index. For frequent deletions,
        consider accumulating deletes and rebuilding periodically.

        Args:
            doc_id: Document identifier

        Returns:
            True if document was found and deleted, False otherwise
        """
        if doc_id not in self.stored_fields_map:
            return False

        # Find index of document
        try:
            idx = self.doc_ids.index(doc_id)
        except ValueError:
            return False

        # Remove from all structures
        del self.corpus_texts[idx]
        del self.doc_ids[idx]
        del self.stored_fields_map[doc_id]

        # Rebuild index
        if self.corpus_texts:
            corpus_tokens = [
                self._tokenize_with_analyzer(text)
                for text in self.corpus_texts
            ]
            self.retriever = bm25s.BM25(method=self.method, k1=self.k1, b=self.b, delta=self.delta)
            self.retriever.index(corpus_tokens)
        else:
            self.retriever = None

        return True

    def doc_count(self) -> int:
        """Get the number of documents in the index.

        Returns:
            Document count
        """
        return len(self.doc_ids)

    def optimize(self) -> None:
        """Optimize the index.

        For BM25s, this rebuilds the index from scratch which can
        improve performance if many documents were deleted.
        """
        if not self.corpus_texts:
            return

        corpus_tokens = [
            self._tokenize_with_analyzer(text)
            for text in self.corpus_texts
        ]
        self.retriever = bm25s.BM25(method=self.method, k1=self.k1, b=self.b, delta=self.delta)
        self.retriever.index(corpus_tokens)

    def close(self) -> None:
        """Close the store and free resources."""
        # bm25s doesn't require explicit closing
        pass

    def __enter__(self) -> "BM25sStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        return (
            f"BM25sStore(index_dir={str(self.index_dir)!r}, "
            f"docs={len(self.doc_ids)}, "
            f"method={self.method!r})"
        )
