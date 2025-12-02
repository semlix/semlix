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

"""Unified index combining BM25 and vector search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from semlix.index import Index
from semlix.fields import Schema
from semlix.bm25 import BM25Index, create_bm25_index, open_bm25_index
from semlix.semantic.stores import PgVectorStore
from semlix.semantic.embeddings import EmbeddingProvider


class UnifiedIndex(Index):
    """Unified index combining BM25 lexical and pgvector semantic search.

    This index provides a single interface for both lexical and semantic
    search, with transactional writes across both stores.

    Example:
        >>> from semlix.unified import create_unified_index
        >>> from semlix.fields import Schema, TEXT, ID
        >>> from semlix.semantic import SentenceTransformerProvider
        >>>
        >>> schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
        >>> embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")
        >>>
        >>> ix = create_unified_index(
        ...     index_dir="my_index",
        ...     schema=schema,
        ...     connection_string="postgresql://localhost/mydb",
        ...     embedder=embedder
        ... )
        >>>
        >>> # Index documents (both lexical and semantic)
        >>> with ix.writer() as w:
        ...     w.add_document(id="1", content="Python programming tutorial")
        ...
        >>> # Hybrid search
        >>> with ix.searcher() as s:
        ...     results = s.hybrid_search("learn python", limit=10)

    Attributes:
        bm25_index: BM25Index for lexical search
        vector_store: PgVectorStore for semantic search
        embedder: EmbeddingProvider for generating embeddings
    """

    def __init__(
        self,
        index_dir: Union[str, Path],
        schema: Schema,
        connection_string: str,
        embedder: EmbeddingProvider,
        bm25_index: Optional[BM25Index] = None,
        vector_store: Optional[PgVectorStore] = None,
        id_field: str = "id",
        searchable_fields: Optional[list] = None
    ):
        """Initialize UnifiedIndex.

        Args:
            index_dir: Directory containing the index
            schema: Schema defining index fields
            connection_string: PostgreSQL connection string
            embedder: Embedding provider for semantic search
            bm25_index: Existing BM25Index (or None to load/create)
            vector_store: Existing PgVectorStore (or None to load/create)
            id_field: Field name containing document IDs
            searchable_fields: Fields to use for embedding generation
        """
        self.index_dir = Path(index_dir)
        self._schema = schema
        self.connection_string = connection_string
        self.embedder = embedder
        self.id_field = id_field

        # Determine searchable fields
        if searchable_fields is None:
            from semlix.fields import TEXT
            searchable_fields = [
                name for name, field in schema.items()
                if isinstance(field, TEXT)
            ]
        self.searchable_fields = searchable_fields

        # Create or load BM25 index
        if bm25_index is None:
            bm25_path = self.index_dir / "bm25"
            if (bm25_path / "metadata.json").exists():
                self.bm25_index = open_bm25_index(bm25_path)
            else:
                self.bm25_index = create_bm25_index(bm25_path, schema)
        else:
            self.bm25_index = bm25_index

        # Create or load vector store
        if vector_store is None:
            table_name = f"semlix_vectors_{Path(index_dir).name}"
            self.vector_store = PgVectorStore(
                connection_string=connection_string,
                dimension=embedder.dimension,
                table_name=table_name
            )
        else:
            self.vector_store = vector_store

        # Save metadata
        self._save_metadata()

    def _save_metadata(self) -> None:
        """Save unified index metadata."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "connection_string": self.connection_string,
            "id_field": self.id_field,
            "searchable_fields": self.searchable_fields,
            "embedding_model": self.embedder.model_name if hasattr(self.embedder, 'model_name') else "unknown",
            "dimension": self.embedder.dimension
        }

        meta_path = self.index_dir / "unified_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @property
    def schema(self) -> Schema:
        """Get the index schema."""
        return self._schema

    def close(self) -> None:
        """Close the index."""
        self.bm25_index.close()
        self.vector_store.close()

    def is_empty(self) -> bool:
        """Check if index is empty."""
        return self.bm25_index.is_empty()

    def doc_count(self) -> int:
        """Get number of documents."""
        return self.bm25_index.doc_count()

    def doc_count_all(self) -> int:
        """Get total number of documents."""
        return self.bm25_index.doc_count_all()

    def optimize(self) -> None:
        """Optimize both indexes."""
        self.bm25_index.optimize()
        # Vector store optimization (rebuild index)
        if hasattr(self.vector_store, 'create_index'):
            self.vector_store.create_index()

    def latest_generation(self) -> int:
        """Get latest generation number."""
        return self.bm25_index.latest_generation()

    def refresh(self) -> "UnifiedIndex":
        """Refresh to latest generation."""
        return self

    def up_to_date(self) -> bool:
        """Check if up to date."""
        return True

    def last_modified(self) -> float:
        """Get last modified time."""
        return self.bm25_index.last_modified()

    def reader(self, reuse=None):
        """Get a reader for this index."""
        return self.bm25_index.reader(reuse)

    def writer(self, **kwargs) -> "UnifiedWriter":
        """Get a unified writer for this index."""
        from .unified_writer import UnifiedWriter
        return UnifiedWriter(self, **kwargs)

    def searcher(self, **kwargs) -> "UnifiedSearcher":
        """Get a unified searcher for this index."""
        from .unified_searcher import UnifiedSearcher
        return UnifiedSearcher(self, **kwargs)

    def __repr__(self) -> str:
        return f"UnifiedIndex({str(self.index_dir)!r}, docs={self.doc_count()})"


def create_unified_index(
    index_dir: Union[str, Path],
    schema: Schema,
    connection_string: str,
    embedder: EmbeddingProvider,
    **kwargs
) -> UnifiedIndex:
    """Create a new UnifiedIndex.

    Args:
        index_dir: Directory to create the index in
        schema: Schema defining index fields
        connection_string: PostgreSQL connection string
        embedder: Embedding provider for semantic search
        **kwargs: Additional arguments

    Returns:
        New UnifiedIndex instance

    Example:
        >>> from semlix.unified import create_unified_index
        >>> from semlix.fields import Schema, TEXT, ID
        >>> from semlix.semantic import SentenceTransformerProvider
        >>>
        >>> schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
        >>> embedder = SentenceTransformerProvider()
        >>>
        >>> ix = create_unified_index(
        ...     "my_index",
        ...     schema,
        ...     "postgresql://localhost/mydb",
        ...     embedder
        ... )
    """
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    # Save schema
    import pickle
    schema_path = index_path / "schema.pkl"
    with open(schema_path, "wb") as f:
        pickle.dump(schema, f)

    return UnifiedIndex(index_dir, schema, connection_string, embedder, **kwargs)


def open_unified_index(
    index_dir: Union[str, Path],
    embedder: Optional[EmbeddingProvider] = None
) -> UnifiedIndex:
    """Open an existing UnifiedIndex.

    Args:
        index_dir: Directory containing the index
        embedder: Embedding provider (or None to use saved model name)

    Returns:
        Opened UnifiedIndex instance

    Raises:
        FileNotFoundError: If index doesn't exist

    Example:
        >>> from semlix.unified import open_unified_index
        >>> from semlix.semantic import SentenceTransformerProvider
        >>>
        >>> embedder = SentenceTransformerProvider()
        >>> ix = open_unified_index("my_index", embedder)
    """
    index_path = Path(index_dir)

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_dir}")

    # Load schema
    import pickle
    schema_path = index_path / "schema.pkl"
    if not schema_path.exists():
        raise FileNotFoundError(f"Index schema not found: {schema_path}")

    with open(schema_path, "rb") as f:
        schema = pickle.load(f)

    # Load metadata
    meta_path = index_path / "unified_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Unified metadata not found: {meta_path}")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    # Create embedder if not provided
    if embedder is None:
        from semlix.semantic import SentenceTransformerProvider
        model_name = metadata.get("embedding_model", "all-MiniLM-L6-v2")
        embedder = SentenceTransformerProvider(model_name)

    return UnifiedIndex(
        index_dir=index_dir,
        schema=schema,
        connection_string=metadata["connection_string"],
        embedder=embedder,
        id_field=metadata.get("id_field", "id"),
        searchable_fields=metadata.get("searchable_fields")
    )
