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

"""BM25-based Index implementation.

This module provides BM25Index, a complete implementation of the Semlix
Index protocol using bm25s for high-performance lexical search.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from semlix.index import Index
from semlix.fields import Schema, ensure_schema
from semlix.stores.bm25s_store import BM25sStore


class BM25Index(Index):
    """BM25-based index implementation.

    This class implements the complete Semlix Index protocol using bm25s
    for fast BM25 scoring. It's compatible with HybridSearcher and other
    Semlix components.

    Example:
        >>> from semlix.bm25 import create_bm25_index
        >>> from semlix.fields import Schema, TEXT, ID
        >>>
        >>> schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
        >>> ix = create_bm25_index("my_index", schema)
        >>>
        >>> with ix.writer() as writer:
        ...     writer.add_document(id="1", content="Python programming")
        ...     writer.add_document(id="2", content="SQL databases")
        >>>
        >>> with ix.searcher() as searcher:
        ...     results = searcher.search("python")

    Attributes:
        index_dir: Directory containing the index
        schema: Schema defining index fields
        store: Underlying BM25sStore instance
    """

    def __init__(
        self,
        index_dir: Union[str, Path],
        schema: Schema,
        store: Optional[BM25sStore] = None
    ):
        """Initialize BM25Index.

        Args:
            index_dir: Directory containing the index
            schema: Schema defining index fields
            store: Existing BM25sStore (or None to load/create)
        """
        self.index_dir = Path(index_dir)
        self._schema = ensure_schema(schema)

        # Get searchable fields
        searchable_fields = [
            name for name, field in self._schema.items()
            if hasattr(field, 'analyzer') and field.analyzer is not None
        ]

        if not searchable_fields:
            # Fall back to all TEXT fields
            from semlix.fields import TEXT
            searchable_fields = [
                name for name, field in self._schema.items()
                if isinstance(field, TEXT)
            ]

        # Get ID field
        from semlix.fields import ID
        id_field = None
        for name, field in self._schema.items():
            if isinstance(field, ID):
                id_field = name
                break
        if id_field is None:
            id_field = "id"  # Default

        # Create or use store
        if store is None:
            # Try to load existing
            if (self.index_dir / "metadata.json").exists():
                self.store = BM25sStore.load(self.index_dir)
            else:
                # Get analyzer from first searchable field
                analyzer = None
                if searchable_fields:
                    field = self._schema[searchable_fields[0]]
                    if hasattr(field, 'analyzer'):
                        analyzer = field.analyzer

                self.store = BM25sStore.create(
                    self.index_dir,
                    analyzer=analyzer,
                    fields=searchable_fields,
                    id_field=id_field
                )
        else:
            self.store = store

    @property
    def schema(self) -> Schema:
        """Get the index schema."""
        return self._schema

    def close(self) -> None:
        """Close the index."""
        self.store.close()

    def is_empty(self) -> bool:
        """Check if index is empty."""
        return self.store.doc_count() == 0

    def doc_count(self) -> int:
        """Get number of documents."""
        return self.store.doc_count()

    def doc_count_all(self) -> int:
        """Get total number of documents (including deleted)."""
        # BM25sStore doesn't track deletions separately
        return self.store.doc_count()

    def optimize(self) -> None:
        """Optimize the index."""
        self.store.optimize()
        self.store.save()

    def latest_generation(self) -> int:
        """Get latest generation number."""
        # BM25sStore doesn't support generations
        return 0

    def refresh(self) -> "BM25Index":
        """Refresh to latest generation."""
        return self

    def up_to_date(self) -> bool:
        """Check if up to date."""
        return True

    def last_modified(self) -> float:
        """Get last modified time."""
        meta_path = self.index_dir / "metadata.json"
        if meta_path.exists():
            return meta_path.stat().st_mtime
        return -1

    def reader(self, reuse=None) -> "BM25Reader":
        """Get a reader for this index."""
        from .bm25_reader import BM25Reader
        return BM25Reader(self)

    def writer(self, **kwargs) -> "BM25Writer":
        """Get a writer for this index."""
        from .bm25_writer import BM25Writer
        return BM25Writer(self, **kwargs)

    def searcher(self, **kwargs) -> "BM25Searcher":
        """Get a searcher for this index."""
        from .bm25_searcher import BM25Searcher
        return BM25Searcher(self, **kwargs)

    def __repr__(self) -> str:
        return f"BM25Index({str(self.index_dir)!r}, docs={self.doc_count()})"


def create_bm25_index(
    index_dir: Union[str, Path],
    schema: Schema,
    **kwargs
) -> BM25Index:
    """Create a new BM25Index.

    Args:
        index_dir: Directory to create the index in
        schema: Schema defining index fields
        **kwargs: Additional arguments passed to BM25sStore

    Returns:
        New BM25Index instance

    Example:
        >>> from semlix.bm25 import create_bm25_index
        >>> from semlix.fields import Schema, TEXT, ID
        >>>
        >>> schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
        >>> ix = create_bm25_index("my_index", schema)
    """
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    # Save schema
    schema_path = index_path / "schema.pkl"
    import pickle
    with open(schema_path, "wb") as f:
        pickle.dump(schema, f)

    return BM25Index(index_dir, schema)


def open_bm25_index(index_dir: Union[str, Path]) -> BM25Index:
    """Open an existing BM25Index.

    Args:
        index_dir: Directory containing the index

    Returns:
        Opened BM25Index instance

    Raises:
        FileNotFoundError: If index doesn't exist

    Example:
        >>> from semlix.bm25 import open_bm25_index
        >>>
        >>> ix = open_bm25_index("my_index")
    """
    index_path = Path(index_dir)

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_dir}")

    # Load schema
    schema_path = index_path / "schema.pkl"
    if not schema_path.exists():
        raise FileNotFoundError(f"Index schema not found: {schema_path}")

    import pickle
    with open(schema_path, "rb") as f:
        schema = pickle.load(f)

    return BM25Index(index_dir, schema)
