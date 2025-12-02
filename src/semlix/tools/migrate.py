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

"""Migration tools for converting indexes between storage backends."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import time


class IndexMigrator:
    """Migrate indexes between different storage backends.

    Supports migration from:
    - FileStorage (Whoosh) → BM25Index
    - FileStorage + NumpyVectorStore → UnifiedIndex
    - Any Index → BM25Index
    - Any Index + VectorStore → UnifiedIndex

    Example:
        >>> from semlix.tools import IndexMigrator
        >>> from semlix.semantic import SentenceTransformerProvider
        >>>
        >>> migrator = IndexMigrator()
        >>>
        >>> # Migrate to BM25
        >>> migrator.migrate_to_bm25(
        ...     source_dir="old_index",
        ...     target_dir="new_index"
        ... )
        >>>
        >>> # Migrate to Unified (with vectors)
        >>> embedder = SentenceTransformerProvider()
        >>> migrator.migrate_to_unified(
        ...     source_dir="old_index",
        ...     target_dir="new_index",
        ...     connection_string="postgresql://localhost/mydb",
        ...     embedder=embedder
        ... )
    """

    def __init__(self, verbose: bool = True):
        """Initialize migrator.

        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """Log a message if verbose."""
        if self.verbose:
            print(message)

    def migrate_to_bm25(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        batch_size: int = 1000
    ) -> None:
        """Migrate an existing index to BM25Index.

        Args:
            source_dir: Directory containing source index
            target_dir: Directory for new BM25 index
            batch_size: Number of documents per batch

        Example:
            >>> migrator = IndexMigrator()
            >>> migrator.migrate_to_bm25("old_index", "new_bm25_index")
        """
        from semlix.index import open_dir
        from semlix.bm25 import create_bm25_index

        self._log(f"Starting migration from {source_dir} to {target_dir}")
        start_time = time.time()

        # Open source index
        self._log("Opening source index...")
        source_ix = open_dir(source_dir)

        # Create target BM25 index
        self._log("Creating BM25 index...")
        target_ix = create_bm25_index(target_dir, source_ix.schema)

        # Migrate documents
        self._log("Migrating documents...")
        doc_count = 0

        with source_ix.searcher() as searcher:
            with target_ix.writer() as writer:
                for docnum in range(searcher.reader().doc_count_all()):
                    try:
                        fields = searcher.stored_fields(docnum)
                        writer.add_document(**fields)
                        doc_count += 1

                        if doc_count % batch_size == 0:
                            self._log(f"  Migrated {doc_count} documents...")

                    except Exception as e:
                        self._log(f"  Warning: Failed to migrate doc {docnum}: {e}")

        elapsed = time.time() - start_time
        self._log(f"Migration complete! Migrated {doc_count} documents in {elapsed:.2f}s")
        self._log(f"Average: {doc_count/elapsed:.0f} docs/sec")

    def migrate_to_unified(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        connection_string: str,
        embedder,
        vector_store_path: Optional[Union[str, Path]] = None,
        batch_size: int = 100
    ) -> None:
        """Migrate an existing index to UnifiedIndex with vectors.

        Args:
            source_dir: Directory containing source index
            target_dir: Directory for new unified index
            connection_string: PostgreSQL connection string
            embedder: Embedding provider for generating vectors
            vector_store_path: Optional path to existing vector store to migrate
            batch_size: Number of documents per batch

        Example:
            >>> from semlix.semantic import SentenceTransformerProvider
            >>>
            >>> embedder = SentenceTransformerProvider()
            >>> migrator = IndexMigrator()
            >>>
            >>> migrator.migrate_to_unified(
            ...     "old_index",
            ...     "new_unified_index",
            ...     "postgresql://localhost/mydb",
            ...     embedder,
            ...     vector_store_path="vectors.pkl"
            ... )
        """
        from semlix.index import open_dir
        from semlix.unified import create_unified_index

        self._log(f"Starting unified migration from {source_dir} to {target_dir}")
        start_time = time.time()

        # Open source index
        self._log("Opening source index...")
        source_ix = open_dir(source_dir)

        # Create target unified index
        self._log("Creating unified index...")
        target_ix = create_unified_index(
            target_dir,
            source_ix.schema,
            connection_string,
            embedder
        )

        # Load existing vector store if provided
        existing_vectors = {}
        if vector_store_path:
            self._log(f"Loading existing vectors from {vector_store_path}...")
            try:
                from semlix.semantic.stores import NumpyVectorStore
                old_store = NumpyVectorStore.load(vector_store_path)
                # Build map of doc_id -> embedding
                for doc_id, embedding in zip(old_store._doc_ids, old_store._embeddings):
                    existing_vectors[doc_id] = embedding
                self._log(f"  Loaded {len(existing_vectors)} existing vectors")
            except Exception as e:
                self._log(f"  Warning: Could not load vector store: {e}")

        # Migrate documents
        self._log("Migrating documents with embeddings...")
        doc_count = 0
        generated_count = 0

        with source_ix.searcher() as searcher:
            with target_ix.writer() as writer:
                batch = []
                batch_ids = []

                for docnum in range(searcher.reader().doc_count_all()):
                    try:
                        fields = searcher.stored_fields(docnum)

                        # Check if we have existing vector
                        id_field = target_ix.id_field
                        doc_id = str(fields.get(id_field, f"doc{docnum}"))

                        if doc_id in existing_vectors:
                            # Use existing vector
                            batch.append(fields)
                            batch_ids.append(doc_id)
                        else:
                            # Will generate new vector
                            batch.append(fields)
                            batch_ids.append(doc_id)
                            generated_count += 1

                        # Process batch
                        if len(batch) >= batch_size:
                            for doc_fields in batch:
                                writer.add_document(**doc_fields)

                            doc_count += len(batch)
                            self._log(f"  Migrated {doc_count} documents ({generated_count} new vectors)...")

                            batch = []
                            batch_ids = []

                    except Exception as e:
                        self._log(f"  Warning: Failed to migrate doc {docnum}: {e}")

                # Process remaining batch
                if batch:
                    for doc_fields in batch:
                        writer.add_document(**doc_fields)
                    doc_count += len(batch)

        elapsed = time.time() - start_time
        self._log(f"Migration complete! Migrated {doc_count} documents in {elapsed:.2f}s")
        self._log(f"Generated {generated_count} new embeddings")
        self._log(f"Average: {doc_count/elapsed:.0f} docs/sec")

    def migrate_vectors_only(
        self,
        source_store_path: Union[str, Path],
        target_connection_string: str,
        table_name: str = "semlix_vectors"
    ) -> None:
        """Migrate vectors from NumpyVectorStore to PgVectorStore.

        Args:
            source_store_path: Path to NumpyVectorStore file
            target_connection_string: PostgreSQL connection string
            table_name: Target table name

        Example:
            >>> migrator = IndexMigrator()
            >>> migrator.migrate_vectors_only(
            ...     "vectors.pkl",
            ...     "postgresql://localhost/mydb",
            ...     "my_vectors"
            ... )
        """
        from semlix.semantic.stores import NumpyVectorStore, PgVectorStore

        self._log(f"Migrating vectors from {source_store_path} to PostgreSQL")
        start_time = time.time()

        # Load source
        self._log("Loading source vector store...")
        source_store = NumpyVectorStore.load(source_store_path)
        doc_count = len(source_store._doc_ids)
        self._log(f"  Found {doc_count} vectors (dimension={source_store.dimension})")

        # Create target
        self._log("Creating PostgreSQL vector store...")
        target_store = PgVectorStore(
            connection_string=target_connection_string,
            dimension=source_store.dimension,
            table_name=table_name
        )

        # Migrate
        self._log("Migrating vectors...")
        target_store.add(
            doc_ids=source_store._doc_ids,
            embeddings=source_store._embeddings,
            metadata=source_store._metadata if hasattr(source_store, '_metadata') else None
        )

        # Create index
        self._log("Creating HNSW index...")
        target_store.create_index(index_type="hnsw")

        elapsed = time.time() - start_time
        self._log(f"Migration complete! Migrated {doc_count} vectors in {elapsed:.2f}s")


def migrate_to_bm25(
    source_dir: Union[str, Path],
    target_dir: Union[str, Path],
    batch_size: int = 1000,
    verbose: bool = True
) -> None:
    """Convenience function to migrate an index to BM25.

    Args:
        source_dir: Directory containing source index
        target_dir: Directory for new BM25 index
        batch_size: Number of documents per batch
        verbose: Print progress messages

    Example:
        >>> from semlix.tools import migrate_to_bm25
        >>> migrate_to_bm25("old_index", "new_bm25_index")
    """
    migrator = IndexMigrator(verbose=verbose)
    migrator.migrate_to_bm25(source_dir, target_dir, batch_size)


def migrate_to_unified(
    source_dir: Union[str, Path],
    target_dir: Union[str, Path],
    connection_string: str,
    embedder,
    vector_store_path: Optional[Union[str, Path]] = None,
    batch_size: int = 100,
    verbose: bool = True
) -> None:
    """Convenience function to migrate an index to UnifiedIndex.

    Args:
        source_dir: Directory containing source index
        target_dir: Directory for new unified index
        connection_string: PostgreSQL connection string
        embedder: Embedding provider
        vector_store_path: Optional existing vector store path
        batch_size: Number of documents per batch
        verbose: Print progress messages

    Example:
        >>> from semlix.tools import migrate_to_unified
        >>> from semlix.semantic import SentenceTransformerProvider
        >>>
        >>> embedder = SentenceTransformerProvider()
        >>> migrate_to_unified(
        ...     "old_index",
        ...     "new_index",
        ...     "postgresql://localhost/mydb",
        ...     embedder
        ... )
    """
    migrator = IndexMigrator(verbose=verbose)
    migrator.migrate_to_unified(
        source_dir,
        target_dir,
        connection_string,
        embedder,
        vector_store_path,
        batch_size
    )
