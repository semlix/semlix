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

"""Unified writer that writes to both BM25 and vector stores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .unified_index import UnifiedIndex


class UnifiedWriter:
    """Writer for UnifiedIndex.

    Writes to both BM25 index and vector store transactionally.

    Example:
        >>> with ix.writer() as w:
        ...     w.add_document(id="1", title="Python", content="Programming...")
        ...     w.update_document(id="2", title="Updated")
    """

    def __init__(self, index: "UnifiedIndex", **kwargs):
        """Initialize writer.

        Args:
            index: UnifiedIndex instance to write to
            **kwargs: Additional writer options
        """
        self.index = index
        self.bm25_writer = index.bm25_index.writer(**kwargs)
        self._documents_to_embed = []
        self._committed = False

    def add_document(self, **fields) -> None:
        """Add a document to both indexes.

        Args:
            **fields: Field name-value pairs
        """
        if self._committed:
            raise RuntimeError("Writer already committed")

        # Add to BM25 index
        self.bm25_writer.add_document(**fields)

        # Queue for embedding
        self._documents_to_embed.append(("add", fields))

    def update_document(self, **fields) -> None:
        """Update a document in both indexes.

        Args:
            **fields: Field name-value pairs (must include ID field)
        """
        if self._committed:
            raise RuntimeError("Writer already committed")

        # Update in BM25 index
        self.bm25_writer.update_document(**fields)

        # Queue for embedding (will delete old and add new)
        self._documents_to_embed.append(("update", fields))

    def delete_document(self, **fields) -> None:
        """Delete a document from both indexes.

        Args:
            **fields: Field name-value pairs (must include ID field)
        """
        if self._committed:
            raise RuntimeError("Writer already committed")

        # Delete from BM25 index
        self.bm25_writer.delete_document(**fields)

        # Delete from vector store
        id_field = self.index.id_field
        if id_field in fields:
            doc_id = str(fields[id_field])
            self._documents_to_embed.append(("delete", {"doc_id": doc_id}))

    def delete_by_term(self, fieldname: str, text: str, searcher=None) -> int:
        """Delete documents where fieldname matches text.

        Args:
            fieldname: Field to match
            text: Value to match
            searcher: Optional searcher

        Returns:
            Number of documents deleted
        """
        if self._committed:
            raise RuntimeError("Writer already committed")

        count = self.bm25_writer.delete_by_term(fieldname, text, searcher)

        # If deleting by ID field, also delete from vector store
        if fieldname == self.index.id_field:
            self._documents_to_embed.append(("delete", {"doc_id": str(text)}))

        return count

    def delete_by_query(self, q, searcher=None) -> int:
        """Delete documents matching a query.

        Args:
            q: Query object
            searcher: Optional searcher

        Returns:
            Number of documents deleted
        """
        if self._committed:
            raise RuntimeError("Writer already committed")

        # Get matching document IDs first
        if searcher is None:
            searcher = self.index.searcher()
            close_searcher = True
        else:
            close_searcher = False

        try:
            results = searcher.search(q, limit=None)
            id_field = self.index.id_field

            doc_ids_to_delete = []
            for hit in results:
                if id_field in hit:
                    doc_id = str(hit[id_field])
                    doc_ids_to_delete.append(doc_id)
                    self._documents_to_embed.append(("delete", {"doc_id": doc_id}))

            # Delete from BM25
            count = self.bm25_writer.delete_by_query(q, searcher)
            return count

        finally:
            if close_searcher:
                searcher.close()

    def add_field(self, fieldname: str, fieldspec) -> None:
        """Add a field to the schema.

        Args:
            fieldname: Name of the field
            fieldspec: Field specification
        """
        self.bm25_writer.add_field(fieldname, fieldspec)

    def remove_field(self, fieldname: str) -> None:
        """Remove a field from the schema.

        Args:
            fieldname: Name of the field to remove
        """
        self.bm25_writer.remove_field(fieldname)

    def commit(self, optimize: bool = False) -> None:
        """Commit changes to both indexes.

        Args:
            optimize: Whether to optimize indexes after commit
        """
        if self._committed:
            raise RuntimeError("Writer already committed")

        try:
            # Commit BM25 index first
            self.bm25_writer.commit(optimize=optimize)

            # Process vector store operations
            self._commit_vector_operations()

            # Optimize if requested
            if optimize:
                self.index.optimize()

            self._committed = True

        except Exception as e:
            # Rollback on error
            self.cancel()
            raise RuntimeError(f"Commit failed: {e}") from e

    def _commit_vector_operations(self) -> None:
        """Commit queued vector store operations."""
        if not self._documents_to_embed:
            return

        # Process operations
        docs_to_add = []
        doc_ids_to_add = []
        doc_ids_to_delete = set()

        for op_type, data in self._documents_to_embed:
            if op_type == "delete":
                doc_ids_to_delete.add(data["doc_id"])

            elif op_type == "add":
                id_field = self.index.id_field
                if id_field in data:
                    doc_id = str(data[id_field])
                    text = self._extract_text(data)
                    docs_to_add.append(text)
                    doc_ids_to_add.append(doc_id)

            elif op_type == "update":
                id_field = self.index.id_field
                if id_field in data:
                    doc_id = str(data[id_field])
                    # Delete old version
                    doc_ids_to_delete.add(doc_id)
                    # Add new version
                    text = self._extract_text(data)
                    docs_to_add.append(text)
                    doc_ids_to_add.append(doc_id)

        # Process deletions
        for doc_id in doc_ids_to_delete:
            try:
                self.index.vector_store.delete([doc_id])
            except Exception:
                pass  # Document may not exist in vector store

        # Process additions
        if docs_to_add:
            # Generate embeddings
            embeddings = self.index.embedder.encode(docs_to_add)

            # Add to vector store
            self.index.vector_store.add(
                doc_ids=doc_ids_to_add,
                embeddings=embeddings
            )

    def _extract_text(self, fields: dict) -> str:
        """Extract text from document fields for embedding.

        Args:
            fields: Document fields

        Returns:
            Combined text for embedding
        """
        texts = []
        for field_name in self.index.searchable_fields:
            if field_name in fields:
                value = fields[field_name]
                if isinstance(value, str):
                    texts.append(value)
                else:
                    texts.append(str(value))

        return " ".join(texts)

    def cancel(self) -> None:
        """Cancel changes (rollback)."""
        if not self._committed:
            self.bm25_writer.cancel()
            self._documents_to_embed.clear()
            self._committed = True

    def __enter__(self) -> "UnifiedWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type is None and not self._committed:
            self.commit()
        elif not self._committed:
            self.cancel()

    def __repr__(self) -> str:
        status = "committed" if self._committed else "active"
        return f"UnifiedWriter(queued={len(self._documents_to_embed)}, status={status})"
