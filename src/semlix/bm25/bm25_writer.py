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

"""BM25 index writer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .bm25_index import BM25Index


class BM25Writer:
    """Writer for BM25Index.

    Implements the IndexWriter protocol for adding, updating, and deleting
    documents in a BM25Index.

    Example:
        >>> with ix.writer() as writer:
        ...     writer.add_document(id="1", title="Python", content="Programming...")
        ...     writer.update_document(id="2", title="Updated")
        ...     writer.delete_by_term("id", "3")
    """

    def __init__(self, index: "BM25Index", **kwargs):
        """Initialize writer.

        Args:
            index: BM25Index instance to write to
            **kwargs: Additional writer options (unused for BM25)
        """
        self.index = index
        self._buffer: List[Dict[str, Any]] = []
        self._deleted_ids: set = set()
        self._committed = False

    def add_document(self, **fields) -> None:
        """Add a document to the index.

        Args:
            **fields: Field name-value pairs
        """
        if self._committed:
            raise RuntimeError("Writer already committed")

        # Validate fields against schema
        for fieldname in fields:
            if fieldname not in self.index.schema:
                raise ValueError(f"No field named {fieldname!r} in schema")

        # Add to buffer
        self._buffer.append(fields)

    def update_document(self, **fields) -> None:
        """Update a document (delete old, add new).

        Args:
            **fields: Field name-value pairs (must include ID field)
        """
        if self._committed:
            raise RuntimeError("Writer already committed")

        # Find ID field
        id_field = self.index.store.id_field
        if id_field not in fields:
            raise ValueError(f"update_document requires {id_field!r} field")

        # Mark old document for deletion
        doc_id = str(fields[id_field])
        self._deleted_ids.add(doc_id)

        # Add new version
        self._buffer.append(fields)

    def delete_document(self, **fields) -> None:
        """Delete a document matching the given fields.

        Args:
            **fields: Field name-value pairs (must include ID field)
        """
        if self._committed:
            raise RuntimeError("Writer already committed")

        id_field = self.index.store.id_field
        if id_field not in fields:
            raise ValueError(f"delete_document requires {id_field!r} field")

        doc_id = str(fields[id_field])
        self._deleted_ids.add(doc_id)

    def delete_by_term(self, fieldname: str, text: str, searcher=None) -> int:
        """Delete documents where fieldname matches text.

        Args:
            fieldname: Field to match
            text: Value to match
            searcher: Optional searcher for finding documents

        Returns:
            Number of documents deleted
        """
        if self._committed:
            raise RuntimeError("Writer already committed")

        # For BM25, we only support deleting by ID field
        if fieldname == self.index.store.id_field:
            self._deleted_ids.add(str(text))
            return 1

        # For other fields, would need to search first
        # This is a simplified implementation
        return 0

    def delete_by_query(self, q, searcher=None) -> int:
        """Delete documents matching a query.

        Args:
            q: Query object
            searcher: Optional searcher for finding documents

        Returns:
            Number of documents deleted
        """
        if self._committed:
            raise RuntimeError("Writer already committed")

        if searcher is None:
            searcher = self.index.searcher()
            close_searcher = True
        else:
            close_searcher = False

        try:
            results = searcher.search(q, limit=None)
            id_field = self.index.store.id_field

            for hit in results:
                if id_field in hit:
                    doc_id = str(hit[id_field])
                    self._deleted_ids.add(doc_id)

            return len(self._deleted_ids)

        finally:
            if close_searcher:
                searcher.close()

    def add_field(self, fieldname: str, fieldspec) -> None:
        """Add a field to the schema.

        Args:
            fieldname: Name of the field
            fieldspec: Field specification

        Note:
            BM25Index doesn't support dynamic schema changes.
            This is a no-op for compatibility.
        """
        # Would need to modify schema and rebuild index
        # For now, this is a no-op
        pass

    def remove_field(self, fieldname: str) -> None:
        """Remove a field from the schema.

        Args:
            fieldname: Name of the field to remove

        Note:
            BM25Index doesn't support dynamic schema changes.
            This is a no-op for compatibility.
        """
        pass

    def commit(self, optimize: bool = False) -> None:
        """Commit changes to the index.

        Args:
            optimize: Whether to optimize the index after commit
        """
        if self._committed:
            raise RuntimeError("Writer already committed")

        # Process deletions
        for doc_id in self._deleted_ids:
            self.index.store.delete_document(doc_id)

        # Process additions
        if self._buffer:
            self.index.store.add_documents(self._buffer)

        # Save to disk
        self.index.store.save()

        # Optimize if requested
        if optimize:
            self.index.optimize()

        self._committed = True

    def cancel(self) -> None:
        """Cancel changes (rollback)."""
        self._buffer.clear()
        self._deleted_ids.clear()
        self._committed = True

    def __enter__(self) -> "BM25Writer":
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
        return f"BM25Writer(buffered={len(self._buffer)}, status={status})"
