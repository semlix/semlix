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

"""BM25 index reader implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional

if TYPE_CHECKING:
    from .bm25_index import BM25Index


class BM25Reader:
    """Reader for BM25Index.

    Implements the IndexReader protocol for reading documents and statistics
    from a BM25Index.

    Example:
        >>> with ix.reader() as reader:
        ...     doc_count = reader.doc_count()
        ...     fields = reader.stored_fields(0)
    """

    def __init__(self, index: "BM25Index"):
        """Initialize reader.

        Args:
            index: BM25Index instance to read from
        """
        self.index = index
        self._closed = False

    def close(self) -> None:
        """Close the reader."""
        self._closed = True

    def __enter__(self) -> "BM25Reader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def doc_count(self) -> int:
        """Get number of documents."""
        return self.index.store.doc_count()

    def doc_count_all(self) -> int:
        """Get total number of documents (including deleted)."""
        return self.index.store.doc_count()

    def stored_fields(self, docnum: int) -> Dict[str, Any]:
        """Get stored fields for a document by number.

        Args:
            docnum: Document number (0-indexed)

        Returns:
            Dictionary of stored field values

        Raises:
            IndexError: If docnum is out of range
        """
        if docnum < 0 or docnum >= len(self.index.store.doc_ids):
            raise IndexError(f"Document number {docnum} out of range")

        doc_id = self.index.store.doc_ids[docnum]
        return self.index.store.stored_fields_map.get(doc_id, {})

    def document_number(self, **kwargs) -> Optional[int]:
        """Get document number for a document matching the given fields.

        Args:
            **kwargs: Field name-value pairs (typically ID field)

        Returns:
            Document number or None if not found
        """
        id_field = self.index.store.id_field

        if id_field in kwargs:
            doc_id = str(kwargs[id_field])
            try:
                return self.index.store.doc_ids.index(doc_id)
            except ValueError:
                return None

        return None

    def all_doc_ids(self) -> Iterator[int]:
        """Iterate over all document numbers.

        Yields:
            Document numbers (0-indexed)
        """
        for i in range(len(self.index.store.doc_ids)):
            yield i

    def iter_docs(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all documents.

        Yields:
            Document field dictionaries
        """
        for doc_id in self.index.store.doc_ids:
            yield self.index.store.stored_fields_map.get(doc_id, {})

    def field_length(self, fieldname: str) -> int:
        """Get total length of a field across all documents.

        Args:
            fieldname: Field name

        Returns:
            Total field length (sum of tokens across all docs)
        """
        # BM25sStore doesn't track field lengths separately
        # This is a simplified implementation
        total = 0
        for doc in self.iter_docs():
            if fieldname in doc:
                value = doc[fieldname]
                if isinstance(value, str):
                    total += len(value.split())
        return total

    def max_field_length(self, fieldname: str) -> int:
        """Get maximum field length across all documents.

        Args:
            fieldname: Field name

        Returns:
            Maximum field length in any document
        """
        max_len = 0
        for doc in self.iter_docs():
            if fieldname in doc:
                value = doc[fieldname]
                if isinstance(value, str):
                    max_len = max(max_len, len(value.split()))
        return max_len

    def has_deletions(self) -> bool:
        """Check if index has deletions.

        Returns:
            Always False (BM25sStore doesn't track deletions separately)
        """
        return False

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"BM25Reader(docs={self.doc_count()}, status={status})"
