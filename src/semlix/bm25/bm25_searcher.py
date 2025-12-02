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

"""BM25 index searcher implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .bm25_index import BM25Index


class BM25Hit:
    """A search result hit from BM25Searcher.

    Compatible with Whoosh Hit objects for use with HybridSearcher.

    Attributes:
        docnum: Document number
        score: Relevance score
        fields: Stored field values
    """

    def __init__(self, docnum: int, score: float, fields: Dict[str, Any]):
        """Initialize hit.

        Args:
            docnum: Document number
            score: Relevance score
            fields: Stored field values
        """
        self.docnum = docnum
        self.score = score
        self._fields = fields

    def __getitem__(self, key: str) -> Any:
        """Get a field value."""
        return self._fields[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get a field value with default."""
        return self._fields.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if field exists."""
        return key in self._fields

    def keys(self):
        """Get field names."""
        return self._fields.keys()

    def items(self):
        """Get field items."""
        return self._fields.items()

    def __repr__(self) -> str:
        return f"BM25Hit(docnum={self.docnum}, score={self.score:.4f})"


class BM25Results:
    """Search results from BM25Searcher.

    Compatible with Whoosh Results for use with HybridSearcher.
    """

    def __init__(self, hits: list[BM25Hit], runtime: float = 0.0):
        """Initialize results.

        Args:
            hits: List of BM25Hit objects
            runtime: Search runtime in seconds
        """
        self._hits = hits
        self.runtime = runtime

    def __len__(self) -> int:
        """Get number of results."""
        return len(self._hits)

    def __iter__(self):
        """Iterate over hits."""
        return iter(self._hits)

    def __getitem__(self, index):
        """Get hit by index."""
        return self._hits[index]

    def scored_length(self) -> int:
        """Get number of scored results."""
        return len(self._hits)

    def __repr__(self) -> str:
        return f"BM25Results({len(self)} hits, {self.runtime:.3f}s)"


class BM25Searcher:
    """Searcher for BM25Index.

    Implements a subset of the Whoosh Searcher protocol, compatible with
    HybridSearcher and other Semlix components.

    Example:
        >>> with ix.searcher() as searcher:
        ...     from semlix.qparser import QueryParser
        ...     qp = QueryParser("content", ix.schema)
        ...     q = qp.parse("python tutorial")
        ...     results = searcher.search(q, limit=10)
        ...     for hit in results:
        ...         print(hit["id"], hit.score)
    """

    def __init__(self, index: "BM25Index", **kwargs):
        """Initialize searcher.

        Args:
            index: BM25Index instance to search
            **kwargs: Additional searcher options (unused)
        """
        self.index = index
        self.ixreader = index.reader()
        self._closed = False

    def close(self) -> None:
        """Close the searcher."""
        if not self._closed:
            self.ixreader.close()
            self._closed = True

    def __enter__(self) -> "BM25Searcher":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def reader(self):
        """Get the index reader."""
        return self.ixreader

    def stored_fields(self, docnum: int) -> Dict[str, Any]:
        """Get stored fields for a document.

        Args:
            docnum: Document number

        Returns:
            Dictionary of stored field values
        """
        return self.ixreader.stored_fields(docnum)

    def document_number(self, **kwargs) -> Optional[int]:
        """Get document number for a document matching fields.

        Args:
            **kwargs: Field name-value pairs

        Returns:
            Document number or None
        """
        return self.ixreader.document_number(**kwargs)

    def search(
        self,
        query,
        limit: Optional[int] = 10,
        filter: Optional[Any] = None,
        **kwargs
    ) -> BM25Results:
        """Search the index.

        Args:
            query: Query object or string
            limit: Maximum number of results (None = unlimited)
            filter: Optional filter query (not implemented)
            **kwargs: Additional search options

        Returns:
            BM25Results object
        """
        import time
        start = time.time()

        # Extract query text
        query_text = self._extract_query_text(query)

        if not query_text:
            return BM25Results([], time.time() - start)

        # Search using BM25sStore
        k = limit if limit is not None else self.index.doc_count()
        if k == 0:
            k = 10  # Default

        search_results = self.index.store.search(query_text, k=k)

        # Convert to BM25Hit objects
        hits = []
        for result in search_results:
            # Find document number
            try:
                docnum = self.index.store.doc_ids.index(result.doc_id)
            except ValueError:
                continue

            hits.append(BM25Hit(
                docnum=docnum,
                score=result.score,
                fields=result.stored_fields
            ))

        runtime = time.time() - start
        return BM25Results(hits, runtime)

    def search_page(
        self,
        query,
        pagenum: int = 1,
        pagelen: int = 10,
        **kwargs
    ) -> BM25Results:
        """Search and return a specific page of results.

        Args:
            query: Query object or string
            pagenum: Page number (1-indexed)
            pagelen: Results per page
            **kwargs: Additional search options

        Returns:
            BM25Results for the requested page
        """
        # Calculate offset
        offset = (pagenum - 1) * pagelen
        limit = offset + pagelen

        # Get all results up to limit
        all_results = self.search(query, limit=limit, **kwargs)

        # Slice to get page
        page_hits = list(all_results)[offset:limit]

        return BM25Results(page_hits, all_results.runtime)

    def documents(self, **kwargs):
        """Iterate over documents matching the given fields.

        Args:
            **kwargs: Field name-value pairs

        Yields:
            Matching documents
        """
        id_field = self.index.store.id_field

        if id_field in kwargs:
            doc_id = str(kwargs[id_field])
            doc = self.index.store.get_document(doc_id)
            if doc:
                yield doc
        else:
            # Return all documents
            for doc in self.ixreader.iter_docs():
                yield doc

    def _extract_query_text(self, query) -> str:
        """Extract text from a query object.

        Args:
            query: Query object or string

        Returns:
            Query text string
        """
        if isinstance(query, str):
            return query

        # Handle Whoosh query objects
        if hasattr(query, '__unicode__'):
            return str(query)

        if hasattr(query, 'text'):
            return query.text

        # Try to convert to string
        return str(query)

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"BM25Searcher(docs={self.index.doc_count()}, status={status})"
