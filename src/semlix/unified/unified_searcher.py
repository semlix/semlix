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

"""Unified searcher for hybrid lexical and semantic search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from semlix.semantic.hybrid import HybridSearcher, HybridSearchResult
from semlix.semantic.fusion import FusionMethod
from semlix.bm25.features import Facets, SortBy, PhraseQuery

if TYPE_CHECKING:
    from .unified_index import UnifiedIndex


class UnifiedSearcher(HybridSearcher):
    """Searcher for UnifiedIndex with enhanced hybrid search capabilities.

    Extends HybridSearcher with additional features like faceting, sorting,
    and phrase queries.

    Example:
        >>> with ix.searcher() as s:
        ...     # Hybrid search
        ...     results = s.hybrid_search("python tutorial", limit=10)
        ...
        ...     # With facets
        ...     results, facets = s.search_with_facets(
        ...         "python",
        ...         facet_fields=["category", "author"]
        ...     )
        ...
        ...     # Phrase search
        ...     results = s.phrase_search("content", "machine learning")
    """

    def __init__(self, index: "UnifiedIndex", **kwargs):
        """Initialize searcher.

        Args:
            index: UnifiedIndex instance to search
            **kwargs: Additional searcher options
        """
        self.unified_index = index

        # Initialize HybridSearcher
        super().__init__(
            index=index.bm25_index,
            vector_store=index.vector_store,
            embedding_provider=index.embedder,
            id_field=index.id_field,
            **kwargs
        )

        # Initialize advanced features
        self.facets = Facets(index.bm25_index)

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        alpha: float = 0.5,
        fusion_method: Optional[FusionMethod] = None,
        **kwargs
    ) -> List[HybridSearchResult]:
        """Execute hybrid search combining BM25 and vector search.

        Args:
            query: Search query string
            limit: Maximum number of results
            alpha: Weight for semantic search (0-1)
            fusion_method: Fusion algorithm to use
            **kwargs: Additional search options

        Returns:
            List of HybridSearchResult sorted by combined score
        """
        return self.search(
            query=query,
            limit=limit,
            alpha=alpha,
            fusion_method=fusion_method,
            **kwargs
        )

    def search_with_facets(
        self,
        query: str,
        facet_fields: List[str],
        limit: int = 100,
        facet_limit: Optional[int] = 10,
        **kwargs
    ) -> tuple[List[HybridSearchResult], dict[str, dict]]:
        """Search with facet aggregations.

        Args:
            query: Search query string
            facet_fields: List of fields to facet on
            limit: Maximum number of search results
            facet_limit: Maximum facet values per field
            **kwargs: Additional search options

        Returns:
            Tuple of (search results, facet counts)

        Example:
            >>> results, facets = searcher.search_with_facets(
            ...     "python",
            ...     facet_fields=["category", "author"],
            ...     limit=100
            ... )
            >>> print(facets["category"])
            >>> # {"tutorial": 45, "reference": 32, "guide": 23}
        """
        # Execute search
        results = self.search(query, limit=limit, **kwargs)

        # Compute facets
        facet_counts = {}
        for field_name in facet_fields:
            facet_counts[field_name] = self.facets.count_by_field(
                results,
                field_name,
                limit=facet_limit
            )

        return results, facet_counts

    def phrase_search(
        self,
        field: str,
        phrase: str,
        slop: int = 0,
        limit: int = 10
    ) -> List[HybridSearchResult]:
        """Search for exact phrase matches.

        Args:
            field: Field to search in
            phrase: Phrase to search for
            slop: Maximum word distance (0 = exact phrase)
            limit: Maximum number of results

        Returns:
            List of matching results

        Example:
            >>> results = searcher.phrase_search(
            ...     "content",
            ...     "machine learning",
            ...     slop=0
            ... )
        """
        # Create phrase query
        words = phrase.split()
        pq = PhraseQuery(field, words, slop=slop)

        # Execute phrase search
        matches = pq.search(self.unified_index.bm25_index, limit=limit)

        # Convert to HybridSearchResult
        results = []
        for match in matches:
            results.append(HybridSearchResult(
                doc_id=match["doc_id"],
                score=match["score"],
                lexical_score=match["score"],
                semantic_score=None,
                stored_fields=match["fields"]
            ))

        return results

    def sort_results(
        self,
        results: List[HybridSearchResult],
        sort_by: List[tuple[str, bool]]
    ) -> List[HybridSearchResult]:
        """Sort search results by specified fields.

        Args:
            results: Search results to sort
            sort_by: List of (field_name, reverse) tuples

        Returns:
            Sorted results

        Example:
            >>> results = searcher.search("python", limit=100)
            >>> sorted_results = searcher.sort_results(
            ...     results,
            ...     [("date", True), ("score", True)]
            ... )
        """
        sorter = SortBy(sort_by)
        return sorter.sort_results(results)

    def search_sorted(
        self,
        query: str,
        sort_by: List[tuple[str, bool]],
        limit: int = 10,
        **kwargs
    ) -> List[HybridSearchResult]:
        """Execute search with custom sorting.

        Args:
            query: Search query string
            sort_by: List of (field_name, reverse) tuples
            limit: Maximum number of results (applied after sorting)
            **kwargs: Additional search options

        Returns:
            Sorted search results

        Example:
            >>> results = searcher.search_sorted(
            ...     "python tutorial",
            ...     sort_by=[("date", True), ("score", True)],
            ...     limit=10
            ... )
        """
        # Search with higher limit for sorting
        all_results = self.search(query, limit=limit * 3, **kwargs)

        # Sort
        sorted_results = self.sort_results(all_results, sort_by)

        # Trim to limit
        return sorted_results[:limit]

    def lexical_only(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> List[HybridSearchResult]:
        """Execute lexical-only (BM25) search.

        Args:
            query: Search query string
            limit: Maximum number of results
            **kwargs: Additional search options

        Returns:
            BM25 search results
        """
        return self.search_lexical_only(query, limit=limit, **kwargs)

    def semantic_only(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> List[HybridSearchResult]:
        """Execute semantic-only (vector) search.

        Args:
            query: Search query string
            limit: Maximum number of results
            **kwargs: Additional search options

        Returns:
            Vector search results
        """
        return self.search_semantic_only(query, limit=limit, **kwargs)

    def close(self) -> None:
        """Close the searcher."""
        if hasattr(self, 'ixreader') and self.ixreader:
            self.ixreader.close()

    def __enter__(self) -> "UnifiedSearcher":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        return (
            f"UnifiedSearcher(alpha={self.alpha}, "
            f"fusion={self.fusion_method.value}, "
            f"docs={self.unified_index.doc_count()})"
        )
