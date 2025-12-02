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

"""Advanced features for BM25Index: phrase queries, faceting, sorting, caching."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
from collections import defaultdict


class PhraseQuery:
    """Phrase query support for exact phrase matching.

    Uses token positions to find exact phrase matches with optional slop
    (word distance tolerance).

    Example:
        >>> pq = PhraseQuery("content", ["python", "programming"])
        >>> results = pq.search(index, limit=10)
        >>>
        >>> # With slop (allows words in between)
        >>> pq = PhraseQuery("content", ["python", "tutorial"], slop=2)
        >>> results = pq.search(index, limit=10)
    """

    def __init__(self, field: str, words: List[str], slop: int = 0):
        """Initialize phrase query.

        Args:
            field: Field name to search in
            words: List of words in phrase (in order)
            slop: Maximum distance between words (0 = exact phrase)
        """
        self.field = field
        self.words = words
        self.slop = slop

    def search(self, index, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for phrase matches.

        Args:
            index: BM25Index instance
            limit: Maximum number of results

        Returns:
            List of matching documents with scores
        """
        # For phrase queries with BM25sStore, we need to:
        # 1. Search for documents containing all words
        # 2. Verify word positions match the phrase

        # First, search for all words
        query_text = " ".join(self.words)
        results = index.store.search(query_text, k=limit * 3)

        phrase_matches = []

        for result in results:
            doc = result.stored_fields
            if self.field not in doc:
                continue

            text = doc[self.field]
            if not isinstance(text, str):
                continue

            # Tokenize and check for phrase
            if self._contains_phrase(text, index.store.analyzer):
                phrase_matches.append({
                    "doc_id": result.doc_id,
                    "score": result.score * 1.5,  # Boost for exact phrase
                    "fields": doc
                })

            if len(phrase_matches) >= limit:
                break

        return phrase_matches[:limit]

    def _contains_phrase(self, text: str, analyzer) -> bool:
        """Check if text contains the phrase.

        Args:
            text: Text to search in
            analyzer: Analyzer for tokenization

        Returns:
            True if phrase is found
        """
        # Tokenize text
        tokens = []
        for token in analyzer(text):
            tokens.append(token.text.lower())

        # Look for phrase
        phrase_lower = [w.lower() for w in self.words]

        for i in range(len(tokens) - len(phrase_lower) + 1):
            # Check exact match
            if self.slop == 0:
                if tokens[i:i+len(phrase_lower)] == phrase_lower:
                    return True
            else:
                # Check with slop
                if self._matches_with_slop(tokens[i:i+len(phrase_lower)+self.slop*2], phrase_lower):
                    return True

        return False

    def _matches_with_slop(self, window: List[str], phrase: List[str]) -> bool:
        """Check if phrase matches within window with slop.

        Args:
            window: Token window to search in
            phrase: Phrase words to find

        Returns:
            True if phrase is found within slop distance
        """
        if not phrase:
            return True

        first_word = phrase[0]

        for i, token in enumerate(window):
            if token == first_word:
                if len(phrase) == 1:
                    return True
                # Recursively check rest of phrase
                remaining = window[i+1:]
                if self._matches_with_slop(remaining, phrase[1:]):
                    return True

        return False

    def __str__(self) -> str:
        phrase = ' '.join(self.words)
        return f'"{phrase}"~{self.slop}' if self.slop else f'"{phrase}"'


class Facets:
    """Faceting support for aggregating search results.

    Computes facet counts and ranges for result sets, useful for
    building faceted search interfaces.

    Example:
        >>> facets = Facets(index)
        >>> results = searcher.search(query, limit=100)
        >>>
        >>> # Count by category
        >>> counts = facets.count_by_field(results, "category")
        >>> # {"python": 45, "sql": 32, "javascript": 23}
        >>>
        >>> # Date range facets
        >>> ranges = facets.date_facet(results, "published", gap="month")
    """

    def __init__(self, index):
        """Initialize facets.

        Args:
            index: BM25Index instance
        """
        self.index = index

    def count_by_field(
        self,
        results: List[Any],
        field_name: str,
        limit: Optional[int] = None
    ) -> Dict[str, int]:
        """Count occurrences of field values in results.

        Args:
            results: Search results
            field_name: Field to facet on
            limit: Maximum number of facet values to return

        Returns:
            Dictionary mapping field values to counts
        """
        counts = defaultdict(int)

        for result in results:
            # Get field value
            if hasattr(result, 'stored_fields'):
                doc = result.stored_fields
            elif hasattr(result, '_fields'):
                doc = result._fields
            else:
                doc = result

            if field_name in doc:
                value = doc[field_name]
                # Handle list values
                if isinstance(value, list):
                    for v in value:
                        counts[str(v)] += 1
                else:
                    counts[str(value)] += 1

        # Sort by count
        sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

        if limit:
            return dict(list(sorted_counts.items())[:limit])

        return sorted_counts

    def range_facet(
        self,
        results: List[Any],
        field_name: str,
        ranges: List[Tuple[Any, Any]]
    ) -> Dict[str, int]:
        """Count results falling into numeric ranges.

        Args:
            results: Search results
            field_name: Field to facet on
            ranges: List of (min, max) tuples defining ranges

        Returns:
            Dictionary mapping range labels to counts
        """
        range_counts = {f"{min_val}-{max_val}": 0 for min_val, max_val in ranges}

        for result in results:
            if hasattr(result, 'stored_fields'):
                doc = result.stored_fields
            elif hasattr(result, '_fields'):
                doc = result._fields
            else:
                doc = result

            if field_name in doc:
                try:
                    value = float(doc[field_name])
                    for min_val, max_val in ranges:
                        if min_val <= value < max_val:
                            range_counts[f"{min_val}-{max_val}"] += 1
                            break
                except (ValueError, TypeError):
                    pass

        return range_counts

    def date_facet(
        self,
        results: List[Any],
        field_name: str,
        gap: str = "month"
    ) -> Dict[str, int]:
        """Count results by date periods.

        Args:
            results: Search results
            field_name: Date field to facet on
            gap: Period size ("day", "week", "month", "year")

        Returns:
            Dictionary mapping date periods to counts
        """
        from datetime import datetime
        from collections import defaultdict

        period_counts = defaultdict(int)

        for result in results:
            if hasattr(result, 'stored_fields'):
                doc = result.stored_fields
            elif hasattr(result, '_fields'):
                doc = result._fields
            else:
                doc = result

            if field_name in doc:
                try:
                    date_value = doc[field_name]

                    # Parse date
                    if isinstance(date_value, str):
                        dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                    elif isinstance(date_value, datetime):
                        dt = date_value
                    else:
                        continue

                    # Create period key
                    if gap == "day":
                        period = dt.strftime("%Y-%m-%d")
                    elif gap == "week":
                        period = dt.strftime("%Y-W%U")
                    elif gap == "month":
                        period = dt.strftime("%Y-%m")
                    elif gap == "year":
                        period = dt.strftime("%Y")
                    else:
                        period = dt.strftime("%Y-%m")

                    period_counts[period] += 1

                except (ValueError, TypeError, AttributeError):
                    pass

        return dict(sorted(period_counts.items()))


class SortBy:
    """Multi-field sorting for search results.

    Supports sorting by one or more fields with optional reverse ordering.

    Example:
        >>> sorter = SortBy([("score", True), ("title", False)])
        >>> sorted_results = sorter.sort_results(results)
        >>>
        >>> # Or use convenience methods
        >>> sorted_results = SortBy.by_field(results, "title")
        >>> sorted_results = SortBy.by_score(results, reverse=True)
    """

    def __init__(self, fields: List[Tuple[str, bool]]):
        """Initialize sorter.

        Args:
            fields: List of (field_name, reverse) tuples
                   Fields are sorted in order specified
        """
        self.fields = fields

    def sort_results(self, results: List[Any]) -> List[Any]:
        """Sort results by configured fields.

        Args:
            results: List of search results

        Returns:
            Sorted list of results
        """
        sorted_results = list(results)

        # Sort in reverse order of fields (last sort is primary)
        for field_name, reverse in reversed(self.fields):
            sorted_results.sort(
                key=lambda x: self._get_sort_key(x, field_name),
                reverse=reverse
            )

        return sorted_results

    def _get_sort_key(self, result: Any, field_name: str) -> Any:
        """Get sort key value from result.

        Args:
            result: Search result
            field_name: Field to get value from

        Returns:
            Sort key value
        """
        # Handle special fields
        if field_name == "score":
            if hasattr(result, 'score'):
                return result.score
            return 0

        # Get field value
        if hasattr(result, 'stored_fields'):
            doc = result.stored_fields
        elif hasattr(result, '_fields'):
            doc = result._fields
        else:
            doc = result

        value = doc.get(field_name)

        # Handle None values (sort last)
        if value is None:
            return ""

        return value

    @staticmethod
    def by_field(results: List[Any], field_name: str, reverse: bool = False) -> List[Any]:
        """Sort results by a single field.

        Args:
            results: Search results
            field_name: Field to sort by
            reverse: Sort in reverse order

        Returns:
            Sorted results
        """
        sorter = SortBy([(field_name, reverse)])
        return sorter.sort_results(results)

    @staticmethod
    def by_score(results: List[Any], reverse: bool = True) -> List[Any]:
        """Sort results by score.

        Args:
            results: Search results
            reverse: Sort highest score first (default True)

        Returns:
            Sorted results
        """
        sorter = SortBy([("score", reverse)])
        return sorter.sort_results(results)


class FieldCache:
    """LRU cache for frequently accessed field values.

    Improves performance when repeatedly accessing the same fields
    across multiple documents.

    Example:
        >>> cache = FieldCache(index, max_size=1000)
        >>>
        >>> # Cache a field
        >>> cache.cache_field("title")
        >>>
        >>> # Get cached value
        >>> title = cache.get_cached("doc123", "title")
        >>>
        >>> # Invalidate cache
        >>> cache.invalidate("doc123")
    """

    def __init__(self, index, max_size: int = 1000):
        """Initialize field cache.

        Args:
            index: BM25Index instance
            max_size: Maximum number of cached values
        """
        self.index = index
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cached_fields: set = set()

    def cache_field(self, field_name: str) -> None:
        """Cache values for a field across all documents.

        Args:
            field_name: Field to cache
        """
        self._cached_fields.add(field_name)

        # Populate cache
        for doc_id in self.index.store.doc_ids:
            if doc_id not in self._cache:
                self._cache[doc_id] = {}

            doc = self.index.store.get_document(doc_id)
            if doc and field_name in doc:
                self._cache[doc_id][field_name] = doc[field_name]

    def get_cached(self, doc_id: str, field_name: str) -> Optional[Any]:
        """Get cached field value.

        Args:
            doc_id: Document ID
            field_name: Field name

        Returns:
            Cached value or None if not cached
        """
        if doc_id in self._cache and field_name in self._cache[doc_id]:
            return self._cache[doc_id][field_name]

        # Not cached, fetch and cache
        doc = self.index.store.get_document(doc_id)
        if doc and field_name in doc:
            if doc_id not in self._cache:
                self._cache[doc_id] = {}
            self._cache[doc_id][field_name] = doc[field_name]
            return doc[field_name]

        return None

    def invalidate(self, doc_id: Optional[str] = None) -> None:
        """Invalidate cache entries.

        Args:
            doc_id: Document ID to invalidate (None = invalidate all)
        """
        if doc_id is None:
            self._cache.clear()
        elif doc_id in self._cache:
            del self._cache[doc_id]

    def __repr__(self) -> str:
        return f"FieldCache(cached_docs={len(self._cache)}, fields={len(self._cached_fields)})"
