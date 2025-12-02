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

"""BM25-based index implementation for Semlix.

This module provides a complete implementation of the Semlix Index protocol
using the bm25s library for high-performance BM25 search.
"""

from .bm25_index import BM25Index, create_bm25_index, open_bm25_index
from .features import PhraseQuery, Facets, SortBy, FieldCache

__all__ = [
    "BM25Index",
    "create_bm25_index",
    "open_bm25_index",
    "PhraseQuery",
    "Facets",
    "SortBy",
    "FieldCache",
]
