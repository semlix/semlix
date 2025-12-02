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

"""Unified index combining BM25 and vector search.

This module provides UnifiedIndex, which combines BM25-based lexical search
with pgvector-based semantic search in a single, transactional interface.
"""

from .unified_index import UnifiedIndex, create_unified_index, open_unified_index

__all__ = [
    "UnifiedIndex",
    "create_unified_index",
    "open_unified_index",
]
