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

"""Storage backends for Semlix indexes.

This module provides alternative storage implementations beyond the default
FileStorage, including BM25-optimized storage using the bm25s library.
"""

__all__ = ["BM25sStore"]


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "BM25sStore":
        from .bm25s_store import BM25sStore
        return BM25sStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
