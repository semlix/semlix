"""Database drivers for Semlix storage backends.

This module provides alternative storage backends for Semlix indexes,
allowing you to use relational databases instead of file-based storage.

Available drivers:
- PostgreSQLDriver: PostgreSQL backend for lexical index storage

Note: These drivers are experimental and intended to demonstrate
the viability of database-backed storage. They may not support
all features of FileStorage initially.
"""

__all__ = []

# Lazy imports for optional drivers
def __getattr__(name: str):
    if name == "PostgreSQLDriver":
        from .postgresql import PostgreSQLDriver
        return PostgreSQLDriver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
