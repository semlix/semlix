"""Hybrid indexing for semlix and vector stores.

This module provides utilities for maintaining both a semlix index
and a vector store in sync.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

from .embeddings import EmbeddingProvider
from .stores.base import VectorStore

if TYPE_CHECKING:
    from semlix.index import Index
    from semlix.writing import IndexWriter


class HybridIndexWriter:
    """Index writer that maintains both semlix index and vector store.
    
    This class ensures that documents are indexed in both the lexical
    (semlix) and semantic (vector) indexes simultaneously.
    
    Example:
        >>> from semlix.index import create_in
        >>> from semlix.fields import Schema, TEXT, ID
        >>> from semlix.semantic import HybridIndexWriter, SentenceTransformerProvider
        >>> from semlix.semantic.stores import NumpyVectorStore
        >>> 
        >>> # Create schema and index
        >>> schema = Schema(id=ID(stored=True, unique=True), content=TEXT(stored=True))
        >>> ix = create_in("my_index", schema)
        >>> 
        >>> # Create semantic components
        >>> embedder = SentenceTransformerProvider()
        >>> vector_store = NumpyVectorStore(dimension=embedder.dimension)
        >>> 
        >>> # Use as context manager
        >>> with HybridIndexWriter(ix, vector_store, embedder) as writer:
        ...     writer.add_document(id="doc1", content="Hello world")
        ...     writer.add_document(id="doc2", content="Goodbye world")
        >>> 
        >>> # Or manually commit
        >>> writer = HybridIndexWriter(ix, vector_store, embedder)
        >>> writer.add_document(id="doc3", content="Another document")
        >>> writer.commit()
    
    Attributes:
        index: The semlix index being written to
        vector_store: The vector store being written to
        embedder: Embedding provider for generating embeddings
    """
    
    def __init__(
        self,
        index: "Index",
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        embedding_field: str = "content",
        id_field: str = "id",
        batch_size: int = 100,
        show_progress: bool = False
    ):
        """Initialize hybrid index writer.
        
        Args:
            index: semlix index to write to
            vector_store: Vector store for embeddings
            embedding_provider: Provider for generating embeddings
            embedding_field: Field to generate embeddings from
            id_field: Field containing document ID
            batch_size: Number of documents to batch before embedding
            show_progress: Show progress during batch embedding
        """
        self.index = index
        self.vector_store = vector_store
        self.embedder = embedding_provider
        self.embedding_field = embedding_field
        self.id_field = id_field
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        self._writer: "IndexWriter | None" = None
        self._pending_docs: List[Dict[str, Any]] = []
        self._doc_count = 0
    
    def __enter__(self) -> "HybridIndexWriter":
        """Enter context manager."""
        self._writer = self.index.writer()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager, committing or canceling based on exception."""
        if exc_type is None:
            self.commit()
        else:
            self.cancel()
        return False
    
    def add_document(self, **fields) -> None:
        """Add a document to both indexes.
        
        The document will be added to the semlix index immediately,
        but embedding generation is batched for efficiency.
        
        Args:
            fields: Document fields (keyword arguments). Must include id_field and embedding_field.
        
        Example:
            >>> writer.add_document(
            ...     id="doc123",
            ...     title="My Document",
            ...     content="This is the content to be embedded"
            ... )
        """
        if self._writer is None:
            self._writer = self.index.writer()
        
        # Add to semlix
        self._writer.add_document(**fields)
        
        # Queue for embedding
        doc_id = fields.get(self.id_field)
        text = fields.get(self.embedding_field, "")
        
        if doc_id and text:
            # Store metadata (everything except id and embedding field)
            metadata = {
                k: v for k, v in fields.items() 
                if k not in (self.id_field, self.embedding_field)
            }
            
            self._pending_docs.append({
                "id": str(doc_id),
                "text": str(text),
                "metadata": metadata
            })
            self._doc_count += 1
        
        # Batch embed if needed
        if len(self._pending_docs) >= self.batch_size:
            self._embed_pending()
    
    def update_document(self, **fields) -> None:
        """Update a document in both indexes.
        
        Uses the id_field to identify which document to update.
        The document is replaced in both indexes.
        
        Args:
            fields: Document fields (keyword arguments). Must include id_field.
        """
        if self._writer is None:
            self._writer = self.index.writer()
        
        doc_id = fields.get(self.id_field)
        if not doc_id:
            raise ValueError(f"update_document requires {self.id_field} field")
        
        # Update in semlix (this does delete + add)
        self._writer.update_document(**fields)
        
        # Queue for re-embedding if content changed
        text = fields.get(self.embedding_field, "")
        if text:
            metadata = {
                k: v for k, v in fields.items()
                if k not in (self.id_field, self.embedding_field)
            }
            
            self._pending_docs.append({
                "id": str(doc_id),
                "text": str(text),
                "metadata": metadata
            })
    
    def delete_by_term(self, fieldname: str, text: str) -> None:
        """Delete documents matching a term.
        
        Note: This only deletes from semlix. Vector store deletion
        depends on whether the backend supports it.
        
        Args:
            fieldname: Field to match on
            text: Value to match
        """
        if self._writer is None:
            self._writer = self.index.writer()
        
        self._writer.delete_by_term(fieldname, text)
        
        # Try to delete from vector store if using id field
        if fieldname == self.id_field:
            try:
                self.vector_store.delete([text])
            except NotImplementedError:
                pass  # Some backends don't support deletion
    
    def delete_by_query(self, query) -> None:
        """Delete documents matching a query.
        
        Args:
            query: semlix query object
        """
        if self._writer is None:
            self._writer = self.index.writer()
        
        self._writer.delete_by_query(query)
        # Note: Vector store deletion would require finding matched doc IDs first
    
    def commit(self, merge: bool = True) -> None:
        """Commit all pending changes.
        
        Args:
            merge: Whether to merge segments (passed to semlix)
        """
        # Embed any remaining documents
        if self._pending_docs:
            self._embed_pending()
        
        # Commit semlix changes
        if self._writer is not None:
            self._writer.commit(merge=merge)
            self._writer = None
    
    def cancel(self) -> None:
        """Cancel all pending changes."""
        if self._writer is not None:
            self._writer.cancel()
            self._writer = None
        self._pending_docs = []
    
    def _embed_pending(self) -> None:
        """Generate embeddings for pending documents and add to vector store."""
        if not self._pending_docs:
            return
        
        # Extract texts and generate embeddings
        texts = [doc["text"] for doc in self._pending_docs]
        embeddings = self.embedder.encode(
            texts, 
            show_progress=self.show_progress
        )
        
        # Add to vector store
        doc_ids = [doc["id"] for doc in self._pending_docs]
        metadata = [doc["metadata"] for doc in self._pending_docs]
        self.vector_store.add(doc_ids, embeddings, metadata)
        
        # Clear pending
        self._pending_docs = []
    
    @property
    def doc_count(self) -> int:
        """Return number of documents added in this session."""
        return self._doc_count


def build_vector_store_from_index(
    index: "Index",
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
    embedding_field: str = "content",
    id_field: str = "id",
    batch_size: int = 100,
    show_progress: bool = True
) -> int:
    """Build a vector store from an existing semlix index.
    
    This utility function indexes all documents from a semlix index
    into a vector store. Useful for adding semantic search to an
    existing application.
    
        Args:
        index: Existing semlix index
        vector_store: Vector store to populate
        embedding_provider: Embedding provider
        embedding_field: Field containing text to embed
        id_field: Field containing document ID
        batch_size: Batch size for embedding generation
        show_progress: Show progress during indexing
        
    Returns:
        Number of documents indexed
        
    Example:
        >>> from semlix.index import open_dir
        >>> from semlix.semantic import build_vector_store_from_index
        >>> from semlix.semantic import SentenceTransformerProvider
        >>> from semlix.semantic.stores import NumpyVectorStore
        >>> 
        >>> ix = open_dir("my_existing_index")
        >>> embedder = SentenceTransformerProvider()
        >>> store = NumpyVectorStore(dimension=embedder.dimension)
        >>> 
        >>> count = build_vector_store_from_index(ix, store, embedder)
        >>> print(f"Indexed {count} documents")
        >>> store.save("vectors.pkl")
    """
    doc_count = 0
    pending_docs = []
    
    def process_batch():
        nonlocal pending_docs
        if not pending_docs:
            return
        
        texts = [doc["text"] for doc in pending_docs]
        embeddings = embedding_provider.encode(texts, show_progress=False)
        
        doc_ids = [doc["id"] for doc in pending_docs]
        metadata = [doc["metadata"] for doc in pending_docs]
        vector_store.add(doc_ids, embeddings, metadata)
        
        pending_docs = []
    
    with index.searcher() as searcher:
        total = searcher.doc_count()
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(total), desc="Indexing", unit="docs")
            except ImportError:
                iterator = range(total)
                print(f"Indexing {total} documents...")
        else:
            iterator = range(total)
        
        for docnum in iterator:
            try:
                stored = searcher.stored_fields(docnum)
            except Exception:
                continue
            
            doc_id = stored.get(id_field)
            text = stored.get(embedding_field, "")
            
            if doc_id and text:
                metadata = {
                    k: v for k, v in stored.items()
                    if k not in (id_field, embedding_field)
                }
                
                pending_docs.append({
                    "id": str(doc_id),
                    "text": str(text),
                    "metadata": metadata
                })
                doc_count += 1
            
            if len(pending_docs) >= batch_size:
                process_batch()
    
    # Process remaining
    process_batch()
    
    return doc_count
