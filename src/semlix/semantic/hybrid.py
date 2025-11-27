"""Hybrid search combining semlix lexical search with semantic vector search.

This module provides the HybridSearcher class, which is the main interface
for performing hybrid search queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

from .embeddings import EmbeddingProvider
from .stores.base import VectorStore
from .fusion import (
    FusionMethod,
    FusedResult,
    reciprocal_rank_fusion,
    linear_fusion,
    distribution_based_score_fusion,
    relative_score_fusion,
)

if TYPE_CHECKING:
    from semlix.index import Index


@dataclass
class HybridSearchResult:
    """Result from hybrid search.
    
    Attributes:
        doc_id: Document identifier
        score: Combined relevance score
        lexical_score: Score from lexical search (if matched)
        semantic_score: Score from semantic search (if matched)
        stored_fields: Fields stored in semlix index
        highlights: Highlighted snippets keyed by field name
    """
    doc_id: str
    score: float
    lexical_score: float | None = None
    semantic_score: float | None = None
    stored_fields: Dict[str, Any] = field(default_factory=dict)
    highlights: Dict[str, str] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"HybridSearchResult(doc_id={self.doc_id!r}, "
            f"score={self.score:.4f})"
        )
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to stored fields."""
        return self.stored_fields[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a stored field with default."""
        return self.stored_fields.get(key, default)


class HybridSearcher:
    """Combines semlix lexical search with semantic vector search.
    
    This is the main interface for performing hybrid search queries.
    It executes both lexical and semantic searches, then fuses the
    results using configurable algorithms.
    
    Example:
        >>> from semlix.index import open_dir
        >>> from semlix.semantic import HybridSearcher, SentenceTransformerProvider
        >>> from semlix.semantic.stores import NumpyVectorStore
        >>> 
        >>> # Open existing semlix index
        >>> semlix_index = open_dir("my_index")
        >>> 
        >>> # Load semantic components
        >>> embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")
        >>> vector_store = NumpyVectorStore.load("my_vectors.pkl")
        >>> 
        >>> # Create hybrid searcher
        >>> searcher = HybridSearcher(
        ...     index=index,
        ...     vector_store=vector_store,
        ...     embedding_provider=embedder,
        ...     alpha=0.5  # Balance between lexical and semantic
        ... )
        >>> 
        >>> # Search
        >>> results = searcher.search("authentication problems", limit=10)
        >>> for r in results:
        ...     print(f"{r.doc_id}: {r.score:.3f}")
    
    Attributes:
        index: The semlix index for lexical search
        vector_store: Vector store for semantic search
        embedder: Embedding provider for query encoding
        alpha: Default weight for semantic search (0-1)
        fusion_method: Default fusion algorithm
    """
    
    def __init__(
        self,
        index: "Index",
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        default_field: str = "content",
        id_field: str = "id",
        alpha: float = 0.5,
        fusion_method: FusionMethod | str = FusionMethod.RRF,
        rrf_k: int = 60
    ):
        """Initialize hybrid searcher.
        
        Args:
            index: semlix index for lexical search
            vector_store: Vector store for semantic search
            embedding_provider: Provider for generating query embeddings
            default_field: Default field for semlix queries
            id_field: Field name containing document IDs
            alpha: Weight for semantic search (0 = all lexical, 1 = all semantic)
            fusion_method: Method for combining results
            rrf_k: K parameter for RRF fusion (only used with RRF method)
            
        Raises:
            ValueError: If embedding dimensions don't match vector store
        """
        self.index = index
        self.vector_store = vector_store
        self.embedder = embedding_provider
        self.default_field = default_field
        self.id_field = id_field
        self.alpha = alpha
        self.rrf_k = rrf_k
        
        # Handle string fusion method
        if isinstance(fusion_method, str):
            fusion_method = FusionMethod(fusion_method)
        self.fusion_method = fusion_method
        
        # Validate embedding dimensions match
        if vector_store.dimension != embedding_provider.dimension:
            raise ValueError(
                f"Vector store dimension ({vector_store.dimension}) doesn't match "
                f"embedding provider dimension ({embedding_provider.dimension})"
            )
    
    def search(
        self,
        query: str,
        limit: int = 10,
        alpha: float | None = None,
        lexical_limit: int | None = None,
        semantic_limit: int | None = None,
        filter_query: str | None = None,
        highlight_fields: List[str] | None = None,
        fusion_method: FusionMethod | str | None = None
    ) -> List[HybridSearchResult]:
        """Execute hybrid search.
        
        Performs both lexical and semantic searches, then fuses the results
        using the configured fusion algorithm.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            alpha: Override default alpha for this search (0-1)
            lexical_limit: Max results from lexical search (default: limit * 3)
            semantic_limit: Max results from semantic search (default: limit * 3)
            filter_query: Additional semlix query string to filter results
            highlight_fields: Fields to generate highlights for
            fusion_method: Override default fusion method for this search
            
        Returns:
            List of HybridSearchResult sorted by combined score
        
        Example:
            >>> # Standard hybrid search
            >>> results = searcher.search("python tutorial", limit=10)
            >>> 
            >>> # Prefer semantic matching for conceptual queries
            >>> results = searcher.search("ways to improve code quality", alpha=0.8)
            >>> 
            >>> # With highlighting
            >>> results = searcher.search("error handling", highlight_fields=["content"])
            >>> for r in results:
            ...     print(r.highlights.get("content", ""))
        """
        alpha = alpha if alpha is not None else self.alpha
        lexical_limit = lexical_limit or limit * 3
        semantic_limit = semantic_limit or limit * 3
        
        if fusion_method is not None:
            if isinstance(fusion_method, str):
                fusion_method = FusionMethod(fusion_method)
        else:
            fusion_method = self.fusion_method
        
        # Execute both searches
        lexical_results = self._lexical_search(query, lexical_limit, filter_query)
        semantic_results = self._semantic_search(query, semantic_limit)
        
        # Fuse results
        fused = self._fuse_results(lexical_results, semantic_results, alpha, fusion_method)
        
        # Get stored fields and highlights
        results = self._build_results(
            fused[:limit], 
            query, 
            highlight_fields
        )
        
        return results
    
    def search_lexical_only(
        self,
        query: str,
        limit: int = 10,
        filter_query: str | None = None,
        highlight_fields: List[str] | None = None
    ) -> List[HybridSearchResult]:
        """Execute lexical-only search (standard semlix behavior).
        
        Useful when you know the query contains specific keywords
        that should be matched exactly.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            filter_query: Additional filter query
            highlight_fields: Fields to highlight
            
        Returns:
            List of HybridSearchResult
        """
        return self.search(
            query, 
            limit, 
            alpha=0.0, 
            filter_query=filter_query,
            highlight_fields=highlight_fields
        )
    
    def search_semantic_only(
        self,
        query: str,
        limit: int = 10
    ) -> List[HybridSearchResult]:
        """Execute semantic-only search.
        
        Useful for conceptual queries where exact keyword matching
        is not important.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of HybridSearchResult
        """
        return self.search(query, limit, alpha=1.0)
    
    def _lexical_search(
        self,
        query: str,
        limit: int,
        filter_query: str | None = None
    ) -> List[tuple[str, float]]:
        """Execute semlix lexical search."""
        from semlix.qparser import QueryParser
        
        results = []
        
        with self.index.searcher() as searcher:
            qp = QueryParser(self.default_field, self.index.schema)
            q = qp.parse(query)
            
            # Apply filter if provided
            if filter_query:
                filter_q = qp.parse(filter_query)
                hits = searcher.search(q, filter=filter_q, limit=limit)
            else:
                hits = searcher.search(q, limit=limit)
            
            for hit in hits:
                doc_id = hit.get(self.id_field)
                if doc_id:
                    results.append((str(doc_id), float(hit.score)))
        
        return results
    
    def _semantic_search(
        self,
        query: str,
        limit: int
    ) -> List[tuple[str, float]]:
        """Execute semantic vector search."""
        # Generate query embedding
        query_embedding = self.embedder.encode([query])[0]
        
        # Search vector store
        vector_results = self.vector_store.search(query_embedding, k=limit)
        
        return [(r.doc_id, r.score) for r in vector_results]
    
    def _fuse_results(
        self,
        lexical: List[tuple[str, float]],
        semantic: List[tuple[str, float]],
        alpha: float,
        method: FusionMethod
    ) -> List[FusedResult]:
        """Fuse lexical and semantic results."""
        if method == FusionMethod.RRF:
            return reciprocal_rank_fusion(lexical, semantic, self.rrf_k, alpha)
        elif method == FusionMethod.LINEAR:
            return linear_fusion(lexical, semantic, alpha)
        elif method == FusionMethod.DBSF:
            return distribution_based_score_fusion(lexical, semantic, alpha)
        elif method == FusionMethod.RELATIVE_SCORE:
            return relative_score_fusion(lexical, semantic, alpha)
        else:
            # Default to RRF
            return reciprocal_rank_fusion(lexical, semantic, self.rrf_k, alpha)
    
    def _build_results(
        self,
        fused: List[FusedResult],
        query: str,
        highlight_fields: List[str] | None
    ) -> List[HybridSearchResult]:
        """Build HybridSearchResult objects with stored fields and highlights."""
        from semlix.qparser import QueryParser
        
        results = []
        
        with self.index.searcher() as searcher:
            for fr in fused:
                stored = {}
                highlights = {}
                
                # Find document in semlix by ID
                doc_num = searcher.document_number(**{self.id_field: fr.doc_id})
                
                if doc_num is not None:
                    stored = dict(searcher.stored_fields(doc_num))
                    
                    # Generate highlights if requested
                    if highlight_fields:
                        qp = QueryParser(self.default_field, self.index.schema)
                        q = qp.parse(query)
                        
                        for field_name in highlight_fields:
                            if field_name in stored:
                                try:
                                    hl = hit_highlights(
                                        searcher, 
                                        doc_num, 
                                        field_name, 
                                        q
                                    )
                                    if hl:
                                        highlights[field_name] = hl
                                except Exception:
                                    pass  # Highlighting failed, continue without
                
                results.append(HybridSearchResult(
                    doc_id=fr.doc_id,
                    score=fr.combined_score,
                    lexical_score=fr.lexical_score,
                    semantic_score=fr.semantic_score,
                    stored_fields=stored,
                    highlights=highlights
                ))
        
        return results
    
    def __repr__(self) -> str:
        return (
            f"HybridSearcher(alpha={self.alpha}, "
            f"fusion={self.fusion_method.value}, "
            f"embedder={self.embedder.model_name!r})"
        )


def hit_highlights(
    searcher, 
    docnum: int, 
    fieldname: str, 
    query,
    top: int = 3,
    maxchars: int = 200,
    surround: int = 50
) -> str:
    """Generate highlighted snippet for a document field.
    
    Args:
        searcher: semlix searcher object
        docnum: Document number
        fieldname: Field to highlight
        query: Parsed query object
        top: Number of top fragments to return
        maxchars: Maximum characters per fragment
        surround: Characters of context around matches
        
    Returns:
        Highlighted text string with matched terms wrapped in <b> tags
    """
    from semlix.highlight import Fragmenter, HtmlFormatter, Highlighter
    
    # Get field content
    stored = searcher.stored_fields(docnum)
    text = stored.get(fieldname, "")
    
    if not text:
        return ""
    
    # Create highlighter
    fragmenter = Fragmenter(maxchars=maxchars, surround=surround)
    formatter = HtmlFormatter(tagname="b")
    highlighter = Highlighter(fragmenter=fragmenter, formatter=formatter)
    
    # Get highlights
    return highlighter.highlight_hit(
        searcher.ixreader,
        docnum,
        fieldname,
        text=text,
        top=top
    )
