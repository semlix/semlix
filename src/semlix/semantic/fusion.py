"""Result fusion algorithms for hybrid search.

This module provides various algorithms for combining results from
lexical (BM25/TF-IDF) and semantic (vector) search.

Available methods:
- Reciprocal Rank Fusion (RRF): Robust, rank-based fusion
- Linear Fusion: Simple weighted score combination
- Distribution-Based Score Fusion (DBSF): Z-score normalization
- Relative Score Fusion: Percentile-based normalization
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Tuple


class FusionMethod(Enum):
    """Available result fusion methods."""
    RRF = "rrf"                    # Reciprocal Rank Fusion
    LINEAR = "linear"              # Linear score combination
    DBSF = "dbsf"                  # Distribution-Based Score Fusion
    RELATIVE_SCORE = "relative"   # Relative Score Fusion


@dataclass
class FusedResult:
    """Result after fusing lexical and semantic scores.
    
    Attributes:
        doc_id: Document identifier
        combined_score: Final fused score
        lexical_score: Original score from lexical search (if present)
        semantic_score: Original score from semantic search (if present)
        lexical_rank: Rank in lexical results (1-indexed, if present)
        semantic_rank: Rank in semantic results (1-indexed, if present)
    """
    doc_id: str
    combined_score: float
    lexical_score: float | None = None
    semantic_score: float | None = None
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    
    def __repr__(self) -> str:
        return (
            f"FusedResult(doc_id={self.doc_id!r}, "
            f"combined={self.combined_score:.4f}, "
            f"lex={self.lexical_score}, sem={self.semantic_score})"
        )


def reciprocal_rank_fusion(
    lexical_results: List[Tuple[str, float]],
    semantic_results: List[Tuple[str, float]],
    k: int = 60,
    alpha: float = 0.5
) -> List[FusedResult]:
    """Reciprocal Rank Fusion (RRF).
    
    Combines rankings using: score = sum(weight / (k + rank))
    
    This method is robust to score scale differences between systems
    and has been shown to be highly effective in practice. It only
    considers rank position, not raw scores.
    
    Args:
        lexical_results: List of (doc_id, score) from lexical search,
            ordered by descending score
        semantic_results: List of (doc_id, score) from semantic search,
            ordered by descending score
        k: Ranking constant (typically 60). Higher values give more
            weight to lower-ranked documents.
        alpha: Weight for semantic vs lexical (0 = all lexical, 1 = all semantic)
        
    Returns:
        List of FusedResult sorted by combined score (descending)
        
    Reference:
        Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
        "Reciprocal rank fusion outperforms condorcet and individual 
        rank learning methods." SIGIR '09.
    
    Example:
        >>> lexical = [("doc1", 10.5), ("doc2", 8.3), ("doc3", 5.1)]
        >>> semantic = [("doc2", 0.95), ("doc4", 0.88), ("doc1", 0.75)]
        >>> results = reciprocal_rank_fusion(lexical, semantic, alpha=0.5)
        >>> for r in results[:3]:
        ...     print(f"{r.doc_id}: {r.combined_score:.4f}")
    """
    scores: Dict[str, FusedResult] = {}
    
    lexical_weight = 1.0 - alpha
    semantic_weight = alpha
    
    # Process lexical results
    for rank, (doc_id, score) in enumerate(lexical_results):
        rrf_score = lexical_weight / (k + rank + 1)
        
        if doc_id not in scores:
            scores[doc_id] = FusedResult(
                doc_id=doc_id,
                combined_score=0.0
            )
        
        scores[doc_id].combined_score += rrf_score
        scores[doc_id].lexical_score = score
        scores[doc_id].lexical_rank = rank + 1
    
    # Process semantic results
    for rank, (doc_id, score) in enumerate(semantic_results):
        rrf_score = semantic_weight / (k + rank + 1)
        
        if doc_id not in scores:
            scores[doc_id] = FusedResult(
                doc_id=doc_id,
                combined_score=0.0
            )
        
        scores[doc_id].combined_score += rrf_score
        scores[doc_id].semantic_score = score
        scores[doc_id].semantic_rank = rank + 1
    
    # Sort by combined score
    return sorted(scores.values(), key=lambda x: x.combined_score, reverse=True)


def linear_fusion(
    lexical_results: List[Tuple[str, float]],
    semantic_results: List[Tuple[str, float]],
    alpha: float = 0.5,
    normalize: bool = True
) -> List[FusedResult]:
    """Linear score combination.
    
    Combines scores using: combined = (1 - alpha) * lexical + alpha * semantic
    
    Simple and intuitive, but sensitive to score scale differences
    between the two systems. Use normalize=True to mitigate this.
    
    Args:
        lexical_results: List of (doc_id, score) from lexical search
        semantic_results: List of (doc_id, score) from semantic search
        alpha: Weight for semantic score (0 = all lexical, 1 = all semantic)
        normalize: Whether to min-max normalize scores to [0, 1] before combining
        
    Returns:
        List of FusedResult sorted by combined score (descending)
    
    Example:
        >>> lexical = [("doc1", 10.5), ("doc2", 8.3)]
        >>> semantic = [("doc2", 0.95), ("doc1", 0.75)]
        >>> results = linear_fusion(lexical, semantic, alpha=0.5, normalize=True)
    """
    # Convert to dicts preserving original scores
    lexical_dict = dict(lexical_results)
    semantic_dict = dict(semantic_results)
    
    # Store original scores before normalization
    lexical_original = lexical_dict.copy()
    semantic_original = semantic_dict.copy()
    
    # Normalize if requested
    if normalize:
        lexical_dict = _minmax_normalize(lexical_dict)
        semantic_dict = _minmax_normalize(semantic_dict)
    
    # Combine scores
    all_doc_ids = set(lexical_dict.keys()) | set(semantic_dict.keys())
    results = []
    
    for doc_id in all_doc_ids:
        lex_score = lexical_dict.get(doc_id, 0.0)
        sem_score = semantic_dict.get(doc_id, 0.0)
        combined = (1 - alpha) * lex_score + alpha * sem_score
        
        results.append(FusedResult(
            doc_id=doc_id,
            combined_score=combined,
            lexical_score=lexical_original.get(doc_id),
            semantic_score=semantic_original.get(doc_id)
        ))
    
    return sorted(results, key=lambda x: x.combined_score, reverse=True)


def distribution_based_score_fusion(
    lexical_results: List[Tuple[str, float]],
    semantic_results: List[Tuple[str, float]],
    alpha: float = 0.5
) -> List[FusedResult]:
    """Distribution-Based Score Fusion (DBSF).
    
    Normalizes scores based on their distribution (z-score normalization)
    before combining. This handles different score distributions better
    than min-max normalization.
    
    The z-score normalization transforms scores to have mean=0 and std=1,
    making them comparable across different scoring systems.
    
    Args:
        lexical_results: List of (doc_id, score) from lexical search
        semantic_results: List of (doc_id, score) from semantic search
        alpha: Weight for semantic score (0 = all lexical, 1 = all semantic)
        
    Returns:
        List of FusedResult sorted by combined score (descending)
    """
    lexical_dict = dict(lexical_results)
    semantic_dict = dict(semantic_results)
    
    # Store originals
    lexical_original = lexical_dict.copy()
    semantic_original = semantic_dict.copy()
    
    # Z-score normalization
    lexical_dict = _zscore_normalize(lexical_dict)
    semantic_dict = _zscore_normalize(semantic_dict)
    
    # Combine
    all_doc_ids = set(lexical_dict.keys()) | set(semantic_dict.keys())
    results = []
    
    for doc_id in all_doc_ids:
        lex_score = lexical_dict.get(doc_id, 0.0)
        sem_score = semantic_dict.get(doc_id, 0.0)
        combined = (1 - alpha) * lex_score + alpha * sem_score
        
        results.append(FusedResult(
            doc_id=doc_id,
            combined_score=combined,
            lexical_score=lexical_original.get(doc_id),
            semantic_score=semantic_original.get(doc_id)
        ))
    
    return sorted(results, key=lambda x: x.combined_score, reverse=True)


def relative_score_fusion(
    lexical_results: List[Tuple[str, float]],
    semantic_results: List[Tuple[str, float]],
    alpha: float = 0.5
) -> List[FusedResult]:
    """Relative Score Fusion using percentile normalization.
    
    Normalizes scores to their percentile rank within each result set,
    then combines. This is robust to outliers and non-linear score
    distributions.
    
    Args:
        lexical_results: List of (doc_id, score) from lexical search
        semantic_results: List of (doc_id, score) from semantic search
        alpha: Weight for semantic score
        
    Returns:
        List of FusedResult sorted by combined score (descending)
    """
    lexical_dict = dict(lexical_results)
    semantic_dict = dict(semantic_results)
    
    # Store originals
    lexical_original = lexical_dict.copy()
    semantic_original = semantic_dict.copy()
    
    # Percentile normalization
    lexical_dict = _percentile_normalize(lexical_dict)
    semantic_dict = _percentile_normalize(semantic_dict)
    
    # Combine
    all_doc_ids = set(lexical_dict.keys()) | set(semantic_dict.keys())
    results = []
    
    for doc_id in all_doc_ids:
        lex_score = lexical_dict.get(doc_id, 0.0)
        sem_score = semantic_dict.get(doc_id, 0.0)
        combined = (1 - alpha) * lex_score + alpha * sem_score
        
        results.append(FusedResult(
            doc_id=doc_id,
            combined_score=combined,
            lexical_score=lexical_original.get(doc_id),
            semantic_score=semantic_original.get(doc_id)
        ))
    
    return sorted(results, key=lambda x: x.combined_score, reverse=True)


def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return scores
    
    min_score = min(scores.values())
    max_score = max(scores.values())
    
    if max_score == min_score:
        return {k: 1.0 for k in scores}
    
    return {
        k: (v - min_score) / (max_score - min_score)
        for k, v in scores.items()
    }


def _zscore_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Z-score normalize scores (mean=0, std=1)."""
    if not scores or len(scores) < 2:
        return {k: 0.0 for k in scores} if scores else scores
    
    values = list(scores.values())
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance) if variance > 0 else 1.0
    
    return {k: (v - mean) / std for k, v in scores.items()}


def _percentile_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores to percentile ranks [0, 1]."""
    if not scores:
        return scores
    
    if len(scores) == 1:
        return {k: 1.0 for k in scores}
    
    # Sort by score
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    n = len(sorted_items)
    
    # Assign percentile ranks
    return {
        doc_id: rank / (n - 1)
        for rank, (doc_id, _) in enumerate(sorted_items)
    }


# Convenience function to get fusion function by method
def get_fusion_function(
    method: FusionMethod | str
) -> Callable[[List[Tuple[str, float]], List[Tuple[str, float]], float], List[FusedResult]]:
    """Get the fusion function for a given method.
    
    Args:
        method: Fusion method (enum or string)
        
    Returns:
        Fusion function
        
    Raises:
        ValueError: If method is not recognized
    """
    if isinstance(method, str):
        method = FusionMethod(method)
    
    functions = {
        FusionMethod.RRF: lambda l, s, a: reciprocal_rank_fusion(l, s, alpha=a),
        FusionMethod.LINEAR: lambda l, s, a: linear_fusion(l, s, alpha=a),
        FusionMethod.DBSF: lambda l, s, a: distribution_based_score_fusion(l, s, alpha=a),
        FusionMethod.RELATIVE_SCORE: lambda l, s, a: relative_score_fusion(l, s, alpha=a),
    }
    
    return functions[method]
