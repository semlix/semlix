"""Tests for the semantic search module."""

import numpy as np
import pytest
import tempfile
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_embeddings():
    """Sample normalized embeddings for testing."""
    np.random.seed(42)
    embeddings = np.random.randn(5, 384).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def sample_doc_ids():
    """Sample document IDs."""
    return ["doc1", "doc2", "doc3", "doc4", "doc5"]


@pytest.fixture
def numpy_store(sample_embeddings, sample_doc_ids):
    """NumpyVectorStore with sample data."""
    from semlix.semantic.stores import NumpyVectorStore
    
    store = NumpyVectorStore(dimension=384)
    metadata = [{"title": f"Document {i}"} for i in range(5)]
    store.add(sample_doc_ids, sample_embeddings, metadata)
    return store


# =============================================================================
# VectorStore Tests
# =============================================================================

class TestNumpyVectorStore:
    """Tests for NumpyVectorStore."""
    
    def test_init(self):
        from semlix.semantic.stores import NumpyVectorStore
        
        store = NumpyVectorStore(dimension=384)
        assert store.dimension == 384
        assert store.count == 0
        assert len(store) == 0
    
    def test_init_invalid_dimension(self):
        from semlix.semantic.stores import NumpyVectorStore
        
        with pytest.raises(ValueError):
            NumpyVectorStore(dimension=0)
        
        with pytest.raises(ValueError):
            NumpyVectorStore(dimension=-1)
    
    def test_add_single(self):
        from semlix.semantic.stores import NumpyVectorStore
        
        store = NumpyVectorStore(dimension=3)
        embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        store.add(["doc1"], embedding.reshape(1, -1))
        
        assert store.count == 1
        assert "doc1" in store
    
    def test_add_batch(self, sample_embeddings, sample_doc_ids):
        from semlix.semantic.stores import NumpyVectorStore
        
        store = NumpyVectorStore(dimension=384)
        store.add(sample_doc_ids, sample_embeddings)
        
        assert store.count == 5
        for doc_id in sample_doc_ids:
            assert doc_id in store
    
    def test_add_with_metadata(self):
        from semlix.semantic.stores import NumpyVectorStore
        
        store = NumpyVectorStore(dimension=3)
        embeddings = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        metadata = [{"title": "Doc 1"}, {"title": "Doc 2"}]
        
        store.add(["d1", "d2"], embeddings, metadata)
        
        result = store.get("d1")
        assert result is not None
        assert result.metadata["title"] == "Doc 1"
    
    def test_add_dimension_mismatch(self):
        from semlix.semantic.stores import NumpyVectorStore
        
        store = NumpyVectorStore(dimension=384)
        wrong_dim = np.random.randn(2, 256).astype(np.float32)
        
        with pytest.raises(ValueError, match="dimension"):
            store.add(["d1", "d2"], wrong_dim)
    
    def test_search_basic(self, numpy_store, sample_embeddings):
        query = sample_embeddings[0]  # Search with first embedding
        results = numpy_store.search(query, k=3)
        
        assert len(results) == 3
        assert results[0].doc_id == "doc1"  # Should match itself
        assert results[0].score > 0.99  # Almost 1.0 (self-similarity)
    
    def test_search_with_filter(self, numpy_store, sample_embeddings):
        query = sample_embeddings[0]
        results = numpy_store.search(query, k=3, filter_ids=["doc2", "doc3", "doc4"])
        
        assert len(results) == 3
        assert all(r.doc_id in ["doc2", "doc3", "doc4"] for r in results)
        assert "doc1" not in [r.doc_id for r in results]
    
    def test_search_empty_store(self):
        from semlix.semantic.stores import NumpyVectorStore
        
        store = NumpyVectorStore(dimension=3)
        query = np.array([1, 0, 0], dtype=np.float32)
        
        results = store.search(query, k=5)
        assert results == []
    
    def test_delete(self, numpy_store):
        assert "doc1" in numpy_store
        assert numpy_store.count == 5
        
        deleted = numpy_store.delete(["doc1", "doc2"])
        
        assert deleted == 2
        assert "doc1" not in numpy_store
        assert "doc2" not in numpy_store
        assert numpy_store.count == 3
    
    def test_delete_nonexistent(self, numpy_store):
        deleted = numpy_store.delete(["nonexistent"])
        assert deleted == 0
        assert numpy_store.count == 5
    
    def test_clear(self, numpy_store):
        assert numpy_store.count == 5
        numpy_store.clear()
        assert numpy_store.count == 0
    
    def test_save_load(self, numpy_store):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store.pkl"
            numpy_store.save(path)
            
            from semlix.semantic.stores import NumpyVectorStore
            loaded = NumpyVectorStore.load(path)
            
            assert loaded.dimension == numpy_store.dimension
            assert loaded.count == numpy_store.count
            
            # Test search works
            query = np.random.randn(384).astype(np.float32)
            original_results = numpy_store.search(query, k=3)
            loaded_results = loaded.search(query, k=3)
            
            assert len(original_results) == len(loaded_results)
            assert original_results[0].doc_id == loaded_results[0].doc_id
    
    def test_update_existing(self):
        from semlix.semantic.stores import NumpyVectorStore
        
        store = NumpyVectorStore(dimension=3)
        
        # Add initial
        store.add(["doc1"], np.array([[1, 0, 0]], dtype=np.float32))
        
        # Update with new embedding
        store.add(["doc1"], np.array([[0, 1, 0]], dtype=np.float32))
        
        # Count should still be 1
        assert store.count == 1
        
        # Search should find the updated embedding
        query = np.array([0, 1, 0], dtype=np.float32)
        results = store.search(query, k=1)
        assert results[0].doc_id == "doc1"
        assert results[0].score > 0.99


# =============================================================================
# Fusion Algorithm Tests
# =============================================================================

class TestFusion:
    """Tests for result fusion algorithms."""
    
    @pytest.fixture
    def lexical_results(self):
        return [
            ("doc1", 10.5),
            ("doc2", 8.3),
            ("doc3", 5.1),
            ("doc4", 3.0),
        ]
    
    @pytest.fixture
    def semantic_results(self):
        return [
            ("doc2", 0.95),
            ("doc5", 0.88),
            ("doc1", 0.75),
            ("doc3", 0.60),
        ]
    
    def test_rrf_basic(self, lexical_results, semantic_results):
        from semlix.semantic.fusion import reciprocal_rank_fusion
        
        results = reciprocal_rank_fusion(
            lexical_results, 
            semantic_results, 
            k=60, 
            alpha=0.5
        )
        
        # doc1 and doc2 should be top (appear in both)
        top_ids = [r.doc_id for r in results[:2]]
        assert "doc1" in top_ids or "doc2" in top_ids
        
        # All docs should be present
        all_ids = {r.doc_id for r in results}
        assert all_ids == {"doc1", "doc2", "doc3", "doc4", "doc5"}
    
    def test_rrf_lexical_only(self, lexical_results, semantic_results):
        from semlix.semantic.fusion import reciprocal_rank_fusion
        
        results = reciprocal_rank_fusion(
            lexical_results, 
            semantic_results, 
            alpha=0.0  # All lexical
        )
        
        # Order should follow lexical
        assert results[0].doc_id == "doc1"
        assert results[1].doc_id == "doc2"
    
    def test_rrf_semantic_only(self, lexical_results, semantic_results):
        from semlix.semantic.fusion import reciprocal_rank_fusion
        
        results = reciprocal_rank_fusion(
            lexical_results, 
            semantic_results, 
            alpha=1.0  # All semantic
        )
        
        # Order should follow semantic
        assert results[0].doc_id == "doc2"
        assert results[1].doc_id == "doc5"
    
    def test_linear_fusion(self, lexical_results, semantic_results):
        from semlix.semantic.fusion import linear_fusion
        
        results = linear_fusion(
            lexical_results, 
            semantic_results, 
            alpha=0.5,
            normalize=True
        )
        
        assert len(results) == 5
        # Scores should be in [0, 1] range after normalization
        for r in results:
            assert 0 <= r.combined_score <= 1
    
    def test_dbsf(self, lexical_results, semantic_results):
        from semlix.semantic.fusion import distribution_based_score_fusion
        
        results = distribution_based_score_fusion(
            lexical_results, 
            semantic_results, 
            alpha=0.5
        )
        
        assert len(results) == 5
        # Results should be sorted by combined score
        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_relative_score_fusion(self, lexical_results, semantic_results):
        from semlix.semantic.fusion import relative_score_fusion
        
        results = relative_score_fusion(
            lexical_results, 
            semantic_results, 
            alpha=0.5
        )
        
        assert len(results) == 5
    
    def test_empty_results(self):
        from semlix.semantic.fusion import reciprocal_rank_fusion
        
        results = reciprocal_rank_fusion([], [], alpha=0.5)
        assert results == []
    
    def test_one_sided_results(self, lexical_results):
        from semlix.semantic.fusion import reciprocal_rank_fusion
        
        results = reciprocal_rank_fusion(lexical_results, [], alpha=0.5)
        assert len(results) == 4
        
        results = reciprocal_rank_fusion([], lexical_results, alpha=0.5)
        assert len(results) == 4


# =============================================================================
# Embedding Provider Tests (Mock)
# =============================================================================

class MockEmbeddingProvider:
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return "mock-model"
    
    def encode(
        self, 
        texts, 
        batch_size=32, 
        show_progress=False, 
        normalize=True
    ):
        np.random.seed(hash(str(texts)) % 2**32)
        embeddings = np.random.randn(len(texts), self._dimension).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        return embeddings


class TestEmbeddingProvider:
    """Tests for embedding providers."""
    
    def test_mock_provider(self):
        provider = MockEmbeddingProvider(dimension=384)
        
        assert provider.dimension == 384
        assert provider.model_name == "mock-model"
        
        embeddings = provider.encode(["hello", "world"])
        assert embeddings.shape == (2, 384)
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])
    
    def test_provider_protocol(self):
        from semlix.semantic.embeddings import EmbeddingProvider
        
        provider = MockEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)


# =============================================================================
# Integration Tests
# =============================================================================

class TestHybridSearchIntegration:
    """Integration tests for hybrid search."""
    
    @pytest.fixture
    def semlix_index(self):
        """Create a temporary semlix index."""
        from semlix.index import create_in
        from semlix.fields import Schema, TEXT, ID
        
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = Schema(
                id=ID(stored=True, unique=True),
                title=TEXT(stored=True),
                content=TEXT(stored=True)
            )
            ix = create_in(tmpdir, schema)
            
            # Add some documents
            writer = ix.writer()
            docs = [
                ("1", "Python Tutorial", "Learn Python programming basics and fundamentals"),
                ("2", "Authentication Guide", "How to fix login and authentication issues"),
                ("3", "Machine Learning", "Introduction to ML and deep learning concepts"),
                ("4", "Web Development", "Building websites with HTML CSS JavaScript"),
                ("5", "Data Science", "Analyzing data with Python pandas numpy"),
            ]
            for doc_id, title, content in docs:
                writer.add_document(id=doc_id, title=title, content=content)
            writer.commit()
            
            yield ix
    
    @pytest.fixture
    def vector_store(self):
        """Create vector store with matching documents."""
        from semlix.semantic.stores import NumpyVectorStore
        
        store = NumpyVectorStore(dimension=384)
        provider = MockEmbeddingProvider(dimension=384)
        
        docs = [
            ("1", "Learn Python programming basics and fundamentals"),
            ("2", "How to fix login and authentication issues"),
            ("3", "Introduction to ML and deep learning concepts"),
            ("4", "Building websites with HTML CSS JavaScript"),
            ("5", "Analyzing data with Python pandas numpy"),
        ]
        
        doc_ids = [d[0] for d in docs]
        texts = [d[1] for d in docs]
        embeddings = provider.encode(texts)
        store.add(doc_ids, embeddings)
        
        return store
    
    def test_hybrid_search(self, semlix_index, vector_store):
        from semlix.semantic import HybridSearcher
        
        provider = MockEmbeddingProvider(dimension=384)
        
        searcher = HybridSearcher(
            index=semlix_index,
            vector_store=vector_store,
            embedding_provider=provider,
            default_field="content",
            alpha=0.5
        )
        
        results = searcher.search("python programming", limit=3)
        
        assert len(results) <= 3
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'stored_fields') for r in results)
    
    def test_lexical_only_search(self, semlix_index, vector_store):
        from semlix.semantic import HybridSearcher
        
        provider = MockEmbeddingProvider(dimension=384)
        
        searcher = HybridSearcher(
            index=semlix_index,
            vector_store=vector_store,
            embedding_provider=provider,
        )
        
        results = searcher.search_lexical_only("Python", limit=5)
        
        # Should find documents with "Python" keyword
        assert len(results) > 0
    
    def test_semantic_only_search(self, semlix_index, vector_store):
        from semlix.semantic import HybridSearcher
        
        provider = MockEmbeddingProvider(dimension=384)
        
        searcher = HybridSearcher(
            index=semlix_index,
            vector_store=vector_store,
            embedding_provider=provider,
        )
        
        results = searcher.search_semantic_only("coding tutorial", limit=5)
        
        assert len(results) > 0
    
    def test_dimension_mismatch_error(self, semlix_index):
        from semlix.semantic import HybridSearcher
        from semlix.semantic.stores import NumpyVectorStore
        
        store = NumpyVectorStore(dimension=768)  # Different dimension
        provider = MockEmbeddingProvider(dimension=384)
        
        with pytest.raises(ValueError, match="dimension"):
            HybridSearcher(
                index=semlix_index,
                vector_store=store,
                embedding_provider=provider,
            )


class TestHybridIndexWriter:
    """Tests for HybridIndexWriter."""
    
    def test_add_and_commit(self):
        from semlix.index import create_in
        from semlix.fields import Schema, TEXT, ID
        from semlix.semantic import HybridIndexWriter
        from semlix.semantic.stores import NumpyVectorStore
        
        provider = MockEmbeddingProvider(dimension=384)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = Schema(
                id=ID(stored=True, unique=True),
                content=TEXT(stored=True)
            )
            ix = create_in(tmpdir, schema)
            store = NumpyVectorStore(dimension=384)
            
            with HybridIndexWriter(ix, store, provider) as writer:
                writer.add_document(id="1", content="Hello world")
                writer.add_document(id="2", content="Goodbye world")
            
            # Check semlix index
            with ix.searcher() as s:
                assert s.doc_count() == 2
            
            # Check vector store
            assert store.count == 2
            assert "1" in store
            assert "2" in store
    
    def test_context_manager_cancel_on_error(self):
        from semlix.index import create_in
        from semlix.fields import Schema, TEXT, ID
        from semlix.semantic import HybridIndexWriter
        from semlix.semantic.stores import NumpyVectorStore
        
        provider = MockEmbeddingProvider(dimension=384)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = Schema(
                id=ID(stored=True, unique=True),
                content=TEXT(stored=True)
            )
            ix = create_in(tmpdir, schema)
            store = NumpyVectorStore(dimension=384)
            
            try:
                with HybridIndexWriter(ix, store, provider) as writer:
                    writer.add_document(id="1", content="Hello world")
                    raise ValueError("Simulated error")
            except ValueError:
                pass
            
            # Should have been rolled back
            with ix.searcher() as s:
                assert s.doc_count() == 0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
