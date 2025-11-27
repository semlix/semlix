==================
Semantic Search
==================

semlix includes optional semantic search capabilities that enable hybrid search,
combining traditional lexical matching (BM25/TF-IDF) with modern vector-based semantic
similarity. This allows queries to match documents based on meaning, not just keywords.

Overview
========

Semantic search uses dense vector embeddings to understand the semantic meaning of text.
When combined with semlix's traditional lexical search (inherited from semlix), you get hybrid search that
leverages the strengths of both approaches:

* **Lexical search** (BM25/TF-IDF): Excellent for exact keyword matching, phrase queries,
  and structured searches
* **Semantic search** (vector embeddings): Understands meaning and context, finds relevant
  documents even without keyword overlap

Hybrid search combines both approaches using result fusion algorithms to produce
superior results compared to either method alone.

Quick Start
===========

Here's a minimal example of using semantic search::

    from semlix.index import create_in
    from semlix.fields import Schema, TEXT, ID
    from semlix.semantic import (
        HybridSearcher,
        HybridIndexWriter,
        SentenceTransformerProvider
    )
    from semlix.semantic.stores import NumpyVectorStore
    from pathlib import Path

    # Create schema and index
    schema = Schema(
        id=ID(stored=True, unique=True),
        title=TEXT(stored=True),
        content=TEXT(stored=True)
    )
    ix = create_in("my_index", schema)

    # Create semantic components
    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")
    vector_store = NumpyVectorStore(dimension=embedder.dimension)

    # Index documents
    with HybridIndexWriter(
        ix, vector_store, embedder,
        embedding_field="content",
        id_field="id"
    ) as writer:
        writer.add_document(
            id="1",
            title="Python Tutorial",
            content="Learn Python programming basics and syntax"
        )
        writer.add_document(
            id="2",
            title="Authentication Guide",
            content="How to fix login and authentication issues"
        )

    # Save vector store
    vector_store.save("vectors.pkl")

    # Search
    searcher = HybridSearcher(
        index=ix,
        vector_store=vector_store,
        embedding_provider=embedder,
        alpha=0.5  # 50% lexical, 50% semantic
    )

    # This query will match "Authentication Guide" even without keyword overlap
    results = searcher.search("password problems", limit=10)
    for r in results:
        print(f"{r['title']}: {r.score:.4f}")

Components
==========

The semantic search module consists of several key components:

Embedding Providers
-------------------

Embedding providers generate dense vector representations of text. semlix supports
multiple providers:

**SentenceTransformerProvider** (Recommended for local use)::

    from semlix.semantic import SentenceTransformerProvider

    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")
    # Popular models:
    # - "all-MiniLM-L6-v2": Fast, good quality (384 dim)
    # - "all-mpnet-base-v2": Higher quality (768 dim)
    # - "multi-qa-MiniLM-L6-dot-v1": Optimized for QA (384 dim)

Requires: ``pip install sentence-transformers``

**OpenAIProvider** (For cloud-based embeddings)::

    from semlix.semantic import OpenAIProvider

    embedder = OpenAIProvider(
        model="text-embedding-3-small",
        api_key="your-api-key"  # or set OPENAI_API_KEY env var
    )

Requires: ``pip install openai``

**CohereProvider**::

    from semlix.semantic import CohereProvider

    embedder = CohereProvider(
        model="embed-english-v3.0",
        api_key="your-api-key"  # or set CO_API_KEY env var
    )

Requires: ``pip install cohere``

**HuggingFaceInferenceProvider**::

    from semlix.semantic import HuggingFaceInferenceProvider

    embedder = HuggingFaceInferenceProvider(
        model="sentence-transformers/all-MiniLM-L6-v2",
        api_key="your-token"  # or set HF_TOKEN env var
    )

Requires: ``pip install huggingface_hub``

Vector Stores
-------------

Vector stores persist and search embeddings. Choose based on your dataset size:

**NumpyVectorStore** (Pure-Python, < 100k vectors)::

    from semlix.semantic.stores import NumpyVectorStore

    store = NumpyVectorStore(dimension=384)
    store.add(doc_ids, embeddings, metadata)
    results = store.search(query_embedding, k=10)
    store.save("vectors.pkl")
    loaded = NumpyVectorStore.load("vectors.pkl")

No additional dependencies required (uses NumPy).

**FaissVectorStore** (High-performance, millions of vectors)::

    from semlix.semantic.stores import FaissVectorStore

    # Flat index for exact search
    store = FaissVectorStore(dimension=384, index_type="Flat")

    # IVF index for approximate search (requires training)
    store = FaissVectorStore(dimension=384, index_type="IVF", nlist=100)
    store.train(training_embeddings)  # Train on representative sample
    store.add(doc_ids, embeddings)

    # HNSW index for best speed
    store = FaissVectorStore(dimension=384, index_type="HNSW")

Requires: ``pip install faiss-cpu`` (or ``faiss-gpu`` for GPU support)

Hybrid Index Writer
--------------------

The :class:`~semlix.semantic.HybridIndexWriter` maintains both a semlix index and
a vector store in sync::

    from semlix.semantic import HybridIndexWriter

    writer = HybridIndexWriter(
        index=ix,
        vector_store=vector_store,
        embedding_provider=embedder,
        embedding_field="content",  # Field to generate embeddings from
        id_field="id",              # Field containing document ID
        batch_size=100               # Batch size for embedding generation
    )

    # Use as context manager
    with writer:
        writer.add_document(id="1", title="Doc 1", content="Text to embed")
        writer.add_document(id="2", title="Doc 2", content="More text")

    # Or manually
    writer.add_document(id="3", content="Another document")
    writer.commit()

The writer automatically:
* Adds documents to the semlix index
* Generates embeddings in batches
* Adds embeddings to the vector store

Hybrid Searcher
---------------

The :class:`~semlix.semantic.HybridSearcher` combines lexical and semantic search::

    from semlix.semantic import HybridSearcher, FusionMethod

    searcher = HybridSearcher(
        index=ix,
        vector_store=vector_store,
        embedding_provider=embedder,
        default_field="content",
        id_field="id",
        alpha=0.5,                    # Weight for semantic (0=lexical, 1=semantic)
        fusion_method=FusionMethod.RRF,  # Result fusion algorithm
        rrf_k=60                      # RRF constant
    )

    # Hybrid search (combines both)
    results = searcher.search("query text", limit=10)

    # Lexical-only search (traditional semlix/Whoosh)
    results = searcher.search_lexical_only("exact keywords", limit=10)

    # Semantic-only search
    results = searcher.search_semantic_only("conceptual query", limit=10)

    # Adjust balance per query
    results = searcher.search("query", alpha=0.8)  # Prefer semantic

Result Fusion
-------------

Result fusion combines rankings from lexical and semantic search. Available methods:

**RRF (Reciprocal Rank Fusion)** - Recommended::

    searcher = HybridSearcher(..., fusion_method=FusionMethod.RRF, rrf_k=60)

Robust to score scale differences, rank-based (doesn't depend on raw scores).

**Linear Fusion**::

    searcher = HybridSearcher(..., fusion_method=FusionMethod.LINEAR)

Simple weighted combination: ``combined = (1-alpha) * lexical + alpha * semantic``

**DBSF (Distribution-Based Score Fusion)**::

    searcher = HybridSearcher(..., fusion_method=FusionMethod.DBSF)

Z-score normalization before combining, handles different score distributions.

**Relative Score Fusion**::

    searcher = HybridSearcher(..., fusion_method=FusionMethod.RELATIVE_SCORE)

Percentile-based normalization, robust to outliers.

Advanced Usage
==============

Building Vector Store from Existing Index
------------------------------------------

If you have an existing semlix/semlix index, you can build a vector store from it::

    from semlix.index import open_dir
    from semlix.semantic import build_vector_store_from_index
    from semlix.semantic import SentenceTransformerProvider
    from semlix.semantic.stores import NumpyVectorStore

    # Open existing index
    ix = open_dir("my_existing_index")

    # Create semantic components
    embedder = SentenceTransformerProvider()
    vector_store = NumpyVectorStore(dimension=embedder.dimension)

    # Build vector store from index
    count = build_vector_store_from_index(
        index=ix,
        vector_store=vector_store,
        embedding_provider=embedder,
        embedding_field="content",
        id_field="id",
        show_progress=True
    )

    print(f"Indexed {count} documents")
    vector_store.save("vectors.pkl")

Using FAISS for Large Datasets
-------------------------------

For datasets with millions of documents, use FAISS with an approximate index::

    from semlix.semantic.stores import FaissVectorStore
    import numpy as np

    # Create IVF index
    vector_store = FaissVectorStore(
        dimension=384,
        index_type="IVF",
        nlist=1000,   # Number of clusters
        nprobe=50     # Clusters to search
    )

    # Train on representative sample (10% of data)
    sample_size = len(all_texts) // 10
    sample_texts = all_texts[:sample_size]
    sample_embeddings = embedder.encode(sample_texts)
    vector_store.train(sample_embeddings)

    # Add all documents
    all_embeddings = embedder.encode(all_texts, show_progress=True)
    vector_store.add(all_doc_ids, all_embeddings)

    # Save
    vector_store.save("large_vectors.faiss")

Adjusting Search Balance
------------------------

The ``alpha`` parameter controls the balance between lexical and semantic search:

* ``alpha=0.0``: Pure lexical search (traditional semlix/Whoosh)
* ``alpha=0.5``: Balanced hybrid search (default)
* ``alpha=1.0``: Pure semantic search

You can adjust per query::

    # Prefer lexical for exact keyword queries
    results = searcher.search("error code 404", alpha=0.2)

    # Prefer semantic for conceptual queries
    results = searcher.search("ways to improve performance", alpha=0.8)

Performance Considerations
==========================

Dataset Size Recommendations
----------------------------

**Small datasets (< 10K docs)**: Use ``NumpyVectorStore`` - Pure Python, no dependencies

**Medium datasets (10K - 100K)**: Use ``FaissVectorStore`` with ``index_type="Flat"`` - Exact search, fast enough

**Large datasets (100K - 1M)**: Use ``FaissVectorStore`` with ``index_type="IVF"`` - Approximate, tunable accuracy

**Very large datasets (> 1M)**: Use ``FaissVectorStore`` with ``index_type="HNSW"`` - Best speed/accuracy tradeoff

Embedding Caching
-----------------

For production, consider caching embeddings to avoid recomputing::

    from functools import lru_cache
    import numpy as np

    class CachedEmbedder:
        def __init__(self, provider, cache_size=10000):
            self._provider = provider
            self._cache = {}
            self._cache_size = cache_size

        @property
        def dimension(self):
            return self._provider.dimension

        def encode(self, texts, **kwargs):
            # Check cache
            uncached = []
            uncached_indices = []
            results = [None] * len(texts)

            for i, text in enumerate(texts):
                if text in self._cache:
                    results[i] = self._cache[text]
                else:
                    uncached.append(text)
                    uncached_indices.append(i)

            # Encode uncached texts
            if uncached:
                embeddings = self._provider.encode(uncached, **kwargs)
                for text, emb, idx in zip(uncached, embeddings, uncached_indices):
                    if len(self._cache) < self._cache_size:
                        self._cache[text] = emb
                    results[idx] = emb

            return np.array(results)

Migration Guide
===============

Existing Whoosh/semlix users can add semantic search without modifying existing code.
Your current indexes and queries continue to work as before.

To add semantic capabilities:

1. Install semantic dependencies::

    pip install semlix[semantic]

2. Create a vector store from your existing index (see "Building Vector Store from
   Existing Index" above)

3. Use HybridSearcher for new queries, or continue using traditional searchers
   for existing code

Example::

    # Existing code (still works)
    from semlix.index import open_dir
    from semlix.qparser import QueryParser

    ix = open_dir("my_index")
    with ix.searcher() as s:
        q = QueryParser("content", ix.schema).parse("search query")
        results = s.search(q)  # Traditional search

    # New semantic search (optional)
    from semlix.semantic import HybridSearcher, SentenceTransformerProvider
    from semlix.semantic.stores import NumpyVectorStore

    embedder = SentenceTransformerProvider()
    vector_store = NumpyVectorStore.load("vectors.pkl")
    searcher = HybridSearcher(ix, vector_store, embedder)
    results = searcher.search("search query")  # Hybrid search

API Reference
=============

The main classes and functions are documented below. For complete API details,
see the source code or use Python's ``help()`` function.

Main Classes:

* :class:`~semlix.semantic.HybridSearcher` - Main search interface
* :class:`~semlix.semantic.HybridIndexWriter` - Index writer for hybrid search
* :class:`~semlix.semantic.SentenceTransformerProvider` - Local embedding provider
* :class:`~semlix.semantic.stores.NumpyVectorStore` - Pure-Python vector store
* :class:`~semlix.semantic.stores.FaissVectorStore` - High-performance vector store

.. note::

   The semantic search module is optional and requires additional dependencies.
   See the installation instructions above.

