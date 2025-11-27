"""Embedding providers for semantic search.

This module defines the EmbeddingProvider protocol and provides
implementations for common embedding services.

Supported providers:
- SentenceTransformerProvider: Local models via sentence-transformers
- OpenAIProvider: OpenAI embedding API
- CohereProvider: Cohere embedding API
- HuggingFaceProvider: HuggingFace Inference API
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding model providers.
    
    Implementations can wrap sentence-transformers, OpenAI, Cohere,
    or any other embedding service.
    
    Example:
        >>> class MyProvider:
        ...     @property
        ...     def dimension(self) -> int:
        ...         return 384
        ...     
        ...     @property
        ...     def model_name(self) -> str:
        ...         return "my-model"
        ...     
        ...     def encode(self, texts: List[str], **kwargs) -> NDArray:
        ...         # Generate embeddings
        ...         return np.random.randn(len(texts), 384).astype(np.float32)
    """
    
    @property
    def dimension(self) -> int:
        """Return the dimensionality of embeddings produced by this provider."""
        ...
    
    @property
    def model_name(self) -> str:
        """Return identifier for the embedding model."""
        ...
    
    def encode(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> NDArray[np.float32]:
        """Encode texts into dense vector embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Number of texts to process at once
            show_progress: Whether to display progress bar
            normalize: Whether to L2-normalize output vectors
            
        Returns:
            Array of shape (len(texts), dimension) containing embeddings
        """
        ...


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of embeddings."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass
    
    @abstractmethod
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> NDArray[np.float32]:
        """Encode texts into embeddings."""
        pass
    
    def encode_single(self, text: str, normalize: bool = True) -> NDArray[np.float32]:
        """Encode a single text string.
        
        Args:
            text: Text to encode
            normalize: Whether to L2-normalize the output
            
        Returns:
            Embedding vector of shape (dimension,)
        """
        return self.encode([text], normalize=normalize)[0]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, dim={self.dimension})"


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Embedding provider using sentence-transformers library.
    
    This is the recommended provider for local embedding generation.
    Supports a wide variety of models from HuggingFace.
    
    Popular models:
    - "all-MiniLM-L6-v2": Fast, good quality (384 dim)
    - "all-mpnet-base-v2": Higher quality (768 dim)
    - "multi-qa-MiniLM-L6-dot-v1": Optimized for QA (384 dim)
    - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual (384 dim)
    
    Example:
        >>> provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        >>> embeddings = provider.encode(["Hello world", "How are you?"])
        >>> print(embeddings.shape)
        (2, 384)
    
    Requires: pip install sentence-transformers
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        cache_dir: str | None = None,
        trust_remote_code: bool = False
    ):
        """Initialize the sentence-transformers provider.
        
        Args:
            model_name: Name of the model to load (from HuggingFace)
            device: Device to run on ("cpu", "cuda", "mps", or None for auto)
            cache_dir: Directory to cache downloaded models
            trust_remote_code: Whether to trust remote code in models
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerProvider. "
                "Install with: pip install sentence-transformers"
            )
        
        self._model_name = model_name
        self._model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_dir,
            trust_remote_code=trust_remote_code
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def device(self) -> str:
        """Return the device the model is running on."""
        return str(self._model.device)
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> NDArray[np.float32]:
        """Encode texts using the sentence-transformer model.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar during encoding
            normalize: L2-normalize embeddings (recommended for cosine similarity)
            
        Returns:
            Embeddings array of shape (len(texts), dimension)
        """
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)


class OpenAIProvider(BaseEmbeddingProvider):
    """Embedding provider using OpenAI API.
    
    Supported models:
    - "text-embedding-3-small": Fast, good quality (1536 dim, configurable)
    - "text-embedding-3-large": Higher quality (3072 dim, configurable)
    - "text-embedding-ada-002": Legacy model (1536 dim)
    
    Example:
        >>> provider = OpenAIProvider(model="text-embedding-3-small")
        >>> embeddings = provider.encode(["Hello world"])
        >>> print(embeddings.shape)
        (1, 1536)
    
    Requires: pip install openai
    """
    
    # Default dimensions for OpenAI models
    DEFAULT_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
        base_url: str | None = None
    ):
        """Initialize the OpenAI embedding provider.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            dimensions: Output dimensions (only for text-embedding-3-* models)
            base_url: Custom API base URL (for Azure or compatible APIs)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAIProvider. "
                "Install with: pip install openai"
            )
        
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        
        # Handle dimensions
        if dimensions is not None:
            self._dimensions = dimensions
        else:
            self._dimensions = self.DEFAULT_DIMENSIONS.get(model, 1536)
        
        # Only text-embedding-3-* models support custom dimensions
        self._supports_dimensions = model.startswith("text-embedding-3")
    
    @property
    def dimension(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = False,
        normalize: bool = True
    ) -> NDArray[np.float32]:
        """Encode texts using OpenAI embedding API.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for API calls (max 2048 for OpenAI)
            show_progress: Ignored (no progress bar for API calls)
            normalize: L2-normalize embeddings (OpenAI embeddings are already normalized)
            
        Returns:
            Embeddings array of shape (len(texts), dimensions)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Build request parameters
            kwargs = {
                "input": batch,
                "model": self._model,
            }
            if self._supports_dimensions:
                kwargs["dimensions"] = self._dimensions
            
            response = self._client.embeddings.create(**kwargs)
            
            # Extract embeddings in order
            batch_embeddings = [None] * len(batch)
            for item in response.data:
                batch_embeddings[item.index] = item.embedding
            
            all_embeddings.extend(batch_embeddings)
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        # OpenAI embeddings are already normalized, but normalize anyway if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)
        
        return embeddings


class CohereProvider(BaseEmbeddingProvider):
    """Embedding provider using Cohere API.
    
    Supported models:
    
    - "embed-english-v3.0": English embeddings (1024 dim)
    - "embed-multilingual-v3.0": Multilingual (1024 dim)
    - "embed-english-light-v3.0": Faster, smaller (384 dim)
    - "embed-multilingual-light-v3.0": Multilingual light (384 dim)
    
    Example::
    
        >>> provider = CohereProvider(model="embed-english-v3.0")
        >>> embeddings = provider.encode(["Hello world"])
    
    Requires: pip install cohere
    """
    
    DEFAULT_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
        "embed-english-v2.0": 4096,
        "embed-multilingual-v2.0": 768,
    }
    
    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: str | None = None,
        input_type: str = "search_document"
    ):
        """Initialize the Cohere embedding provider.
        
        Args:
            model: Cohere embedding model name
            api_key: Cohere API key (uses CO_API_KEY env var if not provided)
            input_type: Type of input ("search_document", "search_query", 
                       "classification", "clustering")
        """
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere is required for CohereProvider. "
                "Install with: pip install cohere"
            )
        
        self._client = cohere.Client(api_key=api_key)
        self._model = model
        self._input_type = input_type
        self._dimensions = self.DEFAULT_DIMENSIONS.get(model, 1024)
    
    @property
    def dimension(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 96,
        show_progress: bool = False,
        normalize: bool = True
    ) -> NDArray[np.float32]:
        """Encode texts using Cohere embedding API."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self._client.embed(
                texts=batch,
                model=self._model,
                input_type=self._input_type
            )
            
            all_embeddings.extend(response.embeddings)
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)
        
        return embeddings


class HuggingFaceInferenceProvider(BaseEmbeddingProvider):
    """Embedding provider using HuggingFace Inference API.
    
    Useful for running models without local GPU resources.
    
    Example:
        >>> provider = HuggingFaceInferenceProvider(
        ...     model="sentence-transformers/all-MiniLM-L6-v2"
        ... )
        >>> embeddings = provider.encode(["Hello world"])
    
    Requires: pip install huggingface_hub
    """
    
    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: str | None = None,
        dimension: int | None = None
    ):
        """Initialize the HuggingFace Inference provider.
        
        Args:
            model: HuggingFace model ID
            api_key: HuggingFace API token (uses HF_TOKEN env var if not provided)
            dimension: Embedding dimension (auto-detected if not provided)
        """
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for HuggingFaceInferenceProvider. "
                "Install with: pip install huggingface_hub"
            )
        
        self._client = InferenceClient(token=api_key)
        self._model = model
        
        # Auto-detect dimension if not provided
        if dimension is None:
            test_embedding = self._client.feature_extraction(
                "test", model=model
            )
            self._dimension = len(test_embedding[0]) if isinstance(test_embedding[0], list) else len(test_embedding)
        else:
            self._dimension = dimension
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> NDArray[np.float32]:
        """Encode texts using HuggingFace Inference API."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # HuggingFace Inference API
            for text in batch:
                embedding = self._client.feature_extraction(
                    text, model=self._model
                )
                # Handle nested list response
                if isinstance(embedding[0], list):
                    # Mean pooling for token embeddings
                    embedding = np.mean(embedding, axis=0)
                all_embeddings.append(embedding)
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)
        
        return embeddings
