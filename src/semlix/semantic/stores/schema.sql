-- PostgreSQL + pgvector schema for Semlix semantic search
-- Version: 1.0.0
-- Requires: PostgreSQL 12+ with pgvector extension

-- ============================================================================
-- EXTENSION SETUP
-- ============================================================================

-- Enable pgvector extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- VECTOR STORAGE TABLE
-- ============================================================================

-- Main table for storing document vectors and metadata
CREATE TABLE IF NOT EXISTS semlix_vectors (
    -- Primary key: document identifier
    doc_id TEXT PRIMARY KEY,

    -- Vector embedding (dimension specified at runtime)
    -- Common dimensions: 384 (MiniLM), 768 (BERT), 1536 (OpenAI)
    -- Replace 384 with your model's dimension
    embedding vector(384) NOT NULL,

    -- JSON metadata for flexible filtering
    -- Examples: {"category": "tech", "lang": "en", "author": "John"}
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps for tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- HNSW index for fast cosine similarity search (RECOMMENDED)
-- This is the primary index for most use cases
-- Parameters:
--   m = 16: Number of connections per layer (trade-off: quality vs memory)
--   ef_construction = 64: Build-time search depth (trade-off: quality vs build speed)
CREATE INDEX IF NOT EXISTS idx_semlix_vectors_embedding_hnsw
    ON semlix_vectors
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat index (lower memory, requires VACUUM)
-- Uncomment if you prefer IVFFlat over HNSW:
-- CREATE INDEX IF NOT EXISTS idx_semlix_vectors_embedding_ivfflat
--     ON semlix_vectors
--     USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

-- GIN index for JSONB metadata filtering
-- Enables fast queries like: WHERE metadata @> '{"category": "tech"}'
CREATE INDEX IF NOT EXISTS idx_semlix_vectors_metadata
    ON semlix_vectors
    USING gin (metadata);

-- B-tree index for timestamp-based queries
CREATE INDEX IF NOT EXISTS idx_semlix_vectors_created_at
    ON semlix_vectors (created_at DESC);

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_semlix_vectors_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to call the update function
DROP TRIGGER IF EXISTS trigger_semlix_vectors_updated_at ON semlix_vectors;
CREATE TRIGGER trigger_semlix_vectors_updated_at
    BEFORE UPDATE ON semlix_vectors
    FOR EACH ROW
    EXECUTE FUNCTION update_semlix_vectors_updated_at();

-- ============================================================================
-- HELPER VIEWS
-- ============================================================================

-- View for vector statistics
CREATE OR REPLACE VIEW semlix_vectors_stats AS
SELECT
    COUNT(*) as total_vectors,
    COUNT(DISTINCT metadata->>'category') as unique_categories,
    pg_size_pretty(pg_total_relation_size('semlix_vectors')) as table_size,
    pg_size_pretty(pg_indexes_size('semlix_vectors')) as indexes_size,
    MIN(created_at) as oldest_vector,
    MAX(created_at) as newest_vector
FROM semlix_vectors;

-- ============================================================================
-- GRANTS (adjust based on your security requirements)
-- ============================================================================

-- Grant privileges to application user
-- Replace 'semlix_user' with your actual database user
-- GRANT SELECT, INSERT, UPDATE, DELETE ON semlix_vectors TO semlix_user;
-- GRANT SELECT ON semlix_vectors_stats TO semlix_user;

-- ============================================================================
-- NOTES FOR PRODUCTION
-- ============================================================================

-- 1. DIMENSION CONFIGURATION:
--    Update vector(384) to match your embedding model's dimension
--    Common values: 384, 768, 1024, 1536, 3072
--
-- 2. INDEX TUNING:
--    For HNSW:
--      - Increase 'm' (16 -> 32) for better recall, more memory
--      - Increase 'ef_construction' (64 -> 128) for better quality, slower build
--    For IVFFlat:
--      - Set 'lists' to sqrt(total_rows) as a starting point
--      - Run VACUUM regularly for optimal performance
--
-- 3. METADATA FILTERING:
--    For complex filters, consider additional indexes:
--      CREATE INDEX idx_category ON semlix_vectors ((metadata->>'category'));
--
-- 4. PARTITIONING (for very large datasets):
--    Consider partitioning by date or category:
--      CREATE TABLE semlix_vectors_2024 PARTITION OF semlix_vectors
--      FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
--
-- 5. MONITORING:
--    Check index usage:
--      SELECT * FROM pg_stat_user_indexes WHERE relname = 'semlix_vectors';
--    Check table bloat:
--      SELECT * FROM semlix_vectors_stats;

-- ============================================================================
-- EXAMPLE QUERIES
-- ============================================================================

-- Search by vector similarity (replace [0.1, 0.2, ...] with actual embedding)
-- SELECT doc_id, embedding <=> '[0.1, 0.2, ...]'::vector as distance
-- FROM semlix_vectors
-- ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
-- LIMIT 10;

-- Search with metadata filter
-- SELECT doc_id, embedding <=> '[0.1, 0.2, ...]'::vector as distance, metadata
-- FROM semlix_vectors
-- WHERE metadata @> '{"category": "tech"}'::jsonb
-- ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
-- LIMIT 10;

-- Get vector statistics
-- SELECT * FROM semlix_vectors_stats;
