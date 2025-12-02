-- PostgreSQL schema for Semlix lexical index storage
-- Version: 1.0.0 (Experimental/Proof of Concept)
-- Requires: PostgreSQL 12+

-- ============================================================================
-- LEXICAL INDEX TABLES
-- ============================================================================

-- Documents table: stores document metadata and stored fields
CREATE TABLE IF NOT EXISTS semlix_documents (
    -- Composite primary key: index_name + doc_id
    index_name TEXT NOT NULL,
    doc_id TEXT NOT NULL,

    -- Document number (internal ID used by Whoosh)
    doc_number INTEGER NOT NULL,

    -- Stored fields as JSONB (flexible schema)
    stored_fields JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Field lengths for BM25 scoring (JSON: {field_name: length})
    field_lengths JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (index_name, doc_id)
);

-- Terms table: stores unique terms per index and field
CREATE TABLE IF NOT EXISTS semlix_terms (
    -- Auto-incrementing term ID
    term_id BIGSERIAL PRIMARY KEY,

    -- Index and field context
    index_name TEXT NOT NULL,
    field_name TEXT NOT NULL,

    -- The actual term
    term TEXT NOT NULL,

    -- Document frequency (number of docs containing this term)
    doc_frequency INTEGER NOT NULL DEFAULT 0,

    -- Collection frequency (total occurrences across all docs)
    collection_frequency INTEGER NOT NULL DEFAULT 0,

    -- Ensure uniqueness within index/field
    UNIQUE (index_name, field_name, term)
);

-- Postings table: term -> document mappings with frequencies
CREATE TABLE IF NOT EXISTS semlix_postings (
    -- Foreign key to term
    term_id BIGINT NOT NULL REFERENCES semlix_terms(term_id) ON DELETE CASCADE,

    -- Document reference
    index_name TEXT NOT NULL,
    doc_id TEXT NOT NULL,

    -- Term frequency in this document
    frequency INTEGER NOT NULL DEFAULT 1,

    -- Term positions (optional, for phrase queries)
    -- Stored as array of integers
    positions INTEGER[],

    -- Composite foreign key to document
    FOREIGN KEY (index_name, doc_id) REFERENCES semlix_documents(index_name, doc_id) ON DELETE CASCADE,

    -- Unique posting per term-document pair
    PRIMARY KEY (term_id, index_name, doc_id)
);

-- Index metadata: stores schema and configuration
CREATE TABLE IF NOT EXISTS semlix_index_meta (
    index_name TEXT PRIMARY KEY,

    -- Schema definition (serialized)
    schema_data JSONB NOT NULL,

    -- Index generation/version
    generation INTEGER NOT NULL DEFAULT 0,

    -- Index statistics
    doc_count INTEGER NOT NULL DEFAULT 0,
    term_count INTEGER NOT NULL DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- B-tree indexes for lookups
CREATE INDEX IF NOT EXISTS idx_documents_index_name
    ON semlix_documents(index_name);

CREATE INDEX IF NOT EXISTS idx_documents_doc_number
    ON semlix_documents(index_name, doc_number);

-- GIN index for stored_fields queries
CREATE INDEX IF NOT EXISTS idx_documents_stored_fields
    ON semlix_documents USING gin(stored_fields);

-- Terms lookup index
CREATE INDEX IF NOT EXISTS idx_terms_lookup
    ON semlix_terms(index_name, field_name, term);

-- Postings indexes
CREATE INDEX IF NOT EXISTS idx_postings_term
    ON semlix_postings(term_id);

CREATE INDEX IF NOT EXISTS idx_postings_document
    ON semlix_postings(index_name, doc_id);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to documents table
DROP TRIGGER IF EXISTS trigger_documents_updated_at ON semlix_documents;
CREATE TRIGGER trigger_documents_updated_at
    BEFORE UPDATE ON semlix_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Apply to index metadata
DROP TRIGGER IF EXISTS trigger_index_meta_updated_at ON semlix_index_meta;
CREATE TRIGGER trigger_index_meta_updated_at
    BEFORE UPDATE ON semlix_index_meta
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Function to update document frequency when postings change
CREATE OR REPLACE FUNCTION update_term_frequencies()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Increment doc_frequency
        UPDATE semlix_terms
        SET doc_frequency = doc_frequency + 1,
            collection_frequency = collection_frequency + NEW.frequency
        WHERE term_id = NEW.term_id;
    ELSIF TG_OP = 'DELETE' THEN
        -- Decrement doc_frequency
        UPDATE semlix_terms
        SET doc_frequency = GREATEST(0, doc_frequency - 1),
            collection_frequency = GREATEST(0, collection_frequency - OLD.frequency)
        WHERE term_id = OLD.term_id;
    ELSIF TG_OP = 'UPDATE' THEN
        -- Update collection_frequency
        UPDATE semlix_terms
        SET collection_frequency = collection_frequency - OLD.frequency + NEW.frequency
        WHERE term_id = NEW.term_id;
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger for automatic frequency updates
DROP TRIGGER IF EXISTS trigger_postings_frequencies ON semlix_postings;
CREATE TRIGGER trigger_postings_frequencies
    AFTER INSERT OR UPDATE OR DELETE ON semlix_postings
    FOR EACH ROW
    EXECUTE FUNCTION update_term_frequencies();

-- ============================================================================
-- HELPER VIEWS
-- ============================================================================

-- View for index statistics
CREATE OR REPLACE VIEW semlix_index_stats AS
SELECT
    d.index_name,
    COUNT(DISTINCT d.doc_id) as doc_count,
    COUNT(DISTINCT t.term_id) as unique_terms,
    COUNT(p.term_id) as total_postings,
    pg_size_pretty(pg_total_relation_size('semlix_documents')) as docs_size,
    pg_size_pretty(pg_total_relation_size('semlix_terms')) as terms_size,
    pg_size_pretty(pg_total_relation_size('semlix_postings')) as postings_size,
    m.generation,
    m.created_at,
    m.updated_at
FROM semlix_index_meta m
LEFT JOIN semlix_documents d ON d.index_name = m.index_name
LEFT JOIN semlix_terms t ON t.index_name = m.index_name
LEFT JOIN semlix_postings p ON p.index_name = m.index_name
GROUP BY d.index_name, m.generation, m.created_at, m.updated_at;

-- View for term statistics
CREATE OR REPLACE VIEW semlix_term_stats AS
SELECT
    index_name,
    field_name,
    term,
    doc_frequency,
    collection_frequency,
    CAST(collection_frequency AS FLOAT) / NULLIF(doc_frequency, 0) as avg_frequency_per_doc
FROM semlix_terms
ORDER BY doc_frequency DESC;

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Function to get BM25 score (simplified version)
CREATE OR REPLACE FUNCTION calculate_bm25_score(
    term_frequency INTEGER,
    doc_frequency INTEGER,
    total_docs INTEGER,
    doc_length INTEGER,
    avg_doc_length FLOAT,
    k1 FLOAT DEFAULT 1.2,
    b FLOAT DEFAULT 0.75
) RETURNS FLOAT AS $$
DECLARE
    idf FLOAT;
    normalized_tf FLOAT;
BEGIN
    -- IDF calculation
    idf := LN(1.0 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5));

    -- Normalized TF
    normalized_tf := (term_frequency * (k1 + 1.0)) /
                     (term_frequency + k1 * (1.0 - b + b * (doc_length / avg_doc_length)));

    RETURN idf * normalized_tf;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- NOTES FOR IMPLEMENTATION
-- ============================================================================

-- 1. SCHEMA STORAGE:
--    The schema is stored as JSONB in semlix_index_meta.schema_data
--    Format: {"fields": {"field_name": {"type": "TEXT", "stored": true, ...}}}
--
-- 2. STORED FIELDS:
--    All stored fields are in semlix_documents.stored_fields as JSONB
--    This allows flexible schema without ALTER TABLE operations
--
-- 3. FIELD LENGTHS:
--    Stored in semlix_documents.field_lengths for BM25 scoring
--    Format: {"content": 150, "title": 5}
--
-- 4. TERM POSITIONS:
--    Optional INTEGER[] array in semlix_postings.positions
--    Used for phrase queries and highlighting
--
-- 5. DOCUMENT FREQUENCY:
--    Automatically maintained by triggers on semlix_postings
--    Ensures consistency without manual updates
--
-- 6. INDEXING STRATEGY:
--    - Use GIN for stored_fields (JSON queries)
--    - Use B-tree for term lookups
--    - Consider partitioning for large datasets
--
-- 7. PERFORMANCE TUNING:
--    - VACUUM ANALYZE after bulk inserts
--    - Monitor pg_stat_user_tables for bloat
--    - Consider table partitioning by index_name for multi-tenant

-- ============================================================================
-- EXAMPLE QUERIES
-- ============================================================================

-- Search for documents containing a term
-- SELECT d.doc_id, d.stored_fields, p.frequency
-- FROM semlix_postings p
-- JOIN semlix_documents d ON d.index_name = p.index_name AND d.doc_id = p.doc_id
-- JOIN semlix_terms t ON t.term_id = p.term_id
-- WHERE t.index_name = 'MAIN'
--   AND t.field_name = 'content'
--   AND t.term = 'python'
-- ORDER BY p.frequency DESC;

-- Get index statistics
-- SELECT * FROM semlix_index_stats WHERE index_name = 'MAIN';

-- Get most common terms
-- SELECT * FROM semlix_term_stats
-- WHERE index_name = 'MAIN' AND field_name = 'content'
-- LIMIT 20;
