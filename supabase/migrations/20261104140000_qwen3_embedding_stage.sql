-- Stage local Qwen3 embeddings without replacing the active vector column.
CREATE SCHEMA IF NOT EXISTS private AUTHORIZATION postgres;
REVOKE ALL ON SCHEMA private FROM PUBLIC, anon, authenticated, service_role;

ALTER TABLE public.document_chunks
    ADD COLUMN IF NOT EXISTS embedding_qwen vector(1024);

COMMENT ON COLUMN public.document_chunks.embedding_qwen IS
    'Staged local qwen3-embedding:0.6b vectors; only cut over after full dimension and checkpoint validation.';

CREATE TABLE IF NOT EXISTS private.embedding_migration_checkpoints (
    tenant_id UUID NOT NULL,
    document_id UUID NOT NULL,
    source_hash CHAR(64) NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL CHECK (embedding_dim = 1024),
    expected_chunk_count INTEGER NOT NULL CHECK (expected_chunk_count >= 0),
    completed_chunk_count INTEGER NOT NULL DEFAULT 0
        CHECK (completed_chunk_count >= 0),
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'in_progress', 'failed', 'completed')),
    error_type TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, document_id),
    CHECK (completed_chunk_count <= expected_chunk_count)
);

ALTER TABLE private.embedding_migration_checkpoints ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON private.embedding_migration_checkpoints
    FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT, INSERT, UPDATE, DELETE
    ON private.embedding_migration_checkpoints TO postgres;
