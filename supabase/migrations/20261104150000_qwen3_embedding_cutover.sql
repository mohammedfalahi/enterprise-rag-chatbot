DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = 'document_chunks'
          AND a.attname = 'embedding_qwen'
          AND format_type(a.atttypid, a.atttypmod) = 'vector(1024)'
          AND NOT a.attisdropped
    ) THEN
        RAISE EXCEPTION 'Qwen staging column public.document_chunks.embedding_qwen vector(1024) is required';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = 'document_chunks'
          AND a.attname = 'embedding'
          AND format_type(a.atttypid, a.atttypmod) = 'halfvec(2048)'
          AND NOT a.attisdropped
    ) THEN
        RAISE EXCEPTION 'Qwen cutover blocked: active column is not the audited halfvec(2048) schema';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM public.document_chunks
        WHERE embedding_qwen IS NULL
           OR vector_dims(embedding_qwen) <> 1024
    ) THEN
        RAISE EXCEPTION 'Qwen cutover blocked: staged embeddings are missing or have invalid dimensions';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM public.documents d
        LEFT JOIN private.embedding_migration_checkpoints cp
          ON cp.tenant_id = d.tenant_id
         AND cp.document_id = d.id
        LEFT JOIN (
            SELECT document_id, count(*) AS chunk_count
            FROM public.document_chunks
            GROUP BY document_id
        ) chunks ON chunks.document_id = d.id
        WHERE cp.status IS DISTINCT FROM 'completed'
           OR cp.embedding_model IS DISTINCT FROM 'qwen3-embedding:0.6b'
           OR cp.embedding_dim IS DISTINCT FROM 1024
           OR cp.expected_chunk_count IS DISTINCT FROM COALESCE(chunks.chunk_count, 0)
           OR cp.completed_chunk_count IS DISTINCT FROM COALESCE(chunks.chunk_count, 0)
    ) THEN
        RAISE EXCEPTION 'Qwen cutover blocked: one or more documents lack a complete matching checkpoint';
    END IF;
END;
$$;

DROP INDEX IF EXISTS public.idx_chunks_embedding;

ALTER TABLE public.document_chunks
    DROP COLUMN embedding;

ALTER TABLE public.document_chunks
    RENAME COLUMN embedding_qwen TO embedding;

CREATE INDEX idx_chunks_embedding
    ON public.document_chunks USING hnsw (embedding vector_cosine_ops);

COMMENT ON COLUMN public.document_chunks.embedding IS
    'Qwen3-Embedding-0.6B local Ollama embeddings with exactly 1024 dimensions.';
