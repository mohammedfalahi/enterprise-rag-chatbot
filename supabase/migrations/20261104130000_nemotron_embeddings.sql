-- Migration: 005_nemotron_embeddings.sql
-- Description: Wipes existing 768-dimensional document chunks and alters embedding column to halfvec(2048) for NVIDIA Nemotron 3.5 Embed 1B.

-- 1. Drop existing HNSW index
DROP INDEX IF EXISTS public.idx_chunks_embedding;

-- 2. Clear all existing chunks to prevent dimension mismatch errors during ALTER COLUMN
DELETE FROM public.document_chunks;

-- 3. Alter the embedding column to halfvec(2048)
ALTER TABLE public.document_chunks
ALTER COLUMN embedding TYPE halfvec(2048);

-- 4. Recreate the HNSW index using halfvec_cosine_ops
CREATE INDEX idx_chunks_embedding ON public.document_chunks USING hnsw (embedding halfvec_cosine_ops);
