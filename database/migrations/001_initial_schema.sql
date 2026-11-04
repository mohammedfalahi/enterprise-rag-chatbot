-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Tenants Table
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL
);

-- User Profiles (Linked to Supabase Auth)
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    roles TEXT[] NOT NULL DEFAULT '{"viewer"}'
);

-- Documents Table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL, -- 'pdf', 'docx', 'txt'
    access_roles TEXT[] NOT NULL DEFAULT '{"viewer"}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Document Chunks (Denormalized tenant_id and roles for performant filtering/RLS)
CREATE TABLE document_chunks (
    id TEXT PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    access_roles TEXT[] NOT NULL DEFAULT '{"viewer"}',
    heading_path TEXT[] NOT NULL DEFAULT '{}',
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    embedding VECTOR(768),
    fts TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Indexes (Reused from dev-ai)
CREATE INDEX idx_chunks_tenant ON document_chunks(tenant_id);
CREATE INDEX idx_chunks_fts ON document_chunks USING gin(fts);
CREATE INDEX idx_chunks_embedding ON document_chunks USING hnsw(embedding vector_cosine_ops);

-- Row Level Security (Defense-in-depth)
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Tenant and Role Isolation for Chunks" ON document_chunks
FOR SELECT USING (
    tenant_id = (SELECT tenant_id FROM user_profiles WHERE id = auth.uid())
    AND
    access_roles && (SELECT roles FROM user_profiles WHERE id = auth.uid())
);
