CREATE TABLE public.evaluation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
    requested_by UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    sample_size INTEGER NOT NULL CHECK (sample_size BETWEEN 1 AND 320),
    mode TEXT NOT NULL CHECK (mode IN ('hybrid_rerank', 'hybrid_rrf', 'dense_only', 'sparse_only')),
    top_k INTEGER NOT NULL CHECK (top_k BETWEEN 1 AND 20),
    candidate_limit INTEGER NOT NULL CHECK (candidate_limit BETWEEN 1 AND 50),
    rerank_pool_size INTEGER NOT NULL CHECK (rerank_pool_size BETWEEN 1 AND 20),
    report JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    CHECK (
        (status = 'completed' AND report IS NOT NULL AND completed_at IS NOT NULL)
        OR status != 'completed'
    )
);

CREATE UNIQUE INDEX idx_evaluation_runs_one_active_per_tenant
    ON public.evaluation_runs (tenant_id)
    WHERE status IN ('pending', 'running');

CREATE INDEX idx_evaluation_runs_tenant_created
    ON public.evaluation_runs (tenant_id, created_at DESC);

ALTER TABLE public.evaluation_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Evaluation admins can view tenant runs" ON public.evaluation_runs
FOR SELECT USING (
    tenant_id = (SELECT tenant_id FROM public.user_profiles WHERE id = auth.uid())
    AND
    (SELECT roles FROM public.user_profiles WHERE id = auth.uid())
        && ARRAY['executive', 'admin', 'manager']::TEXT[]
);

REVOKE ALL ON public.evaluation_runs FROM anon;
GRANT ALL PRIVILEGES ON public.evaluation_runs TO postgres, service_role;
GRANT SELECT ON public.evaluation_runs TO authenticated;
