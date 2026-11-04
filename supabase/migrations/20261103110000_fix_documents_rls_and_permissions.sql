-- Migration: 20261103110000_fix_documents_rls_and_permissions.sql
-- Description: Fix critical security vulnerabilities: enable RLS on documents, enforce tenant/role isolation, and revoke client write privileges.

-- 1. Enable Row Level Security on documents table
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

-- 2. Add SELECT policy enforcing tenant isolation and RBAC on documents
DROP POLICY IF EXISTS "Tenant and Role Isolation for Documents" ON public.documents;
CREATE POLICY "Tenant and Role Isolation for Documents" ON public.documents
FOR SELECT USING (
    tenant_id = (SELECT tenant_id FROM public.user_profiles WHERE id = auth.uid())
    AND
    access_roles && (SELECT roles FROM public.user_profiles WHERE id = auth.uid())
);

-- 3. Revoke all mutation permissions (INSERT, UPDATE, DELETE, TRUNCATE) from anon and authenticated
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.documents FROM anon, authenticated;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.document_chunks FROM anon, authenticated;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.tenants FROM anon, authenticated;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.user_profiles FROM anon, authenticated;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.tenant_invitations FROM anon, authenticated;

-- Also revoke SELECT on tenant_invitations from anon and authenticated (administrative only)
REVOKE ALL ON public.tenant_invitations FROM anon, authenticated;

-- 4. Ensure legitimate server-side administrative and ingestion roles retain full access
GRANT ALL PRIVILEGES ON public.documents TO postgres, service_role;
GRANT ALL PRIVILEGES ON public.document_chunks TO postgres, service_role;
GRANT ALL PRIVILEGES ON public.tenants TO postgres, service_role;
GRANT ALL PRIVILEGES ON public.user_profiles TO postgres, service_role;
GRANT ALL PRIVILEGES ON public.tenant_invitations TO postgres, service_role;

-- Allow authenticated users to SELECT documents, document_chunks, tenants, and user_profiles subject to RLS
GRANT SELECT ON public.documents TO authenticated;
GRANT SELECT ON public.document_chunks TO authenticated;
GRANT SELECT ON public.tenants TO authenticated;
GRANT SELECT ON public.user_profiles TO authenticated;
