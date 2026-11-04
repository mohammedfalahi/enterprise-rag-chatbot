-- Migration: 002_secure_provisioning.sql
-- Description: Enforce RLS on tenants/user_profiles, support enterprise roles (employee, sales_rep, manager, executive), and secure invitation-based provisioning.

-- 1. Allowed roles validation on user_profiles
ALTER TABLE user_profiles ALTER COLUMN roles SET DEFAULT '{"employee"}';

-- Add check constraint for valid enterprise roles
ALTER TABLE user_profiles
ADD CONSTRAINT check_user_profile_roles
CHECK (
    roles <@ ARRAY['employee', 'sales_rep', 'manager', 'executive']::TEXT[] 
    AND cardinality(roles) > 0
);

-- 2. Enable Row Level Security on user_profiles and tenants
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;

-- 3. RLS Policies on user_profiles
DROP POLICY IF EXISTS "Users can view their own profile" ON user_profiles;
CREATE POLICY "Users can view their own profile" ON user_profiles
FOR SELECT USING (id = auth.uid());

-- Authenticated users cannot INSERT, UPDATE, or DELETE user_profiles directly
-- (No policies granted for INSERT/UPDATE/DELETE to authenticated or anon)

-- 4. RLS Policies on tenants
DROP POLICY IF EXISTS "Users can view their own tenant" ON tenants;
CREATE POLICY "Users can view their own tenant" ON tenants
FOR SELECT USING (
    id IN (SELECT tenant_id FROM public.user_profiles WHERE id = auth.uid())
);

-- 5. Tenant Invitations Table (Pre-provisioning allowlist)
CREATE TABLE IF NOT EXISTS tenant_invitations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    roles TEXT[] NOT NULL DEFAULT '{"employee"}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_at TIMESTAMPTZ,
    CONSTRAINT check_invitation_roles CHECK (
        roles <@ ARRAY['employee', 'sales_rep', 'manager', 'executive']::TEXT[] 
        AND cardinality(roles) > 0
    ),
    CONSTRAINT unique_pending_invitation UNIQUE (email, tenant_id)
);

CREATE INDEX IF NOT EXISTS idx_invitations_email ON tenant_invitations(LOWER(email));

ALTER TABLE tenant_invitations ENABLE ROW LEVEL SECURITY;
-- No public policies on tenant_invitations; only service_role/postgres can read or write

-- 6. Trigger Function for Secure Provisioning on auth.users INSERT
CREATE OR REPLACE FUNCTION public.handle_new_user_provisioning()
RETURNS TRIGGER
SECURITY DEFINER
SET search_path = public, auth
LANGUAGE plpgsql
AS $$
DECLARE
    v_invitation RECORD;
BEGIN
    -- Search for an unclaimed invitation matching the new user's email
    SELECT id, tenant_id, roles
    INTO v_invitation
    FROM public.tenant_invitations
    WHERE LOWER(email) = LOWER(NEW.email)
      AND claimed_at IS NULL
    ORDER BY created_at ASC
    LIMIT 1;

    IF FOUND THEN
        -- Securely link the authenticated user to the invited tenant and roles
        INSERT INTO public.user_profiles (id, tenant_id, roles)
        VALUES (NEW.id, v_invitation.tenant_id, v_invitation.roles)
        ON CONFLICT (id) DO UPDATE SET
            tenant_id = EXCLUDED.tenant_id,
            roles = EXCLUDED.roles;

        -- Mark invitation as claimed
        UPDATE public.tenant_invitations
        SET claimed_at = NOW()
        WHERE id = v_invitation.id;
    END IF;

    -- If no invitation is found, do NOT assign any default tenant or profile.
    -- The user remains unassigned in user_profiles.
    RETURN NEW;
END;
$$;

-- Attach trigger to auth.users
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
AFTER INSERT ON auth.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_new_user_provisioning();

-- 7. Secure Administrative Functions
CREATE OR REPLACE FUNCTION public.invite_user_to_tenant(
    p_tenant_id UUID,
    p_email TEXT,
    p_roles TEXT[]
)
RETURNS UUID
SECURITY DEFINER
SET search_path = public, auth
LANGUAGE plpgsql
AS $$
DECLARE
    v_inv_id UUID;
    v_existing_user_id UUID;
BEGIN
    -- Validate roles
    IF NOT (p_roles <@ ARRAY['employee', 'sales_rep', 'manager', 'executive']::TEXT[] AND cardinality(p_roles) > 0) THEN
        RAISE EXCEPTION 'Invalid roles provided. Allowed roles are employee, sales_rep, manager, executive';
    END IF;

    -- Check if user is already registered in auth.users
    SELECT id INTO v_existing_user_id FROM auth.users WHERE LOWER(email) = LOWER(p_email) LIMIT 1;

    IF v_existing_user_id IS NOT NULL THEN
        -- Immediately provision user_profiles
        INSERT INTO public.user_profiles (id, tenant_id, roles)
        VALUES (v_existing_user_id, p_tenant_id, p_roles)
        ON CONFLICT (id) DO UPDATE SET
            tenant_id = EXCLUDED.tenant_id,
            roles = EXCLUDED.roles;
    END IF;

    -- Record or update in tenant_invitations
    INSERT INTO public.tenant_invitations (tenant_id, email, roles, claimed_at)
    VALUES (
        p_tenant_id,
        LOWER(p_email),
        p_roles,
        CASE WHEN v_existing_user_id IS NOT NULL THEN NOW() ELSE NULL END
    )
    ON CONFLICT (email, tenant_id) DO UPDATE SET
        roles = EXCLUDED.roles,
        claimed_at = CASE WHEN v_existing_user_id IS NOT NULL THEN NOW() ELSE NULL END
    RETURNING id INTO v_inv_id;

    RETURN v_inv_id;
END;
$$;

CREATE OR REPLACE FUNCTION public.assign_user_tenant(
    p_user_id UUID,
    p_tenant_id UUID,
    p_roles TEXT[]
)
RETURNS VOID
SECURITY DEFINER
SET search_path = public, auth
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT (p_roles <@ ARRAY['employee', 'sales_rep', 'manager', 'executive']::TEXT[] AND cardinality(p_roles) > 0) THEN
        RAISE EXCEPTION 'Invalid roles provided. Allowed roles are employee, sales_rep, manager, executive';
    END IF;

    INSERT INTO public.user_profiles (id, tenant_id, roles)
    VALUES (p_user_id, p_tenant_id, p_roles)
    ON CONFLICT (id) DO UPDATE SET
        tenant_id = EXCLUDED.tenant_id,
        roles = EXCLUDED.roles;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.invite_user_to_tenant(UUID, TEXT, TEXT[]) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.assign_user_tenant(UUID, UUID, TEXT[]) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.invite_user_to_tenant(UUID, TEXT, TEXT[]) TO service_role, postgres;
GRANT EXECUTE ON FUNCTION public.assign_user_tenant(UUID, UUID, TEXT[]) TO service_role, postgres;
