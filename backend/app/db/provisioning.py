"""
Secure Tenant and User Provisioning Module.
Manages enterprise tenant isolation, role validation, invitations, and user-to-tenant mapping.

Security Guarantees:
- Validates roles against the enterprise allowed list: employee, sales_rep, manager, executive.
- Rejects empty roles, invalid roles, or client-side tampering.
- Ensures users without an assigned profile cannot access tenant data.
"""

from __future__ import annotations

import uuid
from typing import Any
import psycopg

from app.db.connection import get_db_connection

ALLOWED_ROLES: list[str] = ["employee", "sales_rep", "manager", "executive"]


class PendingInvitationExists(Exception):
    """Raised when an unclaimed invitation already exists for a tenant and email."""


class InvitationNotFound(Exception):
    """Raised when no pending invitation matches the authenticated user's email."""


class InvitationEmailMismatch(Exception):
    """Raised when the authenticated user's email does not match their Auth account."""


class InvitationProfileConflict(Exception):
    """Raised when accepting an invitation would change an existing user profile."""


def validate_roles(roles: list[str]) -> None:
    """
    Validates that roles is non-empty and every role is an allowed enterprise role.
    Raises ValueError on validation failure.
    """
    if not roles:
        raise ValueError("Roles list cannot be empty.")
    
    invalid = [r for r in roles if r not in ALLOWED_ROLES]
    if invalid:
        raise ValueError(
            f"Invalid role(s): {invalid}. Allowed roles are: {ALLOWED_ROLES}"
        )


async def create_tenant(name: str, tenant_id: str | None = None) -> str:
    """Create a new tenant with a given name and optional deterministic UUID."""
    tid = tenant_id or str(uuid.uuid4())
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO tenants (id, name)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name
                RETURNING id;
                """,
                (tid, name.strip()),
            )
            row = await cur.fetchone()
            await conn.commit()
            return str(row[0])


async def invite_user_to_tenant(
    tenant_id: str,
    email: str,
    roles: list[str],
) -> str:
    """
    Pre-provisions an invitation for a user by email.
    If the user has already signed up in auth.users, immediately provisions user_profiles.
    """
    clean_email = email.strip().lower()
    validate_roles(roles)

    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT public.invite_user_to_tenant(%s::uuid, %s, %s::text[]);
                """,
                (tenant_id, clean_email, roles),
            )
            row = await cur.fetchone()
            await conn.commit()
            return str(row[0])


async def has_pending_invitation(tenant_id: str, email: str) -> bool:
    """Checks whether this tenant already has an unclaimed invitation for the email."""
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT 1 FROM tenant_invitations
                WHERE tenant_id = %s::uuid
                  AND LOWER(email) = LOWER(%s)
                  AND claimed_at IS NULL
                LIMIT 1;
                """,
                (tenant_id, email),
            )
            return await cur.fetchone() is not None


async def create_pending_invitation(
    tenant_id: str,
    email: str,
    roles: list[str],
) -> str:
    """Stores a pending invitation without provisioning an Auth user prematurely."""
    validate_roles(roles)
    clean_email = email.strip().lower()

    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO tenant_invitations (tenant_id, email, roles)
                VALUES (%s::uuid, %s, %s::text[])
                ON CONFLICT (email, tenant_id) DO NOTHING
                RETURNING id;
                """,
                (tenant_id, clean_email, roles),
            )
            row = await cur.fetchone()
            if not row:
                raise PendingInvitationExists
            await conn.commit()
            return str(row[0])


async def accept_user_invitation(user_id: str, email: str) -> tuple[str, list[str]]:
    """Creates the invited user's profile and claims the invitation atomically."""
    clean_email = email.strip().lower()
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT email FROM auth.users WHERE id = %s::uuid;",
                (user_id,),
            )
            auth_user = await cur.fetchone()
            if not auth_user or (auth_user[0] or "").strip().lower() != clean_email:
                raise InvitationEmailMismatch

            await cur.execute(
                """
                SELECT id, tenant_id, roles
                FROM tenant_invitations
                WHERE LOWER(email) = LOWER(%s)
                  AND claimed_at IS NULL
                ORDER BY created_at ASC
                LIMIT 1
                FOR UPDATE;
                """,
                (clean_email,),
            )
            invitation = await cur.fetchone()
            if not invitation:
                raise InvitationNotFound

            invitation_id, tenant_id, roles = invitation
            await cur.execute(
                """
                SELECT tenant_id, roles
                FROM user_profiles
                WHERE id = %s::uuid
                FOR UPDATE;
                """,
                (user_id,),
            )
            profile = await cur.fetchone()
            if profile:
                if str(profile[0]) != str(tenant_id) or list(profile[1]) != list(roles):
                    raise InvitationProfileConflict
            else:
                await cur.execute(
                    """
                    INSERT INTO user_profiles (id, tenant_id, roles)
                    VALUES (%s::uuid, %s::uuid, %s::text[]);
                    """,
                    (user_id, tenant_id, roles),
                )

            await cur.execute(
                """
                UPDATE tenant_invitations
                SET claimed_at = NOW()
                WHERE id = %s::uuid AND claimed_at IS NULL;
                """,
                (invitation_id,),
            )
            await conn.commit()
            return str(tenant_id), list(roles)


async def assign_user_tenant(
    user_id: str,
    tenant_id: str,
    roles: list[str],
) -> None:
    """Directly assign or update a registered user's tenant and roles."""
    validate_roles(roles)

    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT public.assign_user_tenant(%s::uuid, %s::uuid, %s::text[]);
                """,
                (user_id, tenant_id, roles),
            )
            await conn.commit()


async def get_user_profile(user_id: str) -> tuple[str, list[str]] | None:
    """
    Fetches the tenant_id and roles for a user.
    Returns (tenant_id, roles) or None if the user has no assigned profile.
    """
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT tenant_id, roles FROM user_profiles
                WHERE id = %s;
                """,
                (user_id,),
            )
            row = await cur.fetchone()
            if not row:
                return None
            tenant_id, roles = row
            return (str(tenant_id), list(roles))
