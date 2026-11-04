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
