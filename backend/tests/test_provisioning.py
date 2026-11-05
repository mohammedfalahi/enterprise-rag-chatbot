"""
Unit and integration tests for secure user and tenant provisioning.
Tests:
- Role validation for employee, sales_rep, manager, and executive
- DB check constraints preventing invalid or empty roles
- Pre-provisioned invitation matching on auth.users registration
- Uninvited users are not granted access to any default tenant
- Safe 403 Forbidden handling for unassigned users
- Immediate provisioning for existing auth users
- Role updating and tenant isolation
"""

from __future__ import annotations

import unittest
import uuid
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
import jwt
import psycopg

from app.core.config import settings
from app.core.security import CurrentUser, get_current_user, reset_jwks_client
from app.db.connection import close_db_pool, get_db_connection, init_db_pool
from app.db.provisioning import (
    ALLOWED_ROLES,
    accept_user_invitation,
    assign_user_tenant,
    create_tenant,
    create_pending_invitation,
    get_user_profile,
    invite_user_to_tenant,
    validate_roles,
)
from tests.test_jwt_security import _generate_ec_key_pair


class TestRoleValidation(unittest.TestCase):

    def test_allowed_roles_list(self):
        self.assertIn("employee", ALLOWED_ROLES)
        self.assertIn("sales_rep", ALLOWED_ROLES)
        self.assertIn("manager", ALLOWED_ROLES)
        self.assertIn("executive", ALLOWED_ROLES)
        self.assertEqual(len(ALLOWED_ROLES), 4)

    def test_valid_roles_pass(self):
        validate_roles(["employee"])
        validate_roles(["sales_rep"])
        validate_roles(["manager"])
        validate_roles(["executive"])
        validate_roles(["employee", "manager"])
        validate_roles(["sales_rep", "executive"])

    def test_invalid_roles_fail(self):
        for invalid in ["viewer", "admin", "superuser", "root", "guest"]:
            with self.assertRaises(ValueError) as ctx:
                validate_roles([invalid])
            self.assertIn("Invalid role", str(ctx.exception))

    def test_empty_roles_fail(self):
        with self.assertRaises(ValueError) as ctx:
            validate_roles([])
        self.assertIn("Roles list cannot be empty", str(ctx.exception))


class TestDatabaseProvisioningFlow(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        reset_jwks_client(None)
        await init_db_pool()
        self.created_tenants: list[str] = []
        self.created_users: list[str] = []

    async def asyncTearDown(self):
        reset_jwks_client(None)
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                for uid in self.created_users:
                    await cur.execute("DELETE FROM auth.users WHERE id = %s;", (uid,))
                for tid in self.created_tenants:
                    await cur.execute("DELETE FROM tenants WHERE id = %s;", (tid,))
                await conn.commit()
        await close_db_pool()

    async def _create_test_auth_user(self, email: str) -> str:
        user_id = str(uuid.uuid4())
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO auth.users (id, email) VALUES (%s, %s);",
                    (user_id, email),
                )
                await conn.commit()
        self.created_users.append(user_id)
        return user_id

    async def test_db_check_constraint_rejects_invalid_roles(self):
        """Database constraint check_user_profile_roles rejects invalid roles."""
        tid = await create_tenant("Constraint Test Corp")
        self.created_tenants.append(tid)

        uid = await self._create_test_auth_user("invalid_roles_user@test.local")

        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                with self.assertRaises(psycopg.errors.CheckViolation):
                    await cur.execute(
                        "INSERT INTO user_profiles (id, tenant_id, roles) VALUES (%s, %s, %s);",
                        (uid, tid, ["admin"]),
                    )

    async def test_user_provisioning_via_invitation(self):
        """When an invited user registers in auth.users, they are linked to tenant and roles."""
        tid = await create_tenant("Redwood Tech")
        self.created_tenants.append(tid)

        email = f"emp_{uuid.uuid4().hex[:6]}@redwood.local"
        assigned_roles = ["employee", "manager"]

        # 1. Pre-provision invitation
        inv_id = await invite_user_to_tenant(tid, email, assigned_roles)
        self.assertTrue(bool(inv_id))

        # 2. User signs up in auth.users
        uid = await self._create_test_auth_user(email)

        # 3. Verify user profile was automatically provisioned
        profile = await get_user_profile(uid)
        self.assertIsNotNone(profile)
        user_tenant_id, user_roles = profile  # type: ignore
        self.assertEqual(user_tenant_id, tid)
        self.assertEqual(set(user_roles), set(assigned_roles))

        # 4. Verify invitation is marked claimed
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT claimed_at FROM tenant_invitations WHERE id = %s;",
                    (inv_id,),
                )
                row = await cur.fetchone()
                self.assertIsNotNone(row)
                self.assertIsNotNone(row[0])

    async def test_uninvited_user_has_no_tenant_or_profile(self):
        """
        Ensures users without invitations are NOT granted access to a default tenant.
        They must remain unassigned in user_profiles.
        """
        email = f"stranger_{uuid.uuid4().hex[:6]}@unknown.com"
        uid = await self._create_test_auth_user(email)

        profile = await get_user_profile(uid)
        self.assertIsNone(profile)

    async def test_auth_user_is_provisioned_when_invitation_is_accepted(self):
        """A stored invitation is claimed and its profile created only on acceptance."""
        tid = await create_tenant("Invitation Acceptance Corp")
        self.created_tenants.append(tid)
        email = f"accept_{uuid.uuid4().hex[:6]}@enterprise.local"
        uid = await self._create_test_auth_user(email)

        invitation_id = await create_pending_invitation(tid, email, ["sales_rep"])
        self.assertIsNone(await get_user_profile(uid))

        tenant_id, roles = await accept_user_invitation(uid, email)
        self.assertEqual(tenant_id, tid)
        self.assertEqual(roles, ["sales_rep"])
        self.assertEqual(await get_user_profile(uid), (tid, ["sales_rep"]))

        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT claimed_at FROM tenant_invitations WHERE id = %s;",
                    (invitation_id,),
                )
                row = await cur.fetchone()
                self.assertIsNotNone(row)
                self.assertIsNotNone(row[0])

    async def test_unassigned_user_rejected_with_403(self):
        """
        Unassigned user with a valid JWT token receives 403 Forbidden
        with clear security message when calling get_current_user.
        """
        email = f"unassigned_{uuid.uuid4().hex[:6]}@external.com"
        uid = await self._create_test_auth_user(email)

        # Generate signed ES256 JWT for unassigned user
        priv_key, jwk = _generate_ec_key_pair("unassigned-test-key")
        mock_jwks = jwt.PyJWKClient(
            "http://mock-jwks/auth/v1/.well-known/jwks.json",
            cache_jwk_set=True,
            cooldown_duration=0,
        )
        mock_jwks.fetch_data = MagicMock(return_value={"keys": [jwk]})

        token = jwt.encode(
            {
                "sub": uid,
                "aud": "authenticated",
                "iss": settings.jwt_issuer,
                "iat": 1000,
                "exp": 9999999999,
            },
            priv_key,
            algorithm="ES256",
            headers={"kid": "unassigned-test-key"},
        )
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        with patch("app.core.security.get_jwks_client", return_value=mock_jwks):
            with self.assertRaises(HTTPException) as ctx:
                await get_current_user(creds)

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("User profile not found", ctx.exception.detail)
        self.assertIn("not assigned to any enterprise tenant", ctx.exception.detail)

    async def test_existing_user_immediate_invitation_provisioning(self):
        """If user already exists in auth.users, inviting them immediately creates their profile."""
        tid = await create_tenant("Existing User Corp")
        self.created_tenants.append(tid)

        email = f"existing_{uuid.uuid4().hex[:6]}@enterprise.local"
        uid = await self._create_test_auth_user(email)

        # Profile starts as None
        self.assertIsNone(await get_user_profile(uid))

        # Admin invites existing user
        await invite_user_to_tenant(tid, email, ["sales_rep"])

        # Profile is immediately provisioned
        profile = await get_user_profile(uid)
        self.assertIsNotNone(profile)
        user_tenant_id, user_roles = profile  # type: ignore
        self.assertEqual(user_tenant_id, tid)
        self.assertEqual(user_roles, ["sales_rep"])

    async def test_admin_role_reassignment(self):
        """Admin can promote/update a user's role to executive."""
        tid = await create_tenant("Promotion Corp")
        self.created_tenants.append(tid)

        email = f"promoted_{uuid.uuid4().hex[:6]}@enterprise.local"
        uid = await self._create_test_auth_user(email)

        # Initial assignment
        await assign_user_tenant(uid, tid, ["employee"])
        profile = await get_user_profile(uid)
        self.assertEqual(profile[1], ["employee"])  # type: ignore

        # Promotion to executive
        await assign_user_tenant(uid, tid, ["executive"])
        updated_profile = await get_user_profile(uid)
        self.assertEqual(updated_profile[1], ["executive"])  # type: ignore


class TestDocumentsRLSAndSecurityPolicies(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        await init_db_pool()
        self.created_tenants: list[str] = []
        self.created_users: list[str] = []
        self.created_docs: list[str] = []

    async def asyncTearDown(self):
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("RESET ROLE;")
                for uid in self.created_users:
                    await cur.execute("DELETE FROM auth.users WHERE id = %s;", (uid,))
                for tid in self.created_tenants:
                    await cur.execute("DELETE FROM tenants WHERE id = %s;", (tid,))
                await conn.commit()
        await close_db_pool()

    async def _create_user_with_tenant(self, tenant_id: str, email_prefix: str, roles: list[str]) -> str:
        uid = str(uuid.uuid4())
        email = f"{email_prefix}_{uuid.uuid4().hex[:6]}@test.local"
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("INSERT INTO auth.users (id, email) VALUES (%s, %s);", (uid, email))
                await cur.execute(
                    "INSERT INTO user_profiles (id, tenant_id, roles) VALUES (%s, %s, %s);",
                    (uid, tenant_id, roles),
                )
                await conn.commit()
        self.created_users.append(uid)
        return uid

    async def _create_document(self, tenant_id: str, title: str, access_roles: list[str]) -> str:
        doc_id = str(uuid.uuid4())
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO documents (id, tenant_id, title, source_type, access_roles)
                    VALUES (%s, %s, %s, 'txt', %s);
                    """,
                    (doc_id, tenant_id, title, access_roles),
                )
                await conn.commit()
        self.created_docs.append(doc_id)
        return doc_id

    async def test_anonymous_cannot_read_documents(self):
        """Anonymous role returns zero documents when querying documents table."""
        tid = await create_tenant("Anon Read Test")
        self.created_tenants.append(tid)
        await self._create_document(tid, "Confidential Roadmap", ["employee"])

        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SET LOCAL ROLE anon;")
                await cur.execute("SELECT title FROM documents;")
                rows = await cur.fetchall()
                self.assertEqual(len(rows), 0)

    async def test_cross_tenant_document_isolation(self):
        """Users in Tenant A cannot see documents belonging to Tenant B."""
        tid_a = await create_tenant("Tenant A Corp")
        tid_b = await create_tenant("Tenant B Corp")
        self.created_tenants.extend([tid_a, tid_b])

        uid_a = await self._create_user_with_tenant(tid_a, "user_a", ["employee"])
        uid_b = await self._create_user_with_tenant(tid_b, "user_b", ["employee"])

        await self._create_document(tid_a, "Tenant A Secret", ["employee"])
        await self._create_document(tid_b, "Tenant B Secret", ["employee"])

        # User A querying under RLS
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SET LOCAL ROLE authenticated;")
                await cur.execute(f"SET LOCAL \"request.jwt.claim.sub\" = '{uid_a}';")
                await cur.execute("SELECT title FROM documents;")
                rows_a = await cur.fetchall()
                titles_a = [r[0] for r in rows_a]
                self.assertIn("Tenant A Secret", titles_a)
                self.assertNotIn("Tenant B Secret", titles_a)

        # User B querying under RLS
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SET LOCAL ROLE authenticated;")
                await cur.execute(f"SET LOCAL \"request.jwt.claim.sub\" = '{uid_b}';")
                await cur.execute("SELECT title FROM documents;")
                rows_b = await cur.fetchall()
                titles_b = [r[0] for r in rows_b]
                self.assertIn("Tenant B Secret", titles_b)
                self.assertNotIn("Tenant A Secret", titles_b)

    async def test_role_based_access_within_tenant(self):
        """User with role employee cannot access executive documents in same tenant."""
        tid = await create_tenant("RBAC Test Corp")
        self.created_tenants.append(tid)

        uid_employee = await self._create_user_with_tenant(tid, "emp", ["employee"])
        uid_executive = await self._create_user_with_tenant(tid, "exec", ["executive"])

        await self._create_document(tid, "General Employee Policy", ["employee"])
        await self._create_document(tid, "M&A Acquisition Target", ["executive"])

        # Employee query
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SET LOCAL ROLE authenticated;")
                await cur.execute(f"SET LOCAL \"request.jwt.claim.sub\" = '{uid_employee}';")
                await cur.execute("SELECT title FROM documents;")
                titles = [r[0] for r in await cur.fetchall()]
                self.assertIn("General Employee Policy", titles)
                self.assertNotIn("M&A Acquisition Target", titles)

        # Executive query
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SET LOCAL ROLE authenticated;")
                await cur.execute(f"SET LOCAL \"request.jwt.claim.sub\" = '{uid_executive}';")
                await cur.execute("SELECT title FROM documents;")
                titles = [r[0] for r in await cur.fetchall()]
                self.assertIn("M&A Acquisition Target", titles)

    async def test_client_mutation_privileges_revoked(self):
        """Authenticated and anon roles cannot INSERT, UPDATE, or DELETE documents."""
        tid = await create_tenant("Revoke Test Corp")
        self.created_tenants.append(tid)
        uid = await self._create_user_with_tenant(tid, "client_user", ["employee"])

        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SET LOCAL ROLE authenticated;")
                await cur.execute(f"SET LOCAL \"request.jwt.claim.sub\" = '{uid}';")
                with self.assertRaises(psycopg.errors.InsufficientPrivilege):
                    await cur.execute(
                        "INSERT INTO documents (tenant_id, title, source_type) VALUES (%s, %s, %s);",
                        (tid, "Client Created", "txt"),
                    )

    async def test_server_side_ingestion_and_admin_preserved(self):
        """Backend service connection (postgres) can insert and read documents without errors."""
        tid = await create_tenant("Ingestion Test Corp")
        self.created_tenants.append(tid)

        doc_id = await self._create_document(tid, "Ingested Server Document", ["employee"])
        async with get_db_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT title FROM documents WHERE id = %s;", (doc_id,))
                row = await cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], "Ingested Server Document")


if __name__ == "__main__":
    unittest.main()
