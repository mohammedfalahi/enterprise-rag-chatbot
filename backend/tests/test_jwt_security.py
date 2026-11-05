"""
Unit and integration tests for Supabase ES256 JWKS JWT authentication.
Covers:
- Valid ES256 token verification
- Invalid signature rejection
- Expired token rejection
- Unknown key ID rejection & JWKS refresh
- Invalid audience and issuer validation
- Malformed and unsupported algorithm handling
- FastAPI CurrentUser extraction and authorization flow (403 when profile not found)
- Live local Supabase JWKS endpoint validation
"""

from __future__ import annotations

import base64
import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPAuthorizationCredentials
import jwt

from app.core.config import settings
from app.core.security import (
    CurrentUser,
    get_authenticated_user,
    get_current_user,
    get_jwks_client,
    reset_jwks_client,
    verify_supabase_jwt,
)


def _generate_ec_key_pair(kid: str = "key-test-1"):
    """Helper to generate an EC P-256 key pair and corresponding JWK dict."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    pub_nums = public_key.public_numbers()

    def int_to_b64(val: int) -> str:
        b = val.to_bytes(32, byteorder="big")
        return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")

    jwk = {
        "kty": "EC",
        "crv": "P-256",
        "alg": "ES256",
        "use": "sig",
        "kid": kid,
        "x": int_to_b64(pub_nums.x),
        "y": int_to_b64(pub_nums.y),
    }
    return private_key, jwk


class TestSupabaseJWTAuthentication(unittest.TestCase):

    def setUp(self):
        reset_jwks_client(None)
        self.priv_key, self.jwk = _generate_ec_key_pair("test-key-primary")
        self.mock_client = jwt.PyJWKClient(
            "http://mock-jwks/auth/v1/.well-known/jwks.json",
            cache_jwk_set=True,
            cooldown_duration=0,
        )
        self.mock_client.fetch_data = MagicMock(return_value={"keys": [self.jwk]})

    def tearDown(self):
        reset_jwks_client(None)

    def _create_token(
        self,
        key=None,
        kid: str = "test-key-primary",
        sub: str = "11111111-2222-3333-4444-555555555555",
        aud: str = "authenticated",
        iss: str | None = None,
        exp: int | None = None,
        algorithm: str = "ES256",
        headers: dict | None = None,
    ) -> str:
        signing_key = key or self.priv_key
        now = int(time.time())
        payload = {
            "sub": sub,
            "aud": aud,
            "iss": iss or settings.jwt_issuer,
            "iat": now,
            "exp": exp if exp is not None else now + 3600,
            "email": "testuser@enterprise.local",
            "role": "authenticated",
        }
        token_headers = headers if headers is not None else {"kid": kid}
        return jwt.encode(payload, signing_key, algorithm=algorithm, headers=token_headers)

    def test_valid_token_verification(self):
        """Valid ES256 token signed by key in JWKS successfully verifies."""
        token = self._create_token(sub="user-abc-123")
        payload = verify_supabase_jwt(token, jwks_client=self.mock_client)
        self.assertEqual(payload["sub"], "user-abc-123")
        self.assertEqual(payload["aud"], "authenticated")
        self.assertEqual(payload["email"], "testuser@enterprise.local")

    def test_authenticated_user_extraction_uses_verified_claims(self):
        """Invitation acceptance gets the user ID and email from a verified JWT."""
        token = self._create_token(sub="user-abc-123")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token,
        )

        with patch("app.core.security.get_jwks_client", return_value=self.mock_client):
            user = asyncio.run(get_authenticated_user(credentials))

        self.assertEqual(user.id, "user-abc-123")
        self.assertEqual(user.email, "testuser@enterprise.local")

    def test_invalid_signature_rejection(self):
        """Token signed by an unlisted/different EC key raises 401."""
        other_priv_key, _ = _generate_ec_key_pair("test-key-primary")
        token = self._create_token(key=other_priv_key)

        with self.assertRaises(HTTPException) as ctx:
            verify_supabase_jwt(token, jwks_client=self.mock_client)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid token signature", ctx.exception.detail)

    def test_expired_token_rejection(self):
        """Expired token raises 401."""
        now = int(time.time())
        token = self._create_token(exp=now - 60)

        with self.assertRaises(HTTPException) as ctx:
            verify_supabase_jwt(token, jwks_client=self.mock_client)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Token has expired", ctx.exception.detail)

    def test_unknown_key_id_rejection(self):
        """Token with an unknown kid that is not found after JWKS refresh raises 401."""
        token = self._create_token(kid="non-existent-kid")

        with self.assertRaises(HTTPException) as ctx:
            verify_supabase_jwt(token, jwks_client=self.mock_client)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Unable to find signing key", ctx.exception.detail)

    def test_jwks_cache_and_refresh_on_unknown_kid(self):
        """
        When a token with a new key ID arrives, the client refreshes its JWKS cache
        from the endpoint and successfully verifies if the new key is present.
        """
        priv_key_2, jwk_2 = _generate_ec_key_pair("test-key-secondary")
        
        # Initial fetch returns only primary key; refresh returns both keys
        self.mock_client.fetch_data = MagicMock(
            side_effect=[{"keys": [self.jwk]}, {"keys": [self.jwk, jwk_2]}]
        )

        token_1 = self._create_token(kid="test-key-primary")
        token_2 = self._create_token(key=priv_key_2, kid="test-key-secondary", sub="user-2")

        # First token uses initial JWKS
        payload_1 = verify_supabase_jwt(token_1, jwks_client=self.mock_client)
        self.assertEqual(payload_1["sub"], "11111111-2222-3333-4444-555555555555")
        self.assertEqual(self.mock_client.fetch_data.call_count, 1)

        # Second token with unknown kid triggers refresh
        payload_2 = verify_supabase_jwt(token_2, jwks_client=self.mock_client)
        self.assertEqual(payload_2["sub"], "user-2")
        self.assertEqual(self.mock_client.fetch_data.call_count, 2)

    def test_invalid_audience_rejection(self):
        """Token with invalid audience raises 401."""
        token = self._create_token(aud="wrong_audience")

        with self.assertRaises(HTTPException) as ctx:
            verify_supabase_jwt(token, jwks_client=self.mock_client)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid token audience", ctx.exception.detail)

    def test_invalid_issuer_rejection(self):
        """Token with invalid issuer raises 401."""
        token = self._create_token(iss="http://malicious-issuer.com/auth/v1")

        with self.assertRaises(HTTPException) as ctx:
            verify_supabase_jwt(token, jwks_client=self.mock_client)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid token issuer", ctx.exception.detail)

    def test_invalid_algorithm_rejection(self):
        """Token signed with HS256 instead of ES256 raises 401."""
        now = int(time.time())
        token = jwt.encode(
            {"sub": "u1", "aud": "authenticated", "iss": settings.jwt_issuer, "exp": now + 100},
            "this-is-a-32-byte-secret-key-for-test!!",
            algorithm="HS256",
            headers={"kid": "test-key-primary"},
        )

        with self.assertRaises(HTTPException) as ctx:
            verify_supabase_jwt(token, jwks_client=self.mock_client)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("expected ES256", ctx.exception.detail)

    def test_missing_or_empty_token(self):
        """Empty or whitespace token raises 401."""
        for empty_token in ["", "   ", None]:
            with self.assertRaises(HTTPException) as ctx:
                verify_supabase_jwt(empty_token, jwks_client=self.mock_client)  # type: ignore
            self.assertEqual(ctx.exception.status_code, 401)

    def test_malformed_token_string(self):
        """Malformed token string raises 401."""
        with self.assertRaises(HTTPException) as ctx:
            verify_supabase_jwt("not.a.valid.jwt.string", jwks_client=self.mock_client)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_missing_kid_in_header(self):
        """Token header missing kid raises 401."""
        token = self._create_token(headers={})

        with self.assertRaises(HTTPException) as ctx:
            verify_supabase_jwt(token, jwks_client=self.mock_client)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Token header missing key ID (kid)", ctx.exception.detail)


class TestGetCurrentUserFlow(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        reset_jwks_client(None)
        self.priv_key, self.jwk = _generate_ec_key_pair("flow-key-1")
        self.mock_client = jwt.PyJWKClient(
            "http://mock-jwks/auth/v1/.well-known/jwks.json",
            cache_jwk_set=True,
        )
        self.mock_client.fetch_data = MagicMock(return_value={"keys": [self.jwk]})

    def tearDown(self):
        reset_jwks_client(None)

    def _create_token(self, sub: str) -> str:
        now = int(time.time())
        return jwt.encode(
            {
                "sub": sub,
                "aud": "authenticated",
                "iss": settings.jwt_issuer,
                "iat": now,
                "exp": now + 3600,
            },
            self.priv_key,
            algorithm="ES256",
            headers={"kid": "flow-key-1"},
        )

    async def test_get_current_user_success(self):
        """When token is valid and user_profiles record exists, CurrentUser is returned."""
        token = self._create_token(sub="00000000-0000-0000-0000-000000000001")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        mock_cursor = AsyncMock()
        mock_cursor.fetchone.return_value = ("tenant-alpha-uuid", ["employee", "manager"])
        mock_cursor_ctx = MagicMock()
        mock_cursor_ctx.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor_ctx
        mock_conn_ctx = MagicMock()
        mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("app.core.security.get_jwks_client", return_value=self.mock_client):
            with patch("app.core.security.get_db_connection", return_value=mock_conn_ctx):
                user: CurrentUser = await get_current_user(creds)

        self.assertEqual(user.id, "00000000-0000-0000-0000-000000000001")
        self.assertEqual(user.tenant_id, "tenant-alpha-uuid")
        self.assertEqual(user.roles, ["employee", "manager"])

    async def test_get_current_user_profile_not_found_raises_403(self):
        """When token is valid but user_profiles record does not exist, raises 403 Forbidden."""
        token = self._create_token(sub="00000000-0000-0000-0000-999999999999")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        mock_cursor = AsyncMock()
        mock_cursor.fetchone.return_value = None
        mock_cursor_ctx = MagicMock()
        mock_cursor_ctx.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor_ctx
        mock_conn_ctx = MagicMock()
        mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("app.core.security.get_jwks_client", return_value=self.mock_client):
            with patch("app.core.security.get_db_connection", return_value=mock_conn_ctx):
                with self.assertRaises(HTTPException) as ctx:
                    await get_current_user(creds)

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("User profile not found", ctx.exception.detail)


class TestLiveSupabaseJWKSEndpoint(unittest.TestCase):

    def test_live_supabase_jwks_connectivity(self):
        """Verifies local Supabase JWKS endpoint serves valid ES256 P-256 keys."""
        import urllib.request
        import json

        try:
            req = urllib.request.Request(settings.jwks_url)
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            self.skipTest(f"Local Supabase JWKS endpoint not reachable at {settings.jwks_url}: {e}")

        self.assertIn("keys", data)
        self.assertGreaterEqual(len(data["keys"]), 1)
        key = data["keys"][0]
        self.assertEqual(key.get("alg"), "ES256")
        self.assertEqual(key.get("crv"), "P-256")
        self.assertEqual(key.get("kty"), "EC")
        self.assertTrue(bool(key.get("kid")))


class TestFastAPIAuthEndpointIntegration(unittest.TestCase):

    def setUp(self):
        reset_jwks_client(None)
        self.priv_key, self.jwk = _generate_ec_key_pair("e2e-key-1")
        self.mock_client = jwt.PyJWKClient(
            "http://mock-jwks/auth/v1/.well-known/jwks.json",
            cache_jwk_set=True,
        )
        self.mock_client.fetch_data = MagicMock(return_value={"keys": [self.jwk]})

    def tearDown(self):
        reset_jwks_client(None)

    def test_endpoint_missing_token_returns_401(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        res = client.post("/api/chat", json={"message": "hello"})
        self.assertEqual(res.status_code, 401)

    def test_endpoint_malformed_token_returns_401(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        res = client.post(
            "/api/chat",
            headers={"Authorization": "Bearer not-a-valid-jwt"},
            json={"message": "hello"},
        )
        self.assertEqual(res.status_code, 401)

    def test_endpoint_expired_token_returns_401(self):
        from fastapi.testclient import TestClient
        from app.main import app

        now = int(time.time())
        token = jwt.encode(
            {"sub": "u1", "aud": "authenticated", "iss": settings.jwt_issuer, "exp": now - 30},
            self.priv_key,
            algorithm="ES256",
            headers={"kid": "e2e-key-1"},
        )
        client = TestClient(app)
        with patch("app.core.security.get_jwks_client", return_value=self.mock_client):
            res = client.post(
                "/api/chat",
                headers={"Authorization": f"Bearer {token}"},
                json={"message": "hello"},
            )
        self.assertEqual(res.status_code, 401)


if __name__ == "__main__":
    unittest.main()
