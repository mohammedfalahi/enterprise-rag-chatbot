from __future__ import annotations

import logging
from typing import Any, List
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.core.config import settings
from app.db.connection import get_db_connection

logger = logging.getLogger("app.core.security")

security = HTTPBearer()


class CurrentUser(BaseModel):
    id: str
    tenant_id: str
    roles: List[str]


class AuthenticatedUser(BaseModel):
    id: str
    email: str | None = None


# Global JWKS client singleton with in-memory caching
_jwks_client: jwt.PyJWKClient | None = None


def get_jwks_client() -> jwt.PyJWKClient:
    """
    Returns the cached PyJWKClient instance for the Supabase JWKS endpoint.
    Configured with JWK set caching and automatic refresh on unknown kid.
    """
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = jwt.PyJWKClient(
            uri=settings.jwks_url,
            cache_jwk_set=True,
            lifespan=300,  # Cache JWKS for 5 minutes
            cooldown_duration=10,  # Cooldown between refreshes on unknown kid
        )
    return _jwks_client


def reset_jwks_client(client: jwt.PyJWKClient | None = None) -> None:
    """Reset or override the global PyJWKClient (primarily for testing)."""
    global _jwks_client
    _jwks_client = client


async def get_authenticated_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AuthenticatedUser:
    """Returns the identity from a verified Supabase JWT without requiring a profile."""
    payload = verify_supabase_jwt(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing subject (sub)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    email = payload.get("email")
    return AuthenticatedUser(
        id=str(user_id),
        email=email if isinstance(email, str) else None,
    )


def verify_supabase_jwt(
    token: str,
    jwks_client: jwt.PyJWKClient | None = None,
) -> dict[str, Any]:
    """
    Verifies a Supabase JWT token using the JWKS endpoint and ES256 algorithm.
    Validates token signature, expiration (exp), audience (aud), and issuer (iss).

    Raises:
        HTTPException(401): If token is missing, malformed, expired, has an invalid signature,
                            audience, issuer, or unknown kid.
    """
    if not token or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 1. Validate header structure and algorithm
    try:
        unverified_header = jwt.get_unverified_header(token)
    except jwt.DecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Malformed or invalid token format: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unable to parse token header: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    alg = unverified_header.get("alg")
    if alg != "ES256":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token algorithm: expected ES256, got {alg}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    kid = unverified_header.get("kid")
    if not kid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token header missing key ID (kid)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2. Retrieve signing key from JWKS (with automatic cache refresh on unknown kid)
    client = jwks_client or get_jwks_client()
    try:
        signing_key = client.get_signing_key_from_jwt(token)
    except jwt.PyJWKClientError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unable to find signing key for token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"JWKS lookup failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to retrieve public key for token verification",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Decode and validate claims (signature, expiry, audience, issuer)
    try:
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            audience=settings.SUPABASE_JWT_AUDIENCE,
            issuer=settings.jwt_issuer,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": True,
                "verify_iss": True,
            },
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidAudienceError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidIssuerError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token issuer",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> CurrentUser:
    token = credentials.credentials
    payload = verify_supabase_jwt(token)

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing subject (sub)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Fetch tenant_id and roles from the database
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT tenant_id, roles FROM user_profiles WHERE id = %s",
                (user_id,),
            )
            row = await cur.fetchone()
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User profile not found. User is not assigned to any enterprise tenant.",
                )

            tenant_id, roles = row

    return CurrentUser(id=str(user_id), tenant_id=str(tenant_id), roles=roles)
