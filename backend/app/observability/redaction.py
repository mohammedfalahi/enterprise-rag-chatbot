"""
Sensitive data scrubbing and redaction for tracing and logging.
Prevents credential, API key, auth header, and token leakage into Langfuse traces.
Reused from DEV_AI (packages/observability/redaction.py).
"""

from __future__ import annotations

import re
from typing import Any

# Sensitive dictionary keys (case-insensitive substring matches)
SENSITIVE_KEY_PATTERNS = {
    "authorization",
    "auth",
    "api_key",
    "apikey",
    "x-api-key",
    "secret",
    "token",
    "password",
    "passwd",
    "private_key",
    "client_secret",
    "hmac_secret",
    "cookie",
    "set-cookie",
    "bearer",
    "access_token",
    "refresh_token",
}

# Regex patterns for values that look like credentials or tokens
BEARER_PATTERN = re.compile(r"Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*", re.IGNORECASE)
JWT_PATTERN = re.compile(r"eyJ[A-Za-z0-9-_=]+\.eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_.+/=]+")
API_KEY_PATTERN = re.compile(r"(?:sk|pk)-(?:lf-)?[A-Za-z0-9_\-]{16,}")
GEMINI_KEY_PATTERN = re.compile(r"AIzaSy[A-Za-z0-9_\-]{33}")

REDACTED_PLACEHOLDER = "[REDACTED]"


def is_sensitive_key(key: str) -> bool:
    """Check if key name matches sensitive patterns."""
    normalized = key.lower().replace("-", "_")
    for pattern in SENSITIVE_KEY_PATTERNS:
        if pattern in normalized:
            return True
    return False


def scrub_string(value: str) -> str:
    """Scrub sensitive credential patterns from string content."""
    val = BEARER_PATTERN.sub("Bearer [REDACTED]", value)
    val = JWT_PATTERN.sub("[REDACTED_JWT]", val)
    val = API_KEY_PATTERN.sub("[REDACTED_API_KEY]", val)
    val = GEMINI_KEY_PATTERN.sub("[REDACTED_KEY]", val)
    return val


def scrub_sensitive_data(obj: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """
    Recursively scrubs sensitive keys and values from dictionaries, lists, and primitives.
    Safe for complex nested structures.
    """
    if depth > max_depth:
        return obj

    if isinstance(obj, dict):
        for name_field in ("name", "key", "header"):
            if (
                name_field in obj
                and "value" in obj
                and isinstance(obj[name_field], str)
                and is_sensitive_key(obj[name_field])
            ):
                scrubbed = {}
                for k, v in obj.items():
                    if k == "value":
                        scrubbed[k] = REDACTED_PLACEHOLDER
                    else:
                        scrubbed[k] = scrub_sensitive_data(v, depth + 1, max_depth)
                return scrubbed

        scrubbed = {}
        for k, v in obj.items():
            key_str = str(k)
            if is_sensitive_key(key_str):
                scrubbed[key_str] = REDACTED_PLACEHOLDER
            else:
                scrubbed[key_str] = scrub_sensitive_data(v, depth + 1, max_depth)
        return scrubbed

    if isinstance(obj, list):
        return [scrub_sensitive_data(item, depth + 1, max_depth) for item in obj]

    if isinstance(obj, tuple):
        return tuple(scrub_sensitive_data(item, depth + 1, max_depth) for item in obj)

    if isinstance(obj, set):
        return {scrub_sensitive_data(item, depth + 1, max_depth) for item in obj}

    if isinstance(obj, str):
        return scrub_string(obj)

    return obj
