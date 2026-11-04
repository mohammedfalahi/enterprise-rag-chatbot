"""
Unit tests for sensitive data scrubbing and redaction.
Covers:
- Sensitive key pattern matching
- Bearer token regex scrubbing
- JWT token regex scrubbing
- API key regex scrubbing (Gemini, Langfuse)
- Recursive dictionary, list, and tuple data scrubbing
- HTTP header name/value scrubbing
- Non-sensitive text and data preservation
"""

from __future__ import annotations

import unittest
from app.observability.redaction import (
    REDACTED_PLACEHOLDER,
    is_sensitive_key,
    scrub_sensitive_data,
    scrub_string,
)


class TestRedaction(unittest.TestCase):

    def test_is_sensitive_key(self):
        """Identifies sensitive key patterns case-insensitively with dashes or underscores."""
        sensitive_samples = [
            "authorization",
            "Authorization",
            "x-api-key",
            "api_key",
            "API_KEY",
            "client_secret",
            "hmac_secret",
            "password",
            "passwd",
            "access_token",
            "refresh_token",
            "private_key",
        ]
        for key in sensitive_samples:
            self.assertTrue(is_sensitive_key(key), f"Expected {key} to be sensitive")

        non_sensitive_samples = [
            "id",
            "user_id",
            "tenant_id",
            "title",
            "document_id",
            "content",
            "score",
            "heading_path",
            "created_at",
        ]
        for key in non_sensitive_samples:
            self.assertFalse(is_sensitive_key(key), f"Expected {key} to be non-sensitive")

    def test_scrub_bearer_token(self):
        """Bearer tokens in strings are replaced with Bearer [REDACTED]."""
        raw = "Header: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.xyz.123 in request"
        scrubbed = scrub_string(raw)
        self.assertNotIn("eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9", scrubbed)
        self.assertIn("Bearer [REDACTED]", scrubbed)

    def test_scrub_jwt_pattern(self):
        """Standalone JWT patterns are replaced with [REDACTED_JWT]."""
        raw = "Session token: eyJhbGciOiJFUzI1NiJ9.eyJzdWIiOiIxMjMifQ.abc_123-def"
        scrubbed = scrub_string(raw)
        self.assertIn("[REDACTED_JWT]", scrubbed)
        self.assertNotIn("eyJhbGciOiJFUzI1NiJ9", scrubbed)

    def test_scrub_api_keys(self):
        """Gemini and Langfuse API keys are redacted from text."""
        gemini_key = "AIzaSy" + "B" * 33
        langfuse_key = "sk-lf-1234567890abcdef12345678"

        raw = f"Gemini key is {gemini_key} and Langfuse is {langfuse_key}"
        scrubbed = scrub_string(raw)

        self.assertIn("[REDACTED_KEY]", scrubbed)
        self.assertIn("[REDACTED_API_KEY]", scrubbed)
        self.assertNotIn(gemini_key, scrubbed)
        self.assertNotIn(langfuse_key, scrubbed)

    def test_scrub_sensitive_dict_keys(self):
        """Values under sensitive dictionary keys are replaced with REDACTED_PLACEHOLDER."""
        payload = {
            "title": "Employee Handbook",
            "password": "SuperSecretPassword123!",
            "api_key": "secret-123",
            "tenant_id": "tenant-001",
        }
        scrubbed = scrub_sensitive_data(payload)
        self.assertEqual(scrubbed["title"], "Employee Handbook")
        self.assertEqual(scrubbed["tenant_id"], "tenant-001")
        self.assertEqual(scrubbed["password"], REDACTED_PLACEHOLDER)
        self.assertEqual(scrubbed["api_key"], REDACTED_PLACEHOLDER)

    def test_scrub_http_headers_structure(self):
        """Name/value header dicts with sensitive header names have value redacted."""
        headers = [
            {"name": "Content-Type", "value": "application/json"},
            {"name": "Authorization", "value": "Bearer my-secret-token"},
        ]
        scrubbed = scrub_sensitive_data(headers)
        self.assertEqual(scrubbed[0]["value"], "application/json")
        self.assertEqual(scrubbed[1]["value"], REDACTED_PLACEHOLDER)

    def test_scrub_nested_structures(self):
        """Recursively scrubs nested dictionaries, lists, tuples, and sets."""
        nested = {
            "meta": {
                "credential_items": ["item1", "AIzaSy" + "A" * 33],
                "clean_list": ["apple", "banana"],
                "secret_token_info": "should be entirely replaced by placeholder",
            },
            "tuple_data": ("safe", "Bearer top-secret-token"),
        }
        scrubbed = scrub_sensitive_data(nested)
        self.assertEqual(scrubbed["meta"]["clean_list"], ["apple", "banana"])
        self.assertIn("[REDACTED_KEY]", scrubbed["meta"]["credential_items"][1])
        self.assertEqual(scrubbed["meta"]["secret_token_info"], REDACTED_PLACEHOLDER)
        self.assertIn("Bearer [REDACTED]", scrubbed["tuple_data"][1])

    def test_non_sensitive_data_untouched(self):
        """Standard conversational messages and metrics remain unaltered."""
        text = "How many vacation days do full-time employees receive annually?"
        self.assertEqual(scrub_sensitive_data(text), text)

        number = 42
        self.assertEqual(scrub_sensitive_data(number), 42)
