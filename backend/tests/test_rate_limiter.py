"""
Unit and integration tests for in-memory sliding-window rate limiter.
Covers:
- Limit enforcement (HTTP 429)
- Exact Retry-After header calculation
- Sliding-window expiry
- Multi-tenant and multi-user isolation
- Thread-safe concurrent execution without race conditions
- Memory cleanup of expired entries
- Integration with /api/chat and /api/documents/upload endpoints
"""

from __future__ import annotations

import concurrent.futures
import time
import unittest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.rate_limiter import SlidingWindowRateLimiter, limiter
from app.core.security import CurrentUser, get_current_user
from app.main import app


class TestSlidingWindowRateLimiter(unittest.TestCase):

    def setUp(self):
        limiter.clear()
        self.test_limiter = SlidingWindowRateLimiter(cleanup_interval_ops=5)

    def tearDown(self):
        limiter.clear()
        self.test_limiter.clear()

    def test_limit_enforcement_and_retry_after(self):
        """Allows requests up to limit, then raises 429 with correct Retry-After header."""
        now = 1000.0
        key = "tenant-1:user-1:chat"

        # 3 requests allowed in 10-second window
        self.test_limiter.check_rate_limit(key, max_requests=3, window_seconds=10, now=now)
        self.test_limiter.check_rate_limit(key, max_requests=3, window_seconds=10, now=now + 1.0)
        self.test_limiter.check_rate_limit(key, max_requests=3, window_seconds=10, now=now + 2.0)

        # 4th request must be rejected with 429
        with self.assertRaises(HTTPException) as ctx:
            self.test_limiter.check_rate_limit(key, max_requests=3, window_seconds=10, now=now + 3.0)

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("Retry-After", ctx.exception.headers)

        # Oldest request was at 1000.0 with 10s window (expires at 1010.0)
        # Current time is 1003.0 -> retry_after should be ceil(1010 - 1003) = 7
        retry_after = int(ctx.exception.headers["Retry-After"])
        self.assertEqual(retry_after, 7)

    def test_window_expiry_restores_allowance(self):
        """Requests are allowed again once previous requests slide out of the time window."""
        now = 2000.0
        key = "tenant-a:user-a:upload"

        # Exhaust limit of 2 requests in a 5-second window
        self.test_limiter.check_rate_limit(key, max_requests=2, window_seconds=5, now=now)
        self.test_limiter.check_rate_limit(key, max_requests=2, window_seconds=5, now=now + 1.0)

        # Blocked at now + 2.0
        with self.assertRaises(HTTPException):
            self.test_limiter.check_rate_limit(key, max_requests=2, window_seconds=5, now=now + 2.0)

        # At now + 5.5, the first request at now=2000 has expired; slot opens up
        self.test_limiter.check_rate_limit(key, max_requests=2, window_seconds=5, now=now + 5.5)

    def test_tenant_and_user_isolation(self):
        """Rate limits for one user or tenant do not affect other users or tenants."""
        now = 3000.0

        user_1_key = "tenant-alpha:user-1:chat"
        user_2_key = "tenant-alpha:user-2:chat"
        tenant_beta_key = "tenant-beta:user-1:chat"

        # Exhaust limit for user 1 in tenant alpha
        self.test_limiter.check_rate_limit(user_1_key, max_requests=1, window_seconds=60, now=now)
        with self.assertRaises(HTTPException):
            self.test_limiter.check_rate_limit(user_1_key, max_requests=1, window_seconds=60, now=now)

        # User 2 in same tenant is NOT blocked
        self.test_limiter.check_rate_limit(user_2_key, max_requests=1, window_seconds=60, now=now)

        # Same user ID in different tenant (tenant beta) is NOT blocked
        self.test_limiter.check_rate_limit(tenant_beta_key, max_requests=1, window_seconds=60, now=now)

    def test_concurrent_requests_thread_safety(self):
        """Under concurrent execution, exactly max_requests succeed and excess fail with 429."""
        key = "tenant-conc:user-conc:chat"
        max_requests = 10
        total_workers = 30
        now = time.time()

        success_count = 0
        rate_limited_count = 0

        def worker():
            try:
                self.test_limiter.check_rate_limit(key, max_requests=max_requests, window_seconds=60, now=now)
                return True
            except HTTPException as e:
                if e.status_code == 429:
                    return False
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker) for _ in range(total_workers)]
            for f in concurrent.futures.as_completed(futures):
                if f.result() is True:
                    success_count += 1
                else:
                    rate_limited_count += 1

        self.assertEqual(success_count, max_requests)
        self.assertEqual(rate_limited_count, total_workers - max_requests)

    def test_expired_entry_cleanup(self):
        """Periodic cleanup prunes inactive keys."""
        # Create an entry in the past (> 3600s ago)
        old_time = 1000.0
        self.test_limiter.check_rate_limit("old-key", max_requests=5, window_seconds=10, now=old_time)
        self.assertEqual(self.test_limiter.get_tracked_keys_count(), 1)

        # Advance time by 4000 seconds and run operations to trigger cleanup (cleanup_interval_ops=5)
        new_time = old_time + 4000.0
        for i in range(5):
            self.test_limiter.check_rate_limit("active-key", max_requests=10, window_seconds=60, now=new_time + i)

        # Old key has been purged
        self.assertEqual(self.test_limiter.get_tracked_keys_count(), 1)


class TestRateLimiterApiIntegration(unittest.TestCase):

    def setUp(self):
        limiter.clear()
        app.dependency_overrides = {}
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self):
        limiter.clear()
        app.dependency_overrides = {}

    def test_chat_endpoint_enforces_rate_limit(self):
        """POST /api/chat enforces rate limit and returns 429 with Retry-After header."""
        mock_user = CurrentUser(id="chat-user-1", tenant_id="tenant-chat", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user

        with patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search, \
             patch("app.api.chat.create_conversation", new_callable=AsyncMock) as mock_conv, \
             patch("app.api.chat.append_message", new_callable=AsyncMock) as mock_msg:
            mock_conv.return_value = {"id": "11111111-1111-1111-1111-111111111111", "title": "test"}
            mock_search.return_value = []

            # 30 allowed per 60s
            for _ in range(30):
                res = self.client.post(
                    "/api/chat",
                    json={"message": "test query"},
                    headers={"Authorization": "Bearer mock-token"},
                )
                self.assertEqual(res.status_code, 200)

            # 31st request exceeds rate limit
            res_blocked = self.client.post(
                "/api/chat",
                json={"message": "test query"},
                headers={"Authorization": "Bearer mock-token"},
            )
            self.assertEqual(res_blocked.status_code, 429)
            self.assertIn("Retry-After", res_blocked.headers)
            self.assertIn("Too many requests", res_blocked.json().get("detail", ""))

    def test_documents_upload_enforces_rate_limit(self):
        """POST /api/documents/upload enforces rate limit and returns 429 with Retry-After header."""
        mock_user = CurrentUser(id="upload-user-1", tenant_id="tenant-docs", roles=["manager"])
        app.dependency_overrides[get_current_user] = lambda: mock_user

        with patch("app.api.documents.ingest_document_file", new_callable=AsyncMock) as mock_ingest:
            mock_ingest.return_value = "doc-uuid-1"

            # 10 allowed per 60s
            for _ in range(10):
                res = self.client.post(
                    "/api/documents/upload",
                    data={"title": "Doc", "access_roles": "employee"},
                    files={"file": ("doc.txt", b"Hello world text", "text/plain")},
                    headers={"Authorization": "Bearer mock-token"},
                )
                self.assertEqual(res.status_code, 201)

            # 11th request exceeds rate limit
            res_blocked = self.client.post(
                "/api/documents/upload",
                data={"title": "Doc", "access_roles": "employee"},
                files={"file": ("doc.txt", b"Hello world text", "text/plain")},
                headers={"Authorization": "Bearer mock-token"},
            )
            self.assertEqual(res_blocked.status_code, 429)
            self.assertIn("Retry-After", res_blocked.headers)
