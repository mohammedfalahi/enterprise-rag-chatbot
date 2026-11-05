"""
Integration and RBAC Security Tests for Admin Evaluation Dashboard API.
Covers:
- 401 Unauthorized for missing/invalid Bearer tokens
- 403 Forbidden for unprivileged roles (employee, sales_rep)
- 200 OK for executive/manager on /latest, /history, /queries, /error-analysis
- 202 Accepted on /run with background job queuing
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from app.core.security import CurrentUser, get_current_user
from app.main import app


class TestAdminEvaluationsApi(unittest.TestCase):

    def setUp(self):
        app.dependency_overrides = {}
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self):
        app.dependency_overrides = {}

    def test_unauthenticated_request_returns_401(self):
        """Request without Authorization header returns 401 Unauthorized."""
        response = self.client.get("/api/admin/evaluations/latest")
        self.assertEqual(response.status_code, 401)
        self.assertIn("Not authenticated", response.text)

    def test_employee_role_returns_403_forbidden(self):
        """User with only 'employee' role is denied access with 403 Forbidden."""
        mock_user = CurrentUser(
            id="user-emp",
            tenant_id="tenant-123",
            roles=["employee"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response = self.client.get(
            "/api/admin/evaluations/latest",
            headers={"Authorization": "Bearer mock-token"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("Access denied", response.json().get("detail", ""))

    def test_sales_rep_role_returns_403_forbidden(self):
        """User with only 'sales_rep' role is denied access with 403 Forbidden."""
        mock_user = CurrentUser(
            id="user-sales",
            tenant_id="tenant-123",
            roles=["sales_rep"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response = self.client.get(
            "/api/admin/evaluations/latest",
            headers={"Authorization": "Bearer mock-token"},
        )
        self.assertEqual(response.status_code, 403)

    def test_executive_role_can_fetch_latest_summary(self):
        """Executive role receives 200 OK with real aggregate metrics."""
        mock_user = CurrentUser(
            id="user-exec",
            tenant_id="tenant-123",
            roles=["executive"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response = self.client.get(
            "/api/admin/evaluations/latest",
            headers={"Authorization": "Bearer mock-token"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("generation_metrics", data)
        self.assertIn("retrieval_metrics", data)
        self.assertIn("refusal_metrics", data)
        self.assertIn("faithfulness", data["generation_metrics"])
        self.assertIn("recall_at_k", data["retrieval_metrics"])

    def test_manager_role_can_fetch_history_and_trends(self):
        """Manager role receives 200 OK on historical trends."""
        mock_user = CurrentUser(
            id="user-mgr",
            tenant_id="tenant-123",
            roles=["manager"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response = self.client.get(
            "/api/admin/evaluations/history",
            headers={"Authorization": "Bearer mock-token"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("trends", data)
        self.assertIn("experiments", data)

    def test_executive_can_fetch_and_filter_queries(self):
        """Filtering per-query results by status works properly."""
        mock_user = CurrentUser(
            id="user-exec",
            tenant_id="tenant-123",
            roles=["executive"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response = self.client.get(
            "/api/admin/evaluations/queries?status=failed&limit=5",
            headers={"Authorization": "Bearer mock-token"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("items", data)
        self.assertIn("total", data)
        for item in data["items"]:
            self.assertEqual(item["status"], "failed")

    def test_executive_can_fetch_error_analysis(self):
        """Error analysis diagnostics endpoint returns categorized failure distributions."""
        mock_user = CurrentUser(
            id="user-exec",
            tenant_id="tenant-123",
            roles=["executive"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response = self.client.get(
            "/api/admin/evaluations/error-analysis",
            headers={"Authorization": "Bearer mock-token"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("categories_distribution", data)
        self.assertIn("diagnosed_failures", data)

    def test_trigger_evaluation_run_returns_202_accepted(self):
        """POST /run initiates non-blocking evaluation job."""
        mock_user = CurrentUser(
            id="user-exec",
            tenant_id="tenant-123",
            roles=["executive"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response = self.client.post(
            "/api/admin/evaluations/run",
            json={"sample_size": 5, "mode": "hybrid_rerank"},
            headers={"Authorization": "Bearer mock-token"},
        )
        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertEqual(data["status"], "queued")
        self.assertEqual(data["sample_size"], 5)


if __name__ == "__main__":
    unittest.main()
