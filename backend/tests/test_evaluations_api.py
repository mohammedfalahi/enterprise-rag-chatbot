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
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.core.security import CurrentUser, get_current_user
from app.eval.schemas import EvaluationReport, RefusalMetrics, RetrievalMetrics
from app.main import app
from app.api import evaluations as evaluations_api


class TestAdminEvaluationsApi(unittest.TestCase):

    def setUp(self):
        app.dependency_overrides = {}
        self.client = TestClient(app, raise_server_exceptions=False)
        self.run = {
            "id": "run-123",
            "tenant_id": "tenant-123",
            "requested_by": "user-exec",
            "status": "pending",
            "sample_size": 5,
            "mode": "hybrid_rerank",
            "top_k": 5,
            "candidate_limit": 20,
            "rerank_pool_size": 10,
            "error_message": None,
            "created_at": "2026-11-06T00:00:00+00:00",
            "started_at": None,
            "completed_at": None,
        }
        self.report = EvaluationReport(
            tenant_id="tenant-123",
            tenant_name="tenant-test",
            timestamp="2026-11-06T00:00:00+00:00",
            total_cases=5,
            retrieval=RetrievalMetrics(recall_at_k=0.8),
            refusal=RefusalMetrics(),
        )
        self.patches = [
            patch.object(evaluations_api, "get_latest_evaluation_run", new_callable=AsyncMock, return_value=None),
            patch.object(evaluations_api, "get_latest_completed_evaluation_run", new_callable=AsyncMock, return_value=None),
            patch.object(evaluations_api, "list_evaluation_runs", new_callable=AsyncMock, return_value=[]),
            patch.object(evaluations_api, "get_evaluation_run", new_callable=AsyncMock, return_value=None),
            patch.object(
                evaluations_api,
                "create_evaluation_run",
                new_callable=AsyncMock,
                return_value=(self.run, True),
            ),
            patch.object(evaluations_api, "update_evaluation_run", new_callable=AsyncMock),
            patch.object(
                evaluations_api,
                "evaluate_dataset",
                new_callable=AsyncMock,
                return_value=self.report,
            ),
        ]
        self.mocks = [mock.start() for mock in self.patches]
        (
            self.latest_run,
            self.latest_completed_run,
            self.list_runs,
            self.get_run,
            self.create_run,
            self.update_run,
            self.evaluate_dataset,
        ) = self.mocks

    def tearDown(self):
        app.dependency_overrides = {}
        for mock in reversed(self.patches):
            mock.stop()

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
        """POST /run queues exactly one tenant-scoped evaluation and persists its result."""
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
        self.assertEqual(data["run"]["status"], "pending")
        self.assertEqual(data["run"]["sample_size"], 5)
        self.assertFalse(data["deduplicated"])
        self.create_run.assert_awaited_once()
        self.evaluate_dataset.assert_awaited_once()
        self.assertEqual(
            self.evaluate_dataset.await_args.kwargs["tenant_id"],
            "tenant-123",
        )
        self.assertEqual(self.update_run.await_count, 2)
        self.assertEqual(self.update_run.await_args_list[0].kwargs["status"], "running")
        completed_call = self.update_run.await_args_list[1]
        self.assertEqual(completed_call.kwargs["status"], "completed")
        self.assertEqual(
            completed_call.kwargs["report"]["retrieval"]["recall_at_k"],
            0.8,
        )

    def test_duplicate_trigger_reuses_active_run_without_scheduling_another_task(self):
        mock_user = CurrentUser(id="user-exec", tenant_id="tenant-123", roles=["executive"])
        app.dependency_overrides[get_current_user] = lambda: mock_user
        self.create_run.return_value = (self.run, False)

        response = self.client.post(
            "/api/admin/evaluations/run",
            json={"sample_size": 5, "mode": "hybrid_rerank"},
            headers={"Authorization": "Bearer test-token"},
        )

        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.json()["deduplicated"])
        self.evaluate_dataset.assert_not_awaited()
        self.update_run.assert_not_awaited()

    def test_background_failure_is_saved_and_reported(self):
        self.evaluate_dataset.side_effect = RuntimeError("benchmark dataset could not be loaded")

        import asyncio

        asyncio.run(evaluations_api._run_eval_background_job(
            "run-123",
            evaluations_api.RunEvaluationRequest(sample_size=5),
            "tenant-123",
        ))

        self.assertEqual(self.update_run.await_count, 2)
        failure_call = self.update_run.await_args_list[1]
        self.assertEqual(failure_call.kwargs["status"], "failed")
        self.assertIn("benchmark dataset", failure_call.kwargs["error_message"])

    def test_run_status_is_scoped_to_the_authenticated_tenant(self):
        mock_user = CurrentUser(id="user-exec", tenant_id="tenant-123", roles=["executive"])
        app.dependency_overrides[get_current_user] = lambda: mock_user
        self.get_run.return_value = self.run

        response = self.client.get(
            "/api/admin/evaluations/runs/00000000-0000-0000-0000-000000000123",
            headers={"Authorization": "Bearer test-token"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "run-123")
        self.get_run.assert_awaited_once_with(
            run_id="00000000-0000-0000-0000-000000000123",
            tenant_id="tenant-123",
        )

        self.get_run.return_value = None
        response = self.client.get(
            "/api/admin/evaluations/runs/00000000-0000-0000-0000-000000000123",
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(response.status_code, 404)

    def test_latest_summary_uses_persisted_completed_report(self):
        mock_user = CurrentUser(id="user-exec", tenant_id="tenant-123", roles=["executive"])
        app.dependency_overrides[get_current_user] = lambda: mock_user
        self.latest_completed_run.return_value = {
            **self.run,
            "status": "completed",
            "report": self.report.model_dump(mode="json"),
        }

        response = self.client.get(
            "/api/admin/evaluations/latest",
            headers={"Authorization": "Bearer test-token"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sample_size"], 5)
        self.assertEqual(response.json()["retrieval_metrics"]["recall_at_k"], 0.8)


if __name__ == "__main__":
    unittest.main()
