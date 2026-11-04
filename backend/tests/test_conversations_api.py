"""
Integration tests for the /api/conversations REST API endpoints.
Tests:
- 401 Unauthorized for unauthenticated requests
- GET /api/conversations (list with pagination)
- POST /api/conversations (create)
- GET /api/conversations/{id} (fetch)
- PATCH /api/conversations/{id} (rename)
- DELETE /api/conversations/{id} (delete)
- GET /api/conversations/{id}/messages (history)
- 400 Bad Request on invalid UUIDs
- 404 Not Found on cross-user or nonexistent conversations
"""

from __future__ import annotations

import unittest
import uuid
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.core.security import CurrentUser, get_current_user
from app.main import app


class TestConversationsApiEndpoints(unittest.TestCase):

    def setUp(self):
        app.dependency_overrides = {}
        self.client = TestClient(app, raise_server_exceptions=False)
        self.user_a = CurrentUser(
            id=str(uuid.uuid4()),
            tenant_id=str(uuid.uuid4()),
            roles=["employee"],
        )

    def tearDown(self):
        app.dependency_overrides = {}

    def test_unauthenticated_requests_return_401(self):
        """Endpoints reject requests without Authorization header."""
        res_list = self.client.get("/api/conversations")
        self.assertEqual(res_list.status_code, 401)

        res_create = self.client.post("/api/conversations", json={})
        self.assertEqual(res_create.status_code, 401)

        valid_id = str(uuid.uuid4())
        res_get = self.client.get(f"/api/conversations/{valid_id}")
        self.assertEqual(res_get.status_code, 401)

        res_patch = self.client.patch(f"/api/conversations/{valid_id}", json={"title": "New"})
        self.assertEqual(res_patch.status_code, 401)

        res_del = self.client.delete(f"/api/conversations/{valid_id}")
        self.assertEqual(res_del.status_code, 401)

        res_msgs = self.client.get(f"/api/conversations/{valid_id}/messages")
        self.assertEqual(res_msgs.status_code, 401)

    def test_invalid_uuid_returns_400(self):
        """Passing non-UUID strings returns HTTP 400 Bad Request."""
        app.dependency_overrides[get_current_user] = lambda: self.user_a

        res_get = self.client.get("/api/conversations/not-a-uuid")
        self.assertEqual(res_get.status_code, 400)
        self.assertIn("Invalid conversation_id format", res_get.json()["detail"])

        res_patch = self.client.patch("/api/conversations/not-a-uuid", json={"title": "New Title"})
        self.assertEqual(res_patch.status_code, 400)

        res_del = self.client.delete("/api/conversations/not-a-uuid")
        self.assertEqual(res_del.status_code, 400)

        res_msgs = self.client.get("/api/conversations/not-a-uuid/messages")
        self.assertEqual(res_msgs.status_code, 400)

    def test_list_conversations(self):
        """GET /api/conversations returns list of conversations."""
        app.dependency_overrides[get_current_user] = lambda: self.user_a

        mock_convs = [
            {
                "id": str(uuid.uuid4()),
                "tenant_id": self.user_a.tenant_id,
                "user_id": self.user_a.id,
                "title": "Chat 1",
                "created_at": "2026-11-04T12:00:00Z",
                "updated_at": "2026-11-04T12:05:00Z",
            }
        ]

        with patch("app.api.conversations.list_conversations", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = mock_convs

            response = self.client.get(
                "/api/conversations?limit=10&offset=0",
                headers={"Authorization": "Bearer token"},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["title"], "Chat 1")
            mock_list.assert_called_once_with(
                tenant_id=self.user_a.tenant_id,
                user_id=self.user_a.id,
                limit=10,
                offset=0,
            )

    def test_create_conversation(self):
        """POST /api/conversations creates a new conversation."""
        app.dependency_overrides[get_current_user] = lambda: self.user_a

        cid = str(uuid.uuid4())
        mock_created = {
            "id": cid,
            "tenant_id": self.user_a.tenant_id,
            "user_id": self.user_a.id,
            "title": "Roadmap Brainstorming",
            "created_at": "2026-11-04T12:00:00Z",
            "updated_at": "2026-11-04T12:00:00Z",
        }

        with patch("app.api.conversations.create_conversation", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_created

            response = self.client.post(
                "/api/conversations",
                json={"title": "Roadmap Brainstorming"},
                headers={"Authorization": "Bearer token"},
            )
            self.assertEqual(response.status_code, 201)
            data = response.json()
            self.assertEqual(data["id"], cid)
            self.assertEqual(data["title"], "Roadmap Brainstorming")
            mock_create.assert_called_once_with(
                tenant_id=self.user_a.tenant_id,
                user_id=self.user_a.id,
                title="Roadmap Brainstorming",
            )

    def test_get_conversation_success_and_404(self):
        """GET /api/conversations/{id} returns conversation or 404 if not found."""
        app.dependency_overrides[get_current_user] = lambda: self.user_a
        cid = str(uuid.uuid4())

        mock_conv = {
            "id": cid,
            "tenant_id": self.user_a.tenant_id,
            "user_id": self.user_a.id,
            "title": "Existing Thread",
            "created_at": "2026-11-04T12:00:00Z",
            "updated_at": "2026-11-04T12:00:00Z",
        }

        with patch("app.api.conversations.get_conversation", new_callable=AsyncMock) as mock_get:
            # 1. Success case
            mock_get.return_value = mock_conv
            res_found = self.client.get(
                f"/api/conversations/{cid}",
                headers={"Authorization": "Bearer token"},
            )
            self.assertEqual(res_found.status_code, 200)
            self.assertEqual(res_found.json()["title"], "Existing Thread")

            # 2. Not found / unauthorized case
            mock_get.return_value = None
            res_missing = self.client.get(
                f"/api/conversations/{cid}",
                headers={"Authorization": "Bearer token"},
            )
            self.assertEqual(res_missing.status_code, 404)
            self.assertIn("not found or access denied", res_missing.json()["detail"])

    def test_rename_conversation_success_and_validation(self):
        """PATCH /api/conversations/{id} renames conversation and validates input."""
        app.dependency_overrides[get_current_user] = lambda: self.user_a
        cid = str(uuid.uuid4())

        # Empty title fails validation (422 from Pydantic min_length or 400 from endpoint)
        res_empty = self.client.patch(
            f"/api/conversations/{cid}",
            json={"title": "   "},
            headers={"Authorization": "Bearer token"},
        )
        self.assertIn(res_empty.status_code, (400, 422))

        mock_updated = {
            "id": cid,
            "tenant_id": self.user_a.tenant_id,
            "user_id": self.user_a.id,
            "title": "Renamed Thread",
            "created_at": "2026-11-04T12:00:00Z",
            "updated_at": "2026-11-04T12:10:00Z",
        }

        with patch("app.api.conversations.update_conversation_title", new_callable=AsyncMock) as mock_update:
            mock_update.return_value = mock_updated
            res_success = self.client.patch(
                f"/api/conversations/{cid}",
                json={"title": "Renamed Thread"},
                headers={"Authorization": "Bearer token"},
            )
            self.assertEqual(res_success.status_code, 200)
            self.assertEqual(res_success.json()["title"], "Renamed Thread")

    def test_delete_conversation_success_and_404(self):
        """DELETE /api/conversations/{id} deletes conversation or returns 404."""
        app.dependency_overrides[get_current_user] = lambda: self.user_a
        cid = str(uuid.uuid4())

        with patch("app.api.conversations.delete_conversation", new_callable=AsyncMock) as mock_del:
            # 1. Success case
            mock_del.return_value = True
            res_del = self.client.delete(
                f"/api/conversations/{cid}",
                headers={"Authorization": "Bearer token"},
            )
            self.assertEqual(res_del.status_code, 200)
            self.assertTrue(res_del.json()["success"])

            # 2. Not found case
            mock_del.return_value = False
            res_not_found = self.client.delete(
                f"/api/conversations/{cid}",
                headers={"Authorization": "Bearer token"},
            )
            self.assertEqual(res_not_found.status_code, 404)

    def test_get_conversation_messages_success_and_404(self):
        """GET /api/conversations/{id}/messages returns message history."""
        app.dependency_overrides[get_current_user] = lambda: self.user_a
        cid = str(uuid.uuid4())

        with patch("app.api.conversations.get_conversation", new_callable=AsyncMock) as mock_get_conv:
            with patch("app.api.conversations.get_conversation_messages", new_callable=AsyncMock) as mock_get_msgs:
                # 1. Conversation not found returns 404
                mock_get_conv.return_value = None
                res_missing = self.client.get(
                    f"/api/conversations/{cid}/messages",
                    headers={"Authorization": "Bearer token"},
                )
                self.assertEqual(res_missing.status_code, 404)

                # 2. Conversation exists returns messages
                mock_get_conv.return_value = {"id": cid}
                mock_get_msgs.return_value = [
                    {
                        "id": str(uuid.uuid4()),
                        "conversation_id": cid,
                        "tenant_id": self.user_a.tenant_id,
                        "user_id": self.user_a.id,
                        "role": "user",
                        "content": "What is the policy?",
                        "citations": [],
                        "created_at": "2026-11-04T12:00:00Z",
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "conversation_id": cid,
                        "tenant_id": self.user_a.tenant_id,
                        "user_id": self.user_a.id,
                        "role": "assistant",
                        "content": "The policy requires approvals.",
                        "citations": [{"chunk_id": "c1", "score": 0.9}],
                        "created_at": "2026-11-04T12:00:02Z",
                    },
                ]

                res_msgs = self.client.get(
                    f"/api/conversations/{cid}/messages?limit=10",
                    headers={"Authorization": "Bearer token"},
                )
                self.assertEqual(res_msgs.status_code, 200)
                msgs = res_msgs.json()
                self.assertEqual(len(msgs), 2)
                self.assertEqual(msgs[0]["role"], "user")
                self.assertEqual(msgs[1]["role"], "assistant")
                self.assertEqual(len(msgs[1]["citations"]), 1)
