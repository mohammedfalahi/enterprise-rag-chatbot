"""Tests for manager invitations and invitation acceptance endpoints."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.security import (
    AuthenticatedUser,
    CurrentUser,
    get_authenticated_user,
    get_current_user,
)
from app.main import app


class TestUsersApi(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides = {}
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self):
        app.dependency_overrides = {}

    def test_create_invitation_requires_authentication(self):
        response = self.client.post(
            "/api/users/invitations",
            json={"email": "new.user@example.com", "role": "employee"},
        )
        self.assertEqual(response.status_code, 401)

    def test_non_manager_cannot_send_invitation(self):
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(
            id="employee-id",
            tenant_id="tenant-001",
            roles=["employee"],
        )

        with patch("app.api.users._invite_with_supabase", new_callable=AsyncMock) as invite:
            response = self.client.post(
                "/api/users/invitations",
                json={"email": "new.user@example.com", "role": "employee"},
            )

        self.assertEqual(response.status_code, 403)
        invite.assert_not_called()

    def test_manager_invitation_uses_server_tenant_and_selected_role(self):
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(
            id="manager-id",
            tenant_id="trusted-tenant",
            roles=["manager"],
        )

        with (
            patch("app.api.users.has_pending_invitation", new_callable=AsyncMock) as pending,
            patch("app.api.users._invite_with_supabase", new_callable=AsyncMock) as invite,
            patch("app.api.users.create_pending_invitation", new_callable=AsyncMock) as store,
        ):
            pending.return_value = False
            store.return_value = "invitation-id"
            response = self.client.post(
                "/api/users/invitations",
                json={"email": " NEW.User@Example.com ", "role": "sales_rep"},
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {"message": "Invitation sent."})
        invite.assert_awaited_once_with("new.user@example.com")
        store.assert_awaited_once_with(
            tenant_id="trusted-tenant",
            email="new.user@example.com",
            roles=["sales_rep"],
        )

    def test_invalid_role_is_rejected(self):
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(
            id="manager-id",
            tenant_id="tenant-001",
            roles=["manager"],
        )

        with patch("app.api.users._invite_with_supabase", new_callable=AsyncMock) as invite:
            response = self.client.post(
                "/api/users/invitations",
                json={"email": "new.user@example.com", "role": "administrator"},
            )

        self.assertEqual(response.status_code, 422)
        invite.assert_not_called()

    def test_pending_invitation_is_not_resent(self):
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(
            id="manager-id",
            tenant_id="tenant-001",
            roles=["manager"],
        )

        with (
            patch("app.api.users.has_pending_invitation", new_callable=AsyncMock) as pending,
            patch("app.api.users._invite_with_supabase", new_callable=AsyncMock) as invite,
        ):
            pending.return_value = True
            response = self.client.post(
                "/api/users/invitations",
                json={"email": "new.user@example.com", "role": "employee"},
            )

        self.assertEqual(response.status_code, 409)
        invite.assert_not_called()

    def test_accept_invitation_requires_authentication(self):
        response = self.client.post("/api/users/accept-invitation")
        self.assertEqual(response.status_code, 401)

    def test_authenticated_user_accepts_invitation(self):
        app.dependency_overrides[get_authenticated_user] = lambda: AuthenticatedUser(
            id="invited-user-id",
            email="new.user@example.com",
        )

        with patch("app.api.users.accept_user_invitation", new_callable=AsyncMock) as accept:
            accept.return_value = ("tenant-001", ["employee"])
            response = self.client.post("/api/users/accept-invitation")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Invitation accepted."})
        accept.assert_awaited_once_with(
            user_id="invited-user-id",
            email="new.user@example.com",
        )
