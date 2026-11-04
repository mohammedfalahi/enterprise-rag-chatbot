"""
Integration tests for the /api/documents endpoint.
Covers:
- 401 Unauthorized for unauthenticated requests
- 403 Forbidden for non-manager/executive users
- 422 Unprocessable Entity for invalid access_roles
- 415 Unsupported Media Type for unsupported extensions or spoofed magic bytes
- 413 Request Entity Too Large when file size exceeds limits
- 201 Created on successful document upload and atomic ingestion
- 400 Bad Request when document has no extractable text
"""

from __future__ import annotations

import io
import unittest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.core.security import CurrentUser, get_current_user
from app.main import app


class TestDocumentsApiEndpoint(unittest.TestCase):

    def setUp(self):
        app.dependency_overrides = {}
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self):
        app.dependency_overrides = {}

    def test_unauthenticated_upload_returns_401(self):
        """Upload request without Authorization header returns 401."""
        response = self.client.post("/api/documents/upload")
        self.assertEqual(response.status_code, 401)

    def test_non_manager_upload_returns_403(self):
        """Employee role without manager/executive permission receives 403."""
        mock_user = CurrentUser(
            id="user-emp",
            tenant_id="tenant-001",
            roles=["employee", "sales_rep"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response = self.client.post(
            "/api/documents/upload",
            data={"title": "Q3 Report", "access_roles": "employee"},
            files={"file": ("report.txt", b"Valid report content", "text/plain")},
            headers={"Authorization": "Bearer mock-token"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("Only managers or executives", response.json().get("detail", ""))

    def test_invalid_access_roles_returns_422(self):
        """Submitting non-enterprise roles returns 422."""
        mock_user = CurrentUser(
            id="user-mgr",
            tenant_id="tenant-001",
            roles=["manager"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response = self.client.post(
            "/api/documents/upload",
            data={"title": "Guide", "access_roles": "superadmin,unknown"},
            files={"file": ("guide.txt", b"Valid text", "text/plain")},
            headers={"Authorization": "Bearer mock-token"},
        )
        self.assertEqual(response.status_code, 422)

    def test_unsupported_extension_returns_415(self):
        """Unsupported extensions (e.g. .exe, .sh) return 415."""
        mock_user = CurrentUser(
            id="user-mgr",
            tenant_id="tenant-001",
            roles=["manager"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response = self.client.post(
            "/api/documents/upload",
            data={"title": "Binary", "access_roles": "employee"},
            files={"file": ("payload.exe", b"binary data", "application/octet-stream")},
            headers={"Authorization": "Bearer mock-token"},
        )
        self.assertEqual(response.status_code, 415)
        self.assertIn("Unsupported file extension", response.json().get("detail", ""))

    def test_spoofed_pdf_magic_bytes_rejected_with_415(self):
        """File claiming to be .pdf with invalid header magic bytes returns 415."""
        mock_user = CurrentUser(
            id="user-mgr",
            tenant_id="tenant-001",
            roles=["manager"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        # Spoofed file: .pdf extension with plain text or shell script content
        fake_pdf_content = b"echo 'This is not a real PDF file';"
        response = self.client.post(
            "/api/documents/upload",
            data={"title": "Spoofed Doc", "access_roles": "employee"},
            files={"file": ("fake.pdf", fake_pdf_content, "application/pdf")},
            headers={"Authorization": "Bearer mock-token"},
        )
        self.assertEqual(response.status_code, 415)
        self.assertIn("does not match declared PDF format", response.json().get("detail", ""))

    def test_file_size_exceeded_returns_413(self):
        """Files exceeding MAX_UPLOAD_SIZE_BYTES are rejected with 413."""
        mock_user = CurrentUser(
            id="user-mgr",
            tenant_id="tenant-001",
            roles=["manager"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        # Patch MAX_UPLOAD_SIZE_BYTES to 100 bytes for test speed
        with patch("app.api.documents.MAX_UPLOAD_SIZE_BYTES", 100):
            oversized_content = b"A" * 200
            response = self.client.post(
                "/api/documents/upload",
                data={"title": "Too Large", "access_roles": "employee"},
                files={"file": ("large.txt", oversized_content, "text/plain")},
                headers={"Authorization": "Bearer mock-token"},
            )
            self.assertEqual(response.status_code, 413)
            self.assertIn("File size exceeds maximum allowed limit", response.json().get("detail", ""))

    def test_successful_document_upload(self):
        """Managers uploading valid files successfully ingest and return 201."""
        mock_user = CurrentUser(
            id="user-exec",
            tenant_id="tenant-enterprise",
            roles=["executive"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        valid_text = b"# Executive Strategy\n\nQuarterly goals for the enterprise organization."

        with patch("app.api.documents.ingest_document_file", new_callable=AsyncMock) as mock_ingest:
            mock_ingest.return_value = "doc-uuid-999"

            response = self.client.post(
                "/api/documents/upload",
                data={"title": "Executive Strategy 2026", "access_roles": "employee,manager,executive"},
                files={"file": ("strategy.md", valid_text, "text/markdown")},
                headers={"Authorization": "Bearer mock-token"},
            )

            self.assertEqual(response.status_code, 201)
            data = response.json()
            self.assertEqual(data["document_id"], "doc-uuid-999")
            self.assertEqual(data["status"], "ingested")
            self.assertEqual(data["title"], "Executive Strategy 2026")

            # Verify tenant_id from JWT user was passed to ingest_document_file
            mock_ingest.assert_called_once()
            call_kwargs = mock_ingest.call_args.kwargs
            self.assertEqual(call_kwargs["tenant_id"], "tenant-enterprise")
            self.assertEqual(call_kwargs["title"], "Executive Strategy 2026")
            self.assertEqual(set(call_kwargs["access_roles"]), {"employee", "manager", "executive"})

    def test_empty_content_document_returns_400(self):
        """When ingestion returns None (no extractable chunks), returns 400."""
        mock_user = CurrentUser(
            id="user-mgr",
            tenant_id="tenant-001",
            roles=["manager"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        with patch("app.api.documents.ingest_document_file", new_callable=AsyncMock) as mock_ingest:
            mock_ingest.return_value = None

            response = self.client.post(
                "/api/documents/upload",
                data={"title": "Empty Doc", "access_roles": "employee"},
                files={"file": ("empty.txt", b"   ", "text/plain")},
                headers={"Authorization": "Bearer mock-token"},
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("no extractable text", response.json().get("detail", ""))
