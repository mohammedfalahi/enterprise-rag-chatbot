"""
Document Ingestion REST API Endpoint.
Provides secure multi-tenant document upload with:
- 15MB file size limit enforcement (HTTP 413)
- Magic byte file signature validation (HTTP 415)
- Role-based authorization (managers/executives only)
- Atomic ingestion transaction
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.security import CurrentUser, get_current_user
from app.core.rate_limiter import rate_limit
from app.db.provisioning import validate_roles
from app.rag.ingest import ingest_document_file

router = APIRouter()

MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
CHUNK_READ_SIZE = 64 * 1024  # 64 KB

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


def _validate_file_content_and_signature(file_path: str, ext: str) -> None:
    """
    Validates magic bytes / content signature to prevent extension spoofing.
    Rejects binary executables and mismatched file types with HTTP 415.
    """
    with open(file_path, "rb") as f:
        header = f.read(1024)

    if ext == ".pdf":
        if not header.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="File content does not match declared PDF format (invalid magic bytes).",
            )
    elif ext == ".docx":
        if not header.startswith(b"PK\x03\x04"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="File content does not match declared DOCX format (invalid ZIP archive header).",
            )
        if not zipfile.is_zipfile(file_path):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="File content is not a valid DOCX/ZIP package.",
            )
    elif ext in {".txt", ".md", ".markdown"}:
        if b"\x00" in header:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="File contains binary/null bytes. Expected plain text or markdown.",
            )
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                f.read(4096)
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Text file content cannot be decoded as UTF-8.",
            )


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit(max_requests=10, window_seconds=60, endpoint_tag="documents_upload"))],
)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    access_roles: str = Form(
        ...,
        description="Comma-separated or JSON list of enterprise roles allowed to view this document.",
    ),
    user: CurrentUser = Depends(get_current_user),
):
    """
    Upload and ingest an enterprise document.
    Enforces file size limits, magic byte validation, and atomic database persistence.
    """
    # 1. Authorization: Only managers or executives can upload enterprise documents
    if not any(r in {"manager", "executive"} for r in user.roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Only managers or executives can upload enterprise documents.",
        )

    # 2. Validate declared file extension
    filename = file.filename or "document.txt"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: PDF, DOCX, TXT, MD.",
        )

    # 3. Parse and validate access_roles
    try:
        if access_roles.strip().startswith("[") and access_roles.strip().endswith("]"):
            roles_list = json.loads(access_roles)
        else:
            roles_list = [r.strip() for r in access_roles.split(",") if r.strip()]
        validate_roles(roles_list)
        validated_roles = roles_list
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid access_roles: {e}",
        )

    # 4. Stream upload to a temporary file while enforcing file size limits
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_path = temp_file.name
    total_bytes = 0

    try:
        while True:
            chunk = await file.read(CHUNK_READ_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.",
                )
            temp_file.write(chunk)
        temp_file.close()

        # 5. Content inspection / Magic byte signature verification
        _validate_file_content_and_signature(temp_path, ext)

        # 6. Atomic ingestion: parsing, embedding, and transactional DB insertion
        doc_id = await ingest_document_file(
            file_path=temp_path,
            tenant_id=user.tenant_id,
            title=title.strip() or filename,
            access_roles=validated_roles,
        )

        if not doc_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document contains no extractable text content.",
            )

        return {
            "document_id": doc_id,
            "title": title.strip() or filename,
            "status": "ingested",
            "size_bytes": total_bytes,
        }

    finally:
        # 7. Safe cleanup: Ensure temporary file is always unlinked
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
