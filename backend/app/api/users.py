"""User invitation and acceptance endpoints."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from supabase import create_client
from supabase_auth.errors import AuthApiError, AuthRetryableError

from app.core.config import settings
from app.core.security import (
    AuthenticatedUser,
    CurrentUser,
    get_authenticated_user,
    get_current_user,
)
from app.db.provisioning import (
    InvitationEmailMismatch,
    InvitationNotFound,
    InvitationProfileConflict,
    PendingInvitationExists,
    accept_user_invitation,
    create_pending_invitation,
    has_pending_invitation,
)

logger = logging.getLogger("app.api.users")
router = APIRouter()

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
InvitationRole = Literal["employee", "sales_rep", "manager", "executive"]


class CreateInvitationRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    role: InvitationRole

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not EMAIL_PATTERN.fullmatch(normalized):
            raise ValueError("A valid email address is required.")
        return normalized


class InvitationResponse(BaseModel):
    message: str


def _send_supabase_invitation(email: str) -> None:
    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    response = client.auth.admin.invite_user_by_email(
        email,
        options={"redirect_to": settings.SUPABASE_INVITE_REDIRECT_URL},
    )
    if response.user is None:
        raise AuthRetryableError("Supabase did not return the invited user.")


async def _invite_with_supabase(email: str) -> None:
    await asyncio.to_thread(_send_supabase_invitation, email)


@router.post(
    "/invitations",
    response_model=InvitationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_user_invitation(
    request: CreateInvitationRequest,
    user: CurrentUser = Depends(get_current_user),
) -> InvitationResponse:
    if "manager" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only managers can invite users.",
        )
    if not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User invitations are not configured.",
        )
    if await has_pending_invitation(user.tenant_id, request.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A pending invitation already exists for this email.",
        )

    try:
        await _invite_with_supabase(request.email)
    except AuthApiError as error:
        logger.warning("Supabase rejected an invitation with status %s", error.status)
        if error.status in (409, 422):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This email address cannot be invited.",
            ) from error
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Supabase could not send the invitation.",
        ) from error
    except AuthRetryableError as error:
        logger.warning("Supabase invitation request failed: %s", error)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Supabase could not send the invitation.",
        ) from error

    try:
        await create_pending_invitation(
            tenant_id=user.tenant_id,
            email=request.email,
            roles=[request.role],
        )
    except PendingInvitationExists as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A pending invitation already exists for this email.",
        ) from error

    return InvitationResponse(message="Invitation sent.")


@router.post(
    "/accept-invitation",
    response_model=InvitationResponse,
    status_code=status.HTTP_200_OK,
)
async def accept_invitation(
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> InvitationResponse:
    if not user.email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The authenticated account does not include an email address.",
        )
    try:
        await accept_user_invitation(user_id=user.id, email=user.email)
    except InvitationEmailMismatch as error:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The authenticated email does not match the invited account.",
        ) from error
    except InvitationNotFound as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No pending invitation was found for this account.",
        ) from error
    except InvitationProfileConflict as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This account is already assigned to a different tenant or role.",
        ) from error

    return InvitationResponse(message="Invitation accepted.")
