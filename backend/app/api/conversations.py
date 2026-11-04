"""
Conversation Management REST APIs.
Provides endpoints for listing, creating, retrieving, renaming, and deleting conversations,
as well as fetching message histories. Enforces strict tenant and user isolation.
"""

from __future__ import annotations

import uuid
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.rate_limiter import rate_limit
from app.core.security import CurrentUser, get_current_user
from app.db.conversations import (
    create_conversation,
    delete_conversation,
    get_conversation,
    get_conversation_messages,
    list_conversations,
    update_conversation_title,
)

router = APIRouter()


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, max_length=100)


class UpdateConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)


class ConversationResponse(BaseModel):
    id: str
    tenant_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    tenant_id: str
    user_id: str
    role: str
    content: str
    citations: list[dict[str, Any]] = []
    created_at: str


def _validate_uuid(val: str, field_name: str = "conversation_id") -> str:
    try:
        uuid.UUID(val)
        return val
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {field_name} format: '{val}'. Must be a valid UUID.",
        )


@router.get(
    "",
    response_model=list[ConversationResponse],
    dependencies=[Depends(rate_limit(max_requests=60, window_seconds=60, endpoint_tag="conversations_list"))],
)
async def list_user_conversations(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: CurrentUser = Depends(get_current_user),
):
    """Lists conversations for the authenticated user ordered by most recently updated."""
    conversations = await list_conversations(
        tenant_id=user.tenant_id,
        user_id=user.id,
        limit=limit,
        offset=offset,
    )
    return conversations


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit(max_requests=30, window_seconds=60, endpoint_tag="conversations_create"))],
)
async def create_user_conversation(
    request: CreateConversationRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """Creates a new conversation for the authenticated user."""
    conv = await create_conversation(
        tenant_id=user.tenant_id,
        user_id=user.id,
        title=request.title,
    )
    return conv


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    dependencies=[Depends(rate_limit(max_requests=60, window_seconds=60, endpoint_tag="conversations_get"))],
)
async def get_user_conversation(
    conversation_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Retrieves conversation details ensuring user and tenant match."""
    _validate_uuid(conversation_id)
    conv = await get_conversation(
        conversation_id=conversation_id,
        tenant_id=user.tenant_id,
        user_id=user.id,
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found or access denied.",
        )
    return conv


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
    dependencies=[Depends(rate_limit(max_requests=30, window_seconds=60, endpoint_tag="conversations_update"))],
)
async def rename_user_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """Updates a conversation's title."""
    _validate_uuid(conversation_id)
    clean_title = request.title.strip()
    if not clean_title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conversation title cannot be empty.",
        )

    updated = await update_conversation_title(
        conversation_id=conversation_id,
        tenant_id=user.tenant_id,
        user_id=user.id,
        title=clean_title,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found or access denied.",
        )
    return updated


@router.delete(
    "/{conversation_id}",
    dependencies=[Depends(rate_limit(max_requests=30, window_seconds=60, endpoint_tag="conversations_delete"))],
)
async def delete_user_conversation(
    conversation_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Deletes a conversation and its messages."""
    _validate_uuid(conversation_id)
    deleted = await delete_conversation(
        conversation_id=conversation_id,
        tenant_id=user.tenant_id,
        user_id=user.id,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found or access denied.",
        )
    return {"success": True, "id": conversation_id}


@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
    dependencies=[Depends(rate_limit(max_requests=60, window_seconds=60, endpoint_tag="conversations_messages"))],
)
async def get_user_conversation_messages(
    conversation_id: str,
    limit: int | None = Query(default=None, ge=1, le=100),
    user: CurrentUser = Depends(get_current_user),
):
    """Retrieves messages for a conversation in ascending chronological order."""
    _validate_uuid(conversation_id)
    # First verify conversation exists and user owns it
    conv = await get_conversation(
        conversation_id=conversation_id,
        tenant_id=user.tenant_id,
        user_id=user.id,
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found or access denied.",
        )

    messages = await get_conversation_messages(
        conversation_id=conversation_id,
        tenant_id=user.tenant_id,
        user_id=user.id,
        limit=limit,
    )
    return messages
