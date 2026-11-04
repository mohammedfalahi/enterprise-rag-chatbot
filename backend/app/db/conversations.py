"""
Enterprise Conversation and Message Persistence Data Access Layer.
Manages multi-turn conversation threads, message histories, citations,
and enforces strict tenant and user isolation.

Security & Architectural Guarantees:
- Every query explicitly parameters `tenant_id` and `user_id` preventing cross-tenant or cross-user leakage.
- Database Row-Level Security (RLS) policies provide secondary defense-in-depth.
- Bounded sliding-window history queries select the most recent N messages in chronological order.
- Conversation titles are automatically generated from the first user message (up to 50 characters).
- Never holds long-lived connections during streaming; all functions acquire and release connections promptly.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from app.core.config import settings
from app.db.connection import get_db_connection

MAX_TITLE_LENGTH: int = 50
DEFAULT_TITLE: str = "New Conversation"


def generate_conversation_title(content: str, max_chars: int = MAX_TITLE_LENGTH) -> str:
    """
    Derives a clean, single-line conversation title from a user message.
    Truncates to max_chars characters.
    """
    clean = " ".join(content.strip().split())
    if not clean:
        return DEFAULT_TITLE
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip()


async def create_conversation(
    tenant_id: str,
    user_id: str,
    title: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """
    Creates a new conversation record for a user within a tenant.
    """
    cid = conversation_id or str(uuid.uuid4())
    conv_title = (title.strip() if title and title.strip() else DEFAULT_TITLE)[:MAX_TITLE_LENGTH]

    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO conversations (id, tenant_id, user_id, title)
                VALUES (%s, %s, %s, %s)
                RETURNING id, tenant_id, user_id, title, created_at, updated_at;
                """,
                (cid, tenant_id, user_id, conv_title),
            )
            row = await cur.fetchone()
            await conn.commit()

            if not row:
                raise RuntimeError("Failed to create conversation record.")

            return {
                "id": str(row[0]),
                "tenant_id": str(row[1]),
                "user_id": str(row[2]),
                "title": row[3],
                "created_at": row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4]),
                "updated_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
            }


async def get_conversation(
    conversation_id: str,
    tenant_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    """
    Fetches a single conversation ensuring tenant and user match.
    Returns None if not found or unauthorized.
    """
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, tenant_id, user_id, title, created_at, updated_at
                FROM conversations
                WHERE id = %s AND tenant_id = %s AND user_id = %s;
                """,
                (conversation_id, tenant_id, user_id),
            )
            row = await cur.fetchone()
            if not row:
                return None

            return {
                "id": str(row[0]),
                "tenant_id": str(row[1]),
                "user_id": str(row[2]),
                "title": row[3],
                "created_at": row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4]),
                "updated_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
            }


async def list_conversations(
    tenant_id: str,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    Lists conversations for a user ordered by most recently updated first.
    """
    safe_limit = max(1, min(limit, 100))
    safe_offset = max(0, offset)

    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, tenant_id, user_id, title, created_at, updated_at
                FROM conversations
                WHERE tenant_id = %s AND user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s;
                """,
                (tenant_id, user_id, safe_limit, safe_offset),
            )
            rows = await cur.fetchall()

            return [
                {
                    "id": str(r[0]),
                    "tenant_id": str(r[1]),
                    "user_id": str(r[2]),
                    "title": r[3],
                    "created_at": r[4].isoformat() if hasattr(r[4], "isoformat") else str(r[4]),
                    "updated_at": r[5].isoformat() if hasattr(r[5], "isoformat") else str(r[5]),
                }
                for r in rows
            ]


async def update_conversation_title(
    conversation_id: str,
    tenant_id: str,
    user_id: str,
    title: str,
) -> dict[str, Any] | None:
    """
    Updates a conversation's title. Returns updated record or None if not found/unauthorized.
    """
    clean_title = (title.strip() if title else "")[:MAX_TITLE_LENGTH]
    if not clean_title:
        raise ValueError("Conversation title cannot be empty.")

    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE conversations
                SET title = %s, updated_at = NOW()
                WHERE id = %s AND tenant_id = %s AND user_id = %s
                RETURNING id, tenant_id, user_id, title, created_at, updated_at;
                """,
                (clean_title, conversation_id, tenant_id, user_id),
            )
            row = await cur.fetchone()
            await conn.commit()

            if not row:
                return None

            return {
                "id": str(row[0]),
                "tenant_id": str(row[1]),
                "user_id": str(row[2]),
                "title": row[3],
                "created_at": row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4]),
                "updated_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
            }


async def delete_conversation(
    conversation_id: str,
    tenant_id: str,
    user_id: str,
) -> bool:
    """
    Deletes a conversation and cascades message deletion.
    Returns True if deleted, False if not found or unauthorized.
    """
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                DELETE FROM conversations
                WHERE id = %s AND tenant_id = %s AND user_id = %s;
                """,
                (conversation_id, tenant_id, user_id),
            )
            deleted = cur.rowcount > 0
            await conn.commit()
            return deleted


async def append_message(
    conversation_id: str,
    tenant_id: str,
    user_id: str,
    role: str,
    content: str,
    citations: list[dict[str, Any]] | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    """
    Appends a new message (user or assistant) to an existing conversation.
    Atomically validates that the conversation belongs to tenant_id and user_id.
    Auto-updates conversation title if it is currently 'New Conversation' and the message is from 'user'.
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"Invalid message role '{role}'. Must be 'user' or 'assistant'.")

    clean_content = content.strip()
    if not clean_content:
        raise ValueError("Message content cannot be empty.")

    mid = message_id or str(uuid.uuid4())
    safe_citations = citations if citations is not None else []
    citations_json = json.dumps(safe_citations)

    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            # 1. Fetch current conversation to verify ownership and title
            await cur.execute(
                """
                SELECT id, title
                FROM conversations
                WHERE id = %s AND tenant_id = %s AND user_id = %s;
                """,
                (conversation_id, tenant_id, user_id),
            )
            conv_row = await cur.fetchone()
            if not conv_row:
                raise ValueError(
                    f"Conversation '{conversation_id}' not found or access denied for user."
                )

            current_title = conv_row[1]

            # 2. Insert message
            await cur.execute(
                """
                INSERT INTO messages (id, conversation_id, tenant_id, user_id, role, content, citations)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                RETURNING id, conversation_id, tenant_id, user_id, role, content, citations, created_at;
                """,
                (mid, conversation_id, tenant_id, user_id, role, clean_content, citations_json),
            )
            msg_row = await cur.fetchone()

            # 3. Auto-update title if it is the default title and a user message was posted
            if current_title == DEFAULT_TITLE and role == "user":
                new_title = generate_conversation_title(clean_content)
                await cur.execute(
                    """
                    UPDATE conversations
                    SET title = %s
                    WHERE id = %s AND tenant_id = %s AND user_id = %s;
                    """,
                    (new_title, conversation_id, tenant_id, user_id),
                )

            await conn.commit()

            if not msg_row:
                raise RuntimeError("Failed to insert message.")

            raw_citations = msg_row[6]
            parsed_citations = (
                raw_citations
                if isinstance(raw_citations, list)
                else (json.loads(raw_citations) if raw_citations else [])
            )

            return {
                "id": str(msg_row[0]),
                "conversation_id": str(msg_row[1]),
                "tenant_id": str(msg_row[2]),
                "user_id": str(msg_row[3]),
                "role": msg_row[4],
                "content": msg_row[5],
                "citations": parsed_citations,
                "created_at": msg_row[7].isoformat() if hasattr(msg_row[7], "isoformat") else str(msg_row[7]),
            }


async def get_conversation_messages(
    conversation_id: str,
    tenant_id: str,
    user_id: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Fetches the sliding-window of messages for a conversation in ascending chronological order.
    Selects the most recent `limit` messages (defaulting to settings.conversation_history_window).
    Enforces tenant_id and user_id isolation.
    """
    effective_limit = limit if limit is not None else settings.conversation_history_window
    safe_limit = max(1, min(effective_limit, 100))

    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT m.id, m.conversation_id, m.tenant_id, m.user_id, m.role, m.content, m.citations, m.created_at
                FROM (
                    SELECT id, conversation_id, tenant_id, user_id, role, content, citations, created_at
                    FROM messages
                    WHERE conversation_id = %s AND tenant_id = %s AND user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) m
                ORDER BY m.created_at ASC;
                """,
                (conversation_id, tenant_id, user_id, safe_limit),
            )
            rows = await cur.fetchall()

            results = []
            for r in rows:
                raw_citations = r[6]
                parsed_citations = (
                    raw_citations
                    if isinstance(raw_citations, list)
                    else (json.loads(raw_citations) if raw_citations else [])
                )
                results.append({
                    "id": str(r[0]),
                    "conversation_id": str(r[1]),
                    "tenant_id": str(r[2]),
                    "user_id": str(r[3]),
                    "role": r[4],
                    "content": r[5],
                    "citations": parsed_citations,
                    "created_at": r[7].isoformat() if hasattr(r[7], "isoformat") else str(r[7]),
                })
            return results


async def get_or_create_conversation(
    tenant_id: str,
    user_id: str,
    conversation_id: str | None = None,
    initial_title: str | None = None,
) -> dict[str, Any]:
    """
    Retrieves an existing conversation verifying ownership, or creates a new one if conversation_id is None.
    Raises ValueError if conversation_id is provided but not found or access is denied.
    """
    if conversation_id:
        conv = await get_conversation(conversation_id, tenant_id, user_id)
        if not conv:
            raise ValueError(
                f"Conversation '{conversation_id}' not found or access denied for user."
            )
        return conv

    return await create_conversation(tenant_id, user_id, title=initial_title)
