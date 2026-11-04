import json
import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google import genai

from app.core.security import get_current_user, CurrentUser
from app.rag.hybrid_search import search_documents
from app.rag.query_rewriter import rewrite_query_for_retrieval
from app.rag.intent_router import classify_conversational_intent
from app.core.config import settings
from app.core.rate_limiter import rate_limit
from app.observability.tracing import trace_span
from app.db.conversations import (
    append_message,
    create_conversation,
    generate_conversation_title,
    get_conversation,
    get_conversation_messages,
)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


def _validate_optional_uuid(val: str | None) -> None:
    if val is not None:
        try:
            uuid.UUID(val)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid conversation_id format: '{val}'. Must be a valid UUID.",
            )


@router.post("", dependencies=[Depends(rate_limit(max_requests=30, window_seconds=60, endpoint_tag="chat"))])
async def chat_endpoint(request: ChatRequest, user: CurrentUser = Depends(get_current_user)):
    clean_message = request.message.strip()
    if not clean_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty.",
        )

    _validate_optional_uuid(request.conversation_id)

    # 1. Resolve or create conversation with strict tenant/user ownership
    if request.conversation_id:
        conv = await get_conversation(
            conversation_id=request.conversation_id,
            tenant_id=user.tenant_id,
            user_id=user.id,
        )
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{request.conversation_id}' not found or access denied.",
            )
        # Fetch previous history before appending the current question
        history = await get_conversation_messages(
            conversation_id=conv["id"],
            tenant_id=user.tenant_id,
            user_id=user.id,
            limit=settings.conversation_history_window,
        )
    else:
        initial_title = generate_conversation_title(clean_message)
        conv = await create_conversation(
            tenant_id=user.tenant_id,
            user_id=user.id,
            title=initial_title,
        )
        history = []

    # 2. Persist the incoming user message
    await append_message(
        conversation_id=conv["id"],
        tenant_id=user.tenant_id,
        user_id=user.id,
        role="user",
        content=clean_message,
    )

    conv_id = conv["id"]
    conv_title = conv["title"]

    # 3. Intent-based routing: Fast-path for conversational intents (greetings, gratitude, goodbyes, identity)
    # Bypasses query rewriting, embedding generation, retrieval, and LLM calls
    intent_route = classify_conversational_intent(clean_message)
    if intent_route is not None:
        direct_reply = intent_route.response

        # Persist direct assistant response in conversation history
        await append_message(
            conversation_id=conv_id,
            tenant_id=user.tenant_id,
            user_id=user.id,
            role="assistant",
            content=direct_reply,
            citations=[],
        )

        async def direct_stream():
            yield f"data: {json.dumps({'type': 'citations', 'conversation_id': conv_id, 'title': conv_title, 'citations': []})}\n\n"
            yield f"data: {json.dumps({'type': 'content', 'text': direct_reply})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            direct_stream(),
            media_type="text/event-stream",
            headers={"X-Conversation-Id": conv_id},
        )

    with trace_span(
        name="chat_request",
        as_type="agent",
        input={
            "message": clean_message,
            "conversation_id": conv_id,
            "tenant_id": user.tenant_id,
            "roles": user.roles,
        },
    ) as agent_span:
        # 4. Contextualized query rewriting for follow-up questions (Option B)
        if history:
            search_query = await rewrite_query_for_retrieval(clean_message, history)
        else:
            search_query = clean_message

        # 4. Retrieve candidates enforcing tenant_id and user.roles at the DB level
        results = await search_documents(
            query=search_query,
            tenant_id=user.tenant_id,
            user_roles=user.roles,
            top_k=3,
        )

        if not results:
            refusal_msg = "I do not have enough verified information to answer that question based on your accessible documents."

            # Persist assistant refusal message
            await append_message(
                conversation_id=conv_id,
                tenant_id=user.tenant_id,
                user_id=user.id,
                role="assistant",
                content=refusal_msg,
                citations=[],
            )

            async def empty_stream():
                yield f"data: {json.dumps({'text': refusal_msg, 'conversation_id': conv_id})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                empty_stream(),
                media_type="text/event-stream",
                headers={"X-Conversation-Id": conv_id},
            )

        # 5. Build context string and citation metadata with breadcrumbs
        context_blocks = []
        citations = []
        for r in results:
            breadcrumbs = " > ".join(r.heading_path) if r.heading_path else r.title
            context_blocks.append(f"Document: {r.title} | Path: {breadcrumbs}\n{r.content}")
            citations.append({
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "title": r.title,
                "heading_path": r.heading_path,
                "score": round(r.rerank_score, 4),
            })

        context_str = "\n\n---\n\n".join(context_blocks)

        # 6. Format multi-turn conversation history
        history_section = ""
        if history:
            history_turns = []
            for m in history:
                prefix = "User" if m.get("role") == "user" else "Assistant"
                turn_content = m.get("content", "").strip()
                if turn_content:
                    history_turns.append(f"{prefix}: {turn_content}")
            if history_turns:
                history_section = "Conversation History:\n" + "\n".join(history_turns) + "\n\n"

        prompt = f"""You are a helpful Enterprise Knowledge Assistant.
Answer the user's question accurately using ONLY the provided verified context below.
Verified context takes strict precedence over conversation history.
If the answer cannot be determined from the context, state that you do not have sufficient information.

Context:
{context_str}

{history_section}Current Question:
{clean_message}
"""

        client = genai.Client(
            api_key=settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY"),
            vertexai=False,
        )

        async def stream_generator():
            try:
                # 1. Emit structured citations event first (including conversation metadata)
                yield f"data: {json.dumps({'type': 'citations', 'conversation_id': conv_id, 'title': conv_title, 'citations': citations})}\n\n"

                # 2. Stream generated content chunks without holding any DB connections
                response = client.models.generate_content_stream(
                    model=settings.chat_llm_model,
                    contents=prompt,
                )
                full_text = []

                if hasattr(response, "__aiter__"):
                    async for chunk in response:
                        chunk_text = getattr(chunk, "text", "") or ""
                        if chunk_text:
                            full_text.append(chunk_text)
                            yield f"data: {json.dumps({'type': 'content', 'text': chunk_text})}\n\n"
                else:
                    for chunk in response:
                        chunk_text = getattr(chunk, "text", "") or ""
                        if chunk_text:
                            full_text.append(chunk_text)
                            yield f"data: {json.dumps({'type': 'content', 'text': chunk_text})}\n\n"

                # 3. Persist assistant reply and citations after stream finishes
                assistant_reply = "".join(full_text).strip()
                if assistant_reply:
                    await append_message(
                        conversation_id=conv_id,
                        tenant_id=user.tenant_id,
                        user_id=user.id,
                        role="assistant",
                        content=assistant_reply,
                        citations=citations,
                    )

                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"X-Conversation-Id": conv_id},
        )
