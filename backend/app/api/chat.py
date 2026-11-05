import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google import genai
from openai import AsyncOpenAI

from app.core.security import get_current_user, CurrentUser
from app.rag.hybrid_search import search_documents
from app.rag.query_rewriter import rewrite_query_for_retrieval
from app.rag.intent_router import classify_conversational_intent
from app.core.config import settings
from app.core.rate_limiter import rate_limit
from app.eval.guardrails import evaluate_grounding_async
from app.observability.redaction import scrub_sensitive_data
from app.observability.tracing import trace_span
from app.db.conversations import (
    append_message,
    create_conversation,
    generate_conversation_title,
    get_conversation,
    get_conversation_messages,
)

router = APIRouter()
logger = logging.getLogger(__name__)


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

    async def stream_generator():
        request_started = time.perf_counter()
        with trace_span(
            name="chat_request",
            as_type="agent",
            input={
                "message_chars": len(clean_message),
                "history_turns": len(history),
                "user_roles": user.roles,
            },
        ) as agent_span:
            try:
                # Conversational fast-path remains within the request trace.
                intent_route = classify_conversational_intent(clean_message)
                if intent_route is not None:
                    direct_reply = intent_route.response
                    await append_message(
                        conversation_id=conv_id,
                        tenant_id=user.tenant_id,
                        user_id=user.id,
                        role="assistant",
                        content=direct_reply,
                        citations=[],
                    )
                    yield f"data: {json.dumps({'type': 'citations', 'conversation_id': conv_id, 'title': conv_title, 'citations': []})}\n\n"
                    yield f"data: {json.dumps({'type': 'content', 'text': direct_reply})}\n\n"
                    yield "data: [DONE]\n\n"
                    agent_span.update(
                        output={
                            "outcome": "conversational_intent",
                            "latency_ms": round(
                                (time.perf_counter() - request_started) * 1000, 2
                            ),
                        }
                    )
                    return

                search_query = clean_message
                if history:
                    search_query = await rewrite_query_for_retrieval(clean_message, history)

                results = await search_documents(
                    query=search_query,
                    tenant_id=user.tenant_id,
                    user_roles=user.roles,
                    top_k=3,
                )

                if not results:
                    refusal_msg = (
                        "I do not have enough verified information to answer that question "
                        "based on your accessible documents."
                    )
                    await append_message(
                        conversation_id=conv_id,
                        tenant_id=user.tenant_id,
                        user_id=user.id,
                        role="assistant",
                        content=refusal_msg,
                        citations=[],
                    )
                    yield f"data: {json.dumps({'text': refusal_msg, 'conversation_id': conv_id})}\n\n"
                    yield "data: [DONE]\n\n"
                    agent_span.update(
                        output={
                            "outcome": "retrieval_refusal",
                            "results_count": 0,
                            "latency_ms": round(
                                (time.perf_counter() - request_started) * 1000, 2
                            ),
                        }
                    )
                    return

                with trace_span(
                    name="context_construction",
                    as_type="span",
                    input={"result_count": len(results)},
                ) as context_span:
                    context_blocks = []
                    citations = []
                    for result in results:
                        breadcrumbs = (
                            " > ".join(result.heading_path) if result.heading_path else result.title
                        )
                        context_blocks.append(
                            f"Document: {result.title} | Path: {breadcrumbs}\n{result.content}"
                        )
                        citations.append(
                            {
                                "chunk_id": result.chunk_id,
                                "document_id": result.document_id,
                                "title": result.title,
                                "heading_path": result.heading_path,
                                "score": round(result.rerank_score, 4),
                            }
                        )
                    context_str = "\n\n---\n\n".join(context_blocks)
                    context_span.update(
                        output={
                            "context_count": len(context_blocks),
                            "context_chars": len(context_str),
                            "citation_count": len(citations),
                        }
                    )

                history_section = ""
                if history:
                    history_turns = []
                    for message in history:
                        prefix = "User" if message.get("role") == "user" else "Assistant"
                        turn_content = message.get("content", "").strip()
                        if turn_content:
                            history_turns.append(f"{prefix}: {turn_content}")
                    if history_turns:
                        history_section = (
                            "Conversation History:\n" + "\n".join(history_turns) + "\n\n"
                        )

                system_prompt = """You are a helpful Enterprise Knowledge Assistant.
Answer the user's question accurately using ONLY the provided verified context below.
Verified context takes strict precedence over conversation history.
If the answer cannot be determined from the context, state that you do not have sufficient information."""

                prompt = f"""{system_prompt}

Context:
{context_str}

{history_section}Current Question:
{clean_message}
"""

                yield f"data: {json.dumps({'type': 'citations', 'conversation_id': conv_id, 'title': conv_title, 'citations': citations})}\n\n"
                full_text = []
                usage_details: dict[str, int] = {}
                generation_input: dict[str, Any] = {
                    "prompt_chars": len(prompt),
                    "context_count": len(results),
                    "history_turns": len(history),
                }
                if settings.langfuse_capture_content:
                    generation_input.update(
                        {
                            "system_prompt": scrub_sensitive_data(system_prompt),
                            "user_query": scrub_sensitive_data(clean_message),
                            "retrieved_context": scrub_sensitive_data(context_str),
                            "conversation_history": scrub_sensitive_data(history_section),
                        }
                    )
                use_groq = bool(settings.USE_GROQ_FOR_GENERATION)
                generation_model = (
                    settings.groq_chat_llm_model if use_groq else settings.chat_llm_model
                )
                span_name = "groq_generation" if use_groq else "gemini_generation"

                with trace_span(
                    name=span_name,
                    as_type="generation",
                    model=generation_model,
                    input=generation_input,
                ) as generation_span:
                    generation_started = time.perf_counter()
                    try:
                        if use_groq:
                            groq_client = AsyncOpenAI(
                                api_key=settings.GROQ_API_KEY
                                or os.environ.get("GROQ_API_KEY")
                                or "none",
                                base_url=settings.GROQ_BASE_URL,
                            )
                            stream_response = await groq_client.chat.completions.create(
                                model=generation_model,
                                messages=[{"role": "user", "content": prompt}],
                                stream=True,
                                stream_options={"include_usage": True},
                            )
                            async for chunk in stream_response:
                                usage = getattr(chunk, "usage", None)
                                if usage is not None:
                                    for source, target in (
                                        ("prompt_tokens", "input"),
                                        ("completion_tokens", "output"),
                                        ("total_tokens", "total"),
                                    ):
                                        token_count = getattr(usage, source, None)
                                        if token_count is not None:
                                            usage_details[target] = int(token_count)
                                if chunk.choices and chunk.choices[0].delta:
                                    chunk_text = (
                                        getattr(chunk.choices[0].delta, "content", "") or ""
                                    )
                                    if chunk_text:
                                        full_text.append(chunk_text)
                                        yield (
                                            f"data: {json.dumps({'type': 'content', 'text': chunk_text})}\n\n"
                                        )
                        else:
                            client = genai.Client(
                                api_key=settings.GEMINI_API_KEY
                                or os.environ.get("GEMINI_API_KEY"),
                                vertexai=False,
                            )
                            response = client.models.generate_content_stream(
                                model=generation_model,
                                contents=prompt,
                            )

                            async def emit_chunk(chunk):
                                chunk_text = getattr(chunk, "text", "") or ""
                                usage = getattr(chunk, "usage_metadata", None)
                                if usage is not None:
                                    for source, target in (
                                        ("prompt_token_count", "input"),
                                        ("candidates_token_count", "output"),
                                        ("total_token_count", "total"),
                                    ):
                                        token_count = getattr(usage, source, None)
                                        if token_count is not None:
                                            usage_details[target] = int(token_count)
                                if chunk_text:
                                    full_text.append(chunk_text)
                                    yield (
                                        f"data: {json.dumps({'type': 'content', 'text': chunk_text})}\n\n"
                                    )

                            if hasattr(response, "__aiter__"):
                                async for chunk in response:
                                    async for frame in emit_chunk(chunk):
                                        yield frame
                            else:
                                for chunk in response:
                                    async for frame in emit_chunk(chunk):
                                        yield frame
                    except Exception as exc:
                        generation_span.update(
                            output={
                                "output_chars": sum(map(len, full_text)),
                                "latency_ms": round(
                                    (time.perf_counter() - generation_started) * 1000, 2
                                ),
                            },
                            usage_details=usage_details or None,
                            level="ERROR",
                            status_message=type(exc).__name__,
                        )
                        raise
                    assistant_reply = "".join(full_text).strip()
                    generation_output: dict[str, Any] = {
                        "output_chars": len(assistant_reply),
                        "latency_ms": round(
                            (time.perf_counter() - generation_started) * 1000, 2
                        ),
                    }
                    if settings.langfuse_capture_content:
                        generation_output["generated_answer"] = scrub_sensitive_data(
                            assistant_reply
                        )
                    generation_span.update(
                        output=generation_output,
                        usage_details=usage_details or None,
                    )

                if assistant_reply:
                    await append_message(
                        conversation_id=conv_id,
                        tenant_id=user.tenant_id,
                        user_id=user.id,
                        role="assistant",
                        content=assistant_reply,
                        citations=citations,
                    )
                    # Non-blocking async runtime grounding and hallucination verification
                    asyncio.create_task(
                        evaluate_grounding_async(
                            query=search_query,
                            contexts=[r.content for r in results],
                            response=assistant_reply,
                        )
                    )

                yield "data: [DONE]\n\n"
                agent_span.update(
                    output={
                        "outcome": "completed",
                        "result_count": len(results),
                        "output_chars": len(assistant_reply),
                        "latency_ms": round((time.perf_counter() - request_started) * 1000, 2),
                    }
                )
            except Exception as exc:
                logger.error("Chat stream failed (%s).", type(exc).__name__)
                agent_span.update(
                    output={
                        "outcome": "error",
                        "latency_ms": round((time.perf_counter() - request_started) * 1000, 2),
                    },
                    level="ERROR",
                    status_message=type(exc).__name__,
                )
                yield f"data: {json.dumps({'error': 'The assistant could not complete this request. Please retry.'})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"X-Conversation-Id": conv_id},
    )
