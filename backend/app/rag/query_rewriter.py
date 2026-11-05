"""
Contextualized Query Rewriter for Multi-Turn RAG Retrieval (Option B).
Translates conversational follow-up questions containing pronouns or references
into standalone search queries suitable for hybrid vector and full-text retrieval.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from google import genai
from openai import AsyncOpenAI

from app.core.config import settings
from app.observability.redaction import scrub_sensitive_data
from app.observability.tracing import trace_span

logger = logging.getLogger(__name__)


def is_rate_limit_error(exc: Exception) -> bool:
    """Detects HTTP 429 / resource exhausted rate limit exceptions."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    exc_type = type(exc).__name__
    if "RateLimit" in exc_type:
        return True
    err_msg = str(exc).lower()
    return "429" in err_msg or "rate limit" in err_msg or "resource_exhausted" in err_msg


def is_auth_error(exc: Exception) -> bool:
    """Detects HTTP 401/403 authentication or invalid API key exceptions."""
    status_code = getattr(exc, "status_code", None)
    if status_code in (401, 403):
        return True
    exc_type = type(exc).__name__
    return "Authentication" in exc_type or "PermissionDenied" in exc_type


def _call_gemini_rewrite(prompt: str, model_name: str) -> tuple[str, dict[str, int]]:
    """Synchronous invocation of Gemini for query rewriting."""
    client = genai.Client(
        api_key=settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY"),
        vertexai=False,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    rewritten_text = response.text if (response and response.text) else ""
    usage_details: dict[str, int] = {}
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        for source, target in (
            ("prompt_token_count", "input"),
            ("candidates_token_count", "output"),
            ("total_token_count", "total"),
        ):
            token_count = getattr(usage, source, None)
            if token_count is not None:
                usage_details[target] = int(token_count)
    return rewritten_text, usage_details


async def _call_groq_rewrite(
    prompt: str,
    model_name: str,
    max_retries: int = 2,
    base_delay: float = 0.5,
) -> tuple[str, dict[str, int]]:
    """Invokes primary Groq model with bounded exponential backoff retries on HTTP 429."""
    groq_client = AsyncOpenAI(
        api_key=settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY") or "none",
        base_url=settings.GROQ_BASE_URL,
    )
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = await groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            rewritten_text = ""
            if response and response.choices:
                first_choice = response.choices[0]
                if first_choice.message and first_choice.message.content:
                    rewritten_text = first_choice.message.content

            usage_details: dict[str, int] = {}
            usage = getattr(response, "usage", None)
            if usage is not None:
                for source, target in (
                    ("prompt_tokens", "input"),
                    ("completion_tokens", "output"),
                    ("total_tokens", "total"),
                ):
                    token_count = getattr(usage, source, None)
                    if token_count is not None:
                        usage_details[target] = int(token_count)
            return rewritten_text, usage_details
        except Exception as exc:
            last_exc = exc
            if is_rate_limit_error(exc) and not is_auth_error(exc) and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Primary LLM query rewriter rate limited (429). Retrying in %.2fs (attempt %d/%d)...",
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)
                continue
            raise
    if last_exc:
        raise last_exc
    return "", {}


def build_rewrite_prompt(current_question: str, history_messages: list[dict[str, Any]]) -> str:
    """Builds the instruction prompt for conversational query reformulation."""
    history_turns = []
    for m in history_messages:
        prefix = "User" if m.get("role") == "user" else "Assistant"
        content = m.get("content", "").strip()
        if content:
            history_turns.append(f"{prefix}: {content}")

    history_text = "\n".join(history_turns)

    return f"""You are an expert search query formulation assistant for an enterprise retrieval engine.
Given the conversation history and the latest follow-up question from the user, rewrite the follow-up question into a single, standalone search query suitable for document retrieval.

Rules:
1. Resolve all pronouns (such as "it", "they", "its", "that", "this", "he", "she") into specific named entities from the conversation history.
2. If the user's question is already fully standalone and does not depend on prior turns, return it unchanged.
3. Output ONLY the rewritten search query.
4. Do NOT answer the question.
5. Do NOT add any preamble, quotes, markdown formatting, or explanations.

Conversation History:
{history_text}

Follow-up Question:
{current_question}

Standalone Search Query:"""


async def rewrite_query_for_retrieval(
    current_question: str,
    history_messages: list[dict[str, Any]],
) -> str:
    """
    Rewrites a follow-up question using Gemini if history exists.
    Falls back gracefully to current_question on any failure or if history is empty.
    """
    clean_question = current_question.strip()
    if not clean_question:
        return current_question

    if not history_messages:
        return current_question

    prompt = build_rewrite_prompt(clean_question, history_messages)

    use_groq = bool(settings.USE_GROQ_FOR_QUERY_REWRITE)
    model_name = settings.groq_query_rewrite_model if use_groq else settings.query_rewrite_model

    trace_input: dict[str, Any] = {
        "question_chars": len(clean_question),
        "history_turns": len(history_messages),
    }
    if settings.langfuse_capture_content:
        trace_input["original_query"] = scrub_sensitive_data(clean_question)

    with trace_span(
        name="query_rewriting",
        as_type="generation",
        model=model_name,
        input=trace_input,
    ) as rewrite_span:
        fallback_activated = False
        fallback_model: str | None = None
        try:
            rewritten_text = ""
            usage_details: dict[str, int] = {}

            if use_groq:
                try:
                    rewritten_text, usage_details = await _call_groq_rewrite(prompt, model_name)
                except Exception as exc:
                    if is_rate_limit_error(exc) and not is_auth_error(exc):
                        logger.warning(
                            "Primary LLM query rewriter rate limit (429) persisted. Activating fallback to Gemini (%s)...",
                            settings.query_rewrite_model,
                        )
                        fallback_activated = True
                        fallback_model = settings.query_rewrite_model
                        rewritten_text, usage_details = await asyncio.to_thread(
                            _call_gemini_rewrite, prompt, settings.query_rewrite_model
                        )
                    else:
                        raise
            else:
                rewritten_text, usage_details = await asyncio.to_thread(
                    _call_gemini_rewrite, prompt, model_name
                )

            if rewritten_text:
                rewritten = rewritten_text.strip().strip('"\'')
                if rewritten:
                    rewrite_output: dict[str, Any] = {
                        "rewritten_query_chars": len(rewritten),
                        "fallback": fallback_activated,
                    }
                    if fallback_activated and fallback_model:
                        rewrite_output["fallback_model"] = fallback_model
                    if settings.langfuse_capture_content:
                        rewrite_output["rewritten_query"] = scrub_sensitive_data(rewritten)
                    rewrite_span.update(
                        output=rewrite_output,
                        usage_details=usage_details or None,
                    )
                    return rewritten
        except Exception as exc:
            logger.warning(
                "Query rewriting failed (%s); falling back to original query.",
                type(exc).__name__,
            )
            rewrite_span.update(
                output={"fallback": True, "error": type(exc).__name__},
                level="WARNING",
                status_message=type(exc).__name__,
            )
            return clean_question

        logger.warning("Query rewriting returned no usable text; using the original question.")
        rewrite_span.update(output={"fallback": True})
        return clean_question
