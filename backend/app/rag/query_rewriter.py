"""
Contextualized Query Rewriter for Multi-Turn RAG Retrieval (Option B).
Translates conversational follow-up questions containing pronouns or references
into standalone search queries suitable for hybrid vector and full-text retrieval.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from google import genai

from app.core.config import settings

logger = logging.getLogger(__name__)


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

    try:
        client = genai.Client(
            api_key=settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY"),
            vertexai=False,
        )
        response = client.models.generate_content(
            model=settings.query_rewrite_model,
            contents=prompt,
        )
        if response and response.text:
            rewritten = response.text.strip().strip('"\'')
            if rewritten:
                return rewritten
    except Exception as e:
        logger.warning(
            "Query rewriting failed, falling back to original query: %s",
            e,
            exc_info=True,
        )

    return clean_question
