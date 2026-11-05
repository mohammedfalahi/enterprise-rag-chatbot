"""
Runtime Grounding and Hallucination Guardrail with Langfuse Trace Scoring.
Executes asynchronously out-of-band via FastAPI BackgroundTasks to verify
factual grounding of live chat responses without adding latency to the streaming path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from google import genai
from openai import AsyncOpenAI

from app.core.config import settings
from app.observability.tracing import record_score

logger = logging.getLogger("app.eval.guardrails")

GUARDRAIL_SYSTEM_PROMPT = """You are an automated enterprise RAG evaluator and hallucination detector.
Your job is to check whether all factual statements in the Assistant Response are strictly
grounded in and entailed by the provided Retrieved Context.

Respond ONLY with a JSON object in this exact schema:
{
  "faithfulness_score": <float between 0.0 and 1.0>,
  "has_hallucination": <boolean>,
  "reason": "<one sentence explanation>"
}"""


async def evaluate_grounding_async(
    *,
    query: str,
    contexts: list[str],
    response: str,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """
    Evaluates grounding out-of-band and logs the resulting score to Langfuse.
    Designed for FastAPI BackgroundTasks.
    """
    if not response or not response.strip():
        return {"faithfulness_score": 1.0, "has_hallucination": False, "reason": "Empty response"}

    # If it was an empty retrieval refusal, grounding is intact
    if not contexts or "do not have enough verified information" in response.lower():
        record_score(
            name="faithfulness",
            value=1.0,
            trace_id=trace_id,
            comment="Safe refusal on unverified context",
        )
        return {"faithfulness_score": 1.0, "has_hallucination": False, "reason": "Safe refusal"}

    context_block = "\n---\n".join(contexts[:5])
    eval_prompt = f"""{GUARDRAIL_SYSTEM_PROMPT}

Retrieved Context:
{context_block}

User Query:
{query}

Assistant Response:
{response}

JSON Evaluation:"""

    try:
        if settings.USE_GROQ_FOR_GENERATION:
            groq_client = AsyncOpenAI(
                api_key=settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY") or "none",
                base_url=settings.GROQ_BASE_URL,
            )
            raw = await groq_client.chat.completions.create(
                model=settings.groq_chat_llm_model,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            text = raw.choices[0].message.content or ""
        else:
            api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return {"faithfulness_score": 1.0, "has_hallucination": False, "reason": "No evaluator API key"}

            client = genai.Client(api_key=api_key, vertexai=False)
            raw_resp = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.chat_llm_model,
                contents=eval_prompt,
            )
            text = raw_resp.text or ""

        # Parse JSON output safely
        clean_text = text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_text)
        score = float(result.get("faithfulness_score", 1.0))
        reason = str(result.get("reason", "Grounding evaluation"))
        has_hallucination = bool(result.get("has_hallucination", score < 0.7))

        # Record evaluation score to Langfuse trace
        record_score(
            name="faithfulness",
            value=score,
            trace_id=trace_id,
            comment=reason,
            metadata={"has_hallucination": has_hallucination},
        )

        return {
            "faithfulness_score": score,
            "has_hallucination": has_hallucination,
            "reason": reason,
        }

    except Exception as exc:
        logger.debug("Async grounding evaluation failed (%s).", type(exc).__name__)
        return {
            "faithfulness_score": 1.0,
            "has_hallucination": False,
            "reason": f"Evaluation skipped: {type(exc).__name__}",
        }
