"""
Generate an end-to-end RAG output snapshot for offline Ragas evaluation.
Pulls from golden evaluation cases (/Users/falahi/datasets-test/data/golden.jsonl),
runs live hybrid retrieval with Qwen3 embeddings and MS-MARCO reranking,
invokes the configured generation LLM, and formats authentic
ground-truth references directly from `answer_facts`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from google import genai
from openai import APIConnectionError, AsyncOpenAI, RateLimitError

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.db.connection import init_db_pool, close_db_pool
from app.rag.hybrid_search import search_documents

DEFAULT_GOLDEN_PATH = "/Users/falahi/datasets-test/data/golden.jsonl"
DEFAULT_TENANT_NAME = "redwood_inference"

SYSTEM_PROMPT = """You are the Enterprise Knowledge Assistant for Redwood Inference.
Your duty is to provide strictly accurate, concise, and professional answers to employee questions
using ONLY the provided enterprise documentation context.
Do NOT fabricate facts or extrapolate beyond what is explicitly stated in the context."""

MAX_RATE_LIMIT_RETRIES = 8


def _rate_limit_retry_delay(
    headers: Mapping[str, str], error_message: str, retry_number: int
) -> float:
    retry_after = next(
        (value for key, value in headers.items() if key.lower() == "retry-after"),
        None,
    )
    if retry_after is not None:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            delay = 0.0
        if math.isfinite(delay) and delay > 0:
            return delay

    match = re.search(
        r"try again in\s+(\d+(?:\.\d+)?)\s*seconds",
        error_message,
        flags=re.IGNORECASE,
    )
    if match:
        delay = float(match.group(1))
        if math.isfinite(delay) and delay > 0:
            return delay

    return min(5.0 * (2**retry_number), 60.0)


async def _generate_with_groq(client: AsyncOpenAI, prompt: str) -> str:
    for retry_number in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=settings.groq_chat_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError) as error:
            if retry_number >= MAX_RATE_LIMIT_RETRIES:
                raise
            if isinstance(error, RateLimitError):
                delay = _rate_limit_retry_delay(
                    error.response.headers, str(error), retry_number
                )
                failure_type = "rate limit"
            else:
                delay = min(5.0 * (2**retry_number), 60.0)
                failure_type = "connection error"
            print(
                f"[Groq generation] {failure_type}; retry "
                f"{retry_number + 1}/{MAX_RATE_LIMIT_RETRIES} in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
    raise RuntimeError("Groq generation exhausted retries without a result")


def _reference_for_item(item: dict[str, Any]) -> str:
    return "\n".join(f"- {fact}" for fact in item["answer_facts"])


def _load_snapshot_checkpoint(
    path: Path,
    selected_records: list[dict[str, Any]],
    generation_provider: str,
    generation_model: str,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}

    selected_by_id = {item["test_id"]: item for item in selected_records}
    checkpoint_rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            test_id = row.get("test_id")
            if not isinstance(test_id, str) or test_id not in selected_by_id:
                raise ValueError(
                    f"{path}:{line_number}: checkpoint test_id is not in the selected dataset"
                )
            if test_id in checkpoint_rows:
                raise ValueError(f"{path}:{line_number}: duplicate checkpoint test_id")

            item = selected_by_id[test_id]
            if (
                row.get("user_input") != item["question"]
                or row.get("reference") != _reference_for_item(item)
                or row.get("role") != item["role"]
                or row.get("generation_provider") != generation_provider
                or row.get("generation_model") != generation_model
            ):
                raise ValueError(
                    f"{path}:{line_number}: checkpoint does not match the selected "
                    "dataset and generation model"
                )
            if (
                not isinstance(row.get("response"), str)
                or not row["response"].strip()
                or not isinstance(row.get("retrieved_contexts"), list)
                or not row["retrieved_contexts"]
            ):
                raise ValueError(f"{path}:{line_number}: incomplete checkpoint record")
            checkpoint_rows[test_id] = row

    return checkpoint_rows


def _write_snapshot_checkpoint(
    path: Path,
    selected_records: list[dict[str, Any]],
    checkpoint_rows: dict[str, dict[str, Any]],
) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for item in selected_records:
            row = checkpoint_rows.get(item["test_id"])
            if row is not None:
                file.write(json.dumps(row, allow_nan=False) + "\n")
    temporary_path.replace(path)


async def generate_llm_response(query: str, contexts: list[str], titles: list[str]) -> str:
    context_str = "\n\n---\n\n".join(
        f"Document: {titles[i]}\nContent:\n{ctx}" for i, ctx in enumerate(contexts)
    )
    prompt = f"""{SYSTEM_PROMPT}

Retrieved Context:
{context_str}

User Question:
{query}

Answer:"""

    if settings.USE_GROQ_FOR_GENERATION:
        groq_client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY") or "none",
            base_url=settings.GROQ_BASE_URL,
            timeout=90.0,
            max_retries=0,
        )
        return await _generate_with_groq(groq_client, prompt)
    else:
        client = genai.Client(
            api_key=settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY"),
            vertexai=False,
        )
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model=settings.chat_llm_model,
            contents=prompt,
        )
        return resp.text or ""


async def build_snapshot(
    golden_path: str | Path = DEFAULT_GOLDEN_PATH,
    output_path: str | Path = "evaluation_results/ragas_snapshot_20.jsonl",
    tenant_name: str = DEFAULT_TENANT_NAME,
    sample_size: int = 20,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    tenant_id = str(uuid.uuid5(uuid.NAMESPACE_OID, tenant_name))
    source = Path(golden_path)

    with source.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Filter to cases with authentic answer facts
    answerable = [
        r for r in records
        if r.get("expected_behavior") in ("answer", "partial") and r.get("answer_facts")
    ]
    print(f"Loaded {len(answerable)} answerable cases with authentic answer_facts from {source}")

    # Select balanced sample across categories
    categories = sorted(list({r.get("category", "basic") for r in answerable}))
    sampled_records = []
    
    # Pick round-robin across categories to ensure diverse coverage
    cat_buckets = {c: [r for r in answerable if r.get("category") == c] for c in categories}
    idx = 0
    while len(sampled_records) < sample_size and any(cat_buckets.values()):
        cat = categories[idx % len(categories)]
        if cat_buckets[cat]:
            sampled_records.append(cat_buckets[cat].pop(0))
        idx += 1

    print(f"Selected {len(sampled_records)} representative benchmark cases across categories: {categories}")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    generation_provider = (
        "groq" if settings.USE_GROQ_FOR_GENERATION else "gemini"
    )
    generation_model = (
        settings.groq_chat_llm_model
        if settings.USE_GROQ_FOR_GENERATION
        else settings.chat_llm_model
    )
    output_rows = _load_snapshot_checkpoint(
        out_file,
        sampled_records,
        generation_provider,
        generation_model,
    )
    if output_rows:
        print(f"Resuming from {len(output_rows)} completed checkpoint records")

    await init_db_pool()
    try:
        for i, item in enumerate(sampled_records, start=1):
            test_id = item["test_id"]
            if test_id in output_rows:
                continue
            q = item["question"]
            role = item["role"]
            reference = _reference_for_item(item)

            print(f"[{i}/{len(sampled_records)}] Retrieving & generating for: {q[:70]}...")

            results = await search_documents(
                query=q,
                tenant_id=tenant_id,
                user_roles=[role],
                top_k=top_k,
                mode="hybrid_rerank",
            )

            if not results:
                # Production path: when retrieval is refused, return production refusal with empty context
                print(f"  Note: refusal gate triggered for {test_id} (score below threshold)")
                retrieved_contexts = []
                response_text = "I do not have enough verified information to answer that question based on your accessible documents."
            else:
                retrieved_contexts = [r.content for r in results]
                retrieved_titles = [r.title for r in results]
                response_text = await generate_llm_response(q, retrieved_contexts, retrieved_titles)

            output_rows_row = {
                "user_input": q,
                "response": response_text,
                "retrieved_contexts": retrieved_contexts,
                "reference": reference,
                "test_id": item.get("test_id"),
                "question_id": item.get("question_id"),
                "category": item.get("category"),
                "role": role,
                "generation_provider": generation_provider,
                "generation_model": generation_model,
            }
            output_rows[test_id] = output_rows_row
            _write_snapshot_checkpoint(out_file, sampled_records, output_rows)
            # Small delay to respect rate limits
            await asyncio.sleep(0.5)

    finally:
        await close_db_pool()

    ordered_rows = [output_rows[item["test_id"]] for item in sampled_records]
    print(f"\nSuccessfully generated {len(ordered_rows)} Ragas evaluation snapshot records to {out_file}")
    return ordered_rows


def main():
    parser = argparse.ArgumentParser(description="Generate Ragas evaluation snapshot with live RAG responses")
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output", type=Path, default="evaluation_results/ragas_snapshot_20.jsonl")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(
        build_snapshot(
            golden_path=args.golden_path,
            output_path=args.output,
            sample_size=args.sample_size,
            top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    main()
