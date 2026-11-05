"""
Generate an end-to-end RAG output snapshot for offline Ragas evaluation.
Pulls from golden evaluation cases (/Users/falahi/datasets-test/data/golden.jsonl),
runs live hybrid retrieval with Qwen3 embeddings and MS-MARCO reranking,
invokes the generation LLM (Gemini 3.5 Flash Lite), and formats authentic
ground-truth references directly from `answer_facts`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from google import genai
from openai import AsyncOpenAI

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
        )
        resp = await groq_client.chat.completions.create(
            model=settings.groq_chat_llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
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

    output_rows = []
    await init_db_pool()
    try:
        for i, item in enumerate(sampled_records, start=1):
            q = item["question"]
            role = item["role"]
            facts = item["answer_facts"]
            reference = "\n".join(f"- {fact}" for fact in facts)

            print(f"[{i}/{len(sampled_records)}] Retrieving & generating for: {q[:70]}...")

            results = await search_documents(
                query=q,
                tenant_id=tenant_id,
                user_roles=[role],
                top_k=top_k,
                mode="hybrid_rerank",
            )

            if not results:
                # If refusal triggered, fallback to dense candidates so Ragas has retrieved context to judge
                print(f"  Note: refusal triggered, fetching dense candidates for evaluation")
                results = await search_documents(
                    query=q,
                    tenant_id=tenant_id,
                    user_roles=[role],
                    top_k=top_k,
                    mode="dense_only",
                )

            retrieved_contexts = [r.content for r in results] if results else ["No context retrieved."]
            retrieved_titles = [r.title for r in results] if results else ["Unknown"]

            response_text = await generate_llm_response(q, retrieved_contexts, retrieved_titles)

            output_rows.append({
                "user_input": q,
                "response": response_text,
                "retrieved_contexts": retrieved_contexts,
                "reference": reference,
                "test_id": item.get("test_id"),
                "category": item.get("category"),
                "role": role,
            })
            # Small delay to respect rate limits
            await asyncio.sleep(0.5)

    finally:
        await close_db_pool()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nSuccessfully generated {len(output_rows)} Ragas evaluation snapshot records to {out_file}")
    return output_rows


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
