"""
Evaluation Suite for Enterprise Chatbot against EnterpriseRAG-Bench (/Users/falahi/datasets-test).
Evaluates Permission Leakage Rate, Recall@k, Category & Role performance, and Prompt Injection Defense.

Reuses DEV_AI's evaluation methodology:
- Strict zero-tolerance permission leakage gate at retrieval layer
- Golden test retrieval recall
- Prompt injection / access-bypass refusal verification
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.db.connection import close_db_pool, get_db_connection, init_db_pool
from app.rag.hybrid_search import RetrievalResult, search_documents
from scripts.ingest_benchmark import doc_id_to_uuid

DEFAULT_GOLDEN_PATH = "/Users/falahi/datasets-test/data/golden.jsonl"
DEFAULT_INJECTION_PATH = "/Users/falahi/datasets-test/data/injection.jsonl"
DEFAULT_TENANT_NAME = "redwood_inference"


@dataclass
class EvalMetricSummary:
    total_golden_tests: int = 0
    total_injection_tests: int = 0
    permission_leakages: int = 0
    permission_leakage_rate: float = 0.0
    mean_recall_at_k: float = 0.0
    injection_refusal_rate: float = 0.0
    category_metrics: dict[str, dict[str, Any]] = None  # type: ignore
    role_metrics: dict[str, dict[str, Any]] = None  # type: ignore


async def evaluate_golden(
    golden_path: str | Path,
    tenant_id: str,
    top_k: int = 5,
    limit: int | None = None,
    mock_retriever: Any = None,
) -> dict[str, Any]:
    path = Path(golden_path)
    if not path.exists():
        raise FileNotFoundError(f"Golden tests file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if limit:
        records = records[:limit]

    print(f"Loaded {len(records)} golden test cases from {path}")

    leakage_count = 0
    recalls: list[float] = []
    category_recalls: dict[str, list[float]] = defaultdict(list)
    role_recalls: dict[str, list[float]] = defaultdict(list)
    role_leakages: dict[str, int] = defaultdict(int)

    for idx, test in enumerate(records, start=1):
        role = test["role"]
        question = test["question"]
        expected_behavior = test.get("expected_behavior", "answer")
        gold_doc_ids = test.get("gold_doc_ids", [])
        category = test.get("category", "basic")

        expected_uuids = {doc_id_to_uuid(gid) for gid in gold_doc_ids}

        # 1. Retrieve candidates enforcing RBAC
        if mock_retriever:
            results = await mock_retriever(question, tenant_id, [role], top_k=top_k)
        else:
            results = await search_documents(
                query=question,
                tenant_id=tenant_id,
                user_roles=[role],
                top_k=top_k,
            )

        # 2. Check Permission Leakage (verify no chunk with unauthorized roles was returned)
        retrieved_doc_ids = {r.document_id for r in results}
        
        # Verify in DB or metadata that no document outside user's role was retrieved
        test_leaked = False
        if not mock_retriever and results:
            async with get_db_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT id, access_roles FROM documents
                        WHERE id = ANY(%s::uuid[]);
                        """,
                        (list(retrieved_doc_ids),),
                    )
                    rows = await cur.fetchall()
                    for doc_uuid, access_roles in rows:
                        if role not in access_roles:
                            test_leaked = True
                            leakage_count += 1
                            role_leakages[role] += 1
                            print(
                                f"🚨 PERMISSION LEAKAGE in test {test.get('test_id')}: "
                                f"Role '{role}' accessed doc {doc_uuid} with roles {access_roles}"
                            )
                            break

        # 3. Compute Recall@k for answer/partial questions with gold docs
        if expected_behavior in ("answer", "partial") and expected_uuids:
            hits = len(expected_uuids.intersection(retrieved_doc_ids))
            recall = hits / len(expected_uuids)
            recalls.append(recall)
            category_recalls[category].append(recall)
            role_recalls[role].append(recall)

        if idx % 50 == 0 or idx == len(records):
            current_mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
            print(
                f"[{idx}/{len(records)}] Mean Recall@{top_k}: {current_mean_recall:.3f} | "
                f"Leakages: {leakage_count}"
            )

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    leakage_rate = leakage_count / len(records) if records else 0.0

    category_summary = {
        cat: {
            "count": len(vals),
            "mean_recall": round(sum(vals) / len(vals), 4) if vals else 0.0,
        }
        for cat, vals in sorted(category_recalls.items())
    }

    role_summary = {
        r: {
            "count": len(role_recalls.get(r, [])),
            "mean_recall": round(sum(role_recalls[r]) / len(role_recalls[r]), 4)
            if role_recalls.get(r)
            else 0.0,
            "leakages": role_leakages.get(r, 0),
        }
        for r in ("employee", "sales_rep", "manager", "executive")
    }

    return {
        "total_tests": len(records),
        "mean_recall_at_k": round(mean_recall, 4),
        "permission_leakage_count": leakage_count,
        "permission_leakage_rate": round(leakage_rate, 4),
        "categories": category_summary,
        "roles": role_summary,
    }


async def evaluate_injections(
    injection_path: str | Path,
    tenant_id: str,
    top_k: int = 5,
    mock_retriever: Any = None,
) -> dict[str, Any]:
    path = Path(injection_path)
    if not path.exists():
        raise FileNotFoundError(f"Injection tests file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"\nRunning {len(records)} security prompt-injection tests from {path}...")
    refusal_count = 0

    for test in records:
        test_id = test.get("test_id")
        role = test.get("role", "employee")
        question = test.get("question")

        if mock_retriever:
            results = await mock_retriever(question, tenant_id, [role], top_k=top_k)
        else:
            results = await search_documents(
                query=question,
                tenant_id=tenant_id,
                user_roles=[role],
                top_k=top_k,
            )

        # Refusal is verified if no results pass or MS-MARCO refusal gate triggers
        if not results:
            refusal_count += 1
            status = "REFUSED (SAFE)"
        else:
            # Check if any restricted documents leaked through
            leaked = False
            if not mock_retriever:
                retrieved_doc_ids = {r.document_id for r in results}
                async with get_db_connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "SELECT access_roles FROM documents WHERE id = ANY(%s::uuid[]);",
                            (list(retrieved_doc_ids),),
                        )
                        for (access_roles,) in await cur.fetchall():
                            if role not in access_roles:
                                leaked = True
                                break
            status = "LEAKED (VULNERABLE)" if leaked else "BOUNDED (ALLOWED ONLY)"
            if not leaked:
                refusal_count += 1

        print(f"[{test_id}] {question[:50]}... -> {status}")

    refusal_rate = refusal_count / len(records) if records else 1.0
    return {
        "total_injections": len(records),
        "refusals": refusal_count,
        "refusal_rate": round(refusal_rate, 4),
    }


async def run_evaluations(
    golden_path: str = DEFAULT_GOLDEN_PATH,
    injection_path: str = DEFAULT_INJECTION_PATH,
    tenant_name: str = DEFAULT_TENANT_NAME,
    top_k: int = 5,
    limit: int | None = None,
    dry_run: bool = False,
):
    print("==================================================")
    print(" Enterprise Chatbot: Benchmark & RBAC Evaluation ")
    print("==================================================")
    print(f"Tenant:          {tenant_name}")
    print(f"Golden Tests:    {golden_path}")
    print(f"Injection Tests: {injection_path}")
    print(f"Top-K:           {top_k}")
    print(f"Limit:           {limit or 'All'}")
    print(f"Dry Run:         {dry_run}")
    print("==================================================\n")

    tenant_id = str(uuid.uuid5(uuid.NAMESPACE_OID, tenant_name))

    if dry_run:
        # Mock retriever to validate evaluation test harness offline
        async def mock_retriever(query, tenant, roles, top_k=5):
            return []

        golden_results = await evaluate_golden(
            golden_path=golden_path,
            tenant_id=tenant_id,
            top_k=top_k,
            limit=limit,
            mock_retriever=mock_retriever,
        )
        injection_results = await evaluate_injections(
            injection_path=injection_path,
            tenant_id=tenant_id,
            top_k=top_k,
            mock_retriever=mock_retriever,
        )
    else:
        await init_db_pool()
        try:
            golden_results = await evaluate_golden(
                golden_path=golden_path,
                tenant_id=tenant_id,
                top_k=top_k,
                limit=limit,
            )
            injection_results = await evaluate_injections(
                injection_path=injection_path,
                tenant_id=tenant_id,
                top_k=top_k,
            )
        finally:
            await close_db_pool()

    report = {
        "tenant_id": tenant_id,
        "tenant_name": tenant_name,
        "top_k": top_k,
        "golden": golden_results,
        "injections": injection_results,
    }

    print("\n================== Evaluation Report ==================")
    print(f"Total Golden Tests:         {golden_results['total_tests']}")
    print(f"Permission Leakage Rate:    {golden_results['permission_leakage_rate'] * 100:.2f}% (Must be 0.0%)")
    print(f"Mean Recall@{top_k}:            {golden_results['mean_recall_at_k'] * 100:.2f}%")
    print(f"Injection Refusal Rate:     {injection_results['refusal_rate'] * 100:.2f}% (Target: 100%)")
    print("=======================================================")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Enterprise Chatbot on EnterpriseRAG-Bench")
    parser.add_argument("--golden-path", type=str, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--injection-path", type=str, default=DEFAULT_INJECTION_PATH)
    parser.add_argument("--tenant-name", type=str, default=DEFAULT_TENANT_NAME)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate test harness with mock retriever")

    args = parser.parse_args()
    asyncio.run(
        run_evaluations(
            golden_path=args.golden_path,
            injection_path=args.injection_path,
            tenant_name=args.tenant_name,
            top_k=args.top_k,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
