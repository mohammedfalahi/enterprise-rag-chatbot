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
import math
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.connection import close_db_pool, get_db_connection, init_db_pool
from app.rag.hybrid_search import RetrievalResult, search_documents
from scripts.ingest_benchmark import doc_id_to_uuid

DEFAULT_GOLDEN_PATH = "/Users/falahi/datasets-test/data/golden.jsonl"
DEFAULT_INJECTION_PATH = "/Users/falahi/datasets-test/data/injection.jsonl"
DEFAULT_TENANT_NAME = "redwood_inference"


def ranking_metrics_at_k(
    ranked_document_ids: list[str],
    gold_document_ids: set[str],
    k: int = 5,
) -> dict[str, float]:
    """Compute binary document-level metrics after deduplicating ranked documents."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not gold_document_ids:
        raise ValueError("gold_document_ids must not be empty")

    unique_ranked_ids = list(dict.fromkeys(ranked_document_ids))[:k]
    relevant_ranks = [
        rank
        for rank, document_id in enumerate(unique_ranked_ids, start=1)
        if document_id in gold_document_ids
    ]
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_relevant_count = min(len(gold_document_ids), k)
    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )
    return {
        f"precision@{k}": len(relevant_ranks) / k,
        f"recall@{k}": len(relevant_ranks) / len(gold_document_ids),
        "mrr": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
    }


async def evaluate_golden(
    golden_path: str | Path,
    tenant_id: str,
    top_k: int = 5,
    limit: int | None = None,
    mock_retriever: Any = None,
    mode: str = "hybrid_rerank",
    candidate_limit: int = 10,
    k_rrf: int = 60,
    rerank_pool_size: int = 5,
    save_per_query_path: str | Path | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    path = Path(golden_path)
    if not path.exists():
        raise FileNotFoundError(f"Golden tests file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if limit is not None:
        if limit < 0:
            raise ValueError("limit must not be negative")
        records = records[:limit]

    print(
        f"Loaded {len(records)} golden test cases from {path} "
        f"[mode={mode}, top_k={top_k}, cand_limit={candidate_limit}, k_rrf={k_rrf}, pool={rerank_pool_size}]"
    )

    leakage_count = 0
    query_metrics: list[dict[str, float]] = []
    category_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    role_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    role_leakages: dict[str, int] = defaultdict(int)
    per_query_records: list[dict[str, Any]] = []

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
                mode=mode,
                candidate_limit=candidate_limit,
                k_rrf=k_rrf,
                rerank_pool_size=rerank_pool_size,
            )

        # 2. Check Permission Leakage (verify no chunk with unauthorized roles was returned)
        retrieved_doc_ids = {r.document_id for r in results}
        
        # Verify in DB or metadata that no document outside user's role was retrieved
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
                            leakage_count += 1
                            role_leakages[role] += 1
                            print(
                                f"🚨 PERMISSION LEAKAGE in test {test.get('test_id')}: "
                                f"Role '{role}' accessed doc {doc_uuid} with roles {access_roles}"
                            )
                            break

        # 3. Compute Recall@k for answer/partial questions with gold docs
        metrics = None
        if expected_behavior in ("answer", "partial") and expected_uuids:
            metrics = ranking_metrics_at_k(
                [result.document_id for result in results],
                expected_uuids,
                k=top_k,
            )
            query_metrics.append(metrics)
            category_metrics[category].append(metrics)
            role_metrics[role].append(metrics)

        per_query_records.append(
            {
                "test_id": test.get("test_id"),
                "question_id": test.get("question_id"),
                "role": role,
                "category": category,
                "expected_behavior": expected_behavior,
                "question": question,
                "answer_facts": test.get("answer_facts", []),
                "gold_doc_ids": list(gold_doc_ids),
                "gold_uuids": list(expected_uuids),
                "retrieved_doc_ids": [r.document_id for r in results],
                "retrieved_chunk_ids": [r.chunk_id for r in results],
                "retrieved_titles": [r.title for r in results],
                "metrics": metrics,
            }
        )

        if idx % 50 == 0 or idx == len(records):
            current_mean_recall = (
                sum(metrics[f"recall@{top_k}"] for metrics in query_metrics)
                / len(query_metrics)
                if query_metrics
                else 0.0
            )
            print(
                f"[{idx}/{len(records)}] Mean Recall@{top_k}: {current_mean_recall:.3f} | "
                f"Leakages: {leakage_count}"
            )

    if save_per_query_path:
        out_path = Path(save_per_query_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for item in per_query_records:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(per_query_records)} per-query records to {out_path}")

    mean_recall = (
        sum(metrics[f"recall@{top_k}"] for metrics in query_metrics) / len(query_metrics)
        if query_metrics
        else None
    )
    leakage_rate = leakage_count / len(records) if records else None

    def summarize_metrics(values: list[dict[str, float]]) -> dict[str, float]:
        if not values:
            return {}
        return {
            metric_name: round(
                sum(result[metric_name] for result in values) / len(values), 4
            )
            for metric_name in values[0]
        }

    category_summary = {
        category: {"count": len(values), **summarize_metrics(values)}
        for category, values in sorted(category_metrics.items())
    }

    role_summary = {
        role: {
            "count": len(role_metrics.get(role, [])),
            **summarize_metrics(role_metrics.get(role, [])),
            "leakages": role_leakages.get(role, 0),
        }
        for role in ("employee", "sales_rep", "manager", "executive")
    }

    return {
        "total_tests": len(records),
        "mean_recall_at_k": round(mean_recall, 4) if mean_recall is not None else None,
        "retrieval_metrics": {
            "count": len(query_metrics),
            **summarize_metrics(query_metrics),
        },
        "permission_leakage_count": leakage_count,
        "permission_leakage_rate": round(leakage_rate, 4) if leakage_rate is not None else None,
        "categories": category_summary,
        "roles": role_summary,
    }


async def evaluate_injections(
    injection_path: str | Path,
    tenant_id: str,
    top_k: int = 5,
    mock_retriever: Any = None,
    mode: str = "hybrid_rerank",
    candidate_limit: int = 10,
    k_rrf: int = 60,
    rerank_pool_size: int = 5,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    path = Path(injection_path)
    if not path.exists():
        raise FileNotFoundError(f"Injection tests file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"\nRunning {len(records)} security prompt-injection tests from {path}...")
    refusal_count = 0
    permission_leakage_count = 0
    bounded_unverified_count = 0

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
                mode=mode,
                candidate_limit=candidate_limit,
                k_rrf=k_rrf,
                rerank_pool_size=rerank_pool_size,
            )

        # Refusal is verified if no results pass or MS-MARCO refusal gate triggers
        if not results:
            refusal_count += 1
            status = "REFUSED (SAFE)"
        else:
            leaked = False
            if mock_retriever:
                bounded_unverified_count += 1
                status = "BOUNDED (NOT VERIFIED)"
            else:
                retrieved_doc_ids = {r.document_id for r in results}
                async with get_db_connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "SELECT access_roles FROM documents WHERE id = ANY(%s::uuid[]);",
                            (list(retrieved_doc_ids),),
                        )
                        leaked = any(
                            role not in access_roles for (access_roles,) in await cur.fetchall()
                        )
                if leaked:
                    permission_leakage_count += 1
                    status = "LEAKED (VULNERABLE)"
                else:
                    status = "BOUNDED (NOT REFUSED)"

        print(f"[{test_id}] {question[:50]}... -> {status}")

    refusal_rate = refusal_count / len(records) if records else None
    return {
        "total_injections": len(records),
        "refusals": refusal_count,
        "refusal_rate": round(refusal_rate, 4) if refusal_rate is not None else None,
        "permission_leakage_count": permission_leakage_count,
        "bounded_unverified_count": bounded_unverified_count,
    }


async def run_evaluations(
    golden_path: str = DEFAULT_GOLDEN_PATH,
    injection_path: str = DEFAULT_INJECTION_PATH,
    tenant_name: str = DEFAULT_TENANT_NAME,
    top_k: int = 5,
    limit: int | None = None,
    dry_run: bool = False,
    mode: str = "hybrid_rerank",
    candidate_limit: int = 10,
    k_rrf: int = 60,
    rerank_pool_size: int = 5,
    save_per_query_path: str | Path | None = None,
):
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if limit is not None and limit < 0:
        raise ValueError("limit must not be negative")
    print("==================================================")
    print(" Enterprise Chatbot: Benchmark & RBAC Evaluation ")
    print("==================================================")
    print(f"Tenant:          {tenant_name}")
    print(f"Golden Tests:    {golden_path}")
    print(f"Injection Tests: {injection_path}")
    print(f"Mode:            {mode}")
    print(f"Top-K:           {top_k}")
    print(f"Candidate Limit: {candidate_limit}")
    print(f"RRF k:           {k_rrf}")
    print(f"Rerank Pool:     {rerank_pool_size}")
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
                mode=mode,
                candidate_limit=candidate_limit,
                k_rrf=k_rrf,
                rerank_pool_size=rerank_pool_size,
                save_per_query_path=save_per_query_path,
            )
            injection_results = await evaluate_injections(
                injection_path=injection_path,
                tenant_id=tenant_id,
                top_k=top_k,
                mode=mode,
                candidate_limit=candidate_limit,
                k_rrf=k_rrf,
                rerank_pool_size=rerank_pool_size,
            )
        finally:
            await close_db_pool()

    report = {
        "tenant_id": tenant_id,
        "tenant_name": tenant_name,
        "top_k": top_k,
        "mode": mode,
        "candidate_limit": candidate_limit,
        "k_rrf": k_rrf,
        "rerank_pool_size": rerank_pool_size,
        "golden": golden_results,
        "injections": injection_results,
    }

    print("\n================== Evaluation Report ==================")
    print(f"Total Golden Tests:         {golden_results['total_tests']}")
    leakage_rate = golden_results["permission_leakage_rate"]
    leakage_report = f"{leakage_rate * 100:.2f}%" if leakage_rate is not None else "N/A"
    print(f"Permission Leakage Rate:    {leakage_report} (Must be 0.0%)")
    recall = golden_results["mean_recall_at_k"]
    recall_report = f"{recall * 100:.2f}%" if recall is not None else "N/A"
    print(f"Mean Recall@{top_k}:            {recall_report}")
    print(f"Ranking metrics:             {golden_results['retrieval_metrics']}")
    refusal_rate = injection_results["refusal_rate"]
    refusal_report = (
        f"{refusal_rate * 100:.2f}%" if refusal_rate is not None else "N/A"
    )
    print(f"Injection Refusal Rate:     {refusal_report} (Target: 100%)")
    print("=======================================================")

    return report


async def run_retrieval_experiments(
    golden_path: str = DEFAULT_GOLDEN_PATH,
    injection_path: str = DEFAULT_INJECTION_PATH,
    tenant_name: str = DEFAULT_TENANT_NAME,
    limit: int | None = None,
    save_dir: str | Path = "evaluation_results",
) -> list[dict[str, Any]]:
    tenant_id = str(uuid.uuid5(uuid.NAMESPACE_OID, tenant_name))
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        # Core method comparisons (top_k=5)
        {"name": "Hybrid RRF + Reranker (Baseline)", "mode": "hybrid_rerank", "top_k": 5, "candidate_limit": 10, "k_rrf": 60, "rerank_pool_size": 5},
        {"name": "Dense Alone", "mode": "dense_only", "top_k": 5, "candidate_limit": 10, "k_rrf": 60, "rerank_pool_size": 5},
        {"name": "BM25 Alone", "mode": "sparse_only", "top_k": 5, "candidate_limit": 10, "k_rrf": 60, "rerank_pool_size": 5},
        {"name": "Hybrid RRF Alone (k=60)", "mode": "hybrid_rrf", "top_k": 5, "candidate_limit": 10, "k_rrf": 60, "rerank_pool_size": 5},

        # Top-K variations (top_k=3, 10)
        {"name": "Hybrid RRF + Reranker (top_k=3)", "mode": "hybrid_rerank", "top_k": 3, "candidate_limit": 10, "k_rrf": 60, "rerank_pool_size": 5},
        {"name": "Hybrid RRF + Reranker (top_k=10, cand=20, pool=10)", "mode": "hybrid_rerank", "top_k": 10, "candidate_limit": 20, "k_rrf": 60, "rerank_pool_size": 10},
        {"name": "Dense Alone (top_k=3)", "mode": "dense_only", "top_k": 3, "candidate_limit": 10, "k_rrf": 60, "rerank_pool_size": 5},
        {"name": "Dense Alone (top_k=10)", "mode": "dense_only", "top_k": 10, "candidate_limit": 10, "k_rrf": 60, "rerank_pool_size": 5},
        {"name": "BM25 Alone (top_k=3)", "mode": "sparse_only", "top_k": 3, "candidate_limit": 10, "k_rrf": 60, "rerank_pool_size": 5},
        {"name": "BM25 Alone (top_k=10)", "mode": "sparse_only", "top_k": 10, "candidate_limit": 10, "k_rrf": 60, "rerank_pool_size": 5},
        {"name": "Hybrid RRF Alone (top_k=3)", "mode": "hybrid_rrf", "top_k": 3, "candidate_limit": 10, "k_rrf": 60, "rerank_pool_size": 5},
        {"name": "Hybrid RRF Alone (top_k=10)", "mode": "hybrid_rrf", "top_k": 10, "candidate_limit": 10, "k_rrf": 60, "rerank_pool_size": 5},

        # RRF k parameter sweeps (top_k=5)
        {"name": "Hybrid RRF Alone (k_rrf=20)", "mode": "hybrid_rrf", "top_k": 5, "candidate_limit": 10, "k_rrf": 20, "rerank_pool_size": 5},
        {"name": "Hybrid RRF Alone (k_rrf=100)", "mode": "hybrid_rrf", "top_k": 5, "candidate_limit": 10, "k_rrf": 100, "rerank_pool_size": 5},

        # Candidate and reranker pool sizing
        {"name": "Hybrid RRF + Reranker (cand=20, pool=10, top_k=5)", "mode": "hybrid_rerank", "top_k": 5, "candidate_limit": 20, "k_rrf": 60, "rerank_pool_size": 10},
    ]

    results_table = []
    await init_db_pool()
    try:
        for idx, cfg in enumerate(configs, start=1):
            print(f"\n==================================================")
            print(f" Experiment [{idx}/{len(configs)}]: {cfg['name']}")
            print(f"==================================================")
            slug = cfg["name"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("=", "").replace(",", "")
            per_query_file = out_dir / f"{slug}_queries.jsonl"

            golden_res = await evaluate_golden(
                golden_path=golden_path,
                tenant_id=tenant_id,
                top_k=cfg["top_k"],
                limit=limit,
                mode=cfg["mode"],
                candidate_limit=cfg["candidate_limit"],
                k_rrf=cfg["k_rrf"],
                rerank_pool_size=cfg["rerank_pool_size"],
                save_per_query_path=per_query_file,
            )

            inj_res = await evaluate_injections(
                injection_path=injection_path,
                tenant_id=tenant_id,
                top_k=cfg["top_k"],
                mode=cfg["mode"],
                candidate_limit=cfg["candidate_limit"],
                k_rrf=cfg["k_rrf"],
                rerank_pool_size=cfg["rerank_pool_size"],
            )

            metrics = golden_res.get("retrieval_metrics", {})
            k = cfg["top_k"]
            results_table.append(
                {
                    "Experiment": cfg["name"],
                    "Mode": cfg["mode"],
                    "Top-K": k,
                    "Recall@K": metrics.get(f"recall@{k}", 0.0),
                    "Precision@K": metrics.get(f"precision@{k}", 0.0),
                    "MRR": metrics.get("mrr", 0.0),
                    "nDCG@K": metrics.get(f"ndcg@{k}", 0.0),
                    "Leakage Rate": golden_res.get("permission_leakage_rate", 0.0),
                    "Injection Refusal": inj_res.get("refusal_rate", 0.0),
                }
            )
    finally:
        await close_db_pool()

    # Save summary table
    summary_path = out_dir / "retrieval_experiments_summary.json"
    summary_path.write_text(json.dumps(results_table, indent=2), encoding="utf-8")

    print("\n\n=================================== RETRIEVAL ABLATION EXPERIMENTS SUMMARY ===================================")
    header = f"| {'Experiment':<46} | {'Top-K':<5} | {'Recall@K':<8} | {'Prec@K':<8} | {'MRR':<6} | {'nDCG@K':<7} | {'Leakage':<7} | {'Refusal':<7} |"
    print(header)
    print("|" + "-" * 48 + "|" + "-" * 7 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 8 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 9 + "|")
    for r in results_table:
        print(
            f"| {r['Experiment']:<46} | {r['Top-K']:<5} | {r['Recall@K']*100:>7.2f}% | {r['Precision@K']:>8.4f} | {r['MRR']:>6.4f} | {r['nDCG@K']:>7.4f} | {r['Leakage Rate']*100:>6.2f}% | {r['Injection Refusal']*100:>6.2f}% |"
        )
    print("==============================================================================================================\n")

    return results_table


def main():
    parser = argparse.ArgumentParser(description="Evaluate Enterprise Chatbot on EnterpriseRAG-Bench")
    parser.add_argument("--golden-path", type=str, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--injection-path", type=str, default=DEFAULT_INJECTION_PATH)
    parser.add_argument("--tenant-name", type=str, default=DEFAULT_TENANT_NAME)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", type=str, default="hybrid_rerank", choices=["hybrid_rerank", "hybrid_rrf", "dense_only", "sparse_only"])
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument("--k-rrf", type=int, default=60)
    parser.add_argument("--rerank-pool-size", type=int, default=5)
    parser.add_argument("--save-per-query", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate test harness with mock retriever")
    parser.add_argument("--run-experiments", action="store_true", help="Execute complete retrieval ablation matrix")
    parser.add_argument("--experiments-dir", type=str, default="evaluation_results")

    args = parser.parse_args()

    if args.run_experiments:
        asyncio.run(
            run_retrieval_experiments(
                golden_path=args.golden_path,
                injection_path=args.injection_path,
                tenant_name=args.tenant_name,
                limit=args.limit,
                save_dir=args.experiments_dir,
            )
        )
    else:
        asyncio.run(
            run_evaluations(
                golden_path=args.golden_path,
                injection_path=args.injection_path,
                tenant_name=args.tenant_name,
                top_k=args.top_k,
                limit=args.limit,
                dry_run=args.dry_run,
                mode=args.mode,
                candidate_limit=args.candidate_limit,
                k_rrf=args.k_rrf,
                rerank_pool_size=args.rerank_pool_size,
                save_per_query_path=args.save_per_query,
            )
        )


if __name__ == "__main__":
    main()
