"""
Unified AI Efficacy Evaluation Runner.
Centralized orchestrator for batch evaluations across retrieval quality,
permission leakage, refusal confusion matrix, and generation metrics.
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.db.connection import get_db_connection
from app.eval.schemas import (
    EvaluationReport,
    PerQuestionResult,
    RefusalMetrics,
    RetrievalMetrics,
    GenerationMetrics,
)
from app.eval.metrics.retrieval import (
    compute_ranking_metrics_at_k,
    check_permission_leakage,
    doc_id_to_uuid,
)
from app.eval.metrics.refusal import compute_refusal_metrics
from app.eval.metrics.generation import evaluate_ragas_records
from app.rag.hybrid_search import search_documents

logger = logging.getLogger("app.eval.runner")

DEFAULT_GOLDEN_PATH = "/Users/falahi/datasets-test/data/golden.jsonl"
DEFAULT_INJECTION_PATH = "/Users/falahi/datasets-test/data/injection.jsonl"
DEFAULT_TENANT_NAME = "redwood_inference"


async def evaluate_dataset(
    *,
    golden_path: str | Path = DEFAULT_GOLDEN_PATH,
    injection_path: str | Path | None = DEFAULT_INJECTION_PATH,
    tenant_name: str = DEFAULT_TENANT_NAME,
    top_k: int = 5,
    limit: int | None = None,
    mode: str = "hybrid_rerank",
    candidate_limit: int = 10,
    k_rrf: int = 60,
    rerank_pool_size: int = 5,
    run_generation_eval: bool = False,
    mock_retriever: Any = None,
) -> EvaluationReport:
    """
    Executes unified evaluation across retrieval, RBAC permission leakage,
    refusal accuracy, and generation efficacy.
    """
    tenant_id = str(uuid.uuid5(uuid.NAMESPACE_OID, tenant_name))
    records: list[dict[str, Any]] = []

    # 1. Load Golden records
    gold_path = Path(golden_path)
    if gold_path.is_file():
        with gold_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if limit is not None and limit > 0:
            records = records[:limit]

    # 2. Load Injections if provided
    injection_records = []
    if injection_path:
        inj_path = Path(injection_path)
        if inj_path.is_file():
            with inj_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        r["is_injection"] = True
                        r["expected_behavior"] = "deny"
                        injection_records.append(r)

    total_eval_records = records + injection_records
    query_ranking_metrics: list[dict[str, float]] = []
    category_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    role_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)

    total_leakage_count = 0
    per_question_results: list[PerQuestionResult] = []
    ragas_input_records: list[dict[str, Any]] = []

    for test in total_eval_records:
        role = test.get("role", "employee")
        question = test.get("question", "")
        test_id = test.get("test_id", str(uuid.uuid4()))
        expected_behavior = test.get("expected_behavior", "answer")
        gold_doc_ids = test.get("gold_doc_ids", [])
        category = test.get("category", "basic")
        is_injection = test.get("is_injection", False)

        expected_uuids = {doc_id_to_uuid(gid) for gid in gold_doc_ids}

        # Execute Retrieval
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

        is_refused = len(results) == 0
        retrieved_doc_ids = [r.document_id for r in results]
        retrieved_chunk_ids = [r.chunk_id for r in results]
        retrieved_titles = [r.title for r in results]

        # Check Permission Leakage
        has_leakage = False
        if not mock_retriever and retrieved_doc_ids:
            has_leakage, _ = await check_permission_leakage(retrieved_doc_ids, role)
            if has_leakage:
                total_leakage_count += 1

        # Compute Ranking Metrics for answerable cases
        scores: dict[str, float] = {}
        if expected_behavior in ("answer", "partial") and expected_uuids:
            scores = compute_ranking_metrics_at_k(retrieved_doc_ids, expected_uuids, k=top_k)
            query_ranking_metrics.append(scores)
            category_metrics[category].append(scores)
            role_metrics[role].append(scores)

        per_question_results.append(
            PerQuestionResult(
                test_id=test_id,
                question=question,
                role=role,
                category=category,
                expected_behavior=expected_behavior,
                retrieved_doc_ids=retrieved_doc_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_titles=retrieved_titles,
                is_refused=is_refused,
                permission_leaked=has_leakage,
                retrieval_scores=scores,
            )
        )

        if run_generation_eval and test.get("answer_facts") and results:
            ragas_input_records.append({
                "user_input": question,
                "response": results[0].content[:200],  # Simplified placeholder if not generated
                "retrieved_contexts": [r.content for r in results],
                "reference": "\n".join(f"- {f}" for f in test["answer_facts"]),
            })

    # Aggregate Retrieval
    def mean_of(metrics_list: list[dict[str, float]], key: str) -> float:
        if not metrics_list:
            return 0.0
        return round(sum(m.get(key, 0.0) for m in metrics_list) / len(metrics_list), 4)

    retrieval_metrics = RetrievalMetrics(
        k=top_k,
        precision_at_k=mean_of(query_ranking_metrics, f"precision@{top_k}"),
        recall_at_k=mean_of(query_ranking_metrics, f"recall@{top_k}"),
        mrr=mean_of(query_ranking_metrics, "mrr"),
        ndcg_at_k=mean_of(query_ranking_metrics, f"ndcg@{top_k}"),
        permission_leakage_count=total_leakage_count,
        permission_leakage_rate=round(total_leakage_count / len(records), 4) if records else 0.0,
    )

    # Aggregate Refusal Confusion Matrix
    refusal_eval_records = [
        {
            "expected_behavior": p.expected_behavior,
            "is_refused": p.is_refused,
            "is_injection": p.expected_behavior == "deny" and "inj_" in p.test_id,
        }
        for p in per_question_results
    ]
    refusal_metrics = compute_refusal_metrics(refusal_eval_records)

    # Generation Metrics if requested
    generation_metrics = None
    if run_generation_eval and ragas_input_records:
        ragas_res = evaluate_ragas_records(ragas_input_records)
        agg = ragas_res.get("aggregate_metrics", {})
        generation_metrics = GenerationMetrics(
            faithfulness=agg.get("faithfulness", 0.0),
            answer_relevancy=agg.get("answer_relevancy", 0.0),
            answer_correctness=agg.get("answer_correctness", 0.0),
            context_precision=agg.get("llm_context_precision_with_reference"),
            context_recall=agg.get("context_recall"),
        )

    return EvaluationReport(
        tenant_id=tenant_id,
        tenant_name=tenant_name,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        total_cases=len(total_eval_records),
        retrieval=retrieval_metrics,
        refusal=refusal_metrics,
        generation=generation_metrics,
        per_question=per_question_results,
    )
