"""
Deterministic retrieval ranking and security metrics for RAG evaluations.
Computes:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@K)
- Permission Leakage across enterprise RBAC boundaries
"""

from __future__ import annotations

import math
import uuid
from typing import Any

from app.db.connection import get_db_connection


def doc_id_to_uuid(doc_id: str) -> str:
    """Deterministically map benchmark dsid_* identifier to corpus UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, doc_id))


def compute_ranking_metrics_at_k(
    retrieved_document_ids: list[str],
    gold_document_ids: set[str],
    k: int = 5,
) -> dict[str, float]:
    """
    Computes precision@k, recall@k, mrr, and ndcg@k for a single query.
    Deduplicates retrieved documents to reward retrieving the relevant document
    once regardless of multiple chunk appearances.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not gold_document_ids:
        return {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            "mrr": 0.0,
            f"ndcg@{k}": 0.0,
        }

    # Deduplicate preserving rank order
    unique_ranked_ids: list[str] = []
    seen: set[str] = set()
    for document_id in retrieved_document_ids:
        if document_id not in seen:
            seen.add(document_id)
            unique_ranked_ids.append(document_id)
        if len(unique_ranked_ids) >= k:
            break

    relevant_ranks = [
        rank
        for rank, document_id in enumerate(unique_ranked_ids, start=1)
        if document_id in gold_document_ids
    ]

    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_relevant_count = min(len(gold_document_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))

    precision = len(relevant_ranks) / k
    recall = len(relevant_ranks) / len(gold_document_ids)
    mrr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    ndcg = dcg / idcg if idcg > 0.0 else 0.0

    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        "mrr": mrr,
        f"ndcg@{k}": ndcg,
    }


def compute_chunk_metrics_at_k(
    retrieved_chunk_ids: list[str],
    gold_chunk_ids: set[str],
    k: int = 5,
) -> dict[str, float]:
    """
    Computes precision@k, recall@k, mrr, and hit@k for retrieved chunk IDs against gold chunk IDs.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not gold_chunk_ids:
        return {
            f"chunk_precision@{k}": 0.0,
            f"chunk_recall@{k}": 0.0,
            "chunk_mrr": 0.0,
            f"chunk_hit@{k}": 0.0,
        }

    top_chunks = retrieved_chunk_ids[:k]
    relevant_ranks = [
        rank
        for rank, chunk_id in enumerate(top_chunks, start=1)
        if chunk_id in gold_chunk_ids
    ]

    precision = len(relevant_ranks) / k
    recall = len(relevant_ranks) / len(gold_chunk_ids)
    mrr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    hit = 1.0 if relevant_ranks else 0.0

    return {
        f"chunk_precision@{k}": precision,
        f"chunk_recall@{k}": recall,
        "chunk_mrr": mrr,
        f"chunk_hit@{k}": hit,
    }


def compute_evidence_hit_at_k(
    retrieved_chunk_contents: list[str],
    answer_facts: list[str],
    min_fact_coverage: float = 0.5,
) -> dict[str, float]:
    """
    Computes evidence coverage by checking whether key factual terms from answer_facts
    appear in the retrieved chunk contents. Differentiates superficial document hits
    from actual evidence hits containing the required facts.
    """
    import re

    if not answer_facts:
        return {"evidence_hit": 1.0, "facts_covered": 1.0}
    if not retrieved_chunk_contents:
        return {"evidence_hit": 0.0, "facts_covered": 0.0}

    combined_context = " ".join(retrieved_chunk_contents).lower()
    covered = 0
    stop = {
        "that", "this", "with", "from", "have", "were", "been", "should", "would",
        "could", "must", "when", "what", "which", "about", "after", "before", "also", "than",
    }

    for fact in answer_facts:
        words = [
            w for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{4,}\b", fact.lower())
            if w not in stop
        ]
        if not words:
            covered += 1
            continue
        matches = sum(1 for w in words if w in combined_context)
        if matches / len(words) >= 0.5:
            covered += 1

    coverage_rate = covered / len(answer_facts)
    return {
        "evidence_hit": 1.0 if coverage_rate >= min_fact_coverage else 0.0,
        "facts_covered": round(coverage_rate, 4),
    }


async def check_permission_leakage(
    retrieved_document_ids: list[str],
    user_role: str,
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Verifies that all retrieved documents permit the user's role.
    Returns (has_leakage, leaked_documents_list).
    """
    if not retrieved_document_ids:
        return False, []

    leaked = []
    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, title, access_roles FROM documents
                WHERE id = ANY(%s::uuid[]);
                """,
                (list(set(retrieved_document_ids)),),
            )
            rows = await cur.fetchall()
            for doc_id, title, access_roles in rows:
                if user_role not in access_roles:
                    leaked.append({
                        "document_id": str(doc_id),
                        "title": title,
                        "required_roles": access_roles,
                        "user_role": user_role,
                    })

    return len(leaked) > 0, leaked
