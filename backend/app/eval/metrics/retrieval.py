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
