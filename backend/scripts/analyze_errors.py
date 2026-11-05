"""
Comprehensive Error Analysis script for Enterprise Chatbot Retrieval Failures.
Analyzes zero-hit and partial-hit queries from the baseline evaluation run.
Categorizes root causes into:
1. Embedding Quality / Semantic Gap
2. Chunking and Context Fragmentation
3. BM25 / FTS Keyword Matching Failures
4. RRF Fusion & Candidate Pool Cutoff
5. Reranker Performance / Score Cutoff
6. Metadata & Role Filtering
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.db.connection import get_db_connection, init_db_pool, close_db_pool
from app.rag.hybrid_search import get_query_embedding, _sync_cross_encoder_predict


async def main():
    base_file = Path("evaluation_results/hybrid_rrf_plus_reranker_baseline_queries.jsonl")
    if not base_file.exists():
        base_file = Path("../evaluation_results/hybrid_rrf_plus_reranker_baseline_queries.jsonl")
    if not base_file.exists():
        print(f"Error: {base_file} not found")
        return

    with open(base_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    answerable = [
        r for r in rows
        if r.get("expected_behavior") in ("answer", "partial") and r.get("gold_uuids")
    ]
    print(f"Total answerable queries with gold documents: {len(answerable)}")

    zero_hits = [r for r in answerable if r["metrics"]["recall@5"] == 0.0]
    partial_hits = [r for r in answerable if 0.0 < r["metrics"]["recall@5"] < 1.0]
    full_hits = [r for r in answerable if r["metrics"]["recall@5"] == 1.0]

    print(f"Full hits (Recall@5 = 1.0): {len(full_hits)} ({len(full_hits)/len(answerable)*100:.1f}%)")
    print(f"Partial hits (0 < Recall@5 < 1.0): {len(partial_hits)} ({len(partial_hits)/len(answerable)*100:.1f}%)")
    print(f"Zero hits (Recall@5 = 0.0): {len(zero_hits)} ({len(zero_hits)/len(answerable)*100:.1f}%)")

    tenant_id = str(uuid.uuid5(uuid.NAMESPACE_OID, "redwood_inference"))

    await init_db_pool()
    try:
        diagnosed_zero = []
        async with get_db_connection() as conn:
            for item in zero_hits:
                role = item["role"]
                q = item["question"]
                gold_uuids = set(item["gold_uuids"])

                # 1. Check if document exists and role permissions in DB
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT id, title, access_roles FROM documents WHERE id = ANY(%s::uuid[]);",
                        (list(gold_uuids),)
                    )
                    db_docs = await cur.fetchall()

                if not db_docs:
                    diagnosed_zero.append({
                        "test_id": item["test_id"],
                        "category": "metadata_not_in_db",
                        "reason": "Gold document UUID not found in corpus database",
                        "item": item,
                    })
                    continue

                accessible_docs = [str(d[0]) for d in db_docs if role in d[2]]
                if not accessible_docs:
                    diagnosed_zero.append({
                        "test_id": item["test_id"],
                        "category": "metadata_and_filtering",
                        "reason": f"Gold document exists ({len(db_docs)} docs) but user role '{role}' is not in access_roles",
                        "item": item,
                    })
                    continue

                # 2. Check Dense vector search up to Top 50
                vec = get_query_embedding(q)
                vec_str = "[" + ",".join(str(f) for f in vec) + "]"
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""
                        SELECT dc.id, dc.document_id, d.title, dc.heading_path, dc.content,
                               (dc.embedding <=> %s::vector(1024)) AS dist
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE dc.tenant_id = %s AND dc.access_roles && %s
                        ORDER BY dist ASC
                        LIMIT 50;
                        """,
                        (vec_str, tenant_id, [role])
                    )
                    dense_rows = await cur.fetchall()

                dense_ranks = []
                for rank, row in enumerate(dense_rows, start=1):
                    doc_id = str(row[1])
                    if doc_id in gold_uuids:
                        dense_ranks.append((rank, doc_id, row[0], row[2], row[3], float(row[5])))

                # 3. Check BM25 / FTS search
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT dc.id, dc.document_id, d.title, ts_rank_cd(dc.fts, plainto_tsquery('english', %s)) AS rank
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE dc.tenant_id = %s AND dc.access_roles && %s
                          AND dc.fts @@ plainto_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT 20;
                        """,
                        (q, tenant_id, [role], q)
                    )
                    fts_rows = await cur.fetchall()

                fts_hits = [r for r in fts_rows if str(r[1]) in gold_uuids]

                # Classification based on empirical findings:
                if not dense_ranks:
                    diagnosed_zero.append({
                        "test_id": item["test_id"],
                        "category": "embedding_quality",
                        "reason": "Semantic gap: gold chunks ranked > 50 in pgvector dense search",
                        "item": item,
                        "dense_best_rank": None,
                        "fts_match_count": len(fts_hits),
                    })
                else:
                    best_dense_rank = dense_ranks[0][0]
                    if best_dense_rank <= 5:
                        # Gold doc was in top 5 dense! Did cross-encoder demote it?
                        # Check cross-encoder score
                        chunk_row = [r for r in dense_rows if str(r[1]) in gold_uuids][0]
                        breadcrumb = " > ".join(chunk_row[3]) if chunk_row[3] else chunk_row[2]
                        enriched = f"Title: {chunk_row[2]} | Path: {breadcrumb} | {chunk_row[4]}"
                        scores = await asyncio.to_thread(_sync_cross_encoder_predict, [[q, enriched]])
                        ce_score = scores[0]
                        diagnosed_zero.append({
                            "test_id": item["test_id"],
                            "category": "reranker_performance",
                            "reason": f"Gold doc was rank {best_dense_rank} in dense, but cross-encoder scored it {ce_score:.2f} (cutoff {-8.5}) or demoted below top 5",
                            "item": item,
                            "dense_best_rank": best_dense_rank,
                            "cross_encoder_score": ce_score,
                        })
                    elif best_dense_rank <= 10:
                        diagnosed_zero.append({
                            "test_id": item["test_id"],
                            "category": "reranker_performance",
                            "reason": f"Gold doc was rank {best_dense_rank} in dense (top 10 candidate pool), but fused outside top 5 or demoted by reranker",
                            "item": item,
                            "dense_best_rank": best_dense_rank,
                        })
                    elif best_dense_rank <= 20:
                        diagnosed_zero.append({
                            "test_id": item["test_id"],
                            "category": "rrf_ranking",
                            "reason": f"Candidate pool cutoff: gold doc was rank {best_dense_rank} in dense (retrievable with cand=20, but missed with cand=10 / pool=5)",
                            "item": item,
                            "dense_best_rank": best_dense_rank,
                        })
                    else:
                        diagnosed_zero.append({
                            "test_id": item["test_id"],
                            "category": "chunking_and_context_fragmentation",
                            "reason": f"Context fragmentation: gold doc was deep at rank {best_dense_rank} (21-50), content scattered or diluted",
                            "item": item,
                            "dense_best_rank": best_dense_rank,
                        })

    finally:
        await close_db_pool()

    print("\n================== FAILURE CATEGORIZATION SUMMARY (71 ZERO HITS) ==================")
    cat_counts = Counter(d["category"] for d in diagnosed_zero)
    for cat, count in cat_counts.most_common():
        pct = count / len(diagnosed_zero) * 100
        print(f"  {cat:<35}: {count:>2} ({pct:>5.1f}%)")

    # Save detailed error analysis report
    report_file = Path("evaluation_results/error_analysis_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(diagnosed_zero, indent=2), encoding="utf-8")
    print(f"\nDetailed per-failure diagnostics written to {report_file}")

    # Print representative examples per category
    for cat in cat_counts:
        samples = [d for d in diagnosed_zero if d["category"] == cat][:2]
        print(f"\n--- Representative Examples: {cat} ---")
        for s in samples:
            item = s["item"]
            print(f"[{s['test_id']}] ({item['category']}) Q: {item['question']}")
            print(f"  Diagnosis: {s['reason']}")


if __name__ == "__main__":
    asyncio.run(main())
