"""
Hybrid Search (Dense pgvector + Sparse tsvector FTS + RRF + Cross-Encoder Reranking).
Ported and adapted from DEV_AI (packages/knowledge/hybrid_search.py).

Security requirement: Tenant and document permission filtering happens inside both
dense and sparse retrieval queries before RRF and before any retrieved content reaches the LLM.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.db.connection import get_db_connection
from app.rag.embeddings import embed_texts, require_active_embedding_schema
from app.observability.tracing import trace_span

# Lazy-loaded model to prevent instantiation overhead when not searching
_cross_encoder: CrossEncoder | None = None

# Thread-safe in-memory LRU query embedding cache
_QUERY_EMBEDDING_CACHE: dict[str, list[float]] = {}
_MAX_QUERY_CACHE_SIZE = 1024
_cache_lock = threading.Lock()

RetrievalMode = Literal["hybrid_rerank", "hybrid_rrf", "dense_only", "sparse_only"]

def get_query_embedding(query: str) -> list[float]:
    """
    Retrieve query embedding vector with thread-safe LRU caching.
    Normalizes query strings to maximize cache hit rates for identical queries.
    """
    normalized_query = " ".join(query.strip().lower().split())
    normalized_key = (
        f"{settings.ollama_embedding_model}:"
        f"{settings.ollama_embedding_dimensions}:{normalized_query}"
    )
    with trace_span(
        name="query_embedding",
        as_type="embedding",
        model=settings.ollama_embedding_model,
        input={
            "query_chars": len(query),
            "dimensions": settings.ollama_embedding_dimensions,
        },
    ) as embedding_span:
        with _cache_lock:
            if normalized_key in _QUERY_EMBEDDING_CACHE:
                # Re-insert to mark as recently used (LRU)
                val = _QUERY_EMBEDDING_CACHE.pop(normalized_key)
                _QUERY_EMBEDDING_CACHE[normalized_key] = val
                embedding_span.update(
                    output={"cache_hit": True, "dimensions": len(val)}
                )
                return val

        vec = embed_texts([query], batch_size=1)[0]

        with _cache_lock:
            if len(_QUERY_EMBEDDING_CACHE) >= _MAX_QUERY_CACHE_SIZE:
                first_key = next(iter(_QUERY_EMBEDDING_CACHE))
                del _QUERY_EMBEDDING_CACHE[first_key]
            _QUERY_EMBEDDING_CACHE[normalized_key] = vec

        embedding_span.update(output={"cache_hit": False, "dimensions": len(vec)})
        return vec


def clear_query_embedding_cache() -> None:
    """Clear query embedding cache."""
    with _cache_lock:
        _QUERY_EMBEDDING_CACHE.clear()


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(settings.reranker_model)
    return _cross_encoder


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    heading_path: list[str]
    content: str
    dense_rank: int | None
    dense_distance: float | None = None
    fts_rank: int | None
    fts_score: float | None = None
    rrf_score: float
    rerank_score: float


@dataclass
class _Candidate:
    chunk_id: str
    document_id: str
    title: str
    heading_path: list[str]
    content: str
    dense_rank: int | None = None
    dense_distance: float | None = None
    fts_rank: int | None = None
    fts_score: float | None = None


def _sync_cross_encoder_predict(
    pairs: list[list[str]],
) -> list[float]:
    model = get_cross_encoder()
    scores = model.predict(pairs)
    return [float(s) for s in scores]


async def search_documents(
    query: str,
    tenant_id: str,
    user_roles: list[str],
    top_k: int = 3,
    min_rerank_score: float | None = None,
    *,
    mode: RetrievalMode = "hybrid_rerank",
    candidate_limit: int = 10,
    k_rrf: int = 60,
    rerank_pool_size: int = 5,
) -> list[RetrievalResult]:
    """
    Search enterprise documents using DEV_AI's Retrieve-and-Rerank hybrid pipeline.

    Security Guarantee:
      Both dense and sparse queries enforce `WHERE tenant_id = %s AND access_roles && %s`
      at the database engine level before RRF fusion or LLM synthesis.

    Stage 1:
      a) Fetch Top-N using pgvector (Dense Semantic Search with local Qwen3 embeddings).
      b) Fetch Top-N using tsvector (Sparse Lexical Search via Postgres plainto_tsquery).
      c) Merge ranks using Reciprocal Rank Fusion (RRF, default k=60).
    Stage 2:
      a) Non-blocking Rerank of Top-M fused candidates using MS-MARCO Cross-Encoder.
      b) Apply refusal gate (`min_rerank_score`) to reject irrelevant matches.

    Ablation modes:
      - 'hybrid_rerank': Dense + Sparse + RRF + Cross-Encoder reranking (default)
      - 'hybrid_rrf': Dense + Sparse + RRF (no reranker or score cutoff)
      - 'dense_only': Dense vector search alone
      - 'sparse_only': PostgreSQL full-text search alone
    """
    if min_rerank_score is None:
        min_rerank_score = settings.min_rerank_score

    async with get_db_connection() as conn:
        async with conn.cursor() as cur:
            await require_active_embedding_schema(cur)

    dense_limit = max(candidate_limit, top_k) if mode == "dense_only" else candidate_limit
    sparse_limit = max(candidate_limit, top_k) if mode == "sparse_only" else candidate_limit

    with trace_span(
        name="hybrid_search_documents",
        as_type="retriever",
        input={
            "user_roles": user_roles,
            "query_chars": len(query),
            "top_k": top_k,
            "min_rerank_score": min_rerank_score,
            "mode": mode,
            "candidate_limit": candidate_limit,
            "k_rrf": k_rrf,
            "rerank_pool_size": rerank_pool_size,
        },
    ) as retriever_span:
        candidates: dict[str, _Candidate] = {}

        # 1. Dense retrieval (if mode is hybrid or dense_only)
        if mode in ("hybrid_rerank", "hybrid_rrf", "dense_only"):
            with trace_span(
                name="dense_retrieval_pgvector",
                as_type="retriever",
                input={
                    "query_chars": len(query),
                    "embedding_model": settings.ollama_embedding_model,
                    "limit": dense_limit,
                },
            ) as dense_span:
                query_vector = get_query_embedding(query)

                async with get_db_connection() as conn:
                    async with conn.cursor() as cur:
                        # Enforce tenant isolation and RBAC in dense retrieval
                        await cur.execute(
                            f"""
                            SELECT dc.id, dc.document_id, d.title, dc.heading_path, dc.content,
                                   (dc.embedding <=> %s::vector({settings.ollama_embedding_dimensions})) AS dist
                            FROM document_chunks dc
                            JOIN documents d ON dc.document_id = d.id
                            WHERE dc.tenant_id = %s AND dc.access_roles && %s
                            ORDER BY dist ASC
                            LIMIT {dense_limit};
                            """,
                            (f"[{','.join(str(f) for f in query_vector)}]", tenant_id, user_roles),
                        )

                        dense_rows = await cur.fetchall()
                        for rank, row in enumerate(dense_rows, start=1):
                            chunk_id, document_id, title, heading_path, content, distance = row
                            candidates[chunk_id] = _Candidate(
                                chunk_id=chunk_id,
                                document_id=str(document_id),
                                title=title,
                                heading_path=heading_path,
                                content=content,
                                dense_rank=rank,
                                dense_distance=float(distance),
                            )
                dense_span.update(
                    output={
                        "dense_candidates_count": len(dense_rows),
                        "scores": [
                            {
                                "rank": rank,
                                "chunk_id": str(row[0]),
                                "distance": float(row[5]),
                            }
                            for rank, row in enumerate(dense_rows, start=1)
                        ],
                    }
                )

        # 2. Sparse lexical retrieval (if mode is hybrid or sparse_only)
        if mode in ("hybrid_rerank", "hybrid_rrf", "sparse_only"):
            with trace_span(
                name="sparse_retrieval_postgres_fts",
                as_type="retriever",
                input={"query_chars": len(query), "limit": sparse_limit},
            ) as sparse_span:
                async with get_db_connection() as conn:
                    async with conn.cursor() as cur:
                        # Enforce tenant isolation and RBAC in sparse retrieval
                        await cur.execute(
                            f"""
                            SELECT dc.id, dc.document_id, d.title, dc.heading_path, dc.content,
                                   ts_rank_cd(dc.fts, plainto_tsquery('english', %s)) AS rank
                            FROM document_chunks dc
                            JOIN documents d ON dc.document_id = d.id
                            WHERE dc.tenant_id = %s AND dc.access_roles && %s
                              AND dc.fts @@ plainto_tsquery('english', %s)
                            ORDER BY rank DESC
                            LIMIT {sparse_limit};
                            """,
                            (query, tenant_id, user_roles, query),
                        )

                        fts_rows = await cur.fetchall()
                        for rank, row in enumerate(fts_rows, start=1):
                            chunk_id, document_id, title, heading_path, content, ts_rank = row
                            if chunk_id in candidates:
                                candidates[chunk_id].fts_rank = rank
                                candidates[chunk_id].fts_score = float(ts_rank)
                            else:
                                candidates[chunk_id] = _Candidate(
                                    chunk_id=chunk_id,
                                    document_id=str(document_id),
                                    title=title,
                                    heading_path=heading_path,
                                    content=content,
                                    fts_rank=rank,
                                    fts_score=float(ts_rank),
                                )
                sparse_span.update(
                    output={
                        "sparse_candidates_count": len(fts_rows),
                        "scores": [
                            {
                                "rank": rank,
                                "chunk_id": str(row[0]),
                                "score": float(row[5]),
                            }
                            for rank, row in enumerate(fts_rows, start=1)
                        ],
                    }
                )

        if not candidates:
            retriever_span.update(output={"results_count": 0, "refused": True})
            return []

        # Ablation fast-paths: dense alone and sparse alone
        if mode == "dense_only":
            dense_sorted = sorted(
                [c for c in candidates.values() if c.dense_distance is not None],
                key=lambda x: x.dense_distance or 0.0,
            )
            returned = [
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    title=c.title,
                    heading_path=c.heading_path,
                    content=c.content,
                    dense_rank=c.dense_rank,
                    dense_distance=c.dense_distance,
                    fts_rank=None,
                    fts_score=None,
                    rrf_score=0.0,
                    rerank_score=-c.dense_distance if c.dense_distance is not None else 0.0,
                )
                for c in dense_sorted[:top_k]
            ]
            retriever_span.update(
                output={
                    "results_count": len(returned),
                    "refused": False,
                    "top_rerank_score": returned[0].rerank_score if returned else None,
                }
            )
            return returned

        if mode == "sparse_only":
            sparse_sorted = sorted(
                [c for c in candidates.values() if c.fts_score is not None],
                key=lambda x: x.fts_score or 0.0,
                reverse=True,
            )
            returned = [
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    title=c.title,
                    heading_path=c.heading_path,
                    content=c.content,
                    dense_rank=None,
                    dense_distance=None,
                    fts_rank=c.fts_rank,
                    fts_score=c.fts_score,
                    rrf_score=0.0,
                    rerank_score=c.fts_score if c.fts_score is not None else 0.0,
                )
                for c in sparse_sorted[:top_k]
            ]
            retriever_span.update(
                output={
                    "results_count": len(returned),
                    "refused": False,
                    "top_rerank_score": returned[0].rerank_score if returned else None,
                }
            )
            return returned

        # 3. Merge using Reciprocal Rank Fusion (RRF)
        fused_scores = {}
        with trace_span(
            name="rrf_fusion",
            as_type="span",
            input={"total_candidates": len(candidates), "k_rrf": k_rrf},
        ) as rrf_span:
            for chunk_id, c in candidates.items():
                score = 0.0
                if c.dense_rank is not None:
                    score += 1.0 / (k_rrf + c.dense_rank)
                if c.fts_rank is not None:
                    score += 1.0 / (k_rrf + c.fts_rank)
                fused_scores[chunk_id] = score

            sorted_fused = sorted(
                fused_scores.items(), key=lambda item: item[1], reverse=True
            )
            top_fused = sorted_fused[:rerank_pool_size]
            rrf_span.update(
                output={
                    "fused_top_5": [
                        {
                            "rank": rank,
                            "chunk_id": str(chunk_id),
                            "score": round(score, 8),
                            "dense_rank": candidates[chunk_id].dense_rank,
                            "fts_rank": candidates[chunk_id].fts_rank,
                        }
                        for rank, (chunk_id, score) in enumerate(top_fused, start=1)
                    ]
                }
            )

        # Ablation fast-path: hybrid RRF without cross-encoder reranker or score cutoff
        if mode == "hybrid_rrf":
            returned = [
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    title=c.title,
                    heading_path=c.heading_path,
                    content=c.content,
                    dense_rank=c.dense_rank,
                    dense_distance=c.dense_distance,
                    fts_rank=c.fts_rank,
                    fts_score=c.fts_score,
                    rrf_score=fused_scores[c.chunk_id],
                    rerank_score=fused_scores[c.chunk_id],
                )
                for chunk_id, _ in sorted_fused[:top_k]
                if (c := candidates[chunk_id]) is not None
            ]
            retriever_span.update(
                output={
                    "results_count": len(returned),
                    "refused": False,
                    "top_rerank_score": returned[0].rerank_score if returned else None,
                }
            )
            return returned

        # 4. Score using MS-MARCO Cross-Encoder and apply refusal gate
        final_results = []
        with trace_span(
            name="cross_encoder_rerank_cutoff",
            as_type="span",
            input={
                "candidates_count": len(top_fused),
                "model": settings.reranker_model,
                "min_rerank_score": min_rerank_score,
            },
        ) as rerank_span:
            cross_encoder_inputs = []
            candidates_to_rerank = []

            for chunk_id, _ in top_fused:
                c = candidates[chunk_id]
                breadcrumb = " > ".join(c.heading_path) if c.heading_path else c.title
                enriched_input = f"Title: {c.title} | Path: {breadcrumb} | {c.content}"
                cross_encoder_inputs.append([query, enriched_input])
                candidates_to_rerank.append(c)

            # Non-blocking cross-encoder execution in threadpool for FastAPI
            rerank_scores = await asyncio.to_thread(
                _sync_cross_encoder_predict, cross_encoder_inputs
            )

            for i, c in enumerate(candidates_to_rerank):
                rrf_score = fused_scores[c.chunk_id]
                rerank_score = float(rerank_scores[i])

                final_results.append(
                    RetrievalResult(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        title=c.title,
                        heading_path=c.heading_path,
                        content=c.content,
                        dense_rank=c.dense_rank,
                        dense_distance=c.dense_distance,
                        fts_rank=c.fts_rank,
                        fts_score=c.fts_score,
                        rrf_score=rrf_score,
                        rerank_score=rerank_score,
                    )
                )

            # Sort descending by rerank score
            final_results.sort(key=lambda x: x.rerank_score, reverse=True)
            top_score = final_results[0].rerank_score if final_results else None
            refused = top_score is None or top_score < min_rerank_score
            rerank_span.update(
                output={
                    "top_rerank_score": top_score,
                    "refused": refused,
                    "scores": [
                        {
                            "rank": rank,
                            "chunk_id": result.chunk_id,
                            "rerank_score": result.rerank_score,
                            "rrf_score": result.rrf_score,
                            "dense_rank": result.dense_rank,
                            "fts_rank": result.fts_rank,
                        }
                        for rank, result in enumerate(final_results, start=1)
                    ],
                }
            )

        # Refusal gate check on the absolute best candidate
        if not final_results or final_results[0].rerank_score < min_rerank_score:
            retriever_span.update(
                output={"results_count": 0, "refused": True, "top_rerank_score": top_score}
            )
            return []

        returned = final_results[:top_k]
        retriever_span.update(
            output={
                "results_count": len(returned),
                "refused": False,
                "top_rerank_score": returned[0].rerank_score,
                "scores": [
                    {
                        "rank": rank,
                        "chunk_id": result.chunk_id,
                        "rerank_score": result.rerank_score,
                        "rrf_score": result.rrf_score,
                        "dense_rank": result.dense_rank,
                        "dense_distance": result.dense_distance,
                        "fts_rank": result.fts_rank,
                        "fts_score": result.fts_score,
                    }
                    for rank, result in enumerate(returned, start=1)
                ],
            }
        )
        return returned
