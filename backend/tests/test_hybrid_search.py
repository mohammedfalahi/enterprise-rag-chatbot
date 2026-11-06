"""
Unit tests for the query embedding LRU cache and retrieval components.
Covers:
- Caching of identical and whitespace/case-normalized queries
- Re-querying different questions invokes embedding model
- Cache clearing behavior
- LRU eviction when maximum cache size is reached
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.rag.hybrid_search import (
    clear_query_embedding_cache,
    get_query_embedding,
    search_documents,
)


class TestQueryEmbeddingCache(unittest.TestCase):

    def setUp(self):
        clear_query_embedding_cache()

    def tearDown(self):
        clear_query_embedding_cache()

    @staticmethod
    def _vector(value: float) -> list[float]:
        return [value] * 1024

    @patch("app.rag.hybrid_search.embed_texts")
    def test_identical_and_normalized_queries_cache_hit(self, mock_embed_texts):
        """Identical queries with different casing and whitespace reuse local embeddings."""
        vector_1 = self._vector(0.1)
        mock_embed_texts.return_value = [vector_1]

        # 1. First invocation: misses cache, calls API
        res1 = get_query_embedding("What is the PTO policy?")
        self.assertEqual(res1, vector_1)
        self.assertEqual(mock_embed_texts.call_count, 1)

        # 2. Second invocation: identical query hits cache
        res2 = get_query_embedding("What is the PTO policy?")
        self.assertEqual(res2, vector_1)
        self.assertEqual(mock_embed_texts.call_count, 1)

        # 3. Third invocation: normalized whitespace & lowercase hits cache
        res3 = get_query_embedding("   what IS the pto POLICY?   ")
        self.assertEqual(res3, vector_1)
        self.assertEqual(mock_embed_texts.call_count, 1)

    @patch("app.rag.hybrid_search.embed_texts")
    def test_distinct_queries_invoke_embedding_service(self, mock_embed_texts):
        """Distinct queries trigger separate API calls."""
        mock_embed_texts.side_effect = [
            [self._vector(0.1)],
            [self._vector(0.3)],
        ]

        res1 = get_query_embedding("Question A")
        res2 = get_query_embedding("Question B")

        self.assertEqual(res1, self._vector(0.1))
        self.assertEqual(res2, self._vector(0.3))
        self.assertEqual(mock_embed_texts.call_count, 2)

    @patch("app.rag.hybrid_search.trace_span")
    @patch("app.rag.hybrid_search.embed_texts")
    def test_embedding_trace_records_metadata_not_query_text(
        self, mock_embed_texts, mock_trace_span
    ):
        mock_embed_texts.return_value = [self._vector(0.1)]

        get_query_embedding("private question content")

        trace_input = mock_trace_span.call_args.kwargs["input"]
        self.assertEqual(trace_input["query_chars"], len("private question content"))
        self.assertNotIn("query", trace_input)

    def test_retrieval_traces_include_chunk_ids_and_scores(self):
        cursor = MagicMock()
        cursor.execute = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=("vector(1024)",))
        cursor.fetchall = AsyncMock(
            side_effect=[
                [("chunk-a", "doc-a", "Title A", [], "content", 0.12)],
                [
                    ("chunk-a", "doc-a", "Title A", [], "content", 0.8),
                    ("chunk-b", "doc-b", "Title B", [], "content", 0.4),
                ],
            ]
        )
        connection = MagicMock()
        connection.cursor.return_value.__aenter__.return_value = cursor
        connection_context = MagicMock()
        connection_context.__aenter__.return_value = connection

        spans = {}

        def make_span(name, **_kwargs):
            context = MagicMock()
            span = MagicMock()
            context.__enter__.return_value = span
            spans[name] = span
            return context

        with (
            patch("app.rag.hybrid_search.get_query_embedding", return_value=[0.1]),
            patch(
                "app.rag.hybrid_search.get_db_connection",
                return_value=connection_context,
            ),
            patch(
                "app.rag.hybrid_search._sync_cross_encoder_predict",
                return_value=[2.5, 1.5],
            ),
            patch("app.rag.hybrid_search.trace_span", side_effect=make_span),
        ):
            results = asyncio.run(
                search_documents("question", "tenant", ["employee"], top_k=2)
            )

        self.assertEqual(len(results), 2)
        dense_output = spans["dense_retrieval_pgvector"].update.call_args.kwargs["output"]
        sparse_output = spans["sparse_retrieval_postgres_fts"].update.call_args.kwargs["output"]
        rrf_output = spans["rrf_fusion"].update.call_args.kwargs["output"]
        rerank_output = spans["cross_encoder_rerank_cutoff"].update.call_args.kwargs["output"]
        self.assertEqual(dense_output["scores"][0]["chunk_id"], "chunk-a")
        self.assertEqual(dense_output["scores"][0]["distance"], 0.12)
        self.assertEqual(sparse_output["scores"][1]["chunk_id"], "chunk-b")
        self.assertEqual(sparse_output["scores"][1]["score"], 0.4)
        self.assertIn("chunk_id", rrf_output["fused_top_5"][0])
        self.assertIn("score", rrf_output["fused_top_5"][0])
        self.assertEqual(rerank_output["scores"][0]["chunk_id"], "chunk-a")
        self.assertEqual(rerank_output["scores"][0]["rerank_score"], 2.5)
        retrieval_queries = [
            call for call in cursor.execute.await_args_list if "SELECT dc.id" in call.args[0]
        ]
        self.assertEqual(len(retrieval_queries), 2)
        for call in retrieval_queries:
            self.assertIn("WHERE dc.tenant_id = %s AND dc.access_roles && %s", call.args[0])
            self.assertEqual(call.args[1][1:3], ("tenant", ["employee"]))

    @patch("app.rag.hybrid_search.embed_texts")
    def test_clear_query_embedding_cache(self, mock_embed_texts):
        """Clearing the cache forces the next query to re-embed locally."""
        mock_embed_texts.return_value = [self._vector(0.5)]

        get_query_embedding("Sample Query")
        self.assertEqual(mock_embed_texts.call_count, 1)

        clear_query_embedding_cache()

        get_query_embedding("Sample Query")
        self.assertEqual(mock_embed_texts.call_count, 2)

    @patch("app.rag.hybrid_search.embed_texts")
    def test_lru_eviction(self, mock_embed_texts):
        """Least recently used items are evicted when cache reaches maximum capacity."""
        mock_embed_texts.side_effect = [
            [self._vector(0.1)],  # Q1
            [self._vector(0.2)],  # Q2
            [self._vector(0.3)],  # Q3 (evicts Q1)
            [self._vector(0.1)],  # Q1 again (misses cache)
        ]

        with patch("app.rag.hybrid_search._MAX_QUERY_CACHE_SIZE", 2):
            get_query_embedding("Q1")
            get_query_embedding("Q2")
            self.assertEqual(mock_embed_texts.call_count, 2)

            # Insert Q3 -> evicts Q1 because Q1 was least recently used
            get_query_embedding("Q3")
            self.assertEqual(mock_embed_texts.call_count, 3)

            # Q2 should still be cached
            get_query_embedding("Q2")
            self.assertEqual(mock_embed_texts.call_count, 3)

            # Q1 was evicted, so querying it calls API
            get_query_embedding("Q1")
            self.assertEqual(mock_embed_texts.call_count, 4)

    def test_search_documents_with_diagnostics_captures_all_pipeline_stages(self):
        """search_documents_with_diagnostics returns candidates and scores from all intermediate stages."""
        from app.rag.hybrid_search import search_documents_with_diagnostics

        cursor = MagicMock()
        cursor.execute = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=("vector(1024)",))
        cursor.fetchall = AsyncMock(
            side_effect=[
                # Dense rows: (id, doc_id, title, path, content, distance)
                [("chunk-dense-1", "doc-1", "Title 1", ["Path"], "Dense text", 0.15)],
                # Sparse FTS rows: (id, doc_id, title, path, content, score)
                [("chunk-fts-1", "doc-2", "Title 2", ["Path"], "FTS text", 0.95)],
            ]
        )
        connection = MagicMock()
        connection.cursor.return_value.__aenter__.return_value = cursor
        connection_context = MagicMock()
        connection_context.__aenter__.return_value = connection

        with (
            patch("app.rag.hybrid_search.get_query_embedding", return_value=[0.1]),
            patch(
                "app.rag.hybrid_search.get_db_connection",
                return_value=connection_context,
            ),
            patch(
                "app.rag.hybrid_search._sync_cross_encoder_predict",
                return_value=[3.1, -1.2],
            ),
        ):
            results, diag = asyncio.run(
                search_documents_with_diagnostics(
                    query="What is the onboarding policy?",
                    tenant_id="tenant-123",
                    user_roles=["employee"],
                    top_k=2,
                )
            )

        self.assertEqual(len(results), 2)
        self.assertFalse(diag.refused)
        self.assertEqual(diag.top_rerank_score, 3.1)
        self.assertEqual(len(diag.dense_candidates), 1)
        self.assertEqual(diag.dense_candidates[0]["chunk_id"], "chunk-dense-1")
        self.assertEqual(len(diag.fts_candidates), 1)
        self.assertEqual(diag.fts_candidates[0]["chunk_id"], "chunk-fts-1")
        self.assertEqual(len(diag.rrf_candidates), 2)
        self.assertEqual(len(diag.reranked_candidates), 2)
        self.assertEqual(diag.final_chunk_ids, ["chunk-dense-1", "chunk-fts-1"])
