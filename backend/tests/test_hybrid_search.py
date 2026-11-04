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

    def _mock_gemini_response(self, vector: list[float]):
        mock_embedding = MagicMock()
        mock_embedding.values = vector
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        return mock_response

    @patch("app.rag.hybrid_search.genai.Client")
    def test_identical_and_normalized_queries_cache_hit(self, mock_client_cls):
        """Identical queries with different casing and whitespace hit the cache without calling Gemini API."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        vector_1 = [0.1, 0.2, 0.3]
        mock_client.models.embed_content.return_value = self._mock_gemini_response(vector_1)

        # 1. First invocation: misses cache, calls API
        res1 = get_query_embedding("What is the PTO policy?")
        self.assertEqual(res1, vector_1)
        self.assertEqual(mock_client.models.embed_content.call_count, 1)

        # 2. Second invocation: identical query hits cache
        res2 = get_query_embedding("What is the PTO policy?")
        self.assertEqual(res2, vector_1)
        self.assertEqual(mock_client.models.embed_content.call_count, 1)

        # 3. Third invocation: normalized whitespace & lowercase hits cache
        res3 = get_query_embedding("   what IS the pto POLICY?   ")
        self.assertEqual(res3, vector_1)
        self.assertEqual(mock_client.models.embed_content.call_count, 1)

    @patch("app.rag.hybrid_search.genai.Client")
    def test_distinct_queries_invoke_api(self, mock_client_cls):
        """Distinct queries trigger separate API calls."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.models.embed_content.side_effect = [
            self._mock_gemini_response([0.1, 0.2]),
            self._mock_gemini_response([0.3, 0.4]),
        ]

        res1 = get_query_embedding("Question A")
        res2 = get_query_embedding("Question B")

        self.assertEqual(res1, [0.1, 0.2])
        self.assertEqual(res2, [0.3, 0.4])
        self.assertEqual(mock_client.models.embed_content.call_count, 2)

    @patch("app.rag.hybrid_search.trace_span")
    @patch("app.rag.hybrid_search.genai.Client")
    def test_embedding_trace_records_metadata_not_query_text(
        self, mock_client_cls, mock_trace_span
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.embed_content.return_value = self._mock_gemini_response([0.1])

        get_query_embedding("private question content")

        trace_input = mock_trace_span.call_args.kwargs["input"]
        self.assertEqual(trace_input["query_chars"], len("private question content"))
        self.assertNotIn("query", trace_input)

    def test_retrieval_traces_include_chunk_ids_and_scores(self):
        cursor = MagicMock()
        cursor.execute = AsyncMock()
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

    @patch("app.rag.hybrid_search.genai.Client")
    def test_clear_query_embedding_cache(self, mock_client_cls):
        """Clearing the cache forces the next query to re-fetch from the API."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.models.embed_content.side_effect = [
            self._mock_gemini_response([0.5, 0.5]),
            self._mock_gemini_response([0.5, 0.5]),
        ]

        get_query_embedding("Sample Query")
        self.assertEqual(mock_client.models.embed_content.call_count, 1)

        clear_query_embedding_cache()

        get_query_embedding("Sample Query")
        self.assertEqual(mock_client.models.embed_content.call_count, 2)

    @patch("app.rag.hybrid_search.genai.Client")
    def test_lru_eviction(self, mock_client_cls):
        """Least recently used items are evicted when cache reaches maximum capacity."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.models.embed_content.side_effect = [
            self._mock_gemini_response([0.1]),  # Q1
            self._mock_gemini_response([0.2]),  # Q2
            self._mock_gemini_response([0.3]),  # Q3 (evicts Q1)
            self._mock_gemini_response([0.1]),  # Q1 again (misses cache)
        ]

        with patch("app.rag.hybrid_search._MAX_QUERY_CACHE_SIZE", 2):
            get_query_embedding("Q1")
            get_query_embedding("Q2")
            self.assertEqual(mock_client.models.embed_content.call_count, 2)

            # Insert Q3 -> evicts Q1 because Q1 was least recently used
            get_query_embedding("Q3")
            self.assertEqual(mock_client.models.embed_content.call_count, 3)

            # Q2 should still be cached
            get_query_embedding("Q2")
            self.assertEqual(mock_client.models.embed_content.call_count, 3)

            # Q1 was evicted, so querying it calls API
            get_query_embedding("Q1")
            self.assertEqual(mock_client.models.embed_content.call_count, 4)
