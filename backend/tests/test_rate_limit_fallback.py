"""
Tests for Automatic LLM Rate-Limit Handling and Fallback.
Covers:
1. Primary Groq HTTP 429 triggers exponential backoff retries and switches to Gemini.
2. SSE stream emits notice frame: "Primary model is busy. Switched to fallback model."
3. If both primary and fallback models fail, emits "AI service is temporarily unavailable. Please try again."
4. Mid-stream interruption does not restart generation or duplicate tokens.
5. Non-429 errors (HTTP 401/403 Authentication failures) do NOT trigger fallback.
6. Query rewriter 429 rate limit triggers retries and fallback to Gemini.
7. Langfuse tracing records fallback activation, fallback model, retry attempts, and outcome.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.core.security import CurrentUser, get_current_user
from app.main import app
from app.rag.hybrid_search import RetrievalResult
from app.rag.query_rewriter import rewrite_query_for_retrieval


class MockDelta:
    def __init__(self, content):
        self.content = content


class MockChoice:
    def __init__(self, content):
        self.delta = MockDelta(content)


class MockChunk:
    def __init__(self, content=None):
        self.choices = [MockChoice(content)] if content is not None else []
        self.usage = None


class RateLimitError(Exception):
    status_code = 429


class AuthenticationError(Exception):
    status_code = 401


class TestRateLimitAndFallback(unittest.TestCase):

    def setUp(self):
        app.dependency_overrides = {}
        self.client = TestClient(app, raise_server_exceptions=False)

        # Mock conversation storage layer
        self.conv_patcher = patch("app.api.chat.create_conversation", new_callable=AsyncMock)
        self.mock_create_conv = self.conv_patcher.start()
        self.mock_create_conv.return_value = {
            "id": "c1111111-1111-1111-1111-111111111111",
            "title": "New Conversation",
            "tenant_id": "tenant-abc",
            "user_id": "user-123",
        }

        self.get_conv_patcher = patch("app.api.chat.get_conversation", new_callable=AsyncMock)
        self.mock_get_conv = self.get_conv_patcher.start()
        self.mock_get_conv.return_value = {
            "id": "c1111111-1111-1111-1111-111111111111",
            "title": "New Conversation",
            "tenant_id": "tenant-abc",
            "user_id": "user-123",
        }

        self.get_msgs_patcher = patch("app.api.chat.get_conversation_messages", new_callable=AsyncMock)
        self.mock_get_msgs = self.get_msgs_patcher.start()
        self.mock_get_msgs.return_value = []

        self.append_msg_patcher = patch("app.api.chat.append_message", new_callable=AsyncMock)
        self.mock_append_msg = self.append_msg_patcher.start()
        self.mock_append_msg.return_value = {"id": "m1111111-1111-1111-1111-111111111111"}

    def tearDown(self):
        app.dependency_overrides = {}
        self.conv_patcher.stop()
        self.get_conv_patcher.stop()
        self.get_msgs_patcher.stop()
        self.append_msg_patcher.stop()

    def test_chat_primary_429_retries_and_falls_back_to_gemini(self):
        """When primary Groq hits 429, retries with backoff and switches to Gemini with SSE notice."""
        mock_user = CurrentUser(id="user-123", tenant_id="tenant-abc", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user

        mock_chunks = [
            RetrievalResult(
                chunk_id="chunk-01",
                document_id="doc-01",
                title="PTO Policy",
                heading_path=["Benefits"],
                content="Employees get 20 vacation days.",
                dense_rank=1,
                fts_rank=1,
                rrf_score=0.03,
                rerank_score=3.5,
            )
        ]

        # Groq fails with 429 RateLimitError
        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError("429 Too Many Requests: Rate limit reached")
        )

        # Gemini fallback succeeds
        class MockGeminiStreamChunk:
            def __init__(self, text: str):
                self.text = text

        mock_gemini_client = MagicMock()
        mock_gemini_client.models.generate_content_stream.return_value = [
            MockGeminiStreamChunk("Employees receive 20 days "),
            MockGeminiStreamChunk("of paid vacation."),
        ]

        spans = {}

        def make_span(name, **kwargs):
            context = MagicMock()
            span = MagicMock()
            context.__enter__.return_value = span
            spans[name] = (kwargs, span)
            return context

        with (
            patch("app.api.chat.settings.USE_GROQ_FOR_GENERATION", True),
            patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search,
            patch("app.api.chat.AsyncOpenAI", return_value=mock_groq_client),
            patch("app.api.chat.genai.Client", return_value=mock_gemini_client),
            patch("app.api.chat.trace_span", side_effect=make_span),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_search.return_value = mock_chunks
            response = self.client.post(
                "/api/chat",
                json={"message": "How much PTO do I have?"},
                headers={"Authorization": "Bearer mock-token"},
            )

            self.assertEqual(response.status_code, 200)
            lines = response.text.strip().split("\n\n")

            # Frame 0: Citations
            citations_frame = json.loads(lines[0].replace("data: ", ""))
            self.assertEqual(citations_frame["type"], "citations")

            # Frame 1: Notice frame
            notice_frame = json.loads(lines[1].replace("data: ", ""))
            self.assertEqual(notice_frame["type"], "notice")
            self.assertIn("Primary model is busy. Switched to fallback model.", notice_frame["text"])

            # Frame 2 & 3: Content from Gemini fallback
            content1 = json.loads(lines[2].replace("data: ", ""))
            content2 = json.loads(lines[3].replace("data: ", ""))
            self.assertEqual(content1["text"], "Employees receive 20 days ")
            self.assertEqual(content2["text"], "of paid vacation.")

            # Frame 4: [DONE]
            self.assertEqual(lines[4], "data: [DONE]")

            # Verify Groq retried with exponential backoff (attempt 0, 1, 2 = 3 calls, 2 sleeps)
            self.assertEqual(mock_groq_client.chat.completions.create.await_count, 3)
            self.assertEqual(mock_sleep.await_count, 2)

            # Verify Gemini was invoked
            mock_gemini_client.models.generate_content_stream.assert_called_once()

            # Verify Langfuse trace recorded fallback
            self.assertIn("groq_generation", spans)
            _, span_obj = spans["groq_generation"]
            last_output = span_obj.update.call_args.kwargs.get("output", {})
            self.assertTrue(last_output.get("fallback"))
            self.assertEqual(last_output.get("fallback_model"), "gemini-3.5-flash-lite")

    def test_chat_both_models_fail_displays_service_unavailable(self):
        """When primary Groq hits 429 and fallback Gemini also fails, displays clean error message."""
        mock_user = CurrentUser(id="user-123", tenant_id="tenant-abc", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user

        mock_chunks = [
            RetrievalResult(
                chunk_id="chunk-01",
                document_id="doc-01",
                title="Doc",
                heading_path=["HR"],
                content="Content",
                dense_rank=1,
                fts_rank=1,
                rrf_score=0.03,
                rerank_score=2.0,
            )
        ]

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError("429 Too Many Requests")
        )

        mock_gemini_client = MagicMock()
        mock_gemini_client.models.generate_content_stream.side_effect = RuntimeError(
            "Gemini 503 Unavailable"
        )

        with (
            patch("app.api.chat.settings.USE_GROQ_FOR_GENERATION", True),
            patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search,
            patch("app.api.chat.AsyncOpenAI", return_value=mock_groq_client),
            patch("app.api.chat.genai.Client", return_value=mock_gemini_client),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_search.return_value = mock_chunks
            response = self.client.post(
                "/api/chat",
                json={"message": "Any question"},
                headers={"Authorization": "Bearer mock-token"},
            )

            self.assertEqual(response.status_code, 200)
            lines = response.text.strip().split("\n\n")

            # Must contain citations and then error frame
            citations_frame = json.loads(lines[0].replace("data: ", ""))
            self.assertEqual(citations_frame["type"], "citations")

            # Notice frame may precede or error frame follows
            last_frame = json.loads(lines[-1].replace("data: ", ""))
            self.assertIn("error", last_frame)
            self.assertIn("AI service is temporarily unavailable. Please try again.", last_frame["error"])

    def test_chat_stream_interrupted_mid_generation_does_not_duplicate_or_restart(self):
        """If primary stream fails mid-generation after emitting tokens, does NOT restart from fallback."""
        mock_user = CurrentUser(id="user-123", tenant_id="tenant-abc", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user

        mock_chunks = [
            RetrievalResult(
                chunk_id="chunk-01",
                document_id="doc-01",
                title="Doc",
                heading_path=["HR"],
                content="Content",
                dense_rank=1,
                fts_rank=1,
                rrf_score=0.03,
                rerank_score=2.0,
            )
        ]

        async def failing_stream():
            yield MockChunk("Partial answer before network drop.")
            raise RuntimeError("Connection dropped mid-stream")

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(return_value=failing_stream())

        mock_gemini_client = MagicMock()

        with (
            patch("app.api.chat.settings.USE_GROQ_FOR_GENERATION", True),
            patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search,
            patch("app.api.chat.AsyncOpenAI", return_value=mock_groq_client),
            patch("app.api.chat.genai.Client", return_value=mock_gemini_client),
        ):
            mock_search.return_value = mock_chunks
            response = self.client.post(
                "/api/chat",
                json={"message": "Any question"},
                headers={"Authorization": "Bearer mock-token"},
            )

            self.assertEqual(response.status_code, 200)
            lines = response.text.strip().split("\n\n")

            # Verify Gemini fallback was NOT called because stream already started
            mock_gemini_client.models.generate_content_stream.assert_not_called()

            # Partial content was emitted
            self.assertTrue(any("Partial answer before network drop." in line for line in lines))
            # Ending with error frame
            last_frame = json.loads(lines[-1].replace("data: ", ""))
            self.assertIn("error", last_frame)

    def test_chat_unrelated_auth_error_does_not_trigger_fallback(self):
        """HTTP 401 Authentication errors do NOT trigger fallback to Gemini."""
        mock_user = CurrentUser(id="user-123", tenant_id="tenant-abc", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user

        mock_chunks = [
            RetrievalResult(
                chunk_id="chunk-01",
                document_id="doc-01",
                title="Doc",
                heading_path=["HR"],
                content="Content",
                dense_rank=1,
                fts_rank=1,
                rrf_score=0.03,
                rerank_score=2.0,
            )
        ]

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(
            side_effect=AuthenticationError("401 Unauthorized: Invalid API Key")
        )

        mock_gemini_client = MagicMock()

        with (
            patch("app.api.chat.settings.USE_GROQ_FOR_GENERATION", True),
            patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search,
            patch("app.api.chat.AsyncOpenAI", return_value=mock_groq_client),
            patch("app.api.chat.genai.Client", return_value=mock_gemini_client),
        ):
            mock_search.return_value = mock_chunks
            response = self.client.post(
                "/api/chat",
                json={"message": "Any question"},
                headers={"Authorization": "Bearer mock-token"},
            )

            self.assertEqual(response.status_code, 200)
            # Gemini MUST NOT be called for authentication errors
            mock_gemini_client.models.generate_content_stream.assert_not_called()


class TestQueryRewriterRateLimitFallback(unittest.IsolatedAsyncioTestCase):

    async def test_query_rewriter_429_falls_back_to_gemini(self):
        """When Groq hits 429 during query rewriting, retries and switches to Gemini."""
        history = [
            {"role": "user", "content": "Tell me about Project Odyssey."},
            {"role": "assistant", "content": "Project Odyssey is our core migration."},
        ]

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError("429 Rate limit reached")
        )

        mock_gemini_client = MagicMock()
        mock_gemini_response = MagicMock()
        mock_gemini_response.text = "Project Odyssey infrastructure budget"
        mock_gemini_client.models.generate_content.return_value = mock_gemini_response

        with (
            patch("app.rag.query_rewriter.settings.USE_GROQ_FOR_QUERY_REWRITE", True),
            patch("app.rag.query_rewriter.AsyncOpenAI", return_value=mock_groq_client),
            patch("app.rag.query_rewriter.genai.Client", return_value=mock_gemini_client),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            rewritten = await rewrite_query_for_retrieval("What is its budget?", history)
            self.assertEqual(rewritten, "Project Odyssey infrastructure budget")
            # Groq attempted with retries
            self.assertEqual(mock_groq_client.chat.completions.create.await_count, 3)
            self.assertEqual(mock_sleep.await_count, 2)
            # Gemini was called
            mock_gemini_client.models.generate_content.assert_called_once()

    async def test_query_rewriter_auth_error_does_not_trigger_gemini(self):
        """HTTP 401 on query rewriting does not call Gemini and returns original query."""
        history = [
            {"role": "user", "content": "Tell me about Project Odyssey."},
            {"role": "assistant", "content": "Project Odyssey is our core migration."},
        ]

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(
            side_effect=AuthenticationError("401 Invalid Key")
        )

        mock_gemini_client = MagicMock()

        with (
            patch("app.rag.query_rewriter.settings.USE_GROQ_FOR_QUERY_REWRITE", True),
            patch("app.rag.query_rewriter.AsyncOpenAI", return_value=mock_groq_client),
            patch("app.rag.query_rewriter.genai.Client", return_value=mock_gemini_client),
        ):
            rewritten = await rewrite_query_for_retrieval("What is its budget?", history)
            self.assertEqual(rewritten, "What is its budget?")
            mock_gemini_client.models.generate_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
