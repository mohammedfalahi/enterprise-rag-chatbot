"""
Integration tests for the /api/chat endpoint.
Covers:
- 401 Unauthorized for missing or invalid Bearer tokens
- 403 Forbidden for unassigned enterprise users
- 200 Streaming refusal when retrieval returns no accessible documents
- 200 Successful streaming response with retrieved context and Gemini generation
- Graceful error handling during streaming generation
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.core.security import CurrentUser, get_current_user
from app.main import app
from app.rag.hybrid_search import RetrievalResult


class TestChatApiEndpoint(unittest.TestCase):

    def setUp(self):
        app.dependency_overrides = {}
        self.client = TestClient(app, raise_server_exceptions=False)

        # Mock conversation persistence layer for unit testing chat endpoint
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
        self.mock_append_msg.return_value = {
            "id": "m1111111-1111-1111-1111-111111111111",
        }

        # Ensure unit tests default to Gemini generation unless explicitly overridden
        self.groq_patcher = patch("app.api.chat.settings.USE_GROQ_FOR_GENERATION", False)
        self.mock_use_groq = self.groq_patcher.start()

    def tearDown(self):
        app.dependency_overrides = {}
        self.groq_patcher.stop()
        self.conv_patcher.stop()
        self.get_conv_patcher.stop()
        self.get_msgs_patcher.stop()
        self.append_msg_patcher.stop()

    def test_unauthenticated_request_returns_401(self):
        """Request without Authorization header returns 401."""
        response = self.client.post("/api/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 401)
        self.assertIn("Not authenticated", response.text)

    def test_invalid_token_returns_401(self):
        """Request with invalid Bearer token returns 401."""
        response = self.client.post(
            "/api/chat",
            json={"message": "Hello"},
            headers={"Authorization": "Bearer invalid.token.payload"},
        )
        self.assertEqual(response.status_code, 401)

    def test_unassigned_user_returns_403(self):
        """User without assigned tenant profile receives 403 Forbidden."""
        from fastapi import HTTPException, status

        async def mock_get_current_user():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User profile not found. User is not assigned to any enterprise tenant.",
            )

        app.dependency_overrides[get_current_user] = mock_get_current_user

        response = self.client.post(
            "/api/chat",
            json={"message": "What is the policy?"},
            headers={"Authorization": "Bearer mocked-token"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("User profile not found", response.json().get("detail", ""))

    def test_chat_empty_retrieval_refusal_stream(self):
        """When search_documents returns empty (no matches or refused), returns refusal SSE stream."""
        mock_user = CurrentUser(
            id="user-123",
            tenant_id="tenant-abc",
            roles=["employee"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        with patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            response = self.client.post(
                "/api/chat",
                json={"message": "What is the confidential merger plan?"},
                headers={"Authorization": "Bearer mocked-token"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])

            # Verify refusal message in stream
            lines = response.text.strip().split("\n\n")
            self.assertTrue(len(lines) >= 2)

            data_payload = json.loads(lines[0].replace("data: ", ""))
            self.assertIn(
                "I do not have enough verified information to answer that question",
                data_payload.get("text", ""),
            )
            self.assertEqual(lines[1], "data: [DONE]")

            # Verify tenant_id and roles were passed to search_documents
            mock_search.assert_called_once_with(
                query="What is the confidential merger plan?",
                tenant_id="tenant-abc",
                user_roles=["employee"],
                top_k=3,
            )

    def test_chat_successful_streaming_response(self):
        """When documents are retrieved, formats context, invokes Gemini, and streams chunks."""
        mock_user = CurrentUser(
            id="user-456",
            tenant_id="tenant-redwood",
            roles=["employee", "manager"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        mock_chunks = [
            RetrievalResult(
                chunk_id="chunk-01",
                document_id="doc-01",
                title="PTO Guidelines",
                heading_path=["Benefits", "Paid Time Off"],
                content="Full-time employees receive 20 days of paid vacation annually.",
                dense_rank=1,
                fts_rank=1,
                rrf_score=0.032,
                rerank_score=3.85,
            )
        ]

        # Mock chunk object with text attribute
        class MockStreamChunk:
            def __init__(self, text: str):
                self.text = text

        mock_gemini_client = MagicMock()
        mock_gemini_client.models.generate_content_stream.return_value = [
            MockStreamChunk("Full-time employees "),
            MockStreamChunk("receive 20 days of PTO."),
        ]

        with patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_chunks

            with patch("app.api.chat.genai.Client", return_value=mock_gemini_client):
                response = self.client.post(
                    "/api/chat",
                    json={"message": "How many vacation days do I get?"},
                    headers={"Authorization": "Bearer mocked-token"},
                )

                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers["content-type"])

                lines = response.text.strip().split("\n\n")
                self.assertEqual(len(lines), 4)

                citations_frame = json.loads(lines[0].replace("data: ", ""))
                self.assertEqual(citations_frame["type"], "citations")
                self.assertEqual(len(citations_frame["citations"]), 1)
                self.assertEqual(citations_frame["citations"][0]["document_id"], "doc-01")
                self.assertEqual(citations_frame["citations"][0]["title"], "PTO Guidelines")
                self.assertEqual(citations_frame["citations"][0]["heading_path"], ["Benefits", "Paid Time Off"])
                self.assertEqual(citations_frame["citations"][0]["score"], 3.85)

                part1 = json.loads(lines[1].replace("data: ", ""))
                part2 = json.loads(lines[2].replace("data: ", ""))
                self.assertEqual(part1["text"], "Full-time employees ")
                self.assertEqual(part2["text"], "receive 20 days of PTO.")
                self.assertEqual(lines[3], "data: [DONE]")

                # Verify generate_content_stream was called with context containing breadcrumbs
                call_args = mock_gemini_client.models.generate_content_stream.call_args
                prompt_arg = call_args.kwargs.get("contents")
                self.assertIn("Document: PTO Guidelines", prompt_arg)
                self.assertIn("Path: Benefits > Paid Time Off", prompt_arg)
                self.assertIn("Full-time employees receive 20 days of paid vacation annually.", prompt_arg)

    def test_generation_trace_content_is_opt_in(self):
        mock_user = CurrentUser(id="user-456", tenant_id="tenant-redwood", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user
        results = [
            RetrievalResult(
                chunk_id="chunk-01",
                document_id="doc-01",
                title="PTO Guidelines",
                heading_path=["Benefits"],
                content="Employees receive 20 days of PTO.",
                dense_rank=1,
                fts_rank=1,
                rrf_score=0.03,
                rerank_score=3.0,
            )
        ]

        class MockChunk:
            text = "Employees receive 20 days of PTO."

        mock_gemini = MagicMock()
        mock_gemini.models.generate_content_stream.return_value = [MockChunk()]

        for capture_content in (False, True):
            spans = {}

            def make_span(name, **kwargs):
                context = MagicMock()
                span = MagicMock()
                context.__enter__.return_value = span
                spans[name] = (kwargs, span)
                return context

            with (
                patch(
                    "app.api.chat.settings.langfuse_capture_content",
                    capture_content,
                ),
                patch("app.api.chat.trace_span", side_effect=make_span),
                patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search,
                patch("app.api.chat.genai.Client", return_value=mock_gemini),
            ):
                mock_search.return_value = results
                response = self.client.post(
                    "/api/chat",
                    json={"message": "How many PTO days?"},
                    headers={"Authorization": "******"},
                )

            self.assertEqual(response.status_code, 200)
            generation_input, generation_span = spans["gemini_generation"]
            trace_input = generation_input["input"]
            if capture_content:
                self.assertIn("system_prompt", trace_input)
                self.assertEqual(trace_input["user_query"], "How many PTO days?")
                self.assertIn("Employees receive 20 days of PTO.", trace_input["retrieved_context"])
                self.assertEqual(
                    generation_span.update.call_args.kwargs["output"]["generated_answer"],
                    "Employees receive 20 days of PTO.",
                )
            else:
                self.assertNotIn("system_prompt", trace_input)
                self.assertNotIn("user_query", trace_input)
                self.assertNotIn(
                    "generated_answer",
                    generation_span.update.call_args.kwargs["output"],
                )

    def test_chat_streaming_generation_error(self):
        """Generation errors are traced/logged and returned without provider details."""
        mock_user = CurrentUser(
            id="user-789",
            tenant_id="tenant-xyz",
            roles=["employee"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        mock_chunks = [
            RetrievalResult(
                chunk_id="chunk-01",
                document_id="doc-01",
                title="Doc Title",
                heading_path=["Overview"],
                content="Some content",
                dense_rank=1,
                fts_rank=1,
                rrf_score=0.03,
                rerank_score=1.0,
            )
        ]

        def failing_stream(*args, **kwargs):
            raise RuntimeError("API quota exhausted: 429 RESOURCE_EXHAUSTED")

        mock_gemini_client = MagicMock()
        mock_gemini_client.models.generate_content_stream.side_effect = failing_stream

        with patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_chunks

            with patch("app.api.chat.genai.Client", return_value=mock_gemini_client):
                response = self.client.post(
                    "/api/chat",
                    json={"message": "Any question"},
                    headers={"Authorization": "Bearer mocked-token"},
                )

                self.assertEqual(response.status_code, 200)
                lines = response.text.strip().split("\n\n")
                self.assertEqual(len(lines), 2)
                citations_frame = json.loads(lines[0].replace("data: ", ""))
                self.assertEqual(citations_frame["type"], "citations")
                error_frame = json.loads(lines[1].replace("data: ", ""))
                self.assertIn("could not complete", error_frame["error"])
                self.assertNotIn("API quota exhausted", error_frame["error"])

    def test_chat_empty_message_returns_400(self):
        """Empty message body returns HTTP 400 Bad Request."""
        mock_user = CurrentUser(id="user-123", tenant_id="tenant-abc", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response = self.client.post(
            "/api/chat",
            json={"message": "   "},
            headers={"Authorization": "Bearer mocked-token"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Message content cannot be empty", response.json().get("detail", ""))

    def test_chat_invalid_conversation_id_returns_400(self):
        """Invalid conversation_id UUID format returns HTTP 400 Bad Request."""
        mock_user = CurrentUser(id="user-123", tenant_id="tenant-abc", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response = self.client.post(
            "/api/chat",
            json={"message": "Hello", "conversation_id": "not-a-valid-uuid"},
            headers={"Authorization": "Bearer mocked-token"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid conversation_id format", response.json().get("detail", ""))

    def test_chat_nonexistent_conversation_returns_404(self):
        """Conversation ID that does not exist or belongs to another user returns 404."""
        mock_user = CurrentUser(id="user-123", tenant_id="tenant-abc", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user
        self.mock_get_conv.return_value = None

        response = self.client.post(
            "/api/chat",
            json={"message": "Hello", "conversation_id": "00000000-0000-0000-0000-000000000000"},
            headers={"Authorization": "Bearer mocked-token"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found or access denied", response.json().get("detail", ""))

    def test_chat_follow_up_invokes_query_rewriter(self):
        """When conversation history exists, rewrite_query_for_retrieval is invoked for retrieval."""
        mock_user = CurrentUser(id="user-123", tenant_id="tenant-abc", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user

        # History with prior turn
        prior_history = [
            {"role": "user", "content": "Tell me about Project Odyssey."},
            {"role": "assistant", "content": "Project Odyssey is our next-gen cloud platform."},
        ]
        self.mock_get_msgs.return_value = prior_history

        mock_chunks = [
            RetrievalResult(
                chunk_id="chunk-01",
                document_id="doc-01",
                title="Project Odyssey Budget",
                heading_path=["Odyssey", "Financials"],
                content="Project Odyssey has a 2027 budget of $4.5 million.",
                dense_rank=1,
                fts_rank=1,
                rrf_score=0.032,
                rerank_score=4.12,
            )
        ]

        class MockChunk:
            def __init__(self, text: str):
                self.text = text

        mock_gemini = MagicMock()
        mock_gemini.models.generate_content_stream.return_value = [
            MockChunk("The budget is $4.5 million.")
        ]

        with patch("app.api.chat.rewrite_query_for_retrieval", new_callable=AsyncMock) as mock_rewrite:
            mock_rewrite.return_value = "What is the budget for Project Odyssey?"

            with patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search:
                mock_search.return_value = mock_chunks

                with patch("app.api.chat.genai.Client", return_value=mock_gemini):
                    response = self.client.post(
                        "/api/chat",
                        json={"message": "What is its budget?", "conversation_id": "11111111-1111-1111-1111-111111111111"},
                        headers={"Authorization": "Bearer mocked-token"},
                    )

                    self.assertEqual(response.status_code, 200)
                    # Verify query rewriter was called with question and prior history
                    mock_rewrite.assert_called_once_with("What is its budget?", prior_history)
                    # Verify retrieval used the rewritten query
                    mock_search.assert_called_once_with(
                        query="What is the budget for Project Odyssey?",
                        tenant_id="tenant-abc",
                        user_roles=["employee"],
                        top_k=3,
                    )
                    # Verify X-Conversation-Id header is present
                    self.assertIn("x-conversation-id", response.headers)

    def test_conversational_greeting_bypasses_retrieval_and_gemini(self):
        """Pure conversational greetings return immediate direct response without calling RAG or Gemini."""
        mock_user = CurrentUser(id="user-123", tenant_id="tenant-abc", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user

        with patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search:
            with patch("app.api.chat.genai.Client") as mock_gemini_cls:
                with patch("app.api.chat.rewrite_query_for_retrieval", new_callable=AsyncMock) as mock_rewrite:
                    response = self.client.post(
                        "/api/chat",
                        json={"message": "Hello!"},
                        headers={"Authorization": "Bearer mocked-token"},
                    )

                    self.assertEqual(response.status_code, 200)
                    self.assertIn("x-conversation-id", response.headers)

                    # Retrieval, rewriting, and LLM MUST NOT be called
                    mock_search.assert_not_called()
                    mock_rewrite.assert_not_called()
                    mock_gemini_cls.assert_not_called()

                    # Stream format must match frontend expectations
                    lines = response.text.strip().split("\n\n")
                    self.assertEqual(len(lines), 3)
                    citations_frame = json.loads(lines[0].replace("data: ", ""))
                    self.assertEqual(citations_frame["type"], "citations")
                    self.assertEqual(citations_frame["citations"], [])

                    content_frame = json.loads(lines[1].replace("data: ", ""))
                    self.assertEqual(content_frame["type"], "content")
                    self.assertIn("Enterprise Knowledge Assistant", content_frame["text"])

                    self.assertEqual(lines[2], "data: [DONE]")

    def test_compound_greeting_and_question_executes_full_rag(self):
        """Greeting followed by a knowledge inquiry must route through search_documents and Gemini."""
        mock_user = CurrentUser(id="user-123", tenant_id="tenant-abc", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user

        mock_chunks = [
            RetrievalResult(
                chunk_id="chunk-01",
                document_id="doc-01",
                title="Remote Work Policy",
                heading_path=["HR", "Remote Work"],
                content="Employees may work remotely 2 days per week.",
                dense_rank=1,
                fts_rank=1,
                rrf_score=0.03,
                rerank_score=2.5,
            )
        ]

        class MockChunk:
            def __init__(self, text: str):
                self.text = text

        mock_gemini = MagicMock()
        mock_gemini.models.generate_content_stream.return_value = [
            MockChunk("Employees can work remotely up to 2 days per week.")
        ]

        with patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_chunks
            with patch("app.api.chat.genai.Client", return_value=mock_gemini):
                response = self.client.post(
                    "/api/chat",
                    json={"message": "Hello, what is the remote work policy?"},
                    headers={"Authorization": "Bearer mocked-token"},
                )

                self.assertEqual(response.status_code, 200)
                # Retrieval MUST be called for compound question
                mock_search.assert_called_once()
                self.assertEqual(mock_search.call_args.kwargs.get("query"), "Hello, what is the remote work policy?")

    def test_chat_successful_streaming_response_with_groq(self):
        """When USE_GROQ_FOR_GENERATION is True, invokes Groq OpenAI-compatible client and streams chunks."""
        mock_user = CurrentUser(
            id="user-456",
            tenant_id="tenant-redwood",
            roles=["employee", "manager"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        mock_chunks = [
            RetrievalResult(
                chunk_id="chunk-01",
                document_id="doc-01",
                title="PTO Guidelines",
                heading_path=["Benefits", "Paid Time Off"],
                content="Full-time employees receive 20 days of paid vacation annually.",
                dense_rank=1,
                fts_rank=1,
                rrf_score=0.032,
                rerank_score=3.85,
            )
        ]

        class MockDelta:
            def __init__(self, content):
                self.content = content

        class MockChoice:
            def __init__(self, content):
                self.delta = MockDelta(content)

        class MockUsage:
            prompt_tokens = 50
            completion_tokens = 10
            total_tokens = 60

        class MockChunk:
            def __init__(self, content=None, usage=None):
                self.choices = [MockChoice(content)] if content is not None else []
                self.usage = usage

        async def mock_groq_stream():
            yield MockChunk("Full-time employees ")
            yield MockChunk("receive 20 days of PTO.")
            yield MockChunk(usage=MockUsage())

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(return_value=mock_groq_stream())

        with (
            patch("app.api.chat.settings.USE_GROQ_FOR_GENERATION", True),
            patch("app.api.chat.settings.groq_chat_llm_model", "openai/gpt-oss-120b"),
            patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search,
            patch("app.api.chat.AsyncOpenAI", return_value=mock_groq_client),
        ):
            mock_search.return_value = mock_chunks
            response = self.client.post(
                "/api/chat",
                json={"message": "How many vacation days do I get?"},
                headers={"Authorization": "Bearer mocked-token"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])

            lines = response.text.strip().split("\n\n")
            self.assertEqual(len(lines), 4)

            citations_frame = json.loads(lines[0].replace("data: ", ""))
            self.assertEqual(citations_frame["type"], "citations")
            self.assertEqual(len(citations_frame["citations"]), 1)
            self.assertEqual(citations_frame["citations"][0]["document_id"], "doc-01")

            part1 = json.loads(lines[1].replace("data: ", ""))
            part2 = json.loads(lines[2].replace("data: ", ""))
            self.assertEqual(part1["text"], "Full-time employees ")
            self.assertEqual(part2["text"], "receive 20 days of PTO.")
            self.assertEqual(lines[3], "data: [DONE]")

            # Verify Groq chat completion call arguments
            mock_groq_client.chat.completions.create.assert_awaited_once()
            call_kwargs = mock_groq_client.chat.completions.create.call_args.kwargs
            self.assertEqual(call_kwargs["model"], "openai/gpt-oss-120b")
            self.assertTrue(call_kwargs["stream"])
            self.assertEqual(call_kwargs["stream_options"], {"include_usage": True})
            prompt_content = call_kwargs["messages"][0]["content"]
            self.assertIn("PTO Guidelines", prompt_content)
            self.assertIn("How many vacation days do I get?", prompt_content)

            # Verify database message was persisted
            self.mock_append_msg.assert_awaited()
            last_call = self.mock_append_msg.await_args.kwargs
            self.assertEqual(last_call["role"], "assistant")
            self.assertEqual(last_call["content"], "Full-time employees receive 20 days of PTO.")

    def test_chat_streaming_groq_generation_error(self):
        """Groq generation errors are traced and return safe sanitized error message."""
        mock_user = CurrentUser(
            id="user-789",
            tenant_id="tenant-xyz",
            roles=["employee"],
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user

        mock_chunks = [
            RetrievalResult(
                chunk_id="chunk-01",
                document_id="doc-01",
                title="Doc Title",
                heading_path=["Overview"],
                content="Some content",
                dense_rank=1,
                fts_rank=1,
                rrf_score=0.03,
                rerank_score=1.0,
            )
        ]

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Groq 503 Service Unavailable"))

        with (
            patch("app.api.chat.settings.USE_GROQ_FOR_GENERATION", True),
            patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search,
            patch("app.api.chat.AsyncOpenAI", return_value=mock_groq_client),
        ):
            mock_search.return_value = mock_chunks
            response = self.client.post(
                "/api/chat",
                json={"message": "Any question"},
                headers={"Authorization": "Bearer mocked-token"},
            )

            self.assertEqual(response.status_code, 200)
            lines = response.text.strip().split("\n\n")
            self.assertEqual(len(lines), 2)
            citations_frame = json.loads(lines[0].replace("data: ", ""))
            self.assertEqual(citations_frame["type"], "citations")
            error_frame = json.loads(lines[1].replace("data: ", ""))
            self.assertIn("could not complete", error_frame["error"])
            self.assertNotIn("Groq 503", error_frame["error"])

    def test_chat_groq_trace_span_and_model_metadata(self):
        """Trace span records groq_generation and groq model name when USE_GROQ_FOR_GENERATION is True."""
        mock_user = CurrentUser(id="user-456", tenant_id="tenant-redwood", roles=["employee"])
        app.dependency_overrides[get_current_user] = lambda: mock_user
        results = [
            RetrievalResult(
                chunk_id="chunk-01",
                document_id="doc-01",
                title="PTO Guidelines",
                heading_path=["Benefits"],
                content="Employees receive 20 days of PTO.",
                dense_rank=1,
                fts_rank=1,
                rrf_score=0.03,
                rerank_score=3.0,
            )
        ]

        class MockDelta:
            content = "Employees receive 20 days of PTO."

        class MockChoice:
            delta = MockDelta()

        class MockChunk:
            choices = [MockChoice()]
            usage = None

        async def mock_groq_stream():
            yield MockChunk()

        mock_groq = MagicMock()
        mock_groq.chat.completions.create = AsyncMock(return_value=mock_groq_stream())

        spans = {}

        def make_span(name, **kwargs):
            context = MagicMock()
            span = MagicMock()
            context.__enter__.return_value = span
            spans[name] = (kwargs, span)
            return context

        with (
            patch("app.api.chat.settings.USE_GROQ_FOR_GENERATION", True),
            patch("app.api.chat.settings.groq_chat_llm_model", "openai/gpt-oss-120b"),
            patch("app.api.chat.trace_span", side_effect=make_span),
            patch("app.api.chat.search_documents", new_callable=AsyncMock) as mock_search,
            patch("app.api.chat.AsyncOpenAI", return_value=mock_groq),
        ):
            mock_search.return_value = results
            response = self.client.post(
                "/api/chat",
                json={"message": "How many PTO days?"},
                headers={"Authorization": "******"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("groq_generation", spans)
        generation_kwargs, generation_span = spans["groq_generation"]
        self.assertEqual(generation_kwargs["model"], "openai/gpt-oss-120b")

