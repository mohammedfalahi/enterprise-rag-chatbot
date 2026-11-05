"""
Unit tests for the Contextualized Query Rewriter module.
Tests prompt assembly, empty history bypass, successful rewriting, and resilient exception fallbacks.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.rag.query_rewriter import build_rewrite_prompt, rewrite_query_for_retrieval


class TestQueryRewriter(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Ensure unit tests default to Gemini query rewriter unless explicitly overridden
        self.groq_patcher = patch(
            "app.rag.query_rewriter.settings.USE_GROQ_FOR_QUERY_REWRITE", False
        )
        self.mock_use_groq = self.groq_patcher.start()

    def tearDown(self):
        self.groq_patcher.stop()

    def test_build_rewrite_prompt(self):
        history = [
            {"role": "user", "content": "Tell me about Project Odyssey."},
            {"role": "assistant", "content": "Project Odyssey is our core migration project."},
        ]
        prompt = build_rewrite_prompt("What is its budget?", history)
        self.assertIn("User: Tell me about Project Odyssey.", prompt)
        self.assertIn("Assistant: Project Odyssey is our core migration project.", prompt)
        self.assertIn("Follow-up Question:\nWhat is its budget?", prompt)
        self.assertIn("Standalone Search Query:", prompt)

    async def test_empty_question_returns_as_is(self):
        result = await rewrite_query_for_retrieval("   ", [{"role": "user", "content": "Hello"}])
        self.assertEqual(result, "   ")

    async def test_empty_history_bypasses_llm(self):
        result = await rewrite_query_for_retrieval("What is the travel policy?", [])
        self.assertEqual(result, "What is the travel policy?")

    async def test_rewrite_query_successful_expansion(self):
        history = [
            {"role": "user", "content": "Who is the lead for Project Odyssey?"},
            {"role": "assistant", "content": "Sarah Connor leads the engineering team."},
        ]

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Project Odyssey engineering team lead contact and role"
        mock_client.models.generate_content.return_value = mock_response

        with patch("app.rag.query_rewriter.genai.Client", return_value=mock_client):
            rewritten = await rewrite_query_for_retrieval("What is her title?", history)
            self.assertEqual(rewritten, "Project Odyssey engineering team lead contact and role")

    async def test_query_trace_captures_redacted_content_only_when_enabled(self):
        history = [{"role": "user", "content": "Earlier question"}]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "rewritten query"
        mock_client.models.generate_content.return_value = mock_response
        mock_span = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_span

        with (
            patch("app.rag.query_rewriter.genai.Client", return_value=mock_client),
            patch("app.rag.query_rewriter.trace_span", return_value=mock_context) as trace,
            patch("app.rag.query_rewriter.settings.langfuse_capture_content", False),
        ):
            await rewrite_query_for_retrieval("original query", history)
            self.assertNotIn("original_query", trace.call_args.kwargs["input"])
            self.assertNotIn("rewritten_query", mock_span.update.call_args.kwargs["output"])

        mock_response.text = "rewritten password=example-secret"
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_span
        with (
            patch("app.rag.query_rewriter.genai.Client", return_value=mock_client),
            patch("app.rag.query_rewriter.trace_span", return_value=mock_context) as trace,
            patch("app.rag.query_rewriter.settings.langfuse_capture_content", True),
        ):
            await rewrite_query_for_retrieval("password=example-secret", history)
            trace_input = trace.call_args.kwargs["input"]
            trace_output = mock_span.update.call_args.kwargs["output"]
            self.assertIn("original_query", trace_input)
            self.assertIn("rewritten_query", trace_output)
            self.assertNotIn("example-secret", str(trace_input))
            self.assertNotIn("example-secret", str(trace_output))

    async def test_rewrite_query_exception_falls_back_gracefully(self):
        history = [
            {"role": "user", "content": "Tell me about Q3 earnings."},
            {"role": "assistant", "content": "Revenue grew by 15%."},
        ]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API Quota exceeded: 429")

        with patch("app.rag.query_rewriter.genai.Client", return_value=mock_client):
            rewritten = await rewrite_query_for_retrieval("Did profit margin increase?", history)
            # Must fall back gracefully to original question
            self.assertEqual(rewritten, "Did profit margin increase?")

    async def test_rewrite_query_empty_llm_response_falls_back(self):
        history = [{"role": "user", "content": "Hi"}]

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response

        with patch("app.rag.query_rewriter.genai.Client", return_value=mock_client):
            rewritten = await rewrite_query_for_retrieval("How to login?", history)
            self.assertEqual(rewritten, "How to login?")

    async def test_rewrite_query_successful_with_groq(self):
        history = [
            {"role": "user", "content": "Tell me about Project Odyssey."},
            {"role": "assistant", "content": "Project Odyssey is our core migration project."},
        ]

        mock_choice = MagicMock()
        mock_choice.message.content = "Project Odyssey total project budget"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 30
        mock_usage.completion_tokens = 6
        mock_usage.total_tokens = 36

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_span = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_span

        with (
            patch("app.rag.query_rewriter.settings.USE_GROQ_FOR_QUERY_REWRITE", True),
            patch("app.rag.query_rewriter.settings.groq_query_rewrite_model", "openai/gpt-oss-120b"),
            patch("app.rag.query_rewriter.settings.langfuse_capture_content", False),
            patch("app.rag.query_rewriter.AsyncOpenAI", return_value=mock_groq_client),
            patch("app.rag.query_rewriter.trace_span", return_value=mock_context) as trace,
        ):
            rewritten = await rewrite_query_for_retrieval("What is its budget?", history)
            self.assertEqual(rewritten, "Project Odyssey total project budget")

            # Verify Groq client was called with expected model and prompt
            mock_groq_client.chat.completions.create.assert_awaited_once()
            call_kwargs = mock_groq_client.chat.completions.create.call_args.kwargs
            self.assertEqual(call_kwargs["model"], "openai/gpt-oss-120b")
            self.assertEqual(call_kwargs["temperature"], 0.0)
            self.assertIn("What is its budget?", call_kwargs["messages"][0]["content"])

            # Verify trace span recorded groq model and token counts
            self.assertEqual(trace.call_args.kwargs["model"], "openai/gpt-oss-120b")
            mock_span.update.assert_called_once_with(
                output={"rewritten_query_chars": len("Project Odyssey total project budget"), "fallback": False},
                usage_details={"input": 30, "output": 6, "total": 36},
            )

    async def test_rewrite_query_groq_exception_falls_back_gracefully(self):
        history = [
            {"role": "user", "content": "Tell me about Q3 earnings."},
            {"role": "assistant", "content": "Revenue grew by 15%."},
        ]

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Groq 429 Rate limit"))

        with (
            patch("app.rag.query_rewriter.settings.USE_GROQ_FOR_QUERY_REWRITE", True),
            patch("app.rag.query_rewriter.AsyncOpenAI", return_value=mock_groq_client),
        ):
            rewritten = await rewrite_query_for_retrieval("Did profit margin increase?", history)
            self.assertEqual(rewritten, "Did profit margin increase?")

    async def test_rewrite_query_groq_empty_response_falls_back(self):
        history = [{"role": "user", "content": "Hi"}]

        mock_response = MagicMock()
        mock_response.choices = []

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("app.rag.query_rewriter.settings.USE_GROQ_FOR_QUERY_REWRITE", True),
            patch("app.rag.query_rewriter.AsyncOpenAI", return_value=mock_groq_client),
        ):
            rewritten = await rewrite_query_for_retrieval("How to login?", history)
            self.assertEqual(rewritten, "How to login?")

