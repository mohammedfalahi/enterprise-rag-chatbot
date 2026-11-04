"""
Unit tests for the Contextualized Query Rewriter module.
Tests prompt assembly, empty history bypass, successful rewriting, and resilient exception fallbacks.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.rag.query_rewriter import build_rewrite_prompt, rewrite_query_for_retrieval


class TestQueryRewriter(unittest.IsolatedAsyncioTestCase):

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
