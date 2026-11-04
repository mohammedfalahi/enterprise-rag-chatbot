"""
Unit tests for deterministic intent router.
Verifies:
- Conversational intents (greetings, gratitude, goodbyes, identity) return immediate direct responses.
- Enterprise knowledge questions return None (route to RAG).
- Compound queries (e.g. "Hello, what is the remote work policy?") return None (route to RAG).
- Contextual follow-up questions (e.g. "Why?", "How about that?", "What are the eligibility criteria?") return None (route to RAG).
- Isolated single enterprise keywords (e.g. "policy", "pricing") return None (route to RAG).
"""

from __future__ import annotations

import unittest
from app.rag.intent_router import IntentType, classify_conversational_intent


class TestIntentRouter(unittest.TestCase):

    def test_greetings_detection(self):
        cases = [
            "hello",
            "Hello!",
            "Hi",
            "hey",
            "good morning",
            "Good afternoon!",
            "good evening",
            "hi there",
            "hey there",
            "howdy",
            "hello assistant",
            "hi bot",
        ]
        for msg in cases:
            with self.subTest(msg=msg):
                result = classify_conversational_intent(msg)
                self.assertIsNotNone(result, f"Expected greeting match for: {msg}")
                self.assertEqual(result.intent, IntentType.GREETING)
                self.assertIn("Enterprise Knowledge Assistant", result.response)

    def test_gratitude_detection(self):
        cases = [
            "thank you",
            "Thanks!",
            "thanks a lot",
            "thank you so much",
            "appreciate it",
            "thx",
            "thank you assistant",
        ]
        for msg in cases:
            with self.subTest(msg=msg):
                result = classify_conversational_intent(msg)
                self.assertIsNotNone(result, f"Expected gratitude match for: {msg}")
                self.assertEqual(result.intent, IntentType.GRATITUDE)
                self.assertIn("welcome", result.response.lower())

    def test_goodbye_detection(self):
        cases = [
            "bye",
            "Goodbye!",
            "see you",
            "see you later",
            "take care",
            "have a good day",
            "bye bot",
        ]
        for msg in cases:
            with self.subTest(msg=msg):
                result = classify_conversational_intent(msg)
                self.assertIsNotNone(result, f"Expected goodbye match for: {msg}")
                self.assertEqual(result.intent, IntentType.GOODBYE)
                self.assertIn("Goodbye", result.response)

    def test_bot_identity_detection(self):
        cases = [
            "who are you",
            "Who are you?",
            "what can you do",
            "What can you do?",
            "what are you",
            "how are you",
            "what is your name",
            "help",
        ]
        for msg in cases:
            with self.subTest(msg=msg):
                result = classify_conversational_intent(msg)
                self.assertIsNotNone(result, f"Expected identity match for: {msg}")
                self.assertEqual(result.intent, IntentType.IDENTITY)
                self.assertIn("Enterprise Knowledge Assistant", result.response)

    def test_compound_greeting_and_question_routes_to_rag(self):
        """Messages with greeting + enterprise question MUST NOT be intercepted."""
        cases = [
            "Hello, what is the remote work policy?",
            "Hi, can you explain our healthcare benefits?",
            "Good morning! How do I submit an expense report?",
            "Hey, tell me about the Q4 pricing tiers.",
            "Thanks, but what are the travel reimbursement limits?",
        ]
        for msg in cases:
            with self.subTest(msg=msg):
                result = classify_conversational_intent(msg)
                self.assertIsNone(result, f"Compound query should route to RAG: {msg}")

    def test_follow_up_questions_route_to_rag(self):
        """Conversational follow-up inquiries MUST NOT be intercepted as small talk."""
        cases = [
            "Why?",
            "Why is that?",
            "How about that?",
            "What are the eligibility criteria?",
            "Can you explain further?",
            "Tell me more.",
            "Where can I find that?",
            "Is there an exception?",
        ]
        for msg in cases:
            with self.subTest(msg=msg):
                result = classify_conversational_intent(msg)
                self.assertIsNone(result, f"Follow-up inquiry should route to RAG: {msg}")

    def test_isolated_enterprise_keywords_route_to_rag(self):
        """Single short enterprise keywords MUST NOT be classified as small talk."""
        cases = [
            "policy",
            "pricing",
            "benefits",
            "SLA",
            "headcount",
            "KPIs",
            "reimbursement",
            "travel",
        ]
        for msg in cases:
            with self.subTest(msg=msg):
                result = classify_conversational_intent(msg)
                self.assertIsNone(result, f"Enterprise keyword should route to RAG: {msg}")


if __name__ == "__main__":
    unittest.main()
