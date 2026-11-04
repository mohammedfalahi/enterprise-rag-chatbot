"""
Deterministic, rule-based intent router for conversational query interception.
Detects greetings, gratitude, small talk, goodbyes, and bot identity queries
before query rewriting, embedding generation, or hybrid vector retrieval.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass


class IntentType(enum.Enum):
    GREETING = "greeting"
    GRATITUDE = "gratitude"
    GOODBYE = "goodbye"
    IDENTITY = "identity"


@dataclass(frozen=True)
class IntentRouteResult:
    intent: IntentType
    response: str


GREETING_PATTERNS = {
    "hello",
    "hi",
    "hai",
    "hlo",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "hi there",
    "hey there",
    "howdy",
    "greetings",
    "yo",
}

GRATITUDE_PATTERNS = {
    "thank you",
    "thanks",
    "thanks a lot",
    "thank you so much",
    "many thanks",
    "appreciate it",
    "thank you very much",
    "thx",
    "ty",
}

GOODBYE_PATTERNS = {
    "bye",
    "goodbye",
    "see you",
    "see you later",
    "see ya",
    "take care",
    "have a good day",
    "have a nice day",
    "cya",
}

IDENTITY_PATTERNS = {
    "who are you",
    "what are you",
    "what can you do",
    "what do you do",
    "what is your name",
    "whats your name",
    "how are you",
    "how are you doing",
    "how is it going",
    "hows it going",
    "help",
}

RESPONSES: dict[IntentType, str] = {
    IntentType.GREETING: (
        "Hello! I am your Enterprise Knowledge Assistant. "
        "How can I help you with our verified enterprise policies and documentation today?"
    ),
    IntentType.GRATITUDE: (
        "You're welcome! Let me know if you need any more details from our verified documents."
    ),
    IntentType.GOODBYE: (
        "Goodbye! Feel free to return whenever you have questions about company policies or roadmaps."
    ),
    IntentType.IDENTITY: (
        "I am your Enterprise Knowledge Assistant. "
        "I help you query company policies, departmental workflows, technical specifications, "
        "and internal documentation authorized for your enterprise role."
    ),
}

# Substantive enterprise keywords that indicate an inquiry, not small talk
_ENTERPRISE_INQUIRY_TERMS = {
    "policy",
    "document",
    "documents",
    "doc",
    "docs",
    "procedure",
    "guideline",
    "guidelines",
    "sla",
    "kpi",
    "kpis",
    "metric",
    "metrics",
    "vacation",
    "leave",
    "pto",
    "salary",
    "compensation",
    "benefits",
    "benefit",
    "pricing",
    "price",
    "tiers",
    "discount",
    "headcount",
    "hiring",
    "review",
    "eligibility",
    "criteria",
    "contract",
    "compliance",
    "audit",
    "travel",
    "expense",
    "reimbursement",
    "okr",
    "okrs",
    "budget",
    "department",
}

# Interrogative keywords for follow-ups and inquiries
_INQUIRY_LEAD_TERMS = {
    "why",
    "how",
    "what",
    "when",
    "where",
    "which",
    "whose",
    "whom",
    "can",
    "could",
    "would",
    "should",
    "is",
    "are",
    "does",
    "do",
    "did",
    "tell",
    "explain",
    "show",
    "summarize",
}


def _normalize_text(text: str) -> str:
    """Strip punctuation and normalize whitespace to lowercase single spaces."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(cleaned.split())


def classify_conversational_intent(message: str) -> IntentRouteResult | None:
    """
    Deterministically classifies whether a message is purely conversational
    (greeting, gratitude, goodbye, or bot identity).

    Returns IntentRouteResult with direct response if matched, or None if the message
    is a knowledge inquiry or follow-up question that must route to RAG.
    """
    normalized = _normalize_text(message)
    if not normalized:
        return None

    words = normalized.split()

    # 1. Any mention of company/policy inquiry terms immediately routes to RAG
    if any(term in words for term in _ENTERPRISE_INQUIRY_TERMS):
        return None

    # 2. Check for exact identity questions (e.g. "who are you", "what can you do")
    if normalized in IDENTITY_PATTERNS:
        return IntentRouteResult(
            intent=IntentType.IDENTITY,
            response=RESPONSES[IntentType.IDENTITY],
        )

    # 3. Follow-up inquiries (e.g. "why", "how about that", "can you explain")
    # Must route to RAG and conversational query rewriter
    if any(lead in words for lead in _INQUIRY_LEAD_TERMS):
        return None

    # 4. Strip common polite fillers/vocatives like "bot", "assistant", "there"
    stripped_vocatives = " ".join(
        w for w in words if w not in {"bot", "assistant", "there"}
    )

    # 5. Check exact greeting patterns
    if normalized in GREETING_PATTERNS or stripped_vocatives in GREETING_PATTERNS:
        return IntentRouteResult(
            intent=IntentType.GREETING,
            response=RESPONSES[IntentType.GREETING],
        )

    # 6. Check exact gratitude patterns
    if normalized in GRATITUDE_PATTERNS or stripped_vocatives in GRATITUDE_PATTERNS:
        return IntentRouteResult(
            intent=IntentType.GRATITUDE,
            response=RESPONSES[IntentType.GRATITUDE],
        )

    # 7. Check exact goodbye patterns
    if normalized in GOODBYE_PATTERNS or stripped_vocatives in GOODBYE_PATTERNS:
        return IntentRouteResult(
            intent=IntentType.GOODBYE,
            response=RESPONSES[IntentType.GOODBYE],
        )

    # All other inputs (short keywords, ambiguous inputs, follow-ups) route to RAG
    return None
