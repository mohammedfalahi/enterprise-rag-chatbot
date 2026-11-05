"""
Unified AI Efficacy Evaluation Layer for Enterprise RAG Chatbot.
Provides centralized evaluation for:
- Retrieval Quality (Precision@K, Recall@K, MRR, nDCG)
- Permission Leakage & RBAC Gating
- Generation Quality (Faithfulness, Answer Correctness, Answer Relevancy)
- Refusal Accuracy (Precision, Recall, F1 for unanswerable/injection queries)
- Runtime Grounding Verification & Langfuse Trace Scoring
"""

from app.eval.schemas import (
    RetrievalMetrics,
    RefusalMetrics,
    GenerationMetrics,
    EvaluationReport,
    EvaluationTestCase,
)

__all__ = [
    "RetrievalMetrics",
    "RefusalMetrics",
    "GenerationMetrics",
    "EvaluationReport",
    "EvaluationTestCase",
]
