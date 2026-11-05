"""
Pydantic schemas and data contracts for the Unified AI Efficacy Evaluation layer.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class RetrievalMetrics(BaseModel):
    k: int = Field(default=5, description="Ranking depth K")
    precision_at_k: float = Field(default=0.0, description="Precision@K")
    recall_at_k: float = Field(default=0.0, description="Recall@K")
    mrr: float = Field(default=0.0, description="Mean Reciprocal Rank")
    ndcg_at_k: float = Field(default=0.0, description="Normalized Discounted Cumulative Gain@K")
    permission_leakage_count: int = Field(default=0, description="Count of chunks returned outside user roles")
    permission_leakage_rate: float = Field(default=0.0, description="Leakage rate (must be 0.0%)")


class RefusalMetrics(BaseModel):
    total_unanswerable: int = Field(default=0, description="Total unanswerable/deny/injection cases")
    total_answerable: int = Field(default=0, description="Total answerable cases")
    true_refusals: int = Field(default=0, description="Correctly refused unanswerable queries (TP)")
    false_refusals: int = Field(default=0, description="Inappropriately refused answerable queries (FP)")
    false_answers: int = Field(default=0, description="Inappropriately answered unanswerable queries (FN)")
    true_answers: int = Field(default=0, description="Correctly answered queries (TN)")
    precision: float = Field(default=1.0, description="Refusal Precision = TP / (TP + FP)")
    recall: float = Field(default=1.0, description="Refusal Recall = TP / (TP + FN)")
    f1_score: float = Field(default=1.0, description="Refusal F1 Score")


class GenerationMetrics(BaseModel):
    faithfulness: float = Field(default=0.0, description="Factual entailment to retrieved context (0-1)")
    answer_relevancy: float = Field(default=0.0, description="Responsiveness to user question (0-1)")
    answer_correctness: float = Field(default=0.0, description="Factual and semantic agreement with reference (0-1)")
    context_precision: float | None = Field(default=None, description="Ragas LLM context precision with reference")
    context_recall: float | None = Field(default=None, description="Ragas context recall")


class EvaluationTestCase(BaseModel):
    test_id: str
    question_id: str | None = None
    role: str = "employee"
    category: str = "basic"
    expected_behavior: Literal["answer", "no_answer", "deny", "partial"] = "answer"
    question: str
    gold_doc_ids: list[str] = Field(default_factory=list)
    answer_facts: list[str] = Field(default_factory=list)


class PerQuestionResult(BaseModel):
    test_id: str
    question: str
    role: str
    category: str
    expected_behavior: str
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_titles: list[str] = Field(default_factory=list)
    is_refused: bool = False
    permission_leaked: bool = False
    retrieval_scores: dict[str, float] = Field(default_factory=dict)
    generation_scores: dict[str, float] = Field(default_factory=dict)
    generated_response: str | None = None
    reference_answer: str | None = None


class EvaluationReport(BaseModel):
    tenant_id: str
    tenant_name: str
    timestamp: str
    total_cases: int
    retrieval: RetrievalMetrics
    refusal: RefusalMetrics
    generation: GenerationMetrics | None = None
    per_question: list[PerQuestionResult] = Field(default_factory=list)
    categories: dict[str, Any] = Field(default_factory=dict)
    roles: dict[str, Any] = Field(default_factory=dict)
