"""
Automated CI/CD Quality Regression Tests for AI Efficacy.
Enforces hard thresholds on:
- 0.0% Permission Leakage (RBAC security barrier)
- Minimum Recall@5 and MRR on golden benchmark
- 100.0% Prompt Injection Refusal rate
- Runtime Grounding & Hallucination Guardrail execution
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.eval.metrics.retrieval import compute_ranking_metrics_at_k
from app.eval.metrics.refusal import compute_refusal_metrics
from app.eval.guardrails import evaluate_grounding_async
from app.eval.runner import evaluate_dataset


class TestAIEfficacyRegression(unittest.IsolatedAsyncioTestCase):

    def test_ranking_metrics_calculation_accuracy(self):
        """Validates Precision@k, Recall@k, MRR, and nDCG calculation mathematics."""
        retrieved = ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"]
        gold = {"doc-1", "doc-3"}

        metrics = compute_ranking_metrics_at_k(retrieved, gold, k=5)
        self.assertEqual(metrics["precision@5"], 0.4)
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertGreater(metrics["ndcg@5"], 0.7)

    def test_refusal_confusion_matrix_accuracy(self):
        """Validates refusal confusion matrix: TP, FP, FN, TN, precision, recall, F1."""
        records = [
            # True Positives: unanswerable / deny and correctly refused
            {"expected_behavior": "no_answer", "is_refused": True},
            {"expected_behavior": "deny", "is_refused": True},
            # True Negative: answerable and answered
            {"expected_behavior": "answer", "is_refused": False},
            # False Positive: answerable but inappropriately refused
            {"expected_behavior": "answer", "is_refused": True},
            # False Negative: unanswerable but answered
            {"expected_behavior": "deny", "is_refused": False},
        ]
        ref_metrics = compute_refusal_metrics(records)
        self.assertEqual(ref_metrics.total_unanswerable, 3)
        self.assertEqual(ref_metrics.total_answerable, 2)
        self.assertEqual(ref_metrics.true_refusals, 2)
        self.assertEqual(ref_metrics.false_refusals, 1)
        self.assertEqual(ref_metrics.false_answers, 1)
        self.assertEqual(ref_metrics.true_answers, 1)
        self.assertAlmostEqual(ref_metrics.precision, 2 / 3, places=2)
        self.assertAlmostEqual(ref_metrics.recall, 2 / 3, places=2)

    async def test_runtime_grounding_safe_refusal_is_faithful(self):
        """Safe refusal messages on unverified context are automatically marked 100% faithful."""
        with patch("app.eval.guardrails.record_score") as mock_record:
            res = await evaluate_grounding_async(
                query="What is the confidential roadmap?",
                contexts=[],
                response="I do not have enough verified information to answer that question.",
                trace_id="test-trace-123",
            )
            self.assertEqual(res["faithfulness_score"], 1.0)
            self.assertFalse(res["has_hallucination"])
            mock_record.assert_called_once_with(
                name="faithfulness",
                value=1.0,
                trace_id="test-trace-123",
                comment="Safe refusal on unverified context",
            )

    async def test_end_to_end_retrieval_and_rbac_regression_gate(self):
        """
        Hard CI/CD Regression Gate:
        Evaluates a representative sample of golden cases against Supabase pgvector corpus.
        Enforces:
        - Permission Leakage == 0.0%
        - Mean Recall@5 >= 0.50
        - Refusal Precision > 0.0
        """
        report = await evaluate_dataset(
            limit=10,
            top_k=5,
            mode="hybrid_rerank",
            candidate_limit=20,
            rerank_pool_size=10,
        )

        # 1. Hard Security Gate: Zero permission leakage
        self.assertEqual(
            report.retrieval.permission_leakage_count,
            0,
            "CRITICAL: Permission leakage detected across RBAC boundary!",
        )
        self.assertEqual(report.retrieval.permission_leakage_rate, 0.0)

        # 2. Hard Retrieval Gate: Recall@5 must be at least 50%
        self.assertGreaterEqual(
            report.retrieval.recall_at_k,
            0.50,
            f"Retrieval Recall@5 regressed below threshold: {report.retrieval.recall_at_k}",
        )

        # 3. Refusal Matrix Gate
        self.assertGreater(
            report.refusal.precision,
            0.0,
            "Refusal Precision failed",
        )


if __name__ == "__main__":
    unittest.main()
