from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.evaluate_ragas import load_evaluation_records


class TestRagasEvaluationInput(unittest.TestCase):
    def _write_records(self, records: list[dict]) -> Path:
        temp_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        with temp_file:
            for record in records:
                temp_file.write(json.dumps(record) + "\n")
        self.addCleanup(Path(temp_file.name).unlink, missing_ok=True)
        return Path(temp_file.name)

    def test_loads_complete_saved_evaluation_records(self):
        record = {
            "user_input": "Question",
            "response": "Answer",
            "retrieved_contexts": ["Context"],
            "reference": "Reviewed reference",
            "extra_metadata": "not passed to Ragas",
        }
        loaded = load_evaluation_records(self._write_records([record]))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            loaded[0],
            {
                "user_input": "Question",
                "response": "Answer",
                "retrieved_contexts": ["Context"],
                "reference": "Reviewed reference",
            },
        )

    def test_rejects_missing_reference_instead_of_fabricating_one(self):
        record = {
            "user_input": "Question",
            "response": "Answer",
            "retrieved_contexts": ["Context"],
        }
        with self.assertRaisesRegex(ValueError, "missing required fields: reference"):
            load_evaluation_records(self._write_records([record]))

    def test_rejects_empty_retrieved_contexts(self):
        record = {
            "user_input": "Question",
            "response": "Answer",
            "retrieved_contexts": [],
            "reference": "Reviewed reference",
        }
        with self.assertRaisesRegex(ValueError, "retrieved_contexts"):
            load_evaluation_records(self._write_records([record]))

    def test_preserves_traceability_metadata_fields(self):
        record = {
            "test_id": "qst_0164:employee",
            "question_id": "qst_0164",
            "role": "employee",
            "category": "basic",
            "user_input": "In HyperOnboard...",
            "response": "Answer...",
            "retrieved_contexts": ["Context 1"],
            "reference": "Reference facts...",
            "extra_ignored_field": "dropped",
        }
        loaded = load_evaluation_records(self._write_records([record]))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["test_id"], "qst_0164:employee")
        self.assertEqual(loaded[0]["question_id"], "qst_0164")
        self.assertEqual(loaded[0]["role"], "employee")
        self.assertEqual(loaded[0]["category"], "basic")
        self.assertNotIn("extra_ignored_field", loaded[0])


class TestRagasScoreSanitizationAndReliability(unittest.TestCase):
    def test_sanitize_score_handles_finite_nan_and_inf(self):
        import math
        from app.eval.metrics.generation import sanitize_score, sanitize_for_strict_json

        self.assertEqual(sanitize_score(0.85231), 0.8523)
        self.assertEqual(sanitize_score(1), 1.0)
        self.assertIsNone(sanitize_score(float("nan")))
        self.assertIsNone(sanitize_score(float("inf")))
        self.assertIsNone(sanitize_score(-float("inf")))
        self.assertIsNone(sanitize_score(None))
        self.assertIsNone(sanitize_score("not_a_number"))

        raw = {
            "valid": 0.5,
            "nan_val": float("nan"),
            "nested": [float("inf"), 0.9, None],
        }
        cleaned = sanitize_for_strict_json(raw)
        self.assertEqual(cleaned["valid"], 0.5)
        self.assertIsNone(cleaned["nan_val"])
        self.assertEqual(cleaned["nested"], [None, 0.9, None])

        # Verify strict JSON serialization succeeds with allow_nan=False
        dumped = json.dumps(cleaned, allow_nan=False)
        self.assertIn('"nan_val": null', dumped)

    def test_evaluate_ragas_records_excludes_nan_and_tracks_counts(self):
        from unittest.mock import MagicMock, patch
        from app.eval.metrics.generation import evaluate_ragas_records

        mock_result = MagicMock()
        mock_result.scores = [
            {"faithfulness": 0.8, "answer_correctness": 0.6, "context_recall": float("nan")},
            {"faithfulness": 0.9, "answer_correctness": float("nan"), "context_recall": 1.0},
            {"faithfulness": float("nan"), "answer_correctness": 0.8, "context_recall": 0.8},
        ]

        records = [
            {
                "test_id": "t1",
                "question_id": "q1",
                "role": "employee",
                "category": "basic",
                "user_input": "Q1",
                "response": "R1",
                "retrieved_contexts": ["C1"],
                "reference": "Ref1",
            },
            {
                "test_id": "t2",
                "question_id": "q2",
                "role": "manager",
                "category": "project",
                "user_input": "Q2",
                "response": "R2",
                "retrieved_contexts": ["C2"],
                "reference": "Ref2",
            },
            {
                "test_id": "t3",
                "question_id": "q3",
                "role": "executive",
                "category": "semantic",
                "user_input": "Q3",
                "response": "R3",
                "retrieved_contexts": ["C3"],
                "reference": "Ref3",
            },
        ]

        with (
            patch("ragas.evaluate", return_value=mock_result),
            patch("ragas.EvaluationDataset.from_list"),
            patch("app.eval.metrics.generation.settings.GEMINI_API_KEY", "mock-key"),
        ):
            report = evaluate_ragas_records(records)

            # 1. Faithfulness: [0.8, 0.9], NaN excluded -> mean = 0.85
            self.assertEqual(report["aggregate_metrics"]["faithfulness"], 0.85)
            # 2. Answer Correctness: [0.6, 0.8], NaN excluded -> mean = 0.7
            self.assertEqual(report["aggregate_metrics"]["answer_correctness"], 0.7)
            # 3. Context Recall: [1.0, 0.8], NaN excluded -> mean = 0.9
            self.assertEqual(report["aggregate_metrics"]["context_recall"], 0.9)

            # 4. Metric counts: n_scored and n_failed
            self.assertEqual(report["metric_counts"]["faithfulness"]["n_scored"], 2)
            self.assertEqual(report["metric_counts"]["faithfulness"]["n_failed"], 1)
            self.assertEqual(report["metric_counts"]["answer_correctness"]["n_scored"], 2)
            self.assertEqual(report["metric_counts"]["answer_correctness"]["n_failed"], 1)

            # 5. Traceability: test_id, question_id, role, category preserved per question
            self.assertEqual(report["per_question_scores"][0]["test_id"], "t1")
            self.assertEqual(report["per_question_scores"][0]["question_id"], "q1")
            self.assertEqual(report["per_question_scores"][0]["role"], "employee")
            self.assertEqual(report["per_question_scores"][0]["category"], "basic")

            # 6. Strict JSON serialization without error
            serialized = json.dumps(report, allow_nan=False)
            self.assertNotIn("NaN", serialized)


if __name__ == "__main__":
    unittest.main()
