from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.eval.runner import evaluate_dataset


class TestEvaluationRunner(unittest.IsolatedAsyncioTestCase):
    async def test_missing_or_empty_benchmark_files_fail_the_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.jsonl"
            with self.assertRaises(FileNotFoundError):
                await evaluate_dataset(golden_path=missing_path, injection_path=None)

            empty_path = Path(temp_dir) / "empty.jsonl"
            empty_path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                await evaluate_dataset(golden_path=empty_path, injection_path=None)

    async def test_missing_injection_dataset_fails_the_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            golden_path = Path(temp_dir) / "golden.jsonl"
            golden_path.write_text(
                json.dumps({
                    "test_id": "sample-1",
                    "question": "Find a sample answer.",
                    "expected_behavior": "answer",
                    "gold_doc_ids": [],
                    "role": "employee",
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                await evaluate_dataset(
                    golden_path=golden_path,
                    injection_path=Path(temp_dir) / "missing-injection.jsonl",
                )

    async def test_retrieval_uses_the_authenticated_tenant_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            golden_path = Path(temp_dir) / "golden.jsonl"
            golden_path.write_text(
                json.dumps({
                    "test_id": "sample-1",
                    "question": "Find a sample answer.",
                    "expected_behavior": "answer",
                    "gold_doc_ids": [],
                    "role": "employee",
                }) + "\n",
                encoding="utf-8",
            )
            received_tenant_ids: list[str] = []

            async def mock_retriever(question, tenant_id, roles, top_k=5):
                received_tenant_ids.append(tenant_id)
                return []

            report = await evaluate_dataset(
                golden_path=golden_path,
                injection_path=None,
                tenant_id="authenticated-tenant",
                mock_retriever=mock_retriever,
            )

            self.assertEqual(received_tenant_ids, ["authenticated-tenant"])
            self.assertEqual(report.tenant_id, "authenticated-tenant")
            self.assertEqual(report.total_cases, 1)


if __name__ == "__main__":
    unittest.main()
