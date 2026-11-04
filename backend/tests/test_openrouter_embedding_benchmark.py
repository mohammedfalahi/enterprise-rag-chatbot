from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.openrouter_embedding_benchmark import (
    EXPECTED_DIMENSIONS,
    read_corpus_records,
    run_benchmark,
)


class FakeEmbeddingsClient:
    def __init__(self, *, invalid_dimension_at: set[int] | None = None):
        self.batch_sizes: list[int] = []
        self.invalid_dimension_at = invalid_dimension_at or set()
        self.request_number = 0

    def create(self, *, model: str, input: list[str]):
        self.assert_model(model)
        self.batch_sizes.append(len(input))
        request_number = self.request_number
        self.request_number += 1
        data = []
        for index in range(len(input)):
            dimensions = (
                EXPECTED_DIMENSIONS - 1
                if request_number == 0 and index in self.invalid_dimension_at
                else EXPECTED_DIMENSIONS
            )
            data.append(
                SimpleNamespace(index=index, embedding=[0.0] * dimensions)
            )
        return SimpleNamespace(
            data=data,
            usage=SimpleNamespace(
                prompt_tokens=len(input) * 3,
                total_tokens=len(input) * 3,
            ),
        )

    @staticmethod
    def assert_model(model: str) -> None:
        if model != "nvidia/nemotron-3-embed-1b:free":
            raise AssertionError(f"Unexpected model: {model}")


class FakeOpenRouterClient:
    def __init__(self, *, invalid_dimension_at: set[int] | None = None):
        self.embeddings = FakeEmbeddingsClient(
            invalid_dimension_at=invalid_dimension_at
        )


class FailingOpenRouterClient:
    class Embeddings:
        @staticmethod
        def create(*, model: str, input: list[str]):
            raise RuntimeError("request failed")

    embeddings = Embeddings()


class TestOpenRouterEmbeddingBenchmark(unittest.TestCase):
    def _write_jsonl(self, records: list[object]) -> Path:
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False
        )
        with temp_file:
            for record in records:
                if isinstance(record, str):
                    temp_file.write(record + "\n")
                else:
                    temp_file.write(json.dumps(record) + "\n")
        path = Path(temp_file.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_reads_text_field_and_marks_bad_records(self):
        path = self._write_jsonl(
            [
                {"doc_id": "one", "text": "first document"},
                "{invalid json",
                {"doc_id": "three", "text": ""},
                ["not", "an", "object"],
            ]
        )

        records = read_corpus_records(path)

        self.assertEqual(len(records), 4)
        self.assertEqual(records[0].text, "first document")
        self.assertEqual(records[1].error, "invalid JSON")
        self.assertEqual(records[2].error, "missing or empty string field 'text'")
        self.assertEqual(records[3].error, "record is not a JSON object")

    def test_reads_no_more_than_the_first_100_records(self):
        path = self._write_jsonl([{"text": f"document {i}"} for i in range(105)])

        records = read_corpus_records(path)

        self.assertEqual(len(records), 100)
        self.assertEqual(records[-1].text, "document 99")

    def test_batches_records_and_checks_embedding_dimensions(self):
        path = self._write_jsonl([{"text": f"document {i}"} for i in range(27)])
        client = FakeOpenRouterClient(invalid_dimension_at={1})

        summary = run_benchmark(path, client)

        self.assertEqual(client.embeddings.batch_sizes, [25, 2])
        self.assertEqual(summary.records_read, 27)
        self.assertEqual(summary.successful_records, 26)
        self.assertEqual(summary.failed_records, 1)
        self.assertEqual(summary.prompt_tokens, 81)
        self.assertEqual(summary.total_tokens, 81)

    def test_counts_invalid_records_without_sending_them(self):
        path = self._write_jsonl(
            [
                {"text": "send this"},
                {"title": "missing text"},
            ]
        )
        client = FakeOpenRouterClient()

        summary = run_benchmark(path, client)

        self.assertEqual(client.embeddings.batch_sizes, [1])
        self.assertEqual(summary.successful_records, 1)
        self.assertEqual(summary.failed_records, 1)

    def test_counts_records_when_an_api_batch_fails(self):
        path = self._write_jsonl([{"text": "first"}, {"text": "second"}])

        with redirect_stdout(StringIO()):
            summary = run_benchmark(path, FailingOpenRouterClient())

        self.assertEqual(summary.successful_records, 0)
        self.assertEqual(summary.failed_records, 2)


if __name__ == "__main__":
    unittest.main()
