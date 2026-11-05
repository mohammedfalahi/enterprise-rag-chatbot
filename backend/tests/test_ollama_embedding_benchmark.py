from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.ollama_embedding_benchmark import (
    EXPECTED_DIMENSIONS,
    read_corpus_records,
    run_benchmark,
)


class TestOllamaEmbeddingBenchmark(unittest.TestCase):
    def _write_jsonl(self, records: list[object]) -> Path:
        temp_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        with temp_file:
            for record in records:
                temp_file.write(
                    record + "\n" if isinstance(record, str) else json.dumps(record) + "\n"
                )
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

    def test_embeds_batches_and_counts_invalid_dimensions_as_failures(self):
        path = self._write_jsonl([{"text": f"document {i}"} for i in range(27)])
        batch_sizes: list[int] = []

        def fake_embed(texts, batch_size):
            batch_sizes.append(len(texts))
            dimension = EXPECTED_DIMENSIONS - 1 if len(batch_sizes) == 1 else EXPECTED_DIMENSIONS
            return [[0.0] * dimension for _ in texts]

        with redirect_stdout(StringIO()):
            summary = run_benchmark(path, fake_embed, batch_size=25)
        self.assertEqual(batch_sizes, [25, 2])
        self.assertEqual(summary.records_read, 27)
        self.assertEqual(summary.successful_records, 2)
        self.assertEqual(summary.failed_records, 25)

    def test_invalid_records_are_not_sent_to_ollama(self):
        path = self._write_jsonl(
            [{"text": "send this"}, {"title": "missing text"}]
        )
        batches: list[list[str]] = []

        def fake_embed(texts, batch_size):
            batches.append(texts)
            return [[0.0] * EXPECTED_DIMENSIONS for _ in texts]

        with redirect_stdout(StringIO()):
            summary = run_benchmark(path, fake_embed)
        self.assertEqual(batches, [["send this"]])
        self.assertEqual(summary.successful_records, 1)
        self.assertEqual(summary.failed_records, 1)


if __name__ == "__main__":
    unittest.main()
