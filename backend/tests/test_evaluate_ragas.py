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


if __name__ == "__main__":
    unittest.main()
