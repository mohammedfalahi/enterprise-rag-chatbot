"""Run Ragas answer/context metrics against a saved JSONL evaluation snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.eval.metrics.generation import evaluate_ragas_records

REQUIRED_FIELDS = ("user_input", "response", "retrieved_contexts", "reference")
TRACEABILITY_FIELDS = (
    "test_id",
    "question_id",
    "role",
    "category",
    "generation_provider",
    "generation_model",
)


def load_evaluation_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Ragas evaluation snapshot not found: {source}")

    records = []
    with source.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(
                    f"{source}:{line_number}: missing required fields: {', '.join(missing)}"
                )
            if not isinstance(row["user_input"], str) or not row["user_input"].strip():
                raise ValueError(f"{source}:{line_number}: user_input must be non-empty text")
            if not isinstance(row["response"], str) or not row["response"].strip():
                raise ValueError(f"{source}:{line_number}: response must be non-empty text")
            if not isinstance(row["reference"], str) or not row["reference"].strip():
                raise ValueError(f"{source}:{line_number}: reference must be non-empty text")
            contexts = row["retrieved_contexts"]
            if (
                not isinstance(contexts, list)
                or not contexts
                or any(not isinstance(context, str) or not context.strip() for context in contexts)
            ):
                raise ValueError(
                    f"{source}:{line_number}: retrieved_contexts must be a non-empty list of text"
                )
            rec = {field: row[field] for field in REQUIRED_FIELDS}
            for field in TRACEABILITY_FIELDS:
                if field in row and row[field] is not None:
                    rec[field] = row[field]
            records.append(rec)

    if not records:
        raise ValueError(f"{source}: no evaluation records found")
    return records


def run_ragas(records: list[dict[str, Any]]) -> dict[str, Any]:
    return evaluate_ragas_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved RAG outputs with Ragas; does not call chat or database APIs."
    )
    parser.add_argument("--input", type=Path, required=True, help="Saved evaluation JSONL")
    parser.add_argument("--output", type=Path, help="Optional JSON file for scores report")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on records to evaluate")
    args = parser.parse_args()

    records = load_evaluation_records(args.input)
    if args.limit is not None and args.limit > 0:
        records = records[:args.limit]

    report = run_ragas(records)
    output_str = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(output_str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_str + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
