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

REQUIRED_FIELDS = ("user_input", "response", "retrieved_contexts", "reference")


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
            records.append({field: row[field] for field in REQUIRED_FIELDS})

    if not records:
        raise ValueError(f"{source}: no evaluation records found")
    return records


def run_ragas(records: list[dict[str, Any]]) -> dict[str, float]:
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY to run the offline Ragas evaluator.")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install the evaluation extra with `uv sync --extra evaluation`."
        ) from exc

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=settings.chat_llm_model,
            google_api_key=api_key,
            temperature=0,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
    )
    dataset = EvaluationDataset.from_list(records)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    metric_values: dict[str, list[float]] = {}
    for row in result.scores:
        for name, value in row.items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(name, []).append(float(value))
    return {
        name: sum(values) / len(values)
        for name, values in metric_values.items()
        if values
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved RAG outputs with Ragas; does not call chat or database APIs."
    )
    parser.add_argument("--input", type=Path, required=True, help="Saved evaluation JSONL")
    parser.add_argument("--output", type=Path, help="Optional JSON file for aggregate scores")
    args = parser.parse_args()

    records = load_evaluation_records(args.input)
    scores = run_ragas(records)
    report = {"record_count": len(records), "metrics": scores}
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
