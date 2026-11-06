"""
Generation quality metrics evaluating Faithfulness, Answer Correctness,
Answer Relevancy, Context Precision, and Context Recall via Ragas and Gemini.
"""

from __future__ import annotations

import math
import os
from typing import Any

from app.core.config import settings
from app.rag.embeddings import embed_texts


def sanitize_score(val: Any) -> float | None:
    """Convert a metric score to a rounded float if finite, else None (JSON null)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        if math.isfinite(f):
            return round(f, 4)
    return None


def sanitize_for_strict_json(obj: Any) -> Any:
    """Recursively sanitize data structures replacing NaN and Inf with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_for_strict_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_strict_json(v) for v in obj]
    return obj


def evaluate_ragas_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Evaluates saved RAG outputs against reference answers using Ragas LLM-as-a-judge.
    Expects records with:
      - 'user_input': query text
      - 'response': model generated answer
      - 'retrieved_contexts': list of retrieved chunk text
      - 'reference': verified ground-truth reference text
    Preserves traceability metadata ('test_id', 'question_id', 'role', 'category').
    Excludes NaN/Infinity from aggregate metrics and reports n_scored / n_failed counts.
    """
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY to run the offline Ragas evaluator.")

    try:
        from langchain_core.embeddings import Embeddings
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install evaluation dependencies with `uv sync --extra evaluation`."
        ) from exc

    class LocalOllamaEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embed_texts(texts)

        def embed_query(self, text: str) -> list[float]:
            return embed_texts([text])[0]

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=settings.chat_llm_model,
            google_api_key=api_key,
            temperature=0,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(LocalOllamaEmbeddings())

    dataset = EvaluationDataset.from_list(records)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            AnswerCorrectness(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    metric_values: dict[str, list[float]] = {}
    metric_scored_count: dict[str, int] = {}
    metric_failed_count: dict[str, int] = {}
    timeout_details: list[dict[str, Any]] = []
    per_question_scores: list[dict[str, Any]] = []

    for i, row in enumerate(result.scores):
        row_scores: dict[str, float | None] = {}
        for name, raw_value in row.items():
            clean_val = sanitize_score(raw_value)
            if clean_val is not None:
                metric_values.setdefault(name, []).append(clean_val)
                metric_scored_count[name] = metric_scored_count.get(name, 0) + 1
                row_scores[name] = clean_val
            else:
                metric_failed_count[name] = metric_failed_count.get(name, 0) + 1
                row_scores[name] = None
                if isinstance(raw_value, str) and "timeout" in raw_value.lower():
                    timeout_details.append({
                        "index": i,
                        "test_id": records[i].get("test_id"),
                        "metric": name,
                        "error": raw_value,
                    })

        q_score = {
            "user_input": records[i]["user_input"],
            "response": records[i]["response"],
            "reference": records[i]["reference"],
            "scores": row_scores,
        }
        for field in ("test_id", "question_id", "role", "category"):
            if field in records[i]:
                q_score[field] = records[i][field]

        per_question_scores.append(q_score)

    aggregate = {
        name: round(sum(values) / len(values), 4)
        for name, values in metric_values.items()
        if values
    }
    all_metrics = set(metric_scored_count) | set(metric_failed_count)
    metric_counts = {
        name: {
            "n_scored": metric_scored_count.get(name, 0),
            "n_failed": metric_failed_count.get(name, 0),
        }
        for name in all_metrics
    }

    report = {
        "record_count": len(records),
        "aggregate_metrics": aggregate,
        "metric_counts": metric_counts,
        "timeout_details": timeout_details,
        "per_question_scores": per_question_scores,
    }
    return sanitize_for_strict_json(report)
