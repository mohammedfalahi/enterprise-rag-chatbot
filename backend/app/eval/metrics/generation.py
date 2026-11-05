"""
Generation quality metrics evaluating Faithfulness, Answer Correctness,
Answer Relevancy, Context Precision, and Context Recall via Ragas and Gemini.
"""

from __future__ import annotations

import os
from typing import Any

from app.core.config import settings
from app.rag.embeddings import embed_texts


def evaluate_ragas_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Evaluates saved RAG outputs against reference answers using Ragas LLM-as-a-judge.
    Expects records with:
      - 'user_input': query text
      - 'response': model generated answer
      - 'retrieved_contexts': list of retrieved chunk text
      - 'reference': verified ground-truth reference text
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
    per_question_scores: list[dict[str, Any]] = []

    for i, row in enumerate(result.scores):
        row_scores = {}
        for name, value in row.items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(name, []).append(float(value))
                row_scores[name] = round(float(value), 4)
        per_question_scores.append(
            {
                "user_input": records[i]["user_input"],
                "response": records[i]["response"],
                "reference": records[i]["reference"],
                "scores": row_scores,
            }
        )

    aggregate = {
        name: round(sum(values) / len(values), 4)
        for name, values in metric_values.items()
        if values
    }

    return {
        "record_count": len(records),
        "aggregate_metrics": aggregate,
        "per_question_scores": per_question_scores,
    }
