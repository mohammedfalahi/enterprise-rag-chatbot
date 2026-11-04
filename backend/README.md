# Backend observability and evaluation

## Langfuse

Langfuse is the backend's only observability platform. Set `LANGFUSE_PUBLIC_KEY`
and `LANGFUSE_SECRET_KEY` to enable it; `LANGFUSE_BASE_URL` defaults to
`https://cloud.langfuse.com`. Without both keys, tracing is a no-op.

Request traces cover query rewriting, embedding (including cache hits), dense
retrieval, PostgreSQL full-text retrieval, RRF, reranking, context construction,
Gemini generation and SSE completion. By default, traces include summaries,
chunk IDs and retrieval scores, not prompt or answer text. Set
`LANGFUSE_CAPTURE_CONTENT=true` to include redacted original/rewritten queries,
generation prompt parts, retrieved context and generated answers. Credential,
token and private-key patterns are scrubbed before capture.

## Offline answer evaluation with Ragas

The evaluator reads saved JSONL records only; it does not call the chat API or
database. Ragas uses Gemini as a separate evaluation model, so running this batch
does make provider API calls using `GEMINI_API_KEY`. Supply reviewed references;
the evaluator rejects missing/empty references rather than inventing answers.
The supplied questions, answers, retrieved contexts and references are sent to
Gemini for scoring; use only data approved for that external processing and do
not include production conversation records.

Install the optional evaluation dependencies and run from `backend/`:

```sh
uv sync --extra evaluation
uv run --extra evaluation python scripts/evaluate_ragas.py \
  --input /path/to/saved-evaluation.jsonl \
  --output /path/to/ragas-report.json
```

Each non-empty JSONL record must have this shape:

```json
{
  "user_input": "Question text",
  "response": "Previously saved assistant response",
  "retrieved_contexts": ["Previously saved retrieved passage"],
  "reference": "Reviewed expected answer"
}
```

The runner reports aggregate Faithfulness, Answer Relevancy, Context Precision
and Context Recall scores from those records. It does not create evaluation
records, call production retrieval, or claim results for the benchmark datasets,
which currently do not include saved assistant responses or complete references.

## Custom retrieval/security evaluation

`scripts/run_evals.py` remains a separate benchmark command. It executes
retrieval against the database configured by `DATABASE_URL`; use an isolated
benchmark/staging database populated from the external corpus, never production.
For answer/partial golden cases with known document labels, it computes
document-level Precision@K, Recall@K, MRR and nDCG@K, deduplicating repeated
chunks from the same document. Injection records report retrieval refusal and
permission leakage separately; bounded retrieval is not counted as a refusal.

The external corpus and labels are not bundled in this repository. Supply paths
with `--golden-path` and `--injection-path` as needed. Evaluation and live chat
requests do not share request-time evaluation logic.