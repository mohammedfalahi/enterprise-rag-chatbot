"""
Unit tests for the datasets-test benchmark integration, RBAC mapping, and evaluation logic.
"""

import unittest
import uuid
import json
from pathlib import Path
import sys

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.ingest_benchmark import doc_id_to_uuid
from app.rag.chunker import chunk_document_text
from scripts.run_evals import evaluate_golden, evaluate_injections, ranking_metrics_at_k
from app.rag.hybrid_search import RetrievalResult


class TestDatasetIntegration(unittest.TestCase):

    def test_doc_id_to_uuid_deterministic(self):
        doc_id = "dsid_9c8cb10d65324c03a8ededbd98c1b058"
        u1 = doc_id_to_uuid(doc_id)
        u2 = doc_id_to_uuid(doc_id)
        self.assertEqual(u1, u2)
        # Verify valid UUID format
        parsed = uuid.UUID(u1)
        self.assertEqual(str(parsed), u1)

    def test_chunking_corpus_document(self):
        sample_text = """
# Onboarding Runbook

## Section 1: Setup
Here is how to setup your environment.
```bash
export TOKEN=12345
```

## Section 2: Escalation
If anything goes wrong, contact the SRE team.
"""
        doc_uuid = doc_id_to_uuid("dsid_test_123")
        chunks = chunk_document_text(sample_text, document_id=doc_uuid, title="Onboarding Runbook")
        
        self.assertGreater(len(chunks), 0)
        # Check that heading paths are extracted
        heading_paths = [c.heading_path for c in chunks]
        self.assertTrue(any("Section 1: Setup" in p for p in heading_paths))
        self.assertTrue(any("Section 2: Escalation" in p for p in heading_paths))
        # Check that chunk_id starts with doc_uuid
        for c in chunks:
            self.assertTrue(c.chunk_id.startswith(doc_uuid))
            self.assertEqual(c.document_id, doc_uuid)

    def test_rbac_roles_mapping(self):
        roles_path = Path("/Users/falahi/datasets-test/data/roles.json")
        roles_hierarchy = json.loads(roles_path.read_text(encoding="utf-8"))
        
        # An employee should only see internal documents
        employee_roles = {"employee"}
        self.assertTrue(bool(employee_roles & set(roles_hierarchy["internal"])))
        self.assertFalse(bool(employee_roles & set(roles_hierarchy["confidential"])))
        self.assertFalse(bool(employee_roles & set(roles_hierarchy["restricted"])))

        # A manager should see internal and confidential
        manager_roles = {"manager"}
        self.assertTrue(bool(manager_roles & set(roles_hierarchy["internal"])))
        self.assertTrue(bool(manager_roles & set(roles_hierarchy["confidential"])))
        self.assertFalse(bool(manager_roles & set(roles_hierarchy["restricted"])))

        # An executive should see all
        exec_roles = {"executive"}
        self.assertTrue(bool(exec_roles & set(roles_hierarchy["internal"])))
        self.assertTrue(bool(exec_roles & set(roles_hierarchy["confidential"])))
        self.assertTrue(bool(exec_roles & set(roles_hierarchy["restricted"])))

    def test_document_ranking_metrics_are_deduplicated(self):
        metrics = ranking_metrics_at_k(
            ["irrelevant", "irrelevant", "gold-a", "gold-b"],
            {"gold-a", "gold-b"},
            k=3,
        )
        self.assertEqual(metrics["precision@3"], 2 / 3)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["mrr"], 1 / 2)
        self.assertAlmostEqual(metrics["ndcg@3"], 0.6934264)

    def test_document_ranking_metrics_reject_invalid_k(self):
        with self.assertRaises(ValueError):
            ranking_metrics_at_k(["doc"], {"doc"}, k=0)

    def test_evaluation_recall_metric(self):
        import asyncio
        gold_id = "dsid_5c05d2a74c484906852925d1cf1daa70"
        gold_uuid = doc_id_to_uuid(gold_id)

        # Mock retriever that returns the gold document
        async def mock_retriever_hit(query, tenant, roles, top_k=5):
            return [
                RetrievalResult(
                    chunk_id=f"{gold_uuid}::sec::0",
                    document_id=gold_uuid,
                    title="Mock Doc",
                    heading_path=["Sec"],
                    content="Content",
                    dense_rank=1,
                    fts_rank=1,
                    rrf_score=0.03,
                    rerank_score=2.5,
                )
            ]

        results = asyncio.run(
            evaluate_golden(
                golden_path="/Users/falahi/datasets-test/data/golden.jsonl",
                tenant_id="test_tenant",
                top_k=5,
                limit=5,
                mock_retriever=mock_retriever_hit,
            )
        )
        self.assertEqual(results["permission_leakage_count"], 0)
        self.assertGreater(results["mean_recall_at_k"], 0.0)
        self.assertIn("precision@5", results["retrieval_metrics"])
        self.assertIn("mrr", results["retrieval_metrics"])
        self.assertIn("ndcg@5", results["retrieval_metrics"])

    def test_injection_defense_eval(self):
        import asyncio
        # Mock retriever that refuses / returns no results for injections
        async def mock_refusing_retriever(query, tenant, roles, top_k=5):
            return []

        results = asyncio.run(
            evaluate_injections(
                injection_path="/Users/falahi/datasets-test/data/injection.jsonl",
                tenant_id="test_tenant",
                top_k=5,
                mock_retriever=mock_refusing_retriever,
            )
        )
        self.assertEqual(results["refusals"], 5)
        self.assertEqual(results["refusal_rate"], 1.0)

    def test_bounded_injection_retrieval_is_not_counted_as_refusal(self):
        import asyncio

        async def mock_bounded_retriever(query, tenant, roles, top_k=5):
            return [
                RetrievalResult(
                    chunk_id="chunk-01",
                    document_id="doc-01",
                    title="Mock Doc",
                    heading_path=[],
                    content="Accessible content",
                    dense_rank=1,
                    fts_rank=1,
                    rrf_score=0.03,
                    rerank_score=2.5,
                )
            ]

        results = asyncio.run(
            evaluate_injections(
                injection_path="/Users/falahi/datasets-test/data/injection.jsonl",
                tenant_id="test_tenant",
                top_k=5,
                mock_retriever=mock_bounded_retriever,
            )
        )
        self.assertEqual(results["refusals"], 0)
        self.assertEqual(results["refusal_rate"], 0.0)
        self.assertEqual(results["bounded_unverified_count"], 5)

    def test_empty_evaluation_sets_report_undefined_metrics(self):
        import asyncio
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as golden_file:
            golden_results = asyncio.run(
                evaluate_golden(
                    golden_path=golden_file.name,
                    tenant_id="test_tenant",
                    mock_retriever=None,
                )
            )
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as injection_file:
            injection_results = asyncio.run(
                evaluate_injections(
                    injection_path=injection_file.name,
                    tenant_id="test_tenant",
                    mock_retriever=None,
                )
            )
        self.assertIsNone(golden_results["mean_recall_at_k"])
        self.assertIsNone(golden_results["permission_leakage_rate"])
        self.assertIsNone(injection_results["refusal_rate"])

    def test_chunk_level_recall_and_evidence_hit_metrics(self):
        from app.eval.metrics.retrieval import (
            compute_chunk_metrics_at_k,
            compute_evidence_hit_at_k,
        )

        # 1. Chunk metrics with gold chunk IDs
        retrieved_chunks = [
            "doc-1#0000",  # Irrelevant chunk
            "doc-1#0005",  # Irrelevant chunk
            "doc-1#0003",  # Evidence chunk
        ]
        gold_chunks = {"doc-1#0003"}
        chunk_m = compute_chunk_metrics_at_k(retrieved_chunks, gold_chunks, k=3)
        self.assertEqual(chunk_m["chunk_recall@3"], 1.0)
        self.assertEqual(chunk_m["chunk_hit@3"], 1.0)
        self.assertEqual(chunk_m["chunk_mrr"], 1 / 3)

        # When evidence chunk #0003 is missed (only #0000 and #0005 retrieved)
        missed_m = compute_chunk_metrics_at_k(
            ["doc-1#0000", "doc-1#0005"], gold_chunks, k=2
        )
        self.assertEqual(missed_m["chunk_recall@2"], 0.0)
        self.assertEqual(missed_m["chunk_hit@2"], 0.0)

        # 2. Evidence hit calculation from answer_facts
        facts = [
            "An escalation email should be sent if not shipping a measurable artifact by day 30",
            "An escalation email should be sent if access blockers persist for more than 48 hours",
            "An escalation email should be sent if there are repeated missed meetings",
        ]
        # Chunk #0003 content
        evidence_content = [
            "Escalation triggers (email to p-ops + hiring manager): not shipping a measurable artifact by day 30, ongoing access blockers >48h, repeated missing meetings."
        ]
        hit_result = compute_evidence_hit_at_k(evidence_content, facts)
        self.assertEqual(hit_result["evidence_hit"], 1.0)
        self.assertGreaterEqual(hit_result["facts_covered"], 0.6)

        # Overview chunk #0000 content without escalation triggers
        irrelevant_content = [
            "Purpose: quick, reusable working template to take a candidate through offer acceptance to contribution in 30/60/90 days."
        ]
        miss_result = compute_evidence_hit_at_k(irrelevant_content, facts)
        self.assertEqual(miss_result["evidence_hit"], 0.0)
        self.assertEqual(miss_result["facts_covered"], 0.0)


if __name__ == "__main__":
    unittest.main()
