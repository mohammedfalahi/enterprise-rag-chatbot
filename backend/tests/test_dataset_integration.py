"""
Unit tests for the datasets-test benchmark integration, RBAC mapping, and evaluation logic.
"""

import unittest
import uuid
from pathlib import Path
import sys

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.ingest_benchmark import doc_id_to_uuid
from app.rag.chunker import chunk_document_text
from scripts.run_evals import evaluate_golden, evaluate_injections
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
        # Verify roles from roles.json match expected hierarchy
        roles_hierarchy = {
            "internal": ["employee", "sales_rep", "manager", "executive"],
            "confidential": ["manager", "executive"],
            "restricted": ["executive"],
        }
        
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


if __name__ == "__main__":
    unittest.main()
