"""
Unit tests for enterprise RAG chunking:
- Monotonic sequential chunk IDs (doc_id#0000)
- Source-aware parsing for Slack, Gmail, Fireflies
- Speaker, timestamp, subject and message boundary preservation
- Clean content vs context-enriched search_text
- Token budget targeting (350-400 tokens)
- Safe oversized chunk splitting without truncation
- Backwards compatibility for Markdown, Confluence, etc.
"""

import unittest
from app.rag.chunker import (
    DocumentChunk,
    chunk_document_text,
    count_tokens,
    split_oversized_text,
    slugify,
)


class TestRAGChunker(unittest.TestCase):

    def test_sequential_monotonic_chunk_ids(self):
        """Chunk IDs must follow doc_id#0000, doc_id#0001 monotonically across sections."""
        markdown = """
# Enterprise Handbook

## Section 1: Introduction
This is the first section of the company handbook.

## Section 2: Security
Security policies must be adhered to at all times.

## Section 3: Benefits
All employees receive health insurance and PTO.
"""
        doc_id = "doc-uuid-12345"
        chunks = chunk_document_text(markdown, document_id=doc_id, title="Enterprise Handbook")
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].chunk_id, f"{doc_id}#0000")
        self.assertEqual(chunks[1].chunk_id, f"{doc_id}#0001")
        self.assertEqual(chunks[2].chunk_id, f"{doc_id}#0002")
        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.document_id, doc_id)
            self.assertTrue(chunk.chunk_id.startswith(doc_id))

    def test_slack_message_aware_chunking(self):
        """Slack threads must preserve speakers, timestamps and group turns into ~350-400 tokens."""
        slack_thread = """
[2024-02-10 09:15:00] @alice: Morning team, has anyone reviewed the Q1 infra roadmap?
[2024-02-10 09:16:30] @bob: Looking at it now. The Kafka partition rebalance seems risky.
[2024-02-10 09:18:10] @charlie: Agreed with Bob. We should schedule a separate maintenance window.
[2024-02-10 09:20:00] @alice: Good point, let's target Saturday at 02:00 UTC.
"""
        doc_id = "slack-doc-99"
        chunks = chunk_document_text(
            slack_thread,
            document_id=doc_id,
            title="Q1 Infra Slack Thread",
            source_type="slack",
            max_tokens=380,
        )
        self.assertGreater(len(chunks), 0)
        # Verify first chunk contains speakers
        first_chunk = chunks[0]
        self.assertIn("@alice:", first_chunk.content)
        self.assertIn("@bob:", first_chunk.content)
        # Content must be clean
        self.assertFalse(first_chunk.content.startswith("Title:"))
        # search_text must have title and breadcrumbs
        self.assertTrue(first_chunk.search_text.startswith("Title: Q1 Infra Slack Thread"))
        self.assertIn(first_chunk.content, first_chunk.search_text)
        self.assertEqual(first_chunk.chunk_id, f"{doc_id}#0000")

    def test_gmail_subject_preservation_and_email_chunking(self):
        """Gmail threads must extract and preserve email Subject in heading path and context."""
        email_thread = """
From: hr@company.com
To: all@company.com
Date: Mon, 15 Jan 2024 08:00:00 -0800
Subject: Annual Benefits Enrollment 2024

Please complete your benefits selections by January 31st through the HR portal.

--------------------------------------------------

From: employee@company.com
To: hr@company.com
Date: Mon, 15 Jan 2024 09:30:00 -0800
Subject: Re: Annual Benefits Enrollment 2024

Does the vision plan include laser eye surgery discounts?

--------------------------------------------------

From: hr@company.com
To: employee@company.com
Date: Mon, 15 Jan 2024 10:15:00 -0800
Subject: Re: Annual Benefits Enrollment 2024

Yes, there is a 15% discount with in-network providers.
"""
        doc_id = "gmail-doc-42"
        chunks = chunk_document_text(
            email_thread,
            document_id=doc_id,
            title="HR Benefits Email",
            source_type="gmail",
            max_tokens=380,
        )
        self.assertGreater(len(chunks), 0)
        # Check that Subject is preserved in heading_path
        for c in chunks:
            self.assertTrue(any("Annual Benefits Enrollment 2024" in h for h in c.heading_path))
            self.assertIn("Subject: Annual Benefits Enrollment 2024", c.search_text)
            self.assertTrue(c.chunk_id.startswith(f"{doc_id}#"))

    def test_fireflies_transcript_turn_preservation(self):
        """Fireflies meetings must preserve speaker tags and pack turns into chunks."""
        transcript = """
Speaker 1 (00:00): Welcome everyone to the quarterly product roadmap review.
Speaker 2 (00:35): Thanks John. We achieved 94% of our OKR targets for the ingestion pipeline.
Alice Cooper (01:10): That's great news. What about the cross-encoder latency optimization?
Speaker 2 (01:45): The reranker latency dropped from 240ms to 45ms with batching.
"""
        doc_id = "fireflies-meeting-7"
        chunks = chunk_document_text(
            transcript,
            document_id=doc_id,
            title="Q1 Product Roadmap Review",
            source_type="fireflies",
            max_tokens=380,
        )
        self.assertGreater(len(chunks), 0)
        first_chunk = chunks[0]
        self.assertIn("Speaker 1", first_chunk.content)
        self.assertIn("Speaker 2", first_chunk.content)
        self.assertIn("Alice Cooper", first_chunk.content)
        self.assertTrue(any("Transcript" in h for h in first_chunk.heading_path))

    def test_oversized_text_split_without_truncation(self):
        """Oversized continuous text must be safely split at line boundaries without losing words."""
        # Create a single paragraph with 500 words
        lines = [f"This is line number {i} detailing critical architectural specification." for i in range(100)]
        long_text = "\n".join(lines)
        total_original_tokens = count_tokens(long_text)

        doc_id = "doc-oversized"
        chunks = chunk_document_text(
            long_text,
            document_id=doc_id,
            title="Long Specification",
            source_type="generic",
            max_tokens=200,
        )
        self.assertGreater(len(chunks), 1)
        # Verify no truncation: all lines are accounted for across chunks
        reconstructed = "\n".join(c.content for c in chunks)
        reconstructed_tokens = count_tokens(reconstructed)
        # The sum of tokens across chunks must equal or exceed original (accounting for join spacing)
        self.assertGreaterEqual(reconstructed_tokens, total_original_tokens - 10)
        for i in range(100):
            target_phrase = f"line number {i}"
            self.assertTrue(
                any(target_phrase in c.content for c in chunks),
                f"Missing '{target_phrase}' in chunks!",
            )

    def test_markdown_confluence_backwards_compatibility(self):
        """Structured markdown/Confluence documents retain section hierarchy and code blocks."""
        markdown = """
# Deployment Guide

## Prerequisites
Ensure docker and python 3.12 are installed.
```bash
docker --version
python3 --version
```

## Running Migrations
Execute the migrations script.
```sql
SELECT * FROM document_chunks;
```
"""
        doc_id = "doc-compat"
        chunks = chunk_document_text(
            markdown,
            document_id=doc_id,
            title="Deployment Guide",
            source_type="confluence",
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].heading_path, ["Prerequisites"])
        self.assertIn("```bash", chunks[0].content)
        self.assertEqual(chunks[1].heading_path, ["Running Migrations"])
        self.assertIn("```sql", chunks[1].content)
        self.assertEqual(chunks[0].chunk_id, f"{doc_id}#0000")
        self.assertEqual(chunks[1].chunk_id, f"{doc_id}#0001")


if __name__ == "__main__":
    unittest.main()
