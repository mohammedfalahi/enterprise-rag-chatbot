"""
Structural chunking, code-fence protection, subdivision and contextual breadcrumb logic.
Adapted from DEV_AI (packages/knowledge/chunker.py) for enterprise documents.
"""

from __future__ import annotations

import re
from typing import Any
import yaml
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str = Field(
        description="Unique deterministic chunk ID: {document_id}::{section_slug}::{idx}"
    )
    document_id: str = Field(description="Identifier of parent document")
    title: str = Field(description="Document title")
    heading_path: list[str] = Field(
        description="Breadcrumb heading path (e.g. ['Section 1', 'Sub-heading'])"
    )
    content: str = Field(description="Clean content for LLM synthesis")
    search_text: str = Field(
        description="Context-enriched text for vector and BM25 search"
    )
    token_count: int = Field(
        description="Estimated token count using regex tokenization"
    )


def slugify(text: str) -> str:
    """Generate a clean URL/identifier-friendly slug from a section title."""
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[-\s]+", "-", cleaned)
    return slug or "section"


def count_tokens(text: str) -> int:
    """Fast regex-based token counter (matches words or individual punctuation symbols).

    Exact DEV_AI implementation.
    """
    return len(re.findall(r"\w+|[^\w\s]", text))


def split_into_blocks(text: str) -> list[str]:
    """Split section content into atomic markdown blocks (paragraphs and code fences).

    Code fences (``` ... ```) are treated as single indivisible atomic blocks.
    Exact DEV_AI implementation.
    """
    lines = text.splitlines(keepends=True)
    blocks: list[str] = []
    current_block: list[str] = []
    in_code_fence = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            current_block.append(line)
            if not in_code_fence:
                # Closed code fence: end current block
                blocks.append("".join(current_block).strip())
                current_block = []
            continue

        if in_code_fence:
            current_block.append(line)
            continue

        # Outside code fence: blank line denotes paragraph boundary
        if not stripped:
            if current_block:
                blocks.append("".join(current_block).strip())
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        blocks.append("".join(current_block).strip())

    return [b for b in blocks if b]


def subdivide_section(content: str, max_words: int = 600) -> list[str]:
    """Subdivide a section into sub-chunks if total words exceed max_words.

    Maintains code block integrity by operating over atomic blocks.
    Exact DEV_AI implementation.
    """
    blocks = split_into_blocks(content)
    if not blocks:
        return []

    chunks: list[str] = []
    current_chunk_blocks: list[str] = []
    current_word_count = 0

    for block in blocks:
        block_words = len(block.split())
        if current_chunk_blocks and (current_word_count + block_words > max_words):
            chunks.append("\n\n".join(current_chunk_blocks))
            current_chunk_blocks = [block]
            current_word_count = block_words
        else:
            current_chunk_blocks.append(block)
            current_word_count += block_words

    if current_chunk_blocks:
        chunks.append("\n\n".join(current_chunk_blocks))

    return chunks


def parse_markdown_sections(
    body: str, default_title: str = "Document"
) -> list[tuple[list[str], str]]:
    """Parse markdown body into sections based on ## and ### headers.

    Code fences are strictly protected so comments or code lines starting with
    ## or ### are never treated as markdown headers.
    Exact DEV_AI implementation.
    """
    lines = body.splitlines()
    sections: list[tuple[list[str], str]] = []

    current_h2: str | None = None
    current_h3: str | None = None
    current_lines: list[str] = []
    in_code_fence = False

    def finalize_current():
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            if current_h2 and current_h3:
                heading_path = [current_h2, current_h3]
            elif current_h2:
                heading_path = [current_h2]
            elif current_h3:
                heading_path = [current_h3]
            else:
                heading_path = ["Overview"]
            sections.append((heading_path, content))
        current_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            current_lines.append(line)
            continue

        if not in_code_fence:
            # Check for H1 (document title) - ignore header line itself if seen at start
            if stripped.startswith("# ") and not stripped.startswith("## "):
                continue

            # Check for H2
            if stripped.startswith("## ") and not stripped.startswith("### "):
                finalize_current()
                current_h2 = stripped[3:].strip()
                current_h3 = None
                continue

            # Check for H3
            if stripped.startswith("### "):
                finalize_current()
                current_h3 = stripped[4:].strip()
                continue

        current_lines.append(line)

    finalize_current()
    return sections


def create_chunks_from_sections(
    document_id: str,
    title: str,
    sections: list[tuple[list[str], str]],
    max_words: int = 600,
) -> list[DocumentChunk]:
    """Convert parsed sections with heading paths into structured DocumentChunks.

    Preserves DEV_AI's breadcrumbs and search_text generation.
    """
    chunks: list[DocumentChunk] = []

    for heading_path, section_content in sections:
        sub_contents = subdivide_section(section_content, max_words=max_words)
        section_slug = slugify(heading_path[-1] if heading_path else "section")

        for idx, content in enumerate(sub_contents):
            clean_content = content.strip()
            if not clean_content:
                continue

            chunk_id = f"{document_id}::{section_slug}::{idx}"
            breadcrumbs = " > ".join(heading_path) if heading_path else title
            search_text = f"Title: {title} | Path: {breadcrumbs}\n\n{clean_content}"
            tokens = count_tokens(clean_content)

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    title=title,
                    heading_path=heading_path,
                    content=clean_content,
                    search_text=search_text,
                    token_count=tokens,
                )
            )

    return chunks


def chunk_document_text(
    text: str,
    document_id: str,
    title: str = "Document",
) -> list[DocumentChunk]:
    """Chunk generic raw/markdown text using DEV_AI's structural section parser and subdivision."""
    sections = parse_markdown_sections(text, default_title=title)
    if not sections:
        sections = [(["Overview"], text.strip())]
    return create_chunks_from_sections(document_id, title, sections)
