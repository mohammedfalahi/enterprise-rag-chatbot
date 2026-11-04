"""
Enterprise Document Parsers (PDF, DOCX, TXT/Markdown).
Integrates with DEV_AI's structural block & breadcrumb chunking pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import pymupdf
from docx import Document

from app.rag.chunker import (
    DocumentChunk,
    create_chunks_from_sections,
    parse_markdown_sections,
)


def parse_pdf_to_sections(file_path: str | Path, title: str) -> list[tuple[list[str], str]]:
    """
    Parse PDF into sections using font size heuristics for headings.
    Preserves heading hierarchy for breadcrumbs.
    """
    doc = pymupdf.open(file_path)
    sections: list[tuple[list[str], str]] = []

    # First pass: find median or body font size across the first few pages
    font_sizes = []
    for page_idx in range(min(5, len(doc))):
        page = doc[page_idx]
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            if b.get("type") == 0:
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        if s.get("text", "").strip():
                            font_sizes.append(round(s.get("size", 10), 1))

    base_size = max(set(font_sizes), key=font_sizes.count) if font_sizes else 10.0

    current_heading_path: list[str] = []
    current_lines: list[str] = []

    def finalize_section():
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            path = list(current_heading_path) if current_heading_path else ["Overview"]
            sections.append((path, content))
        current_lines = []

    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            if b.get("type") == 0:
                for l in b.get("lines", []):
                    line_text = "".join(s.get("text", "") for s in l.get("spans", [])).strip()
                    if not line_text:
                        continue
                    
                    max_span_size = max(s.get("size", base_size) for s in l.get("spans", []))
                    
                    # If font size is noticeably larger than base, treat as header
                    if max_span_size >= base_size * 1.3:
                        finalize_section()
                        if max_span_size >= base_size * 1.6:
                            current_heading_path = [line_text]
                        else:
                            current_heading_path = current_heading_path[:1] + [line_text]
                    else:
                        current_lines.append(line_text)

    finalize_section()
    doc.close()
    return sections


def parse_docx_to_sections(file_path: str | Path, title: str) -> list[tuple[list[str], str]]:
    """
    Parse DOCX into sections based on built-in paragraph styles (Heading 1, 2, etc.).
    Preserves structural breadcrumbs.
    """
    doc = Document(file_path)
    sections: list[tuple[list[str], str]] = []

    current_h1: str | None = None
    current_h2: str | None = None
    current_lines: list[str] = []

    def finalize_section():
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            path = []
            if current_h1:
                path.append(current_h1)
            if current_h2:
                path.append(current_h2)
            if not path:
                path = ["Overview"]
            sections.append((path, content))
        current_lines = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style_name = para.style.name if para.style else ""
        if style_name == "Heading 1":
            finalize_section()
            current_h1 = text
            current_h2 = None
        elif style_name in ("Heading 2", "Heading 3"):
            finalize_section()
            current_h2 = text
        else:
            current_lines.append(text)

    finalize_section()
    return sections


def parse_document(
    file_path: str | Path,
    document_id: str,
    title: str,
) -> list[DocumentChunk]:
    """Parse document according to its file extension using DEV_AI's structural chunking."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        sections = parse_pdf_to_sections(path, title)
    elif ext in (".docx", ".doc"):
        sections = parse_docx_to_sections(path, title)
    else:
        text = path.read_text(encoding="utf-8")
        sections = parse_markdown_sections(text, default_title=title)

    if not sections:
        content = path.read_text(encoding="utf-8", errors="ignore")
        sections = [(["Overview"], content.strip())]

    return create_chunks_from_sections(document_id, title, sections)
