"""
Structural and source-aware chunking for enterprise documents and conversations.
Supports:
- Monotonic sequential chunk IDs (doc_id#0000)
- Source-aware parsing for Slack, Gmail, and Fireflies (~78% of corpus)
- Clean content vs context-enriched search_text
- Token budget targeting (350-400 tokens)
- Safe oversized chunk splitting without truncation
- Code-fence protection and breadcrumb hierarchy
"""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field

DEFAULT_MAX_TOKENS: int = 380


class DocumentChunk(BaseModel):
    chunk_id: str = Field(
        description="Unique deterministic chunk ID: {document_id}#{idx:04d}"
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
    source: str = Field(
        default="generic", description="Origin source system (slack, gmail, etc.)"
    )


def slugify(text: str) -> str:
    """Generate a clean URL/identifier-friendly slug from a section title."""
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[-\s]+", "-", cleaned)
    return slug or "section"


def count_tokens(text: str) -> int:
    """Fast regex-based token counter (matches words or individual punctuation symbols)."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def split_oversized_text(text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[str]:
    """
    Safely split text exceeding max_tokens at line, sentence, or word boundaries.
    Guarantees zero truncation of content.
    """
    text = text.strip()
    if not text:
        return []

    if count_tokens(text) <= max_tokens:
        return [text]

    # Level 1: Split on double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for p in paragraphs:
            p_tokens = count_tokens(p)
            if p_tokens > max_tokens:
                # Recursively split the oversized paragraph
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                chunks.extend(split_oversized_text(p, max_tokens=max_tokens))
            elif current and (current_tokens + p_tokens > max_tokens):
                chunks.append("\n\n".join(current))
                current = [p]
                current_tokens = p_tokens
            else:
                current.append(p)
                current_tokens += p_tokens

        if current:
            chunks.append("\n\n".join(current))
        return chunks

    # Level 2: Split on single newlines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        chunks = []
        current = []
        current_tokens = 0

        for line in lines:
            line_tokens = count_tokens(line)
            if line_tokens > max_tokens:
                if current:
                    chunks.append("\n".join(current))
                    current = []
                    current_tokens = 0
                chunks.extend(split_oversized_text(line, max_tokens=max_tokens))
            elif current and (current_tokens + line_tokens > max_tokens):
                chunks.append("\n".join(current))
                current = [line]
                current_tokens = line_tokens
            else:
                current.append(line)
                current_tokens += line_tokens

        if current:
            chunks.append("\n".join(current))
        return chunks

    # Level 3: Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        chunks = []
        current = []
        current_tokens = 0

        for s in sentences:
            s_tokens = count_tokens(s)
            if s_tokens > max_tokens:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_tokens = 0
                chunks.extend(split_oversized_text(s, max_tokens=max_tokens))
            elif current and (current_tokens + s_tokens > max_tokens):
                chunks.append(" ".join(current))
                current = [s]
                current_tokens = s_tokens
            else:
                current.append(s)
                current_tokens += s_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks

    # Level 4: Split on words as a final fallback
    words = text.split()
    chunks = []
    current_words: list[str] = []
    current_tokens = 0

    for w in words:
        w_tokens = count_tokens(w)
        if current_words and (current_tokens + w_tokens > max_tokens):
            chunks.append(" ".join(current_words))
            current_words = [w]
            current_tokens = w_tokens
        else:
            current_words.append(w)
            current_tokens += w_tokens

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def split_into_blocks(text: str) -> list[str]:
    """
    Split section content into atomic markdown blocks (paragraphs and code fences).
    Code fences (``` ... ```) are treated as single indivisible atomic blocks.
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
                blocks.append("".join(current_block).strip())
                current_block = []
            continue

        if in_code_fence:
            current_block.append(line)
            continue

        if not stripped:
            if current_block:
                blocks.append("".join(current_block).strip())
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        blocks.append("".join(current_block).strip())

    return [b for b in blocks if b]


def subdivide_section(content: str, max_words: int | None = None, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[str]:
    """
    Subdivide a section into sub-chunks maintaining code block integrity.
    If a block exceeds max_tokens, it is safely split without truncating code or content.
    """
    token_limit = max_tokens if max_words is None else int(max_words * 0.75)
    blocks = split_into_blocks(content)
    if not blocks:
        return []

    chunks: list[str] = []
    current_chunk_blocks: list[str] = []
    current_token_count = 0

    for raw_block in blocks:
        block_tokens = count_tokens(raw_block)

        # Handle atomic blocks that themselves exceed the budget
        sub_blocks = [raw_block] if block_tokens <= token_limit else split_oversized_text(raw_block, max_tokens=token_limit)

        for block in sub_blocks:
            b_tokens = count_tokens(block)
            if current_chunk_blocks and (current_token_count + b_tokens > token_limit):
                chunks.append("\n\n".join(current_chunk_blocks))
                current_chunk_blocks = [block]
                current_token_count = b_tokens
            else:
                current_chunk_blocks.append(block)
                current_token_count += b_tokens

    if current_chunk_blocks:
        chunks.append("\n\n".join(current_chunk_blocks))

    return chunks


def parse_markdown_sections(
    body: str, default_title: str = "Document"
) -> list[tuple[list[str], str]]:
    """Parse markdown body into sections based on ## and ### headers while protecting code fences."""
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
            if stripped.startswith("# ") and not stripped.startswith("## "):
                continue

            if stripped.startswith("## ") and not stripped.startswith("### "):
                finalize_current()
                current_h2 = stripped[3:].strip()
                current_h3 = None
                continue

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
    max_words: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    source_type: str = "generic",
) -> list[DocumentChunk]:
    """Convert parsed sections with heading paths into structured DocumentChunks with monotonic sequential IDs."""
    chunks: list[DocumentChunk] = []
    global_idx = 0

    for heading_path, section_content in sections:
        sub_contents = subdivide_section(section_content, max_words=max_words, max_tokens=max_tokens)

        for content in sub_contents:
            clean_content = content.strip()
            if not clean_content:
                continue

            chunk_id = f"{document_id}#{global_idx:04d}"
            global_idx += 1

            breadcrumbs = " > ".join(heading_path) if heading_path else title
            search_text = f"Title: {title} | Path: {breadcrumbs}\n\n{clean_content}"
            tokens = count_tokens(search_text)

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    title=title,
                    heading_path=heading_path,
                    content=clean_content,
                    search_text=search_text,
                    token_count=tokens,
                    source=source_type,
                )
            )

    return chunks


# ==============================================================================
# Source-Aware Message Chunking: Slack, Gmail, Fireflies
# ==============================================================================

# Slack turn starts with timestamp and/or username: e.g. [2024-01-15 10:00:00] @user: or user:
_SLACK_TURN_REGEX = re.compile(
    r"^(?:\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?\]?\s*<@?[\w\.-]+>:|"
    r"\[?\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?\]?\s*<@?[\w\.-]+>:|"
    r"<@?[\w\.-]+>:|"
    r"@?[\w\.-]+(?:\s+[\w\.-]+)?:\s+)",
    re.MULTILINE,
)

# Gmail message header starts
_GMAIL_HEADER_REGEX = re.compile(
    r"^(?:From:\s+|Date:\s+|Subject:\s+|On\s+.+wrote:|--{3,}|=={3,})",
    re.MULTILINE | re.IGNORECASE,
)

# Fireflies speaker turn: Speaker 1 (00:00): or [00:15] Speaker: or Name (01:23):
_FIREFLIES_TURN_REGEX = re.compile(
    r"^(?:\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*(?:Speaker\s+\d+|[\w\s\.-]+):|"
    r"(?:Speaker\s+\d+|[\w\s\.-]+)\s*\(\d{1,2}:\d{2}(?::\d{2})?\):)",
    re.MULTILINE,
)


def _split_into_turns(text: str, turn_regex: re.Pattern[str]) -> list[str]:
    """Split text into message/speaker turns using a boundary regex while preserving content."""
    lines = text.splitlines(keepends=True)
    turns: list[str] = []
    current_turn: list[str] = []

    for line in lines:
        if turn_regex.match(line):
            if current_turn:
                turn_str = "".join(current_turn).strip()
                if turn_str:
                    turns.append(turn_str)
                current_turn = []
        current_turn.append(line)

    if current_turn:
        turn_str = "".join(current_turn).strip()
        if turn_str:
            turns.append(turn_str)

    # Fallback to paragraph splitting if turn_regex did not match any turn boundaries
    if not turns or len(turns) <= 1:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paras) > 1:
            return paras

    return turns if turns else [text.strip()]


def chunk_slack_messages(
    text: str,
    document_id: str,
    title: str = "Slack Thread",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[DocumentChunk]:
    """
    Message-aware chunking for Slack conversations.
    Preserves speaker handles, timestamps, and packs complete turns up to ~350-400 tokens.
    """
    turns = _split_into_turns(text, _SLACK_TURN_REGEX)
    channel_match = re.search(r"(#[a-zA-Z0-9_\-]+)", title + " " + text[:200])
    channel_name = channel_match.group(1) if channel_match else "Messages"
    heading_path = [title, channel_name]

    chunks: list[DocumentChunk] = []
    current_turns: list[str] = []
    current_tokens = 0
    global_idx = 0

    def finalize_chunk():
        nonlocal current_turns, current_tokens, global_idx
        if not current_turns:
            return
        clean_content = "\n\n".join(current_turns).strip()
        if not clean_content:
            return
        chunk_id = f"{document_id}#{global_idx:04d}"
        global_idx += 1
        breadcrumbs = " > ".join(heading_path)
        search_text = f"Title: {title} | Path: {breadcrumbs}\n\n{clean_content}"
        tokens = count_tokens(search_text)

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                title=title,
                heading_path=heading_path,
                content=clean_content,
                search_text=search_text,
                token_count=tokens,
                source="slack",
            )
        )
        current_turns = []
        current_tokens = 0

    for turn in turns:
        turn_tokens = count_tokens(turn)
        # If single turn is oversized, split safely
        atomic_units = [turn] if turn_tokens <= max_tokens else split_oversized_text(turn, max_tokens=max_tokens)

        for unit in atomic_units:
            u_tokens = count_tokens(unit)
            if current_turns and (current_tokens + u_tokens > max_tokens):
                finalize_chunk()
            current_turns.append(unit)
            current_tokens += u_tokens

    finalize_chunk()
    return chunks


def chunk_gmail_thread(
    text: str,
    document_id: str,
    title: str = "Email Thread",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[DocumentChunk]:
    """
    Message-aware chunking for Gmail / email chains.
    Extracts Subject, preserves email headers, and packs email messages up to ~350-400 tokens.
    """
    # Extract email Subject if present
    subj_match = re.search(r"^Subject:\s*(.+)$", text, flags=re.MULTILINE | re.IGNORECASE)
    subject = subj_match.group(1).strip() if subj_match else None
    heading_path = [title, f"Subject: {subject}"] if subject else [title, "Email Thread"]

    # Split into individual email messages
    turns = _split_into_turns(text, _GMAIL_HEADER_REGEX)

    chunks: list[DocumentChunk] = []
    current_turns: list[str] = []
    current_tokens = 0
    global_idx = 0

    def finalize_chunk():
        nonlocal current_turns, current_tokens, global_idx
        if not current_turns:
            return
        clean_content = "\n\n".join(current_turns).strip()
        if not clean_content:
            return
        chunk_id = f"{document_id}#{global_idx:04d}"
        global_idx += 1
        breadcrumbs = " > ".join(heading_path)
        search_text = f"Title: {title} | Path: {breadcrumbs}\n\n{clean_content}"
        tokens = count_tokens(search_text)

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                title=title,
                heading_path=heading_path,
                content=clean_content,
                search_text=search_text,
                token_count=tokens,
                source="gmail",
            )
        )
        current_turns = []
        current_tokens = 0

    for turn in turns:
        turn_tokens = count_tokens(turn)
        atomic_units = [turn] if turn_tokens <= max_tokens else split_oversized_text(turn, max_tokens=max_tokens)

        for unit in atomic_units:
            u_tokens = count_tokens(unit)
            if current_turns and (current_tokens + u_tokens > max_tokens):
                finalize_chunk()
            current_turns.append(unit)
            current_tokens += u_tokens

    finalize_chunk()
    return chunks


def chunk_fireflies_transcript(
    text: str,
    document_id: str,
    title: str = "Meeting Transcript",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[DocumentChunk]:
    """
    Message-aware chunking for Fireflies meeting transcripts.
    Preserves speaker turns, timestamps, and packs turns into ~350-400 token chunks.
    """
    turns = _split_into_turns(text, _FIREFLIES_TURN_REGEX)
    heading_path = [title, "Meeting Transcript"]

    chunks: list[DocumentChunk] = []
    current_turns: list[str] = []
    current_tokens = 0
    global_idx = 0

    def finalize_chunk():
        nonlocal current_turns, current_tokens, global_idx
        if not current_turns:
            return
        clean_content = "\n\n".join(current_turns).strip()
        if not clean_content:
            return
        chunk_id = f"{document_id}#{global_idx:04d}"
        global_idx += 1
        breadcrumbs = " > ".join(heading_path)
        search_text = f"Title: {title} | Path: {breadcrumbs}\n\n{clean_content}"
        tokens = count_tokens(search_text)

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                title=title,
                heading_path=heading_path,
                content=clean_content,
                search_text=search_text,
                token_count=tokens,
                source="fireflies",
            )
        )
        current_turns = []
        current_tokens = 0

    for turn in turns:
        turn_tokens = count_tokens(turn)
        atomic_units = [turn] if turn_tokens <= max_tokens else split_oversized_text(turn, max_tokens=max_tokens)

        for unit in atomic_units:
            u_tokens = count_tokens(unit)
            if current_turns and (current_tokens + u_tokens > max_tokens):
                finalize_chunk()
            current_turns.append(unit)
            current_tokens += u_tokens

    finalize_chunk()
    return chunks


def chunk_document_text(
    text: str,
    document_id: str,
    title: str = "Document",
    source_type: str = "generic",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[DocumentChunk]:
    """
    Source-aware document chunking entry point.
    Routes Slack, Gmail, and Fireflies to dedicated message/turn-aware chunkers.
    Retains structured markdown & code-fence chunking for all other sources.
    """
    normalized_source = source_type.lower().strip()

    if normalized_source in ("slack",):
        return chunk_slack_messages(text, document_id, title=title, max_tokens=max_tokens)

    if normalized_source in ("gmail", "email"):
        return chunk_gmail_thread(text, document_id, title=title, max_tokens=max_tokens)

    if normalized_source in ("fireflies", "transcript"):
        return chunk_fireflies_transcript(text, document_id, title=title, max_tokens=max_tokens)

    # Confluence, Jira, Linear, GitHub, Google Drive, Markdown, Generic Text
    sections = parse_markdown_sections(text, default_title=title)
    if not sections:
        sections = [(["Overview"], text.strip())]

    return create_chunks_from_sections(
        document_id=document_id,
        title=title,
        sections=sections,
        max_tokens=max_tokens,
        source_type=normalized_source,
    )
