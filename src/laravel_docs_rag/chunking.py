"""Markdown-aware chunking of Laravel documentation pages.

Pipeline for one page:
1. `split_sections` walks the file line by line and starts a new section at every heading.
   It remembers the heading path ("Queues > Job Batching > Dispatching Batches") and the
   `<a name="...">` anchor that the Laravel docs put right before each heading.
2. `split_blocks` cuts a section body into paragraphs, keeping fenced code blocks whole.
3. `pack_blocks` greedily packs blocks into chunks of at most `max_chars`, carrying the last
   small block into the next chunk as overlap.
"""

import re
from dataclasses import dataclass, field

from laravel_docs_rag.config import DOCS_VERSION

ANCHOR_RE = re.compile(r'^<a name="([^"]+)"></a>\s*$')
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING_ATTRS_RE = re.compile(r"\s*\{\.[^}]*\}\s*$")  # e.g. "{.collection-method}"
TOC_LINE_RE = re.compile(r"^\s*- \[.*\]\(#.*\)\s*$")  # "- [Introduction](#introduction)"
HTML_WRAPPER_RE = re.compile(r"^\s*</?div\b[^>]*>\s*$")

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP_CHARS = 300


@dataclass
class Section:
    """A heading and the raw lines below it (up to the next heading)."""

    headings: list[str]
    anchor: str | None
    lines: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """The unit we index and retrieve. `headings` is the full path, page title first."""

    id: str
    file: str
    headings: list[str]
    anchor: str | None
    text: str

    @property
    def title(self) -> str:
        """Human-readable location, e.g. "Queues > Job Batching"."""
        return " > ".join(self.headings)

    def embedding_text(self) -> str:
        """Text given to the embedder, BM25 and reranker: heading path + body.

        Prepending the heading path lets a chunk that only shows code still match
        a question phrased with the section's title.
        """
        return f"{self.title}\n\n{self.text}"


def clean_heading(raw: str) -> str:
    """Strip markdown attribute suffixes like `{.collection-method}` from a heading."""
    return HEADING_ATTRS_RE.sub("", raw).strip()


def closes_fence(line: str, fence: str) -> bool:
    """True if `line` closes a code block opened with `fence` (e.g. "```" or "````")."""
    stripped = line.strip()
    return stripped.startswith(fence) and set(stripped) == {fence[0]}


def split_sections(markdown: str) -> list[Section]:
    """Split a markdown page into sections, one per heading.

    Lines inside fenced code blocks are never treated as headings (PHP attributes such as
    `#[Signature('mail:send')]` would otherwise look like headings).
    """
    sections: list[Section] = []
    current = Section(headings=[], anchor=None)
    path: list[tuple[int, str, str | None]] = []  # (level, title, anchor) of enclosing headings
    pending_anchor: str | None = None
    fence: str | None = None  # the opening fence marker while inside a code block
    in_style = False

    for line in markdown.splitlines():
        fence_match = FENCE_RE.match(line)
        if fence is not None:
            current.lines.append(line)
            if closes_fence(line, fence):
                fence = None
            continue
        if fence_match:
            fence = fence_match.group(1)
            current.lines.append(line)
            continue

        # Outside code: drop presentational HTML the docs site uses.
        if in_style or line.strip().startswith("<style"):
            in_style = "</style>" not in line
            continue
        if HTML_WRAPPER_RE.match(line):
            continue

        anchor_match = ANCHOR_RE.match(line)
        if anchor_match:
            pending_anchor = anchor_match.group(1)
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            sections.append(current)
            level = len(heading_match.group(1))
            title = clean_heading(heading_match.group(2))
            path = [entry for entry in path if entry[0] < level]
            # A few headings have no anchor of their own; link to the nearest parent that does.
            anchor = pending_anchor or next((a for _, _, a in reversed(path) if a), None)
            path.append((level, title, anchor))
            current = Section(headings=[t for _, t, _ in path], anchor=anchor)
            pending_anchor = None
            continue

        if line.strip():
            pending_anchor = None  # an anchor only belongs to a heading directly below it
        current.lines.append(line)

    sections.append(current)
    return [s for s in sections if s.headings]


def split_blocks(lines: list[str]) -> list[str]:
    """Group lines into blocks separated by blank lines; a fenced code block is one block."""
    blocks: list[str] = []
    current: list[str] = []
    fence: str | None = None

    def flush() -> None:
        text = "\n".join(current).strip("\n")
        if text.strip():
            blocks.append(text)
        current.clear()

    for line in lines:
        fence_match = FENCE_RE.match(line)
        if fence is None and fence_match:
            flush()  # code starts a new block
            fence = fence_match.group(1)
            current.append(line)
        elif fence is not None:
            current.append(line)
            if closes_fence(line, fence):
                fence = None
                flush()  # code ends its block
        elif not line.strip():
            flush()
        else:
            current.append(line)
    flush()
    return blocks


def split_oversized(blocks: list[str], max_chars: int) -> list[str]:
    """Split long *non-code* blocks (big tables, long link lists) on line boundaries.

    Code blocks are returned untouched: half a code example is worse than a long chunk.
    """
    result: list[str] = []
    for block in blocks:
        if len(block) <= max_chars or FENCE_RE.match(block):
            result.append(block)
            continue
        piece: list[str] = []
        for line in block.splitlines():
            if piece and sum(len(p) + 1 for p in piece) + len(line) > max_chars:
                result.append("\n".join(piece))
                piece = []
            piece.append(line)
        if piece:
            result.append("\n".join(piece))
    return result


def pack_blocks(
    blocks: list[str],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Greedily pack blocks into chunks of roughly `max_chars`.

    Blocks are never split, so a single code block longer than `max_chars` becomes its own
    (oversized) chunk rather than being cut in half. When a chunk is closed, its last block is
    repeated at the start of the next chunk if it is at most `overlap_chars` long, so a
    sentence like "For example:" stays next to the code it introduces.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    carried = 0  # how many blocks at the start of `current` are overlap from the last chunk

    for block in split_oversized(blocks, max_chars):
        if current and current_len + len(block) > max_chars and len(current) > carried:
            chunks.append("\n\n".join(current))
            tail = current[-1]
            current = [tail] if len(tail) <= overlap_chars else []
            current_len = sum(len(b) for b in current)
            carried = len(current)
        current.append(block)
        current_len += len(block)

    if len(current) > carried:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_markdown(
    markdown: str,
    file: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Turn one documentation page into retrieval chunks with heading/anchor metadata."""
    markdown = markdown.replace("{{version}}", DOCS_VERSION)
    chunks: list[Chunk] = []
    for section_index, section in enumerate(split_sections(markdown)):
        lines = section.lines
        if len(section.headings) == 1:
            # The page intro sits under the H1 together with a table of contents; drop the TOC.
            lines = [line for line in lines if not TOC_LINE_RE.match(line)]
        blocks = split_blocks(lines)
        for part_index, text in enumerate(pack_blocks(blocks, max_chars, overlap_chars)):
            chunks.append(
                Chunk(
                    id=f"{file}:{section_index}:{part_index}",
                    file=file,
                    headings=section.headings,
                    anchor=section.anchor,
                    text=text,
                )
            )
    return chunks
