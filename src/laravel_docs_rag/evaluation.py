"""Retrieval evaluation: hit@1, hit@5 and MRR@10 per retrieval mode, plus decline checks.

Question file format (JSONL, one object per line):
    {"id": "q1", "question": "...", "type": "in_scope" | "out_of_scope",
     "gold": [{"file": "queues.md", "heading": "Job Batching"}]}

A retrieved chunk is a HIT if its file matches a gold entry and the gold heading equals
(case-insensitive, whitespace-normalised, backticks ignored) any heading in the chunk's
heading path. So a chunk from "Queues > Job Batching > Dispatching Batches" is a hit for
gold heading "Job Batching".
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from laravel_docs_rag.chunking import Chunk
from laravel_docs_rag.retrieval import Mode

MRR_DEPTH = 10


@dataclass
class GoldRef:
    file: str
    heading: str


@dataclass
class EvalQuestion:
    id: str
    question: str
    type: str  # "in_scope" or "out_of_scope"
    gold: list[GoldRef]


@dataclass
class RetrievalMetrics:
    hit_at_1: float
    hit_at_5: float
    mrr_at_10: float
    count: int


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read a JSONL question file, skipping blank lines."""
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(
            EvalQuestion(
                id=str(row["id"]),
                question=row["question"],
                type=row["type"],
                gold=[GoldRef(g["file"], g["heading"]) for g in row.get("gold", [])],
            )
        )
    return questions


def normalize_heading(text: str) -> str:
    """'  The `queue:work`  Command ' -> 'the queue:work command'."""
    return " ".join(text.replace("`", "").split()).casefold()


def is_hit(chunk: Chunk, gold: list[GoldRef]) -> bool:
    """Does this chunk belong to (a subsection of) any gold section?"""
    chunk_headings = {normalize_heading(h) for h in chunk.headings}
    return any(
        chunk.file == ref.file and normalize_heading(ref.heading) in chunk_headings for ref in gold
    )


def first_hit_rank(chunks: list[Chunk], gold: list[GoldRef]) -> int | None:
    """1-based rank of the first relevant chunk, or None if none was retrieved."""
    for rank, chunk in enumerate(chunks, start=1):
        if is_hit(chunk, gold):
            return rank
    return None


def compute_metrics(ranks: list[int | None]) -> RetrievalMetrics:
    """Aggregate first-hit ranks into hit@1, hit@5 and MRR@10."""
    n = len(ranks)
    if n == 0:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0)
    return RetrievalMetrics(
        hit_at_1=sum(1 for r in ranks if r is not None and r <= 1) / n,
        hit_at_5=sum(1 for r in ranks if r is not None and r <= 5) / n,
        mrr_at_10=sum(1 / r for r in ranks if r is not None and r <= MRR_DEPTH) / n,
        count=n,
    )


def evaluate_retrieval(
    retriever, questions: list[EvalQuestion], modes: list[Mode]
) -> tuple[dict[Mode, RetrievalMetrics], dict[Mode, dict[str, int | None]]]:
    """Run every in-scope question through every mode.

    Returns the metrics per mode and the per-question first-hit ranks (useful for debugging).
    """
    in_scope = [q for q in questions if q.type == "in_scope"]
    metrics: dict[Mode, RetrievalMetrics] = {}
    ranks_by_mode: dict[Mode, dict[str, int | None]] = {}
    for mode in modes:
        ranks = {
            q.id: first_hit_rank(retriever.search(q.question, mode, MRR_DEPTH), q.gold)
            for q in in_scope
        }
        ranks_by_mode[mode] = ranks
        metrics[mode] = compute_metrics(list(ranks.values()))
    return metrics, ranks_by_mode


@dataclass
class AnswerResults:
    out_of_scope_declined: int
    out_of_scope_total: int
    in_scope_answered: int
    in_scope_total: int
    in_scope_with_citation: int


def evaluate_answers(pipeline, questions: list[EvalQuestion]) -> AnswerResults:
    """Generate answers: out-of-scope questions should be declined, in-scope ones answered."""
    results = AnswerResults(0, 0, 0, 0, 0)
    for q in questions:
        answer = pipeline.ask(q.question)
        if q.type == "out_of_scope":
            results.out_of_scope_total += 1
            results.out_of_scope_declined += answer.declined
        else:
            results.in_scope_total += 1
            results.in_scope_answered += not answer.declined
            results.in_scope_with_citation += any(s.cited for s in answer.sources)
    return results


def _pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "n/a"


def to_markdown(
    question_file: Path,
    metrics: dict[Mode, RetrievalMetrics],
    answers: AnswerResults | None = None,
    answers_note: str = "",
) -> str:
    """Render results as a markdown report."""
    count = next(iter(metrics.values())).count if metrics else 0
    lines = [
        f"# Eval results: `{question_file}`",
        "",
        f"Date: {date.today().isoformat()} - in-scope questions: {count}",
        "",
        "| Mode | hit@1 | hit@5 | MRR@10 |",
        "|---|---|---|---|",
    ]
    for mode, m in metrics.items():
        lines.append(f"| {mode.value} | {m.hit_at_1:.2f} | {m.hit_at_5:.2f} | {m.mrr_at_10:.2f} |")
    if answers is not None:
        lines += [
            "",
            answers_note,
            "",
            "| Generation check | Result |",
            "|---|---|",
            "| Out-of-scope correctly declined | "
            f"{_pct(answers.out_of_scope_declined, answers.out_of_scope_total)} |",
            f"| In-scope answered (not declined) | "
            f"{_pct(answers.in_scope_answered, answers.in_scope_total)} |",
            f"| In-scope answers with >=1 citation | "
            f"{_pct(answers.in_scope_with_citation, answers.in_scope_total)} |",
        ]
    return "\n".join(lines) + "\n"
