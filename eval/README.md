# Held-out evaluation set

`questions.jsonl` has 50 questions about the official Laravel docs (laravel/docs, branch `13.x`). There is one JSON object per line:

```json
{"id": "q01", "question": "...", "type": "in_scope", "gold": [{"file": "routing.md", "heading": "Route Caching"}], "answer_hint": "..."}
```

- `type`: `in_scope` (44 questions) or `out_of_scope` (6 questions; these have `gold: []` and the docs cannot answer them, so the system should decline or say it doesn't know).
- `gold`: the places the answer lives. `file` is the markdown filename at the repo root. `heading` is the exact text of the most specific `##`/`###`/`####` heading whose section holds the answer, with the trailing `<a name=...>` anchor removed. A question has one gold, or two when the answer really is split across two sections.
- `answer_hint`: a short sentence with the key fact, for checking answers by hand or with an LLM judge.

In-scope questions come in three kinds: easy keyword questions, paraphrased questions that avoid the heading's wording, and questions about a specific detail (a method name, a config key, a CLI flag).

## Hit definition

A retrieved chunk counts as a **hit** for a question when:

1. the chunk's source file equals a gold `file`, and
2. that gold's `heading` equals one of the headings in the chunk's heading path (its own heading and all its ancestors). The comparison ignores case and normalises whitespace: trim, then collapse runs of whitespace to one space.

Suggested metrics are Recall@k / Hit@k: does any of the top-k chunks hit any gold? MRR uses the rank of the first hit. For out-of-scope questions, measure whether the system declines to answer.
