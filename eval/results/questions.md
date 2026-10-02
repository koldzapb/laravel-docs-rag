# Eval results: `eval/questions.jsonl`

Date: 2026-10-02 - in-scope questions: 44

| Mode | hit@1 | hit@5 | MRR@10 |
|---|---|---|---|
| vector | 0.61 | 0.89 | 0.73 |
| bm25 | 0.61 | 0.80 | 0.69 |
| hybrid | 0.70 | 0.86 | 0.77 |
| hybrid_rerank | 0.59 | 0.86 | 0.72 |

Answers: mode `hybrid`, model `openai/gpt-oss-120b`.

| Generation check | Result |
|---|---|
| Out-of-scope correctly declined | 6/6 (100%) |
| In-scope answered (not declined) | 40/44 (91%) |
| In-scope answers with >=1 citation | 40/44 (91%) |
