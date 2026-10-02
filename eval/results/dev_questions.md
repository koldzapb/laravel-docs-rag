# Eval results: `eval/dev_questions.jsonl`

Date: 2026-10-02 - in-scope questions: 30

| Mode | hit@1 | hit@5 | MRR@10 |
|---|---|---|---|
| vector | 0.70 | 0.97 | 0.79 |
| bm25 | 0.50 | 0.90 | 0.68 |
| hybrid | 0.73 | 0.93 | 0.81 |
| hybrid_rerank | 0.63 | 0.93 | 0.75 |

Answers: mode `hybrid`, model `openai/gpt-oss-120b`.

| Generation check | Result |
|---|---|
| Out-of-scope correctly declined | 5/5 (100%) |
| In-scope answered (not declined) | 26/30 (87%) |
| In-scope answers with >=1 citation | 26/30 (87%) |
