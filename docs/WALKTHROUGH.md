# Code walkthrough

Read the modules in this order. Each section says what the module does, why it exists, and what
you might be asked about it in an interview. All the code is in `src/laravel_docs_rag/`.

If you know Laravel, these mappings help: `config.py` plays the role of `config/*.php` plus
`env()`, `cli.py` is an Artisan console kernel, `api.py` is `routes/api.php` together with
controllers, and the dataclasses are like small DTOs.

## 1. `config.py`: settings

- `get_settings()` reads environment variables, with `python-dotenv` loading `.env` first, and
  returns a frozen `Settings` dataclass. No other module touches `os.environ`.
- `PROVIDERS` maps `groq`, `openai` and `ollama` to a base URL, the name of the API-key
  variable, and a default model. All three speak the OpenAI chat-completions protocol, so
  switching provider is only configuration.
- `DOCS_COMMIT` pins the docs snapshot so that ingest is reproducible.
- `_env()` strips quotes in case a key was written as `GROQ_API_KEY="..."`.

## 2. `docs_source.py`: getting the docs

- `download_docs()` downloads the GitHub tarball for the pinned commit with only the standard
  library (`urllib` and `tarfile`). It writes the top-level `*.md` pages to `data/docs/` and
  skips `readme.md`, `license.md` and `documentation.md` (the navigation page).

## 3. `chunking.py`: the most "domain" part

The goal is chunks that each map to one section on laravel.com.

- `split_sections()` walks a page line by line:
  - It tracks whether it is inside a fenced code block. This matters because PHP attributes
    like `#[Signature('mail:send')]` start with `#` and would otherwise look like headings.
  - A line `<a name="job-batching"></a>` is remembered as the anchor for the heading directly
    below it.
  - On a heading, it updates the heading path. The path is a list of `(level, title, anchor)`
    entries, and opening a heading removes every entry at the same or a deeper level. A heading
    with no anchor of its own inherits its parent's anchor, so the link still lands nearby.
  - It strips `{.collection-method}` suffixes, `<style>` blocks and `<div>` wrapper lines.
- `split_blocks()` cuts a section body into paragraphs at blank lines. A fenced code block is
  always one block, even if it contains blank lines.
- `pack_blocks()` greedily fills chunks up to `max_chars` (1500). When it closes a chunk, it
  repeats the last block at the start of the next one if that block is short (≤ 300 chars).
  That is the overlap: an intro like "For example:" stays next to the code it introduces. Code
  blocks are never split. `split_oversized()` only splits long non-code blocks, such as big
  tables, at line boundaries.
- `chunk_markdown()` ties it together. It also removes the table of contents under the H1 and
  replaces `{{version}}` in links with `13.x`.
- `Chunk.embedding_text()` is `"Queues > Job Batching > Dispatching Batches\n\n<body>"`. The
  heading path is prepended so that a chunk which is mostly code still matches a question
  phrased in words.

## 4. `citations.py`: links and `[n]` markers

- `doc_url("queues.md", "job-batching")` returns
  `https://laravel.com/docs/13.x/queues#job-batching`.
- `cited_numbers()` finds `[1]`, `[3]`, ... in the answer with a regex.
- `normalize_citations()` rewrites the citation style `【5†L1-L5】`, which gpt-oss models
  sometimes produce, into `[5]`.
- `build_sources()` numbers the chunks 1..n in the same order they appeared in the prompt,
  which is why the numbers match, and flags the ones the answer cited.

## 5. `embeddings.py` and `vector_store.py`: dense retrieval

- `Embedder` wraps fastembed's `TextEmbedding`. fastembed runs ONNX models on the CPU: no
  PyTorch and no API key. BGE models expect queries and passages to be embedded slightly
  differently, so there are two methods, `embed_query` and `embed_documents`.
- `VectorStore` wraps a persistent Chroma collection that uses cosine distance. We compute
  embeddings ourselves and pass them in, so Chroma never embeds anything. Chroma metadata must
  be scalar values, so the heading list is stored as a JSON string.
- `all_chunks()` loads every chunk back. Chroma is the single source of truth for chunk text.

## 6. `bm25.py`: keyword retrieval

- `tokenize()` lowercases the text, splits it on non-word characters (so `queue:work` becomes
  `queue` and `work`) and drops a short stopword list.
- `BM25Index` uses `rank_bm25.BM25Okapi`. It is built in memory at startup from the chunks in
  Chroma, which takes about a second for 4.5k chunks.

## 7. `fusion.py`: Reciprocal Rank Fusion

```python
for ranking in rankings:
    for rank, item_id in enumerate(ranking, start=1):
        scores[item_id] += 1.0 / (k + rank)
```

- Each list votes 1/(60 + rank) for each item. Only ranks matter, so a cosine similarity and a
  BM25 score never have to be put on the same scale.
- k = 60 comes from the original paper. It flattens the curve, so an item ranked 3rd by both
  lists beats an item ranked 1st by one list and absent from the other.

## 8. `rerank.py`: cross-encoder

- A bi-encoder (the embedder) encodes the question and the passage separately. A cross-encoder
  reads them together and scores relevance, which is more accurate but costs a model call per
  pair. So it only reorders the top ~30 hybrid candidates.
- It is off by default because it did not improve the dev-set metrics (see the README).

## 9. `retrieval.py`: one interface, four modes

- `Mode` is the enum `vector | bm25 | hybrid | hybrid_rerank`.
- `Retriever.from_settings()` opens Chroma, raises `IndexNotReadyError` if the index is empty,
  builds BM25 and loads the models.
- `search(query, mode, k)` returns `Chunk`s. Hybrid takes `candidates` (30) results from each
  retriever, fuses them, and keeps `k`. Rerank mode reranks the fused top 30.

## 10. `generation.py`: the prompt and the LLM call

- `SYSTEM_PROMPT` sets the rules: use only the context, cite `[n]`, and if the answer isn't in
  the context, reply with exactly `DECLINE_MESSAGE`. A fixed sentence makes `is_decline()` a
  plain string check, which the eval relies on.
- `build_context()` formats the passages as `[1] Queues > Job Batching (queues.md)\n<text>`
  and truncates the rare huge code sample to 4000 characters.
- `LLMClient` is the official `openai` SDK pointed at any `base_url`. `temperature=0` keeps
  answers factual, and `max_retries=5` makes the SDK back off on HTTP 429 rate limits.

## 11. `pipeline.py`: putting it together

- `RagPipeline.ask()` retrieves chunks. If there are none, it declines without calling the LLM.
  Otherwise it builds the messages, calls the LLM, normalises the citations and attaches the
  sources (none if the model declined).
- The retriever and the LLM are typed as `Protocol`s (structural interfaces, like PHP
  interfaces but implicit). That is what lets the tests pass in fakes.
- `build_pipeline()` wires up the real components from settings.

## 12. `ingest.py`: the offline job

- `run_ingest()` downloads the docs if needed, chunks them, embeds them in batches, resets the
  Chroma collection and stores everything.
- `download_models()` pre-fetches both ONNX models. The Dockerfile calls it so the image
  contains the models.

## 13. `evaluation.py`: measuring retrieval

- `is_hit()`: the file matches and the gold heading (normalised for case, whitespace and
  backticks) appears anywhere in the chunk's heading path. A subsection of the gold section
  therefore counts.
- `compute_metrics()`:
  - hit@1 is the share of questions whose first result is relevant.
  - hit@5 is the share with a relevant result in the top 5.
  - MRR@10 is the mean of 1/rank of the first relevant result, counting 0 if it isn't in the
    top 10.
- `evaluate_answers()` (the `--answers` flag) checks that out-of-scope questions are declined,
  and that in-scope ones are answered with at least one citation.
- `eval/dev_questions.jsonl` is my own tuning set. `eval/questions.jsonl` is a held-out set
  written by someone else, and it is never used for tuning.

## 14. `api.py` and `cli.py`: interfaces

- `api.py` is FastAPI.
  - `get_pipeline()` is cached with `lru_cache`, so the models load once on the first request.
    It is injected with `Depends`, so the tests override it with
    `app.dependency_overrides[get_pipeline] = ...`. This is the equivalent of swapping a binding
    in Laravel's container.
  - Pydantic models validate the request (`question` must be 3 to 1000 characters) and shape
    the response.
  - `IndexNotReadyError` becomes HTTP 503 with an "ingest first" message.
  - `/` serves `static/index.html`, which is plain JS calling `/ask`.
- `cli.py` uses `argparse` with the subcommands `ingest`, `ask`, `eval`, `serve` and
  `download-models`. The `rag` command comes from `[project.scripts]` in `pyproject.toml`.
  Heavy imports happen inside each command, so `rag --help` is instant.

## 15. Tests, Docker, CI

- `tests/` run offline in about 3 seconds. `test_api.py` builds a real `RagPipeline` with a
  fake retriever and a fake LLM. It checks the request and response shape, that the chunk
  reaches the prompt as `[1] ...`, decline handling, validation errors and the 503 status.
- The `Dockerfile` has two stages. The first is a uv image that installs locked dependencies
  and downloads the models into `/opt/models`. The second is `python:3.12-slim` with only the
  virtualenv and the models, running as a non-root user. The index lives in the `rag-data`
  volume mounted at `/data`.
- `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check` and `pytest` with uv.
