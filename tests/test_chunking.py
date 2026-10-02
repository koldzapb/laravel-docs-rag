from laravel_docs_rag.chunking import (
    chunk_markdown,
    pack_blocks,
    split_blocks,
    split_sections,
)

PAGE = """# Queues

- [Introduction](#introduction)
- [Job Batching](#job-batching)

Some intro text about queues, linking to [events](/docs/{{version}}/events).

<a name="introduction"></a>
## Introduction

Queues let you defer work.

<a name="job-batching"></a>
## Job Batching

Batches group jobs.

<a name="dispatching-batches"></a>
### Dispatching Batches

Use the batch method:

```php
#[Tries(3)]
class ImportCsv implements ShouldQueue
{
    # not a heading either

    use Batchable;
}
```

### Heading Without Anchor

Text.

<a name="method-chunk"></a>
#### `chunk()` {.collection-method}

Chunks a collection.
"""


def test_sections_track_heading_path_and_anchor():
    sections = split_sections(PAGE)
    paths = [s.headings for s in sections]
    assert ["Queues", "Job Batching", "Dispatching Batches"] in paths
    batching = next(s for s in sections if s.headings[-1] == "Dispatching Batches")
    assert batching.anchor == "dispatching-batches"


def test_code_lines_starting_with_hash_are_not_headings():
    titles = [s.headings[-1] for s in split_sections(PAGE)]
    assert not any("Tries" in t or "not a heading" in t for t in titles)


def test_heading_attributes_are_stripped():
    titles = [s.headings[-1] for s in split_sections(PAGE)]
    assert "`chunk()`" in titles


def test_heading_without_anchor_falls_back_to_parent_anchor():
    section = next(s for s in split_sections(PAGE) if s.headings[-1] == "Heading Without Anchor")
    assert section.anchor == "job-batching"


def test_toc_is_dropped_and_version_placeholder_replaced():
    intro = chunk_markdown(PAGE, "queues.md")[0]
    assert intro.headings == ["Queues"]
    assert "- [Introduction]" not in intro.text
    assert "/docs/13.x/events" in intro.text


def test_chunks_keep_file_and_metadata():
    chunks = chunk_markdown(PAGE, "queues.md")
    chunk = next(c for c in chunks if c.anchor == "dispatching-batches")
    assert chunk.file == "queues.md"
    assert chunk.title == "Queues > Job Batching > Dispatching Batches"
    assert chunk.embedding_text().startswith("Queues > Job Batching > Dispatching Batches\n\n")
    assert len({c.id for c in chunks}) == len(chunks)


def test_code_block_with_blank_lines_is_one_block():
    lines = ["Intro:", "", "```php", "$a = 1;", "", "$b = 2;", "```", "", "After."]
    assert split_blocks(lines) == ["Intro:", "```php\n$a = 1;\n\n$b = 2;\n```", "After."]


def test_pack_respects_size_cap():
    blocks = ["a" * 400, "b" * 400, "c" * 400]
    chunks = pack_blocks(blocks, max_chars=900, overlap_chars=50)
    # "b"*400 is longer than overlap_chars, so it is not repeated in the second chunk.
    assert chunks == ["a" * 400 + "\n\n" + "b" * 400, "c" * 400]


def test_pack_repeats_small_last_block_as_overlap():
    blocks = ["a" * 500, "For example:", "b" * 500]
    chunks = pack_blocks(blocks, max_chars=600, overlap_chars=50)
    assert chunks == ["a" * 500 + "\n\nFor example:", "For example:\n\n" + "b" * 500]


def test_oversized_code_block_is_never_split():
    code = "```php\n" + "\n".join(f"$x{i} = {i};" for i in range(300)) + "\n```"
    chunks = pack_blocks(["Intro.", code], max_chars=200, overlap_chars=0)
    assert code in chunks


def test_oversized_table_is_split_on_lines():
    table = "\n".join(f"| row {i} | value {i} |" for i in range(100))
    chunks = pack_blocks([table], max_chars=300, overlap_chars=0)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert "\n".join(chunks) == table
