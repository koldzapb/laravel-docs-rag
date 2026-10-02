"""Download the pinned snapshot of github.com/laravel/docs as plain markdown files."""

import io
import tarfile
import urllib.request
from pathlib import Path

from laravel_docs_rag.config import DOCS_COMMIT

TARBALL_URL = "https://codeload.github.com/laravel/docs/tar.gz/{commit}"

# Repository files that are not documentation pages.
SKIPPED_FILES = {"readme.md", "license.md", "documentation.md"}


def download_docs(target_dir: Path, commit: str = DOCS_COMMIT) -> list[Path]:
    """Fetch the repo tarball at `commit` and extract the top-level .md pages into target_dir.

    Returns the paths of the written files. Existing files are overwritten.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(TARBALL_URL.format(commit=commit), timeout=60) as response:
        archive = io.BytesIO(response.read())

    written: list[Path] = []
    with tarfile.open(fileobj=archive, mode="r:gz") as tar:
        for member in tar.getmembers():
            # Members look like "docs-<commit>/queues.md"; we only want top-level pages.
            parts = member.name.split("/")
            if len(parts) != 2 or not member.isfile():
                continue
            name = parts[1]
            if not name.endswith(".md") or name in SKIPPED_FILES:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            path = target_dir / name
            path.write_bytes(extracted.read())
            written.append(path)
    return sorted(written)


def list_doc_files(docs_dir: Path) -> list[Path]:
    """Return the markdown pages already present on disk, sorted by name."""
    return sorted(p for p in docs_dir.glob("*.md") if p.name not in SKIPPED_FILES)
