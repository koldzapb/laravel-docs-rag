"""Application settings, read from environment variables (and an optional .env file).

Everything configurable lives here so the rest of the code never touches os.environ.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# The documentation snapshot we index. Pinning a commit makes ingest reproducible:
# the same command always produces the same chunks, so eval numbers are comparable.
DOCS_VERSION = "13.x"
DOCS_COMMIT = "9c2295fc4b3c0f87d5fe3192a76075c85d9071d7"

# All three providers speak the OpenAI chat-completions protocol, so one client works
# for each of them; only the base URL, the API-key variable and the default model differ.
PROVIDERS: dict[str, dict[str, str | None]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,  # Ollama runs locally and ignores the key.
        "model": "llama3.1",
    },
}


@dataclass(frozen=True)
class Settings:
    """All runtime configuration in one immutable object."""

    data_dir: Path
    model_cache_dir: Path
    embedding_model: str
    rerank_model: str
    llm_provider: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    top_k: int  # how many chunks are sent to the LLM as context
    candidates: int  # how many chunks each retriever returns before fusion/reranking
    use_rerank: bool  # off by default: it did not improve the dev-set numbers (see README)

    @property
    def docs_dir(self) -> Path:
        """Where the downloaded markdown files live."""
        return self.data_dir / "docs"

    @property
    def chroma_dir(self) -> Path:
        """Where Chroma persists the vector index."""
        return self.data_dir / "chroma"


def _env(name: str, default: str) -> str:
    """Read an env var, trimming whitespace and accidental surrounding quotes."""
    value = os.environ.get(name, default).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    """Build Settings from the environment. Values in .env never override real env vars."""
    load_dotenv()

    provider = _env("LLM_PROVIDER", "groq").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got {provider!r}")
    defaults = PROVIDERS[provider]
    key_env = defaults["api_key_env"]

    return Settings(
        data_dir=Path(_env("DATA_DIR", "data")),
        model_cache_dir=Path(_env("MODEL_CACHE_DIR", ".cache/models")),
        embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        rerank_model=_env("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        llm_provider=provider,
        llm_base_url=_env("LLM_BASE_URL", str(defaults["base_url"])),
        llm_api_key=_env(key_env, "") if key_env else "not-needed",
        llm_model=_env("LLM_MODEL", str(defaults["model"])),
        top_k=int(_env("TOP_K", "5")),
        candidates=int(_env("CANDIDATES", "30")),
        use_rerank=_env_bool("USE_RERANK", False),
    )
