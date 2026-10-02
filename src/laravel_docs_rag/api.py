"""HTTP API: POST /ask, GET /health, and a tiny HTML page at /."""

from functools import lru_cache
from importlib import resources
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from laravel_docs_rag.pipeline import RagPipeline, build_pipeline
from laravel_docs_rag.retrieval import IndexNotReadyError, Mode, RerankerDisabledError

app = FastAPI(title="Ask the Laravel docs", version="0.1.0")


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    """Build the pipeline once, on first use (loading models takes a few seconds).

    lru_cache does not cache exceptions, so if the index is empty the next request retries.
    Tests replace this dependency with a pipeline that uses a fake LLM.
    """
    return build_pipeline()


@app.exception_handler(IndexNotReadyError)
async def index_not_ready(request: Request, error: IndexNotReadyError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(RerankerDisabledError)
async def reranker_disabled(request: Request, error: RerankerDisabledError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


# FastAPI dependency: "give this endpoint the pipeline". Tests override get_pipeline.
PipelineDep = Annotated[RagPipeline, Depends(get_pipeline)]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Mode | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    number: int
    title: str
    url: str
    cited: bool


class AskResponse(BaseModel):
    answer: str
    declined: bool
    mode: Mode
    sources: list[SourceOut]


@app.get("/health")
def health(pipeline: PipelineDep) -> dict[str, str]:
    """Liveness + readiness: returns 503 (via the handler above) until the index exists."""
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest, pipeline: PipelineDep) -> AskResponse:
    answer = pipeline.ask(request.question, mode=request.mode, top_k=request.top_k)
    return AskResponse(
        answer=answer.answer,
        declined=answer.declined,
        mode=answer.mode,
        sources=[
            SourceOut(number=s.number, title=s.title, url=s.url, cited=s.cited)
            for s in answer.sources
        ],
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return resources.files("laravel_docs_rag").joinpath("static/index.html").read_text("utf-8")
