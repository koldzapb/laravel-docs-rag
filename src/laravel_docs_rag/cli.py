"""Command-line interface: `rag ingest`, `rag ask "..."`, `rag eval`, `rag serve`."""

import argparse
import sys
from pathlib import Path

from laravel_docs_rag.config import get_settings
from laravel_docs_rag.retrieval import IndexNotReadyError, Mode, RerankerDisabledError


def cmd_ingest(args: argparse.Namespace) -> None:
    from laravel_docs_rag.ingest import run_ingest

    run_ingest(get_settings(), refresh_docs=args.refresh)


def cmd_download_models(args: argparse.Namespace) -> None:
    from laravel_docs_rag.ingest import download_models

    download_models(get_settings())


def cmd_ask(args: argparse.Namespace) -> None:
    from laravel_docs_rag.pipeline import build_pipeline

    pipeline = build_pipeline()
    mode = Mode(args.mode) if args.mode else None
    answer = pipeline.ask(args.question, mode=mode, top_k=args.k)
    print(answer.answer)
    if answer.sources:
        print("\nSources:")
        for source in answer.sources:
            marker = "*" if source.cited else " "
            print(f" {marker}[{source.number}] {source.title}\n      {source.url}")


def cmd_eval(args: argparse.Namespace) -> None:
    from laravel_docs_rag.evaluation import (
        evaluate_answers,
        evaluate_retrieval,
        load_questions,
        to_markdown,
    )
    from laravel_docs_rag.retrieval import Retriever

    settings = get_settings()
    questions = load_questions(args.file)
    retriever = Retriever.from_settings(settings, load_reranker=True)
    modes = list(Mode)
    metrics, ranks = evaluate_retrieval(retriever, questions, modes)

    if args.verbose:
        for q in questions:
            if q.type == "in_scope":
                row = "  ".join(f"{m.value}={ranks[m][q.id] or '-'}" for m in modes)
                print(f"{q.id:>8}  {row}  {q.question}")

    answers, note = None, ""
    if args.answers:
        from laravel_docs_rag.generation import LLMClient
        from laravel_docs_rag.pipeline import RagPipeline

        pipeline = RagPipeline(retriever, LLMClient(settings), top_k=settings.top_k)
        answers = evaluate_answers(pipeline, questions)
        note = f"Answers: mode `{pipeline.default_mode}`, model `{settings.llm_model}`."

    report = to_markdown(args.file, metrics, answers, note)
    output = args.output or Path("eval/results") / f"{args.file.stem}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Written to {output}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("laravel_docs_rag.api:app", host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Ask the Laravel docs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="download, chunk, embed and index the docs")
    p.add_argument("--refresh", action="store_true", help="re-download the docs first")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("download-models", help="pre-fetch the embedding and rerank models")
    p.set_defaults(func=cmd_download_models)

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p.add_argument("--mode", choices=[m.value for m in Mode], help="retrieval mode")
    p.add_argument("-k", type=int, default=None, help="number of context chunks")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="evaluate retrieval (and optionally answers)")
    p.add_argument("--file", type=Path, default=Path("eval/questions.jsonl"))
    p.add_argument("--output", type=Path, default=None, help="markdown report path")
    p.add_argument("--answers", action="store_true", help="also run generation (uses the LLM)")
    p.add_argument("-v", "--verbose", action="store_true", help="print per-question ranks")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (IndexNotReadyError, RerankerDisabledError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
