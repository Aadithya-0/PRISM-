"""Command-line runner for the standalone L4 RAG subsystem."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .rag import L4Config, LocalRAG, L4Error


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.suffix.lower() in {".pdf", ".txt", ".md"})
    raise FileNotFoundError(path)


def main() -> int:
    _load_environment()
    parser = argparse.ArgumentParser(description="Index and query PRISM's local L4 RAG store.")
    parser.add_argument("path", type=Path, help="Document file or directory to index")
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Skip indexing and query the existing local ChromaDB collection",
    )
    parser.add_argument("--store", default=".prism/l4_chroma", help="ChromaDB persistence directory")
    parser.add_argument("--collection", default="prism_l4_documents")
    parser.add_argument("--provider", choices=("google", "ollama"), default="google")
    parser.add_argument("--embedding-model")
    parser.add_argument("--generation-model")
    parser.add_argument("--top-k", type=int, default=5)
    retrieval_group = parser.add_mutually_exclusive_group()
    retrieval_group.add_argument(
        "--cross-document",
        dest="cross_document",
        action="store_true",
        help="Use source-balanced retrieval (default)",
    )
    retrieval_group.add_argument(
        "--single-source",
        dest="cross_document",
        action="store_false",
        help="Use the globally nearest chunks without source balancing",
    )
    parser.set_defaults(cross_document=True)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--max-chunks-per-source", type=int, default=3)
    parser.add_argument("--max-total-chunks", type=int, default=8)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    if args.provider == "google":
        embedding_model = args.embedding_model or os.environ.get(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        )
        generation_model = args.generation_model or os.environ.get(
            "GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite"
        )
    else:
        embedding_model = args.embedding_model or os.environ.get(
            "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
        )
        generation_model = args.generation_model or os.environ.get(
            "OLLAMA_GENERATION_MODEL", "gemma3:4b"
        )
    config = L4Config(
        persist_directory=args.store,
        collection_name=args.collection,
        embedding_provider=args.provider,
        generation_provider=args.provider,
        embedding_model=embedding_model,
        generation_model=generation_model,
    )
    try:
        rag = LocalRAG(config)
        if not args.query_only:
            documents = _files(args.path)
            for document in documents:
                ids = rag.chunk_and_insert(str(document))
                print(f"Indexed {len(ids)} chunks from {document}")
        else:
            print("Using the existing local ChromaDB collection; no documents will be indexed.")
        print("Enter queries; press Ctrl+C or Ctrl+D to exit.")
        while True:
            question = input("\nQuery> ").strip()
            if not question:
                continue
            if args.cross_document:
                results = rag.query_cross_document(
                    question,
                    candidate_k=args.candidate_k,
                    max_chunks_per_source=args.max_chunks_per_source,
                    max_total_chunks=args.max_total_chunks,
                    threshold=args.threshold,
                )
            elif args.threshold is not None:
                results = rag.get_chunks_above_threshold_with_limit(question, args.threshold, args.top_k)
            else:
                results = rag.query(question, args.top_k)
            for index, result in enumerate(results, 1):
                metadata = result.metadata
                print(
                    f"\n[{index}] score={result.score:.3f} "
                    f"source={metadata.get('source_path')} page={metadata.get('page_number', 'n/a')}"
                )
                print(result.content)
            print("\nAnswer:")
            print(
                rag.generate(
                    question,
                    args.top_k,
                    args.threshold,
                    cross_document=args.cross_document,
                    candidate_k=args.candidate_k,
                    max_chunks_per_source=args.max_chunks_per_source,
                    max_total_chunks=args.max_total_chunks,
                )
            )
    except (L4Error, FileNotFoundError, ValueError) as exc:
        print(f"L4 error: {exc}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
