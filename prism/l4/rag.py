"""Persistent local RAG services for PRISM's standalone L4 subsystem."""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)


class L4Error(RuntimeError):
    """Base class for expected L4 failures."""


class SourceDocumentError(L4Error):
    """A source document cannot be read or has no indexable content."""


class ProviderError(L4Error):
    """An embedding or generation provider failed or is not configured."""


class RetrievalError(L4Error):
    """The vector store could not be initialized or queried."""


@dataclass(frozen=True)
class ChunkResult:
    """A retrieved document chunk and its normalized cosine similarity."""

    content: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class L4Config:
    """Configuration for the local vector store and provider selection."""

    persist_directory: str = ".prism/l4_chroma"
    collection_name: str = "prism_l4_documents"
    embedding_provider: str = "google"
    generation_provider: str = "google"
    embedding_model: str = "gemini-embedding-001"
    generation_model: str = "gemini-3.5-flash-lite"
    chunk_size: int = 1200
    chunk_overlap: int = 150


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class GenerationProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...


def _load_source(file_path: Path) -> list[tuple[str, int | None]]:
    if not file_path.exists():
        raise SourceDocumentError(f"Source file does not exist: {file_path}")
    if not file_path.is_file():
        raise SourceDocumentError(f"Source path is not a file: {file_path}")
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader

            pages = PdfReader(str(file_path)).pages
            return [(page.extract_text() or "", index + 1) for index, page in enumerate(pages)]
        if suffix in {".txt", ".md"}:
            return [(file_path.read_text(encoding="utf-8"), None)]
    except Exception as exc:
        raise SourceDocumentError(f"Could not parse {file_path}: {exc}") from exc
    raise SourceDocumentError(f"Unsupported document type: {suffix or '(none)'}")


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be smaller")
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        if end < len(cleaned):
            boundary = max(cleaned.rfind(". ", start, end), cleaned.rfind("\n", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _cosine_similarity(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(distance)))


class GoogleProvider:
    def __init__(self, api_key: str | None, embedding_model: str, generation_model: str):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self._client: Any = None

    def _get_client(self) -> Any:
        if not self.api_key:
            raise ProviderError("Set GEMINI_API_KEY before using the Google provider.")
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError as exc:
                raise ProviderError("Install google-genai to use the Google provider.") from exc
        return self._client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = self._get_client().models.embed_content(
                model=self.embedding_model,
                contents=list(texts),
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            return [list(item.values) for item in response.embeddings]
        except Exception as exc:
            raise ProviderError(f"Google embedding request failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            response = self._get_client().models.embed_content(
                model=self.embedding_model,
                contents=text,
                config={"task_type": "RETRIEVAL_QUERY"},
            )
            return list(response.embeddings[0].values)
        except Exception as exc:
            raise ProviderError(f"Google query embedding failed: {exc}") from exc

    def generate(self, prompt: str) -> str:
        try:
            response = self._get_client().models.generate_content(
                model=self.generation_model,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if not text:
                raise ProviderError("Google generation returned no text.")
            return text
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Google generation request failed: {exc}") from exc


class OllamaProvider:
    """Provider adapter for an Ollama server running locally."""

    def __init__(self, embedding_model: str, generation_model: str, host: str | None = None):
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        import json
        from urllib.error import URLError
        from urllib.request import Request, urlopen

        try:
            request = Request(
                f"{self.host}/api/{endpoint}",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except (OSError, URLError, ValueError) as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._request("embed", {"model": self.embedding_model, "input": list(texts)})
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ProviderError("Ollama returned an invalid embedding response.")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def generate(self, prompt: str) -> str:
        response = self._request("generate", {"model": self.generation_model, "prompt": prompt, "stream": False})
        text = response.get("response")
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Ollama returned no generated text.")
        return text


class LocalRAG:
    """Index and retrieve documents from a persistent ChromaDB collection."""

    def __init__(
        self,
        config: L4Config | None = None,
        *,
        api_key: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        generation_provider: GenerationProvider | None = None,
        collection: Any = None,
    ):
        self.config = config or L4Config()
        self._embedding_provider = embedding_provider or self._make_provider(api_key, "embedding")
        self._generation_provider = generation_provider or self._make_provider(api_key, "generation")
        self._collection = collection or self._open_collection()

    def _make_provider(self, api_key: str | None, purpose: str) -> Any:
        provider = self.config.embedding_provider if purpose == "embedding" else self.config.generation_provider
        if provider == "google":
            return GoogleProvider(api_key, self.config.embedding_model, self.config.generation_model)
        if provider == "ollama":
            return OllamaProvider(self.config.embedding_model, self.config.generation_model)
        raise ProviderError(f"Unsupported {purpose} provider: {provider}")

    def _open_collection(self) -> Any:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.config.persist_directory)
            collection = client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            if (collection.metadata or {}).get("hnsw:space") != "cosine":
                raise RetrievalError(
                    f"Collection {self.config.collection_name!r} is not configured for cosine similarity."
                )
            return collection
        except ImportError as exc:
            raise RetrievalError("Install chromadb to use the L4 vector store.") from exc
        except Exception as exc:
            raise RetrievalError(f"Could not initialize ChromaDB: {exc}") from exc

    def chunk_and_insert(self, file_path: str, metadata: dict[str, Any] | None = None) -> list[str]:
        path = Path(file_path)
        source_pages = _load_source(path)
        base_metadata = dict(metadata or {})
        timestamp = datetime.now(UTC).isoformat()
        contents: list[str] = []
        records: list[dict[str, Any]] = []
        ids: list[str] = []
        for text, page_number in source_pages:
            for index, content in enumerate(_split_text(text, self.config.chunk_size, self.config.chunk_overlap)):
                chunk_id = hashlib.sha256(f"{path.resolve()}:{page_number}:{index}:{content}".encode()).hexdigest()[:24]
                chunk_metadata = {
                    **base_metadata,
                    "source_path": str(path.resolve()),
                    "chunk_id": chunk_id,
                    "created_at": timestamp,
                }
                if page_number is not None:
                    chunk_metadata["page_number"] = page_number
                ids.append(chunk_id)
                contents.append(content)
                records.append(chunk_metadata)
        if not contents:
            raise SourceDocumentError(f"No indexable text found in {path}")
        try:
            embeddings = self._embedding_provider.embed_documents(contents)
            self._collection.upsert(ids=ids, documents=contents, metadatas=records, embeddings=embeddings)
            if hasattr(self._collection, "get") and hasattr(self._collection, "delete"):
                existing = self._collection.get(
                    where={"source_path": str(path.resolve())},
                    include=["metadatas"],
                )
                stale_ids = [
                    existing_id
                    for existing_id in existing.get("ids", [])
                    if existing_id not in ids
                ]
                if stale_ids:
                    self._collection.delete(ids=stale_ids)
        except L4Error:
            raise
        except Exception as exc:
            raise RetrievalError(f"Could not insert chunks for {path}: {exc}") from exc
        return ids

    def query(self, user_query: str, top_k: int = 5) -> list[ChunkResult]:
        if not user_query or not user_query.strip():
            raise ValueError("user_query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        try:
            result = self._collection.query(
                query_embeddings=[self._embedding_provider.embed_query(user_query.strip())],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            return [
                ChunkResult(str(document), _cosine_similarity(distance), dict(metadata or {}))
                for document, metadata, distance in zip(documents, metadatas, distances)
            ]
        except L4Error:
            raise
        except Exception as exc:
            raise RetrievalError(f"Could not query ChromaDB: {exc}") from exc

    def get_chunks_above_threshold(self, user_query: str, threshold: float) -> list[ChunkResult]:
        return self.get_chunks_above_threshold_with_limit(user_query, threshold)

    def get_chunks_above_threshold_with_limit(
        self, user_query: str, threshold: float, top_k: int = 5
    ) -> list[ChunkResult]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        return [result for result in self.query(user_query, top_k) if result.score >= threshold]

    @staticmethod
    def _select_balanced(
        results: Sequence[ChunkResult], max_chunks_per_source: int, max_total_chunks: int
    ) -> list[ChunkResult]:
        if max_chunks_per_source <= 0 or max_total_chunks <= 0:
            raise ValueError("source and total chunk limits must be positive")
        groups: dict[str, list[ChunkResult]] = {}
        for result in results:
            source = str(result.metadata.get("source_path", "unknown"))
            groups.setdefault(source, []).append(result)
        for group in groups.values():
            group.sort(key=lambda item: item.score, reverse=True)
        selected: list[ChunkResult] = []
        for offset in range(max_chunks_per_source):
            for source in sorted(groups):
                group = groups[source]
                if offset < len(group) and len(selected) < max_total_chunks:
                    selected.append(group[offset])
            if len(selected) >= max_total_chunks:
                break
        return selected

    def query_cross_document(
        self,
        user_query: str,
        candidate_k: int = 20,
        max_chunks_per_source: int = 3,
        max_total_chunks: int = 8,
        threshold: float | None = None,
    ) -> list[ChunkResult]:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        candidates = self.query(user_query, candidate_k)
        if threshold is not None:
            candidates = [result for result in candidates if result.score >= threshold]
        return self._select_balanced(candidates, max_chunks_per_source, max_total_chunks)

    def generate(
        self,
        user_query: str,
        top_k: int = 5,
        threshold: float | None = None,
        cross_document: bool = True,
        candidate_k: int = 20,
        max_chunks_per_source: int = 3,
        max_total_chunks: int = 8,
    ) -> str:
        if cross_document:
            results = self.query_cross_document(
                user_query,
                candidate_k=candidate_k,
                max_chunks_per_source=max_chunks_per_source,
                max_total_chunks=max_total_chunks,
                threshold=threshold,
            )
        elif threshold is not None:
            results = self.get_chunks_above_threshold_with_limit(user_query, threshold, top_k)
        else:
            results = self.query(user_query, top_k)
        context = "\n\n".join(
            f"SOURCE: {item.metadata.get('source_path', 'unknown')}\n"
            f"PAGE: {item.metadata.get('page_number', 'n/a')}\n{item.content}"
            for item in results
        )
        if not context:
            context = "(No relevant source chunks were retrieved.)"
        prompt = (
            "Answer the question using only the supplied context. If the context does not contain "
            "the answer, say that the available documents do not provide enough information.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION:\n{user_query}"
        )
        if self._generation_provider is None:
            raise ProviderError("No generation provider is configured.")
        return self._generation_provider.generate(prompt)


__all__ = ["ChunkResult", "L4Config", "LocalRAG", "L4Error", "ProviderError", "RetrievalError", "SourceDocumentError"]
