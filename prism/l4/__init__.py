"""Standalone Level 4 document retrieval and generation."""

from .rag import (
    ChunkResult,
    L4Config,
    L4Error,
    LocalRAG,
    ProviderError,
    RetrievalError,
    SourceDocumentError,
)

__all__ = [
    "ChunkResult",
    "L4Config",
    "L4Error",
    "LocalRAG",
    "ProviderError",
    "RetrievalError",
    "SourceDocumentError",
]
