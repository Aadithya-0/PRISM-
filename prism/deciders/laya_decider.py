"""Laya-based deciders for fast System One decisions (convaiinnovations/laya).

Drop-in replacements for JevSegmenter and JevRouter. Uses the Laya SDK's
Router for non-autoregressive, calibrated-probability decisions in ~33 ms.

Install:  pip install laya
Docs:     https://nandhakishorm.github.io/laya/
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ..models import Message, Topic, TopicCard
from .interfaces import SegmentDecision

logger = logging.getLogger(__name__)

# Ensure Hugging Face Hub uses cached checkpoints offline without re-checking commits
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ---------------------------------------------------------------------------
# Recycled singleton router: avoid reloading checkpoints on every instantiation
# ---------------------------------------------------------------------------
_laya_router = None


def get_shared_router(preload: bool = True, **kwargs: Any) -> Any:
    """Return the shared, recycled laya.Router instance.

    Reuses model weights across all PRISM decider calls and tests,
    avoiding checkpoint reload latency on every turn or instantiation.
    """
    global _laya_router
    if _laya_router is None:
        try:
            from laya import Router as LayaSDKRouter
        except ImportError as exc:
            raise ImportError(
                "The 'laya' package is required for LayaSegmenter / LayaRouter. "
                "Install it with:  pip install laya"
            ) from exc
        _laya_router = LayaSDKRouter(preload=preload, **kwargs)
        logger.info("Laya Router initialised (recycled singleton instance)")
    return _laya_router


# Backward-compatible internal alias for tests and existing callers
_get_laya_router = get_shared_router


def reset_shared_router() -> None:
    """Tear down the shared router (useful in tests)."""
    global _laya_router
    if _laya_router is not None:
        try:
            _laya_router.unload()
        except Exception:
            pass
    _laya_router = None


def __getattr__(name: str) -> Any:
    """Allow convenient imports: from prism.deciders.laya_decider import router, shared_router."""
    if name in ("router", "shared_router"):
        return get_shared_router()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# ---------------------------------------------------------------------------
# Helpers for effective context formatting
# ---------------------------------------------------------------------------

def format_topic_card(card: TopicCard) -> str:
    """Build an effective natural summary of a topic card for Laya.

    Presents the topic label, description/summary, key facts, and named entities
    cleanly so the Laya classification head attends to specific context.
    """
    parts: list[str] = [card.label]
    if card.description:
        parts.append(f"Summary: {card.description}")
    if card.key_facts:
        parts.append("Key facts: " + "; ".join(card.key_facts))
    if card.entities:
        parts.append("Entities: " + ", ".join(card.entities))
    return " | ".join(parts)


def _conversation_text(recent: list[Message], new_msg: Message) -> str:
    """Build a single text state from recent messages + the new user message."""
    lines = [f"{m.role}: {m.content}" for m in recent]
    lines.append(f"NEWEST user message: {new_msg.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LayaSegmenter  (replaces JevSegmenter)
# ---------------------------------------------------------------------------

class LayaSegmenter:
    """Segment using one Laya `choice` question:
    'continue current topic / start new / return to an earlier topic.'

    The Laya SDK answers all questions in a single forward pass (~33 ms on GPU).
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        return_threshold: float = 0.8,
        model: str | None = None,
        router_kwargs: dict | None = None,
    ):
        self.min_confidence = min_confidence
        self.return_threshold = return_threshold
        self.model = model                         # e.g. "english", "multilingual"
        self._router_kwargs = router_kwargs or {}

    # -- public interface (matches Segmenter protocol) -----------------------

    def decide(
        self,
        open_topic: Topic | None,
        closed_topics: list[Topic],
        recent: list[Message],
        new_msg: Message,
    ) -> SegmentDecision:
        if open_topic is None:
            return SegmentDecision("new")

        # Build the choice criteria: continue / new / return:<id>
        criteria: dict[str, str] = {
            "continue": f"Continues the current topic: {format_topic_card(open_topic.card)}",
            "new": "Starts a completely new topic",
        }
        for t in closed_topics:
            criteria[f"return:{t.id}"] = f"Returns to an earlier topic: {format_topic_card(t.card)}"

        questions = {
            "segment": {
                "type": "choice",
                "instructions": "Which best describes the newest user message?",
                "criteria": criteria,
            }
        }

        state = _conversation_text(recent, new_msg)
        router_inst = _get_laya_router(**self._router_kwargs)

        predict_kwargs: dict[str, Any] = {}
        if self.model is not None:
            predict_kwargs["model"] = self.model

        result = router_inst.predict(state, questions, **predict_kwargs)
        seg = result["answers"]["segment"]

        choice = seg["choice"]
        # Laya returns probabilities per option under the key "probabilities"
        probabilities: dict[str, float] = seg.get("probabilities", {})
        confidence = seg.get("confidence", probabilities.get(choice, 0.0))

        if confidence < self.min_confidence:
            return SegmentDecision("continue", confidence=confidence)

        if choice.startswith("return:"):
            p = probabilities.get(choice, 0.0)
            if p < self.return_threshold:
                return SegmentDecision("new", confidence=p)
            topic_id = int(choice.split(":")[1])
            return SegmentDecision("return", topic_id, p)

        return SegmentDecision(choice, confidence=confidence)


# ---------------------------------------------------------------------------
# LayaRouter  (replaces JevRouter)
# ---------------------------------------------------------------------------

class LayaRouter:
    """Score each closed topic with a Laya `noul` question:
    'Does the newest message need information from this topic?'

    All topics are batched into one predict() call — answered in a single
    forward pass regardless of count.
    """

    def __init__(
        self,
        model: str | None = None,
        router_kwargs: dict | None = None,
    ):
        self.model = model
        self._router_kwargs = router_kwargs or {}

    # -- public interface (matches Router protocol) --------------------------

    def score(
        self,
        topics: list[Topic],
        recent: list[Message],
        new_msg: Message,
    ) -> dict[int, float]:
        if not topics:
            return {}

        # Build one noul question per topic — all answered in one forward pass
        questions = {
            f"topic_{t.id}": {
                "type": "noul",
                "instructions": (
                    "The newest user message needs information from this earlier "
                    f"topic: {format_topic_card(t.card)}"
                ),
            }
            for t in topics
        }

        state = _conversation_text(recent, new_msg)
        router_inst = _get_laya_router(**self._router_kwargs)

        predict_kwargs: dict[str, Any] = {}
        if self.model is not None:
            predict_kwargs["model"] = self.model

        result = router_inst.predict(state, questions, **predict_kwargs)
        answers = result["answers"]

        return {
            t.id: answers[f"topic_{t.id}"]["noul"]
            for t in topics
        }


__all__ = [
    "LayaSegmenter",
    "LayaRouter",
    "SegmentDecision",
    "format_topic_card",
    "get_shared_router",
    "reset_shared_router",
    "router",
    "shared_router",
]
