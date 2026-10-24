"""Card writer powered by an LLM (Claude Haiku or similar small model)."""
from __future__ import annotations

import json

from ..models import PageBlock, TopicCard

CARD_PROMPT = """You maintain an index card for one conversation topic.
You get the previous card (may be empty) and the new pages since it was written.
Return ONLY JSON: {"label": str, "description": str, "key_facts": [str], "entities": [str]}

Rules:
1. label: 8 words or fewer naming the active topic.
2. description: 2-3 sentences stating ONLY the currently valid, agreed-upon factual state of this topic.
   - Ground-truth snapshot only: State what is currently true, active, and decided right now.
   - NO chronological storytelling or changelogs: NEVER write statements describing past revision history (e.g. do NOT say "user initially wanted X but later adjusted to Y", "user changed mind", or "decided against Z").
   - Drop obsolete or superseded context completely so downstream semantic routing is never misled by outdated details.
3. key_facts: Concrete values only (names, IDs, numbers, file paths, finalized decisions, preferences), one per item.
4. Overwrite contradicted facts: If new information contradicts or updates an old fact, replace the old one completely. Never keep both past and present versions.
5. Never drop a still-valid fact to save space. Never invent anything not stated in the pages.
6. entities: Key proper nouns and specific identifiers that remain currently relevant.
7. Keep the whole card concise (under ~150 tokens).
"""


class LLMCardWriter:
    """PLACEHOLDER. Implement `_call_llm` with a small fast model (e.g. Claude Haiku)."""

    def write(self, old_card: TopicCard | None, pages: list[PageBlock]) -> TopicCard:
        if not pages:
            return old_card if old_card is not None else TopicCard(label="Empty topic")
        old = json.dumps(old_card.__dict__) if old_card else "(none)"
        new = "\n".join(f"{m.role}: {m.content}" for p in pages for m in p.data)
        raw = self._call_llm(f"{CARD_PROMPT}\nPREVIOUS CARD:\n{old}\n\nNEW PAGES:\n{new}")
        data = json.loads(raw)
        return TopicCard(
            label=data["label"],
            description=data["description"],
            key_facts=data["key_facts"],
            entities=data["entities"],
            covered_through=max(p.seq for p in pages),
        )

    def _call_llm(self, prompt: str) -> str:
        raise NotImplementedError("Connect this to a small LLM. It runs when a topic closes.")
