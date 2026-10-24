"""Card writer powered by Gemini (or another small LLM) using Structured Outputs."""
from __future__ import annotations

import json
import os
from typing import Any, List

from pydantic import BaseModel, Field

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


class TopicCardOutput(BaseModel):
    label: str = Field(description="8 words or fewer naming the active topic.")
    description: str = Field(description="2-3 sentences stating ONLY the currently valid, agreed-upon factual state of this topic.")
    key_facts: List[str] = Field(description="Concrete values only (names, IDs, numbers, file paths, finalized decisions, preferences), one per item.")
    entities: List[str] = Field(description="Key proper nouns and specific identifiers that remain currently relevant.")


def _clean_json_text(text: str) -> str:
    """Strips markdown code fences (```json ... ```) if returned by the LLM."""
    s = text.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
            if s.startswith("json"):
                s = s[4:]
            s = s.rstrip("`").strip()
    return s


class LLMCardWriter:
    """Card writer powered by an LLM (Gemini 3.8 Flash / Claude Haiku or similar).
    
    Uses Gemini Structured Outputs via Google GenAI SDK when api_key or GEMINI_API_KEY is available.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        generation_config: dict[str, Any] | None = None,
        auto_load_env: bool = False,
    ):
        if auto_load_env and api_key is None and "GEMINI_API_KEY" not in os.environ:
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)
            except ImportError:
                pass

        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = client
        self.generation_config = generation_config or {
            "max_output_tokens": 4096,
            "thinking_level": "low",
        }

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise NotImplementedError(
                "Connect this to a small LLM or set GEMINI_API_KEY. It runs when a topic closes."
            )
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except ImportError as exc:
            raise NotImplementedError(
                "google-genai package is required. Install with `uv add google-genai`."
            ) from exc

    def write(self, old_card: TopicCard | None, pages: list[PageBlock]) -> TopicCard:
        if not pages:
            return old_card if old_card is not None else TopicCard(label="Empty topic")
        old = json.dumps(old_card.__dict__) if old_card else "(none)"
        new = "\n".join(f"{m.role}: {m.content}" for p in pages for m in p.data)
        prompt = f"{CARD_PROMPT}\nPREVIOUS CARD:\n{old}\n\nNEW PAGES:\n{new}"
        raw = self._call_llm(prompt)
        cleaned = _clean_json_text(raw)
        data = json.loads(cleaned)
        return TopicCard(
            label=data["label"],
            description=data["description"],
            key_facts=data["key_facts"],
            entities=data["entities"],
            covered_through=max(p.seq for p in pages),
        )

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        interaction = client.interactions.create(
            model=self.model,
            input=prompt,
            system_instruction=CARD_PROMPT,
            generation_config=self.generation_config,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": TopicCardOutput.model_json_schema(),
            },
        )
        return interaction.output_text


class GeminiCardWriter(LLMCardWriter):
    """Gemini-specialized CardWriter that automatically loads .env if available."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        generation_config: dict[str, Any] | None = None,
        auto_load_env: bool = True,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            client=client,
            generation_config=generation_config,
            auto_load_env=auto_load_env,
        )


__all__ = [
    "CARD_PROMPT",
    "TopicCardOutput",
    "LLMCardWriter",
    "GeminiCardWriter",
]
