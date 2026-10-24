import os
from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

generation_config = {
    'max_output_tokens': 65536,
    'thinking_level': 'low',
}

from pydantic import BaseModel, Field
from typing import List

class TopicCardOutput(BaseModel):
    label: str = Field(description="8 words or fewer naming the active topic.")
    description: str = Field(description="2-3 sentences stating ONLY the currently valid, agreed-upon factual state of this topic.")
    key_facts: List[str] = Field(description="Concrete values only (names, IDs, numbers, file paths, finalized decisions, preferences), one per item.")
    entities: List[str] = Field(description="Key proper nouns and specific identifiers that remain currently relevant.")

sample_input = """PREVIOUS CARD:
(none)

NEW PAGES:
User: We are organizing a 4-day trip to Goa from November 14 to 18.
Assistant: Sounds exciting! Where will you stay?
User: Staying at Taj Exotica in Benaulim, South Goa. Booking ref is GOA-9921.
Assistant: Got it. Any vehicle rentals planned?
User: Yes, rented a Mahindra Thar from GoaCarRentals. Total budget is 85000 INR.
"""

interaction = client.interactions.create(
    model='gemini-2.5-flash',
    input=sample_input,
    system_instruction="""You maintain an index card for one conversation topic.
You get the previous card (may be empty) and the new pages since it was written.
Return ONLY JSON: {\"label\": str, \"description\": str, \"key_facts\": [str], \"entities\": [str]}

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
7. Keep the whole card concise (under ~150 tokens).""",
    generation_config=generation_config,
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": TopicCardOutput.model_json_schema(),
    },
)

raw_text = interaction.output_text.strip()
if raw_text.startswith("```"):
    parts = raw_text.split("```")
    if len(parts) >= 2:
        raw_text = parts[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.rstrip("`").strip()

card = TopicCardOutput.model_validate_json(raw_text)
print("=== PARSED TOPIC CARD ===")
print("Label:      ", card.label)
print("Description:", card.description)
print("Key Facts:  ", card.key_facts)
print("Entities:   ", card.entities)


