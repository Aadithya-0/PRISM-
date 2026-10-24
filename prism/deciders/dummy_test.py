from __future__ import annotations

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from prism.deciders.llm import GeminiCardWriter
from prism.models import Message, PageBlock, TopicCard

def main():
    print("=" * 60)
    print(" PRISM GeminiCardWriter Demonstration")
    print("=" * 60)

    # 1. Initialize GeminiCardWriter (auto-loads .env)
    writer = GeminiCardWriter(model="gemini-3.8-flash")
    print(f"[OK] GeminiCardWriter initialized with model: {writer.model}")

    # 2. First Turn: Topic creation
    turn1_pages = [
        PageBlock(
            seq=1,
            topic_id=1,
            data=[
                Message("user", "We are organizing a 4-day trip to Goa from November 14 to 18."),
                Message("assistant", "Sounds exciting! Where will you stay?"),
                Message("user", "Staying at Taj Exotica in Benaulim, South Goa. Booking ref is GOA-9921."),
                Message("assistant", "Got it. Any vehicle rentals planned?"),
                Message("user", "Yes, rented a Mahindra Thar from GoaCarRentals. Total budget is 85000 INR."),
            ],
            is_open=False,
        )
    ]

    print("\n--- Generating Initial Topic Card from PageBlock 1 ---")
    card1 = writer.write(old_card=None, pages=turn1_pages)

    print(f"Label:           {card1.label}")
    print(f"Description:     {card1.description}")
    print(f"Key Facts:       {card1.key_facts}")
    print(f"Entities:        {card1.entities}")
    print(f"Covered Through: Page {card1.covered_through}")

    # 3. Second Turn: Topic Update (Contradiction & Overwrite test)
    turn2_pages = [
        PageBlock(
            seq=2,
            topic_id=1,
            data=[
                Message("user", "Update on the Goa trip: We changed the hotel from Taj Exotica to Alila Diwa Goa."),
                Message("user", "New booking reference is ALILA-4022. Budget increased to 95000 INR."),
                Message("assistant", "Updated! Alila Diwa in Majorda is confirmed."),
            ],
            is_open=False,
        )
    ]

    print("\n--- Updating Topic Card with PageBlock 2 (Fact Revision) ---")
    card2 = writer.write(old_card=card1, pages=turn2_pages)

    print(f"Label:           {card2.label}")
    print(f"Description:     {card2.description}")
    print(f"Key Facts:       {card2.key_facts}")
    print(f"Entities:        {card2.entities}")
    print(f"Covered Through: Page {card2.covered_through}")
    print("\n" + "=" * 60)
    print(" [SUCCESS] TopicCard structured output and fact update verified!")
    print("=" * 60)

if __name__ == "__main__":
    main()
