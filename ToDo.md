# PRISM Project Roadmap & Integration Tracking

> [!IMPORTANT]
> **NOTE FOR AI AGENTS & DEVELOPERS:**
> This file is **NOT** a prompt or an execution script for autonomous agents to automatically parse and act upon.
> It serves exclusively as a **human-readable visual tracker and integration checklist** to document how subsystems work together and track architectural roadmap items.

---

## 📋 Architectural Roadmap & TODOs

### 1. CardWriter Background Thread Offloading
- [ ] **Asynchronous / Background Topic Card Generation**
  - **Current State**: When an active topic closes (`ContextManager._close_current_topic()`), `card_writer.write(topic.card, new_pages)` is called **synchronously**. For LLM-based card writers (`GeminiCardWriter` / `LLMCardWriter`), this blocks the thread for ~1–2 seconds while waiting for the Gemini API response.
  - **Target Implementation**:
    - Offload card generation to a background worker thread (`concurrent.futures.ThreadPoolExecutor` or `threading.Thread`) upon topic close.
    - Allow the current user turn and reply generation to complete immediately with `0ms` blocking overhead.
    - Attach the newly generated `TopicCard` to `topic.card` atomically upon background thread completion.
    - Preserve synchronous execution mode (`blocking=True`) for deterministic unit testing (`tests/fulltest.py`).

---

## 🧩 Subsystem Integration Status

| Component | Status | Description | Tests |
| :--- | :---: | :--- | :---: |
| **L1 Working Memory** | ✅ Done | Bounded LRU-style token buffer with high/low watermarks | `U1–U3`, `P1–P6` |
| **L2 Archival Storage** | ✅ Done | Unbounded in-memory cold storage for evicted pages | `U4` |
| **PageTable & LocationBook** | ✅ Done | Bidirectional tracking of pages across L1 and L2 | `U5`, `U6` |
| **Laya System-One Decider** | ✅ Done | Fast heuristic topic router & segmenter using ONNX/Laya | `L1–L5` |
| **Gemini Structured Output** | ✅ Done | `GeminiCardWriter` using Google GenAI SDK & Pydantic schema | Verified in `dummy_test.py` |
| **Unit Test Suite** | ✅ Done | 47 tests verifying invariants, watermarks, paging, & recovery | **47/47 Passing** |
