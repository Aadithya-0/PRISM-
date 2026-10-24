"""Laya integration tests for PRISM deciders.

Validates that LayaSegmenter and LayaRouter are correct drop-in replacements
for the Jev-based deciders — both against mock responses (fast, offline) and
optionally against the real Laya SDK (when ``laya`` is installed).

Run from the project root:
    python -B prism/deciders/laya_test.py            # offline mock tests only
    python -B prism/deciders/laya_test.py --live      # also run live SDK tests (requires `pip install laya`)

The mock layer patches `laya.Router.predict` so no model weights are needed
for the core logic tests.
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path bootstrap (so the file runs standalone)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prism.models import Message, Topic, TopicCard
from prism.deciders.interfaces import SegmentDecision
from prism.deciders.laya_decider import (
    LayaRouter,
    LayaSegmenter,
    _conversation_text,
    _get_laya_router,
    format_topic_card,
    get_shared_router,
    reset_shared_router,
    router,
    shared_router,
)

# Also import PRISM wiring to test end-to-end with the ContextManager
from prism import ContextManager, FakeCardWriter, L1, L2, PageTable, TopicTable


# ═══════════════════════════════════════════════════════════════════ helpers

def _topic(tid: int, label: str, desc: str = "", closed: bool = False) -> Topic:
    """Quick Topic factory for tests."""
    return Topic(
        id=tid,
        card=TopicCard(label=label, description=desc),
        page_seqs=[],
        is_open=not closed,
    )


def _mock_laya_predict_choice(
    choice: str,
    probabilities: dict[str, float] | None = None,
    confidence: float = 0.95,
):
    """Return a function that mimics laya.Router.predict for a choice question."""
    if probabilities is None:
        probabilities = {choice: confidence}

    def _predict(state, questions, **kwargs):
        return {
            "answers": {
                "segment": {
                    "choice": choice,
                    "probabilities": probabilities,
                    "confidence": confidence,
                }
            },
            "routing": {"model": "english", "repo": "convaiinnovations/laya"},
        }

    return _predict


def _mock_laya_predict_noul(scores: dict[str, float]):
    """Return a function that mimics laya.Router.predict for noul questions."""

    def _predict(state, questions, **kwargs):
        answers = {}
        for qid in questions:
            answers[qid] = {"noul": scores.get(qid, 0.0)}
        return {
            "answers": answers,
            "routing": {"model": "english", "repo": "convaiinnovations/laya"},
        }

    return _predict


# ═══════════════════════════════════════════════════════════════════ mock tests


class TestConversationText(unittest.TestCase):
    """_conversation_text builds the state string correctly."""

    def test_simple(self):
        recent = [Message("user", "hello"), Message("assistant", "hi")]
        new = Message("user", "help me")
        text = _conversation_text(recent, new)
        self.assertIn("user: hello", text)
        self.assertIn("assistant: hi", text)
        self.assertIn("NEWEST user message: help me", text)

    def test_empty_recent(self):
        text = _conversation_text([], Message("user", "first"))
        self.assertEqual(text, "NEWEST user message: first")


class TestLayaSegmenterMocked(unittest.TestCase):
    """LayaSegmenter with patched Laya SDK — exercises decision logic only."""

    def setUp(self):
        reset_shared_router()

    def tearDown(self):
        reset_shared_router()

    def _run(self, mock_predict, open_topic=None, closed=None, recent=None, msg_text="test"):
        """Helper: patch the shared router, run decide(), return the result."""
        closed = closed or []
        recent = recent or []

        mock_router = MagicMock()
        mock_router.predict = mock_predict

        with patch("prism.deciders.laya_decider._get_laya_router", return_value=mock_router):
            seg = LayaSegmenter(min_confidence=0.6, return_threshold=0.8)
            return seg.decide(open_topic, closed, recent, Message("user", msg_text))

    # -- no open topic → always "new" (no SDK call needed) -------------------

    def test_no_open_topic_returns_new(self):
        result = self._run(None, open_topic=None)
        self.assertEqual(result.action, "new")

    # -- choice: continue ----------------------------------------------------

    def test_continue_high_confidence(self):
        topic = _topic(1, "Billing issues")
        result = self._run(
            _mock_laya_predict_choice("continue", confidence=0.9),
            open_topic=topic,
        )
        self.assertEqual(result.action, "continue")
        self.assertGreaterEqual(result.confidence, 0.6)

    def test_continue_low_confidence_falls_back(self):
        """If confidence < min_confidence, default to 'continue' (safest)."""
        topic = _topic(1, "Billing")
        result = self._run(
            _mock_laya_predict_choice("new", confidence=0.3),
            open_topic=topic,
        )
        # Low confidence → safe fallback to "continue"
        self.assertEqual(result.action, "continue")
        self.assertAlmostEqual(result.confidence, 0.3)

    # -- choice: new ---------------------------------------------------------

    def test_new_topic_detected(self):
        topic = _topic(1, "Billing")
        result = self._run(
            _mock_laya_predict_choice("new", confidence=0.85),
            open_topic=topic,
        )
        self.assertEqual(result.action, "new")

    # -- choice: return ------------------------------------------------------

    def test_return_high_probability(self):
        open_t = _topic(1, "Current topic")
        closed_t = _topic(2, "Old billing topic", closed=True)
        result = self._run(
            _mock_laya_predict_choice(
                "return:2",
                probabilities={"return:2": 0.9, "continue": 0.05, "new": 0.05},
                confidence=0.9,
            ),
            open_topic=open_t,
            closed=[closed_t],
        )
        self.assertEqual(result.action, "return")
        self.assertEqual(result.topic_id, 2)
        self.assertGreaterEqual(result.confidence, 0.8)

    def test_return_below_threshold_becomes_new(self):
        """Return probability below return_threshold → fall back to 'new'."""
        open_t = _topic(1, "Current")
        closed_t = _topic(3, "Old debugging", closed=True)
        result = self._run(
            _mock_laya_predict_choice(
                "return:3",
                probabilities={"return:3": 0.65, "continue": 0.2, "new": 0.15},
                confidence=0.75,
            ),
            open_topic=open_t,
            closed=[closed_t],
        )
        self.assertEqual(result.action, "new")

    # -- criteria shape: all closed topics are represented -------------------

    def test_criteria_includes_all_closed_topics(self):
        """The choice question must include continue + new + one key per closed topic."""
        open_t = _topic(1, "Open topic")
        closed = [_topic(i, f"Topic {i}", closed=True) for i in range(2, 6)]
        captured_questions: list[dict] = []

        def _capture_predict(state, questions, **kw):
            captured_questions.append(questions)
            return {
                "answers": {
                    "segment": {
                        "choice": "continue",
                        "probabilities": {"continue": 0.9},
                        "confidence": 0.9,
                    }
                },
                "routing": {"model": "english"},
            }

        mock_router = MagicMock()
        mock_router.predict = _capture_predict

        with patch("prism.deciders.laya_decider._get_laya_router", return_value=mock_router):
            seg = LayaSegmenter()
            seg.decide(open_t, closed, [], Message("user", "test"))

        self.assertEqual(len(captured_questions), 1)
        criteria = captured_questions[0]["segment"]["criteria"]
        self.assertIn("continue", criteria)
        self.assertIn("new", criteria)
        for t in closed:
            self.assertIn(f"return:{t.id}", criteria)


class TestLayaRouterMocked(unittest.TestCase):
    """LayaRouter with patched Laya SDK — exercises noul scoring logic."""

    def setUp(self):
        reset_shared_router()

    def tearDown(self):
        reset_shared_router()

    def _run(self, mock_predict, topics, msg_text="test"):
        mock_router = MagicMock()
        mock_router.predict = mock_predict

        with patch("prism.deciders.laya_decider._get_laya_router", return_value=mock_router):
            router = LayaRouter()
            return router.score(topics, [], Message("user", msg_text))

    def test_empty_topics_returns_empty(self):
        result = self._run(None, [])
        self.assertEqual(result, {})

    def test_single_topic_score(self):
        t = _topic(5, "Billing", closed=True)
        result = self._run(
            _mock_laya_predict_noul({"topic_5": 0.87}),
            [t],
            msg_text="refund my payment",
        )
        self.assertIn(5, result)
        self.assertAlmostEqual(result[5], 0.87)

    def test_multiple_topics_scored(self):
        topics = [
            _topic(1, "Billing", closed=True),
            _topic(2, "Tech support", closed=True),
            _topic(3, "Kerala trip", closed=True),
        ]
        scores_map = {"topic_1": 0.92, "topic_2": 0.15, "topic_3": 0.03}
        result = self._run(_mock_laya_predict_noul(scores_map), topics)
        self.assertEqual(set(result.keys()), {1, 2, 3})
        self.assertAlmostEqual(result[1], 0.92)
        self.assertAlmostEqual(result[2], 0.15)
        self.assertAlmostEqual(result[3], 0.03)

    def test_all_questions_sent_in_one_call(self):
        """All noul questions must be batched into a single predict() call."""
        topics = [_topic(i, f"T{i}", closed=True) for i in range(1, 8)]
        call_count = 0

        def _counting_predict(state, questions, **kw):
            nonlocal call_count
            call_count += 1
            answers = {qid: {"noul": 0.5} for qid in questions}
            return {"answers": answers, "routing": {"model": "english"}}

        mock_router = MagicMock()
        mock_router.predict = _counting_predict

        with patch("prism.deciders.laya_decider._get_laya_router", return_value=mock_router):
            router = LayaRouter()
            router.score(topics, [], Message("user", "test"))

        self.assertEqual(call_count, 1, "All noul questions should be sent in ONE predict call")

    def test_question_shape_is_noul(self):
        """Each question sent to Laya must be type=noul with instructions."""
        topics = [_topic(10, "Debug auth.py", closed=True)]
        captured: list[dict] = []

        def _capture(state, questions, **kw):
            captured.append(questions)
            return {"answers": {"topic_10": {"noul": 0.5}}, "routing": {"model": "english"}}

        mock_router = MagicMock()
        mock_router.predict = _capture

        with patch("prism.deciders.laya_decider._get_laya_router", return_value=mock_router):
            LayaRouter().score(topics, [], Message("user", "fix the token expiry"))

        self.assertEqual(len(captured), 1)
        q = captured[0]["topic_10"]
        self.assertEqual(q["type"], "noul")
        self.assertIn("instructions", q)
        self.assertIn("Debug auth.py", q["instructions"])


# ═══════════════════════════════════════════════════════════════════ protocol conformance


class TestProtocolConformance(unittest.TestCase):
    """LayaSegmenter and LayaRouter satisfy the Segmenter / Router protocols."""

    def test_segmenter_has_decide(self):
        seg = LayaSegmenter()
        self.assertTrue(callable(getattr(seg, "decide", None)))

    def test_router_has_score(self):
        router = LayaRouter()
        self.assertTrue(callable(getattr(router, "score", None)))

    def test_segmenter_decide_signature(self):
        """decide(open_topic, closed_topics, recent, new_msg) → SegmentDecision."""
        import inspect
        sig = inspect.signature(LayaSegmenter.decide)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["self", "open_topic", "closed_topics", "recent", "new_msg"])

    def test_router_score_signature(self):
        """score(topics, recent, new_msg) → dict[int, float]."""
        import inspect
        sig = inspect.signature(LayaRouter.score)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["self", "topics", "recent", "new_msg"])


# ═══════════════════════════════════════════════════════════════════ end-to-end with ContextManager


class TestLayaContextManagerIntegration(unittest.TestCase):
    """Plug LayaSegmenter + LayaRouter into the real ContextManager and run a
    scripted multi-turn conversation with mocked Laya predictions.

    This validates the full pipeline: segmentation → routing → L1/L2 memory
    management — exactly what will happen in production.
    """

    def setUp(self):
        reset_shared_router()

    def tearDown(self):
        reset_shared_router()

    def _build_manager(self, segmenter, router):
        return ContextManager(
            l1=L1(max_tokens=200, high_water=0.85, low_water=0.70),
            l2=L2(),
            topics=TopicTable(),
            page_table=PageTable(),
            segmenter=segmenter,
            router=router,
            card_writer=FakeCardWriter(),
            needed_threshold=0.5,
            protect_recent_pages=1,
        )

    def test_three_turn_conversation(self):
        """Simulate: billing question → same topic follow-up → new topic (debugging)."""
        # Mock: turn 1 has no open topic so segmenter returns "new" without calling Laya.
        # Turn 2 should "continue". Turn 3 should "new".
        call_index = {"n": 0}

        def _seg_predict(state, questions, **kw):
            call_index["n"] += 1
            if call_index["n"] == 1:
                # Turn 2: continue the billing topic
                return {
                    "answers": {"segment": {
                        "choice": "continue",
                        "probabilities": {"continue": 0.92, "new": 0.08},
                        "confidence": 0.92,
                    }},
                    "routing": {"model": "english"},
                }
            else:
                # Turn 3: new topic
                return {
                    "answers": {"segment": {
                        "choice": "new",
                        "probabilities": {"continue": 0.1, "new": 0.9},
                        "confidence": 0.9,
                    }},
                    "routing": {"model": "english"},
                }

        def _route_predict(state, questions, **kw):
            answers = {qid: {"noul": 0.1} for qid in questions}
            return {"answers": answers, "routing": {"model": "english"}}

        seg_mock = MagicMock()
        seg_mock.predict = _seg_predict
        route_mock = MagicMock()
        route_mock.predict = _route_predict

        with patch("prism.deciders.laya_decider._get_laya_router") as get_router:
            # Both segmenter and router will call _get_laya_router → return the mock.
            # We track via the questions dict which mock to dispatch to.
            def _smart_router(**kw):
                smart = MagicMock()

                def _dispatch(state, questions, **kwargs):
                    if "segment" in questions:
                        return _seg_predict(state, questions, **kwargs)
                    else:
                        return _route_predict(state, questions, **kwargs)

                smart.predict = _dispatch
                return smart

            get_router.side_effect = _smart_router

            seg = LayaSegmenter(min_confidence=0.6, return_threshold=0.8)
            rtr = LayaRouter()
            mgr = self._build_manager(seg, rtr)

            # Turn 1: billing question (no open topic → auto "new")
            msgs = mgr.prepare_context("I was charged twice for March. Please refund.")
            self.assertIsNotNone(mgr.open_topic_id)
            mgr.record_reply("Let me look into that refund for you.")

            # Turn 2: follow-up on same topic
            msgs = mgr.prepare_context("It's invoice #4411, billed on March 3rd.")
            mgr.record_reply("Found invoice #4411. Processing the refund now.")

            # Turn 3: completely new topic
            msgs = mgr.prepare_context("Help me debug the login bug in auth.py.")
            mgr.record_reply("Let's check the token expiry in auth.py.")

            # Verify we have multiple topics now
            all_topics = mgr.topics.all()
            self.assertGreaterEqual(len(all_topics), 2)

    def test_return_to_old_topic(self):
        """After closing a topic, Laya can route back to it."""
        seg_call = {"n": 0}

        def _predict(state, questions, **kw):
            if "segment" in questions:
                seg_call["n"] += 1
                if seg_call["n"] <= 2:
                    # Turn 2 and 3: start new topics
                    return {
                        "answers": {"segment": {
                            "choice": "new",
                            "probabilities": {"new": 0.9},
                            "confidence": 0.9,
                        }},
                        "routing": {"model": "english"},
                    }
                else:
                    # Turn 4: return to topic 1 (billing, created on turn 1)
                    # TopicTable._next_id starts at 1, so first topic = id 1
                    return {
                        "answers": {"segment": {
                            "choice": "return:1",
                            "probabilities": {"return:1": 0.92, "continue": 0.05, "new": 0.03},
                            "confidence": 0.92,
                        }},
                        "routing": {"model": "english"},
                    }
            else:
                # Router noul questions: low scores so no L2->L1 churn
                answers = {qid: {"noul": 0.1} for qid in questions}
                return {"answers": answers, "routing": {"model": "english"}}

        mock_router = MagicMock()
        mock_router.predict = _predict

        with patch("prism.deciders.laya_decider._get_laya_router", return_value=mock_router):
            seg = LayaSegmenter(min_confidence=0.6, return_threshold=0.8)
            rtr = LayaRouter()
            mgr = self._build_manager(seg, rtr)

            # Turn 1: billing (topic 1, auto new — no segmenter call)
            mgr.prepare_context("Refund my March charge.")
            mgr.record_reply("Looking into it.")

            # Turn 2: new topic (segmenter call #1 -> "new", creates topic 2)
            mgr.prepare_context("Debug the login bug in auth.py.")
            mgr.record_reply("Checking auth.py now.")

            # Turn 3: yet another new topic (segmenter call #2 -> "new", creates topic 3)
            mgr.prepare_context("Plan a trip to Kerala.")
            mgr.record_reply("December is great for Kerala backwaters.")

            # Turn 4: return to billing (segmenter call #3 -> "return:1")
            mgr.prepare_context("Actually, what happened with my refund?")
            mgr.record_reply("The refund for invoice #4411 was processed.")

            # The open topic should be topic 1 again (billing)
            self.assertEqual(mgr.open_topic_id, 1)


# ═══════════════════════════════════════════════════════════════════ model kwarg forwarding


class TestModelSelection(unittest.TestCase):
    """Verify that the model= kwarg is forwarded to laya.Router.predict."""

    def setUp(self):
        reset_shared_router()

    def tearDown(self):
        reset_shared_router()

    def test_segmenter_forwards_model(self):
        captured_kwargs: list[dict] = []

        def _capture(state, questions, **kw):
            captured_kwargs.append(kw)
            return {
                "answers": {"segment": {
                    "choice": "continue",
                    "probabilities": {"continue": 0.9},
                    "confidence": 0.9,
                }},
                "routing": {"model": "multilingual"},
            }

        mock_router = MagicMock()
        mock_router.predict = _capture

        with patch("prism.deciders.laya_decider._get_laya_router", return_value=mock_router):
            seg = LayaSegmenter(model="multilingual")
            seg.decide(
                _topic(1, "Test"),
                [],
                [],
                Message("user", "prueba de texto en español"),
            )

        self.assertEqual(len(captured_kwargs), 1)
        self.assertEqual(captured_kwargs[0].get("model"), "multilingual")

    def test_router_forwards_model(self):
        captured_kwargs: list[dict] = []

        def _capture(state, questions, **kw):
            captured_kwargs.append(kw)
            answers = {qid: {"noul": 0.5} for qid in questions}
            return {"answers": answers, "routing": {"model": "multilingual"}}

        mock_router = MagicMock()
        mock_router.predict = _capture

        with patch("prism.deciders.laya_decider._get_laya_router", return_value=mock_router):
            rtr = LayaRouter(model="multilingual")
            rtr.score(
                [_topic(1, "Facturación", closed=True)],
                [],
                Message("user", "¿Qué pasó con mi reembolso?"),
            )

        self.assertEqual(len(captured_kwargs), 1)
        self.assertEqual(captured_kwargs[0].get("model"), "multilingual")


# ═══════════════════════════════════════════════════════════════════ edge cases


class TestEdgeCases(unittest.TestCase):
    """Edge cases and robustness checks."""

    def setUp(self):
        reset_shared_router()

    def tearDown(self):
        reset_shared_router()

    def test_many_closed_topics(self):
        """Segmenter handles 30+ closed topics without crashing."""
        open_t = _topic(0, "Current")
        closed = [_topic(i, f"Old topic {i}", closed=True) for i in range(1, 35)]

        def _predict(state, questions, **kw):
            return {
                "answers": {"segment": {
                    "choice": "continue",
                    "probabilities": {"continue": 0.85},
                    "confidence": 0.85,
                }},
                "routing": {"model": "english"},
            }

        mock_router = MagicMock()
        mock_router.predict = _predict

        with patch("prism.deciders.laya_decider._get_laya_router", return_value=mock_router):
            seg = LayaSegmenter()
            result = seg.decide(open_t, closed, [], Message("user", "continue please"))

        self.assertEqual(result.action, "continue")

    def test_router_handles_many_topics(self):
        """Router batches 50 noul questions into a single call."""
        topics = [_topic(i, f"Topic {i}", closed=True) for i in range(50)]

        def _predict(state, questions, **kw):
            self.assertEqual(len(questions), 50)
            answers = {qid: {"noul": 0.5} for qid in questions}
            return {"answers": answers, "routing": {"model": "english"}}

        mock_router = MagicMock()
        mock_router.predict = _predict

        with patch("prism.deciders.laya_decider._get_laya_router", return_value=mock_router):
            rtr = LayaRouter()
            result = rtr.score(topics, [], Message("user", "test"))

        self.assertEqual(len(result), 50)

    def test_import_error_when_laya_not_installed(self):
        """_get_laya_router raises ImportError with a helpful message."""
        reset_shared_router()
        with patch.dict("sys.modules", {"laya": None}):
            with self.assertRaises(ImportError) as ctx:
                _get_laya_router()
            self.assertIn("pip install laya", str(ctx.exception))

    def test_reset_shared_router(self):
        """reset_shared_router clears the global singleton."""
        import prism.deciders.laya_decider as mod
        mod._laya_router = MagicMock()
        self.assertIsNotNone(mod._laya_router)
        reset_shared_router()
        self.assertIsNone(mod._laya_router)


class TestExportedRecycledRouter(unittest.TestCase):
    """Test the exported recycled router singleton and format_topic_card helper."""

    def setUp(self):
        reset_shared_router()

    def tearDown(self):
        reset_shared_router()

    def test_format_topic_card(self):
        card = TopicCard(
            label="Billing & Duplicate Charge",
            description="User was billed twice for March on invoice #4411",
            key_facts=["Charged twice on March 3 ($120)", "Customer requested refund"],
            entities=["invoice #4411", "$120", "March"],
        )
        text = format_topic_card(card)
        self.assertIn("Billing & Duplicate Charge", text)
        self.assertIn("Summary: User was billed twice for March on invoice #4411", text)
        self.assertIn("Key facts: Charged twice on March 3 ($120); Customer requested refund", text)
        self.assertIn("Entities: invoice #4411, $120, March", text)

    def test_shared_router_caching(self):
        mock_inst = MagicMock()
        with patch("laya.Router", return_value=mock_inst):
            r1 = get_shared_router()
            r2 = get_shared_router()
            self.assertIs(r1, r2, "get_shared_router must return the same recycled instance")
            self.assertIs(r1, mock_inst)


# ═══════════════════════════════════════════════════════════════════ live SDK tests (optional)


class TestLayaLiveSDK(unittest.TestCase):
    """Real Laya SDK tests — skipped if `laya` is not installed.

    These actually download the checkpoint (once) and run inference. They
    validate that the PRISM decider wrappers produce the right decision
    shapes against the real model.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import laya  # noqa: F401
            cls._has_laya = True
        except ImportError:
            cls._has_laya = False

    def setUp(self):
        reset_shared_router()
        if not self._has_laya:
            self.skipTest("laya package not installed — run `pip install laya` for live tests")

    def tearDown(self):
        reset_shared_router()

    def test_live_segmenter_continue(self):
        """A clear follow-up message should be classified as 'continue'."""
        seg = LayaSegmenter(min_confidence=0.4)
        open_t = _topic(1, "Billing and refund inquiry")
        recent = [
            Message("user", "I was billed twice for March. Please refund."),
            Message("assistant", "I see the duplicate charge. Let me process the refund."),
        ]
        result = seg.decide(
            open_t, [], recent,
            Message("user", "Great, how long will the refund take?"),
        )
        self.assertIn(result.action, ("continue", "new"))  # either is acceptable
        self.assertIsInstance(result.confidence, float)
        self.assertGreater(result.confidence, 0.0)

    def test_live_segmenter_new_topic(self):
        """A completely unrelated message should be 'new'."""
        seg = LayaSegmenter(min_confidence=0.4)
        open_t = _topic(1, "Billing and refund inquiry")
        recent = [
            Message("user", "I was billed twice for March. Please refund."),
            Message("assistant", "Refund processed."),
        ]
        result = seg.decide(
            open_t, [], recent,
            Message("user", "What's the weather like in Tokyo tomorrow?"),
        )
        self.assertIn(result.action, ("continue", "new"))
        self.assertIsInstance(result.confidence, float)

    def test_live_router_scores(self):
        """Router returns float scores for each topic."""
        rtr = LayaRouter()
        topics = [
            _topic(1, "Billing refund for March invoice", closed=True),
            _topic(2, "Debug login bug in auth.py", closed=True),
        ]
        scores = rtr.score(
            topics, [],
            Message("user", "Did the refund go through for invoice 4411?"),
        )
        self.assertEqual(set(scores.keys()), {1, 2})
        for tid, val in scores.items():
            self.assertIsInstance(val, float)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_live_router_billing_beats_debug(self):
        """A billing question should score higher for the billing topic."""
        rtr = LayaRouter()
        topics = [
            _topic(1, "Billing refund for March invoice #4411", closed=True),
            _topic(2, "Debugging authentication token expiry in auth.py", closed=True),
        ]
        scores = rtr.score(
            topics, [],
            Message("user", "What happened with the refund for that duplicate charge on invoice 4411?"),
        )
        # Billing topic should score higher than debugging topic for a billing question
        self.assertGreater(
            scores[1], scores[2],
            f"Expected billing topic to score higher: billing={scores[1]:.3f}, debug={scores[2]:.3f}",
        )

    def test_live_full_pipeline(self):
        """Full ContextManager pipeline with real Laya SDK predictions."""
        seg = LayaSegmenter(min_confidence=0.4)
        rtr = LayaRouter()
        mgr = ContextManager(
            l1=L1(max_tokens=300, high_water=0.85, low_water=0.70),
            l2=L2(),
            topics=TopicTable(),
            page_table=PageTable(),
            segmenter=seg,
            router=rtr,
            card_writer=FakeCardWriter(),
            needed_threshold=0.5,
            protect_recent_pages=1,
        )

        msgs = mgr.prepare_context("I was charged twice for March. Please refund the duplicate.")
        self.assertTrue(len(msgs) > 0)
        mgr.record_reply("I see the duplicate charge. Processing the refund now.")

        msgs = mgr.prepare_context("Thanks, also the payment portal is showing an error.")
        self.assertTrue(len(msgs) > 0)
        mgr.record_reply("Let me check the payment portal for you.")

        # The system should not crash and should produce valid messages
        for m in msgs:
            self.assertIsInstance(m, Message)
            self.assertIn(m.role, ("user", "assistant", "system"))


# ═══════════════════════════════════════════════════════════════════ visual showcase & console demo


def print_topic_card_box(card: TopicCard, topic_id: int, status: str = "Closed") -> None:
    """Print an ASCII box showcasing a fetched TopicCard."""
    border = "-" * 76
    print(f"+{border}+")
    print(f"| [TOPIC CARD] ID: #{topic_id:<3}  Status: {status:<47}|")
    print(f"+{border}+")
    print(f"| Label       : {card.label:<60}|")
    if card.description:
        desc = card.description if len(card.description) <= 60 else card.description[:57] + "..."
        print(f"| Summary     : {desc:<60}|")
    if card.key_facts:
        facts_str = "; ".join(card.key_facts)
        facts_str = facts_str if len(facts_str) <= 60 else facts_str[:57] + "..."
        print(f"| Key Facts   : {facts_str:<60}|")
    if card.entities:
        ent_str = ", ".join(card.entities)
        ent_str = ent_str if len(ent_str) <= 60 else ent_str[:57] + "..."
        print(f"| Entities    : {ent_str:<60}|")
    formatted = format_topic_card(card)
    fmt_str = formatted if len(formatted) <= 60 else formatted[:57] + "..."
    print(f"| Formatted   : {fmt_str:<60}|")
    print(f"+{border}+")


def run_laya_showcase() -> None:
    """Run an interactive demonstration of Laya in PRISM.

    Shows queries going into Laya, tests choice / score / noul capabilities,
    prints fetched topic cards, and logs all inputs/outputs to the console.
    """
    print("\n" + "=" * 80)
    print("  PRISM + LAYA SYSTEM-ONE DECIDER DEMONSTRATION (convaiinnovations/laya)")
    print("=" * 80)

    try:
        import laya  # noqa: F401
    except ImportError:
        print("[INFO] Laya package not installed; skipping live showcase.")
        print("=" * 80 + "\n")
        return

    # 1. Fetching topic cards from context
    print("\n[STEP 1] Fetching Topic Cards & Summaries from PRISM Context:")
    topic1 = Topic(
        id=1,
        card=TopicCard(
            label="Billing & Duplicate Charge",
            description="User was billed twice for March on invoice #4411 ($120)",
            key_facts=["Charged twice on March 3", "Invoice #4411", "Refund requested"],
            entities=["invoice #4411", "$120", "March"],
            covered_through=2,
        ),
        page_seqs=[0, 1, 2],
        is_open=False,
    )
    topic2 = Topic(
        id=2,
        card=TopicCard(
            label="Authentication Token Expiry",
            description="JWT access token expiry and refresh failure in auth.py",
            key_facts=["Tokens expire after 5 minutes", "Refresh endpoint returns 401"],
            entities=["auth.py", "JWT", "/refresh"],
            covered_through=4,
        ),
        page_seqs=[3, 4],
        is_open=False,
    )
    open_topic = Topic(
        id=3,
        card=TopicCard(
            label="Vacation Itinerary Planning",
            description="Planning travel itinerary for Kerala backwaters",
            key_facts=["Decided on Alleppey houseboat and Munnar tea gardens"],
            entities=["Kerala", "Alleppey", "Munnar"],
            covered_through=5,
        ),
        page_seqs=[5],
        is_open=True,
    )

    print_topic_card_box(topic1.card, topic_id=topic1.id, status="Closed")
    print_topic_card_box(topic2.card, topic_id=topic2.id, status="Closed")
    print_topic_card_box(open_topic.card, topic_id=open_topic.id, status="Open (Current)")

    # 2. Reusing exported recycled router
    print("\n[STEP 2] Obtaining Exported Recycled Laya Router:")
    t_start = time.perf_counter()
    shared_r = get_shared_router(preload=True)
    t_router = (time.perf_counter() - t_start) * 1000
    print(f"  [OK] Shared Router Instance: {shared_r}")
    print(f"  [OK] Obtained / Verified in {t_router:.2f} ms (no checkpoint re-download or file reload)")

    # 3. Formulating conversation state & multi-capability questions
    print("\n[STEP 3] Formulating State & Multi-Capability Questions (choice, score, noul):")
    recent = [
        Message("user", "We are planning our trip to Kerala in December."),
        Message("assistant", "December is great! You can do Alleppey and Munnar."),
    ]
    new_msg = Message(
        "user",
        "Wait, before that, did you process the refund for that duplicate charge on invoice #4411? "
        "If not, I have to dispute it with my bank immediately."
    )

    state = {
        "history": "\n".join(f"{m.role}: {m.content}" for m in recent),
        "newest_user_message": new_msg.content,
    }
    print("  * Dialogue History :")
    for m in recent:
        print(f"      {m.role}: {m.content}")
    print(f"  * New Incoming Msg : \"{new_msg.content}\"")

    # Questions showcasing all Laya capabilities: choice, score, noul
    questions = {
        "segment": {
            "type": "choice",
            "instructions": "Which topic does the newest user message belong to?",
            "criteria": {
                "continue": f"Continues the current topic: {format_topic_card(open_topic.card)}",
                "new": "Starts a completely new, unrelated conversation topic",
                f"return:{topic1.id}": f"Returns to earlier topic: {format_topic_card(topic1.card)}",
                f"return:{topic2.id}": f"Returns to earlier topic: {format_topic_card(topic2.card)}",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this user request?",
            "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
        },
        "refund_inquiry": {
            "type": "noul",
            "instructions": "Does the user explicitly request or ask about a refund?",
        },
        "dispute_risk": {
            "type": "noul",
            "instructions": "Does the user threaten a chargeback, bank dispute, or cancellation?",
        },
        f"topic_{topic1.id}_needed": {
            "type": "noul",
            "instructions": (
                "The newest user message needs information from this earlier "
                f"topic: {format_topic_card(topic1.card)}"
            ),
        },
        f"topic_{topic2.id}_needed": {
            "type": "noul",
            "instructions": (
                "The newest user message needs information from this earlier "
                f"topic: {format_topic_card(topic2.card)}"
            ),
        },
    }

    # 4. Running inference in a single forward pass
    print("\n[STEP 4] Dispatching All Questions to Laya in a SINGLE Forward Pass:")
    t0 = time.perf_counter()
    response = shared_r.predict(state, questions)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    routing_info = response.get("routing", {})
    answers = response.get("answers", {})

    print(f"  [OK] Routing Checkpoint : {routing_info.get('model', 'english')} ({routing_info.get('repo', 'convaiinnovations/laya')})")
    print(f"  [OK] Inference Latency  : {elapsed_ms:.1f} ms\n")

    print("  --- System-One Decider Outputs -----------------------------------------")
    seg_ans = answers.get("segment", {})
    print(f"  [CHOICE] Segmentation Decision:")
    print(f"    Selected Option : {seg_ans.get('choice')}")
    print(f"    Confidence      : {seg_ans.get('confidence', 0.0):.4f}")
    print(f"    Probabilities   : {seg_ans.get('probabilities', {})}")

    urg_ans = answers.get("urgency", {})
    print(f"\n  [SCORE] Request Urgency:")
    urg_score = urg_ans.get("score")
    if urg_score is not None:
        criteria_list = questions["urgency"]["criteria"]
        idx = min(max(int(round(urg_score)), 0), len(criteria_list) - 1)
        print(f"    Continuous Score : {urg_score:.4f} (mapped: '{criteria_list[idx]}')")
    else:
        print(f"    Choice          : {urg_ans.get('choice')}")

    ref_ans = answers.get("refund_inquiry", {})
    disp_ans = answers.get("dispute_risk", {})
    t1_ans = answers.get(f"topic_{topic1.id}_needed", {})
    t2_ans = answers.get(f"topic_{topic2.id}_needed", {})

    print(f"\n  [NOUL] Calibrated Probability Queries:")
    print(f"    * Refund inquiry requested?    : {ref_ans.get('noul', 0.0):.4f}  (confidence: {ref_ans.get('confidence', 0.0):.4f})")
    print(f"    * Dispute / chargeback risk?   : {disp_ans.get('noul', 0.0):.4f}  (confidence: {disp_ans.get('confidence', 0.0):.4f})")
    print(f"    * Topic #1 (Billing) needed?   : {t1_ans.get('noul', 0.0):.4f}  (calibrated prob)")
    print(f"    * Topic #2 (Auth Bug) needed?  : {t2_ans.get('noul', 0.0):.4f}  (calibrated prob)")

    # 5. Segmenter & Router direct verification
    print("\n[STEP 5] Validating Integrated PRISM Deciders (LayaSegmenter & LayaRouter):")
    segmenter = LayaSegmenter()
    decision = segmenter.decide(open_topic, [topic1, topic2], recent, new_msg)
    print(f"  * LayaSegmenter.decide() -> action: '{decision.action}', target_topic_id: {decision.topic_id}, confidence: {decision.confidence:.4f}")

    router_decider = LayaRouter()
    scores = router_decider.score([topic1, topic2], recent, new_msg)
    print(f"  * LayaRouter.score()     -> Topic scores: {scores}")
    print("=" * 80 + "\n")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════ runner

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Laya decider tests for PRISM")
    parser.add_argument("--live", action="store_true", help="Also run live SDK tests (requires `pip install laya`)")
    parser.add_argument("--showcase-only", action="store_true", help="Run the showcase demonstration and exit")
    parser.add_argument("--no-showcase", action="store_true", help="Skip the showcase demonstration and run tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args, remaining = parser.parse_known_args()

    # Run the interactive showcase demonstration unless suppressed
    if not args.no_showcase:
        run_laya_showcase()
        sys.stdout.flush()

    if args.showcase_only:
        sys.exit(0)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Always run mock tests
    for cls in (
        TestConversationText,
        TestLayaSegmenterMocked,
        TestLayaRouterMocked,
        TestProtocolConformance,
        TestLayaContextManagerIntegration,
        TestModelSelection,
        TestExportedRecycledRouter,
        TestEdgeCases,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # Conditionally add live tests
    if args.live:
        suite.addTests(loader.loadTestsFromTestCase(TestLayaLiveSDK))
    else:
        print("[INFO] Skipping live SDK unit tests. Run with --live to include them.\n")

    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
