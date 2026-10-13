"""Transcript mode contracts, including interim revisions and late decisions."""

import asyncio
from unittest.mock import patch

import pytest
from pipecat.classifiers.base_classifier import BaseClassifier, ChoiceResult
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from conftest import (
    FakeSTT,
    FakeTTSService,
    MemoryCache,
    _fired,
    _ready_library,
    _transports,
    _wire,
)
from pipecat_backchannel import Backchannel, BackchannelCandidate
from pipecat_backchannel.candidates import normalize_candidates
from pipecat_backchannel.player import ClipPlayer
from pipecat_backchannel.transcript import PunctuationTracker, TranscriptBackchannelProcessor

DOWN = FrameDirection.DOWNSTREAM
UP = FrameDirection.UPSTREAM


class Classifier(BaseClassifier):
    def __init__(self, *, choice=None, hold=False, fail=False):
        super().__init__()
        self.choice_id = choice
        self.calls = []
        self.release = asyncio.Event()
        if not hold:
            self.release.set()
        self.fail = fail
        self.cleaned = False

    async def _ask(self, state, questions):
        self.calls.append((state, questions))
        await self.release.wait()
        if self.fail:
            raise RuntimeError("offline")
        options = questions["backchannel"].options
        choice = self.choice_id or next(k for k in options if k != "skip")
        return {
            "backchannel": ChoiceResult(
                choice=choice,
                probabilities={k: float(k == choice) for k in options},
                confidence=1.0,
            )
        }, None

    async def cleanup(self):
        self.cleaned = True
        await super().cleanup()


async def make(classifier=None, **kwargs):
    classifier = classifier or Classifier()
    gate = TranscriptBackchannelProcessor(
        classifier=classifier,
        context_provider=lambda: [{"role": "assistant", "content": "どうぞ"}],
        candidates=normalize_candidates(["はい"]),
        clips=await _ready_library(groups={"backchannel": ["はい"]}),
        min_classification_interval_s=0,
        **kwargs,
    )
    sink = await _wire(gate)
    return gate, sink, classifier


async def interim(gate, text):
    await gate.queue_frame(InterimTranscriptionFrame(text=text, user_id="u", timestamp=""), DOWN)
    await asyncio.sleep(0.03)


async def close(gate, sink):
    await gate.cleanup()
    await sink.cleanup()


def test_candidates_shorthand_stable_and_validated():
    first = normalize_candidates(["はい", "ああ"])
    reordered = normalize_candidates(["ああ", "はい"])
    assert first[0].id == reordered[1].id
    assert (
        normalize_candidates([BackchannelCandidate("はい", id="backchannel_hai")])[0].id
        == "backchannel_hai"
    )
    for candidates in [
        [],
        "はい",
        [" "],
        ["はい", "はい"],
        [BackchannelCandidate("はい", id="x"), BackchannelCandidate("ああ", id="x")],
    ]:
        with pytest.raises(ValueError):
            normalize_candidates(candidates)
    with pytest.raises(ValueError, match="reserved"):
        BackchannelCandidate("はい", id="skip")


def test_boundary_tracker_handles_cumulative_interims_and_corrections():
    tracker = PunctuationTracker("、,。.")
    assert tracker.update("昨日は")[0] is None
    assert tracker.update("昨日は、")[0] == 3
    assert tracker.update("昨日は、東京に")[0] is None
    assert tracker.update("昨日は, 東京に")[0] is None
    # Correction grows text before the consumed comma; that comma is still consumed.
    assert tracker.update("一昨日は、東京に")[0] is None
    assert tracker.update("一昨日は、東京に行きました。")[0] is not None
    assert tracker.update("一昨日は、東京に行きました。")[0] is None


async def test_simple_candidates_classify_once_with_context_and_skip_override():
    gate, sink, classifier = await make(skip_description="聞き続ける")
    try:
        await interim(gate, "昨日は、")
        await interim(gate, "昨日は、東京に")
        assert len(classifier.calls) == 1
        state, questions = classifier.calls[0]
        assert state["conversation"][0]["content"] == "どうぞ"
        assert state["seconds_since_last_backchannel"] is None
        assert state["current_utterance"]["interim"] == "昨日は、"
        assert questions["backchannel"].options["skip"] == "聞き続ける"
        assert [f.text for f in _fired(sink)] == ["はい"]
    finally:
        await close(gate, sink)
    assert classifier.cleaned


async def test_skip_does_not_start_playback_cooldown():
    gate, sink, classifier = await make(Classifier(choice="skip"))
    try:
        await interim(gate, "昨日は、")
        await interim(gate, "昨日は、東京に。")
        assert len(classifier.calls) == 2
        assert not _fired(sink)
        assert gate._last_played is None
    finally:
        await close(gate, sink)


async def test_playback_feedback_starts_cooldown_and_history():
    gate, sink, classifier = await make(cooldown_s=5)
    try:
        await interim(gate, "昨日は、")
        request = _fired(sink)[0]
        assert gate.allow_play(request.request_id)
        gate.on_clip_emitted(request.request_id)
        assert gate._last_played is None
        started = BotStartedSpeakingFrame()
        gate.observe_output(started, UP)
        gate.observe_output(BotStoppedSpeakingFrame(), UP)
        assert started.metadata["backchannel"]
        await interim(gate, "昨日は、東京に。")
        assert len(classifier.calls) == 1
        gate._last_played -= 6
        await interim(gate, "昨日は、東京に。今日は、")
        assert len(classifier.calls) == 2
        state = classifier.calls[-1][0]
        assert state["seconds_since_last_backchannel"] >= 6
        assert state["backchannels"][0]["text"] == "はい"
        assert state["backchannels"][0]["interim"] == "昨日は、"
    finally:
        await close(gate, sink)


@pytest.mark.parametrize("event", ["turn_end", "reply", "disabled", "revision", "new_boundary"])
async def test_inflight_result_is_discarded_when_no_longer_applicable(event):
    gate, sink, classifier = await make(Classifier(hold=True))
    try:
        await interim(gate, "昨日は、")
        if event == "turn_end":
            await gate.process_frame(UserStoppedSpeakingFrame(), UP)
        elif event == "reply":
            gate.observe_output(LLMFullResponseStartFrame(), DOWN)
        elif event == "disabled":
            gate.enabled = False
        elif event == "revision":
            await interim(gate, "明日は、")
        else:
            await interim(gate, "昨日は、東京に。")
        classifier.release.set()
        await asyncio.sleep(0.03)
        assert not _fired(sink)
    finally:
        await close(gate, sink)


async def test_continuing_speech_without_new_boundary_keeps_result_valid():
    gate, sink, classifier = await make(Classifier(hold=True))
    try:
        await interim(gate, "昨日は、")
        await interim(gate, "昨日は、東京に行って")
        classifier.release.set()
        await asyncio.sleep(0.03)
        assert len(_fired(sink)) == 1
    finally:
        await close(gate, sink)


@pytest.mark.parametrize("fail", [True, False])
async def test_failure_or_timeout_keeps_audio_path_live(fail):
    gate, sink, classifier = await make(
        Classifier(fail=fail, hold=not fail), classification_timeout_s=0.05
    )
    try:
        await interim(gate, "昨日は、")
        await asyncio.sleep(0.06)
        assert not _fired(sink)
        assert gate._task.done()
        assert any(isinstance(f, InterimTranscriptionFrame) for f in sink.frames)
    finally:
        await close(gate, sink)


async def test_final_segments_are_accumulated_without_final_retrigger():
    gate, sink, classifier = await make(Classifier(choice="skip"))
    try:
        await interim(gate, "昨日は、")
        await gate.process_frame(
            TranscriptionFrame(text="昨日は、", user_id="u", timestamp=""), DOWN
        )
        assert len(classifier.calls) == 1
        await interim(gate, "東京に行きました。")
        assert classifier.calls[-1][0]["current_utterance"]["final_segments"] == ["昨日は、"]
    finally:
        await close(gate, sink)


async def test_player_rechecks_queued_decision():
    gate, sink, classifier = await make()
    player = ClipPlayer(clips=gate._clips, playback_gate=gate)
    player_sink = await _wire(player)
    try:
        await interim(gate, "昨日は、")
        request = _fired(sink)[0]
        gate.enabled = False
        await player.process_frame(request, DOWN)
        assert not any(type(f).__name__ == "TTSAudioRawFrame" for f in player_sink.frames)
    finally:
        await close(player, player_sink)
        await close(gate, sink)


def test_transcript_plugin_avoids_audio_models_and_requires_dependencies():
    tin, tout = _transports()
    stt, tts = FakeSTT(), FakeTTSService()
    with (
        patch("pipecat_backchannel.plugin.HeuristicClipSelector", side_effect=AssertionError),
        patch("pipecat_backchannel.plugin.shared_turn_analyzer", side_effect=AssertionError),
    ):
        bc = Backchannel(mode="transcript", candidates=["はい"], cache=MemoryCache(prefilled=True))
        processors = bc([tin, stt, tts, tout], classifier=Classifier(), context_provider=lambda: [])
        gate = next(p for p in processors if isinstance(p, TranscriptBackchannelProcessor))
        assert gate._cooldown_s == 5
        assert not hasattr(gate, "_vad")
        with pytest.raises(ValueError, match="classifier"):
            bc([tin, stt, tts, tout])
        with pytest.raises(ValueError, match="STT"):
            bc([tin, tts, tout], classifier=Classifier(), context_provider=lambda: [])


async def test_real_transport_feedback_records_only_played_backchannel():
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.tests.utils import SleepFrame, run_test
    from pipecat.transports.base_output import BaseOutputTransport
    from pipecat.transports.base_transport import TransportParams

    class Output(BaseOutputTransport):
        async def start(self, frame):
            await super().start(frame)
            await self.set_transport_ready(frame)

        async def write_audio_frame(self, frame):
            return True

    classifier = Classifier()
    library = await _ready_library(groups={"backchannel": ["はい"]})
    gate = TranscriptBackchannelProcessor(
        classifier=classifier,
        context_provider=lambda: [],
        candidates=normalize_candidates(["はい"]),
        clips=library,
    )
    player = ClipPlayer(clips=library, playback_gate=gate)
    output = Output(TransportParams(audio_out_enabled=True))
    _, upstream = await run_test(
        Pipeline([gate, player, output]),
        frames_to_send=[
            InterimTranscriptionFrame(text="昨日は、", user_id="u", timestamp=""),
            SleepFrame(sleep=0.2),
        ],
    )
    assert len(gate._history) == 1
    assert gate._history[0]["text"] == "はい"
    assert any(
        isinstance(f, BotStartedSpeakingFrame) and f.metadata.get("backchannel") for f in upstream
    )


async def test_minimum_call_interval_and_consumed_cooldown_boundaries():
    gate, sink, classifier = await make(Classifier(choice="skip"))
    gate._interval = 5
    try:
        await interim(gate, "昨日は、")
        await interim(gate, "昨日は、東京へ。")
        assert len(classifier.calls) == 1
        gate._last_classified -= 6
        # Expiry alone does not revisit an old boundary.
        await interim(gate, "昨日は、東京へ。")
        assert len(classifier.calls) == 1
        await interim(gate, "昨日は、東京へ。今日は、")
        assert len(classifier.calls) == 2
    finally:
        await close(gate, sink)


async def test_cleanup_cancels_inflight_call():
    gate, sink, classifier = await make(Classifier(hold=True))
    await interim(gate, "昨日は、")
    task = gate._task
    await close(gate, sink)
    assert task.done()
    assert classifier.cleaned
    assert not _fired(sink)


@pytest.mark.parametrize("punctuation", ["、", ",", "。", "."])
def test_all_default_punctuation_triggers(punctuation):
    tracker = PunctuationTracker("、,。.")
    assert tracker.update("説明" + punctuation)[0] == 2
    assert tracker.update("説明" + punctuation)[0] is None


async def test_context_is_snapshotted_before_async_inference():
    messages = [{"role": "user", "content": "以前の発話"}]
    gate, sink, classifier = await make(Classifier(hold=True))
    gate._context_provider = lambda: messages
    try:
        await interim(gate, "昨日は、")
        messages[0]["content"] = "更新後"
        assert classifier.calls[0][0]["conversation"][0]["content"] == "以前の発話"
    finally:
        await close(gate, sink)


def test_transcript_sessions_share_audio_but_not_decision_state():
    tin, tout = _transports()
    stt, tts = FakeSTT(), FakeTTSService()
    bc = Backchannel(mode="transcript", candidates=["はい"])
    gates = []
    for _ in range(2):
        processors = bc([tin, stt, tts, tout], classifier=Classifier(), context_provider=lambda: [])
        gates.append(next(p for p in processors if isinstance(p, TranscriptBackchannelProcessor)))
    assert gates[0]._clips is gates[1]._clips
    assert gates[0]._classifier is not gates[1]._classifier
    gates[0]._last_played = 1
    assert gates[1]._last_played is None


@pytest.mark.parametrize("suffix", ["", "東京"])
def test_insertion_before_consumed_punctuation_keeps_same_boundary(suffix):
    tracker = PunctuationTracker("、,。.")
    assert tracker.update("昨日は、" + suffix) == (3, False)
    assert tracker.update("昨日はね、" + suffix) == (None, True)
    assert tracker.update("昨日はね、" + suffix + "続き。")[0] is not None


def test_removed_and_restored_punctuation_is_consumed_but_invalidates():
    tracker = PunctuationTracker("、,。.")
    assert tracker.update("昨日は、") == (3, False)
    assert tracker.update("昨日は") == (None, True)
    assert tracker.update("昨日は、") == (None, False)
    assert tracker.update("昨日は、東京へ。")[0] is not None


async def test_removed_punctuation_invalidates_inflight_decision():
    gate, sink, classifier = await make(Classifier(hold=True))
    try:
        await interim(gate, "昨日は、")
        await interim(gate, "昨日は")
        classifier.release.set()
        await asyncio.sleep(0.03)
        assert not _fired(sink)
    finally:
        await close(gate, sink)


def test_boundary_count_preserves_maximum_across_replacement_and_resets():
    tracker = PunctuationTracker("、,。.")
    assert tracker.update("昨日は、東京で。")[0] == 6
    assert tracker.update("昨日は東京で")[0] is None
    assert tracker.update("昨日は東京で、買い物。")[0] is None
    assert tracker.update("昨日は東京で、買い物。そのあと、")[0] == 13
    tracker.reset()
    assert tracker.update("今日は、") == (3, False)


def test_consecutive_punctuation_counts_as_one_boundary():
    tracker = PunctuationTracker("、,。.")
    assert tracker.update("説明、 。,") == (2, False)
    assert tracker.update("説明。その続き、,。") == (6, False)


def test_retroactive_punctuation_insertion_counts_as_new_opportunity():
    tracker = PunctuationTracker("、,。.")
    assert tracker.update("昨日は東京で、買い物")[0] == 6
    assert tracker.update("昨日は、東京で、買い物") == (6, True)
