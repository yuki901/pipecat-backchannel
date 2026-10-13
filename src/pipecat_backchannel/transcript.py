"""Punctuation-triggered, asynchronous classification of backchannels."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import math
import time
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from loguru import logger
from pipecat.classifiers.base_classifier import BaseClassifier, ChoiceQuestion
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_backchannel.candidates import BackchannelCandidate
from pipecat_backchannel.frames import PlayCachedClipFrame
from pipecat_backchannel.library import ClipLibrary

ContextProvider = Callable[[], Sequence[dict[str, Any]]]
DEFAULT_SKIP_DESCRIPTION = "Continue listening; a spoken acknowledgment would be less natural now."


class PunctuationTracker:
    """Track consumed boundaries across cumulative interim revisions.

    Punctuation participates in alignment as a common marker. The cursor sits
    after a consumed marker, so inserting words before it moves the cursor with
    the marker. Returned offsets count only content characters. Remembered
    prefix hashes cover a boundary that disappears and later reappears.
    """

    def __init__(self, punctuation: str):
        self.punctuation = frozenset(punctuation)
        self.text = ""
        self.cursor = 0
        self._seen: set[bytes] = set()

    def reset(self):
        self.text = ""
        self.cursor = 0
        self._seen.clear()

    def update(self, text: str) -> tuple[int | None, bool]:
        characters: list[str] = []
        for char in text:
            if char in self.punctuation:
                if characters and characters[-1] != "\0":
                    characters.append("\0")
            elif not char.isspace():
                characters.append(char)
        normalized = "".join(characters)
        revised = not normalized.startswith(self.text)
        if revised and self.cursor:
            for tag, i1, i2, j1, j2 in SequenceMatcher(
                None, self.text, normalized, autojunk=False
            ).get_opcodes():
                if i1 < self.cursor <= i2:
                    self.cursor = j1 + self.cursor - i1 if tag == "equal" else j2
                    break
        self.text = normalized
        offset = 0
        newest = None
        prefix = hashlib.sha256()
        for position, char in enumerate(normalized, start=1):
            prefix.update(char.encode("utf-8"))
            if char != "\0":
                offset += 1
                continue
            key = prefix.digest()
            if position > self.cursor:
                if key not in self._seen:
                    newest = offset
                self.cursor = position
            self._seen.add(key)
        return newest, revised


@dataclass
class _Request:
    id: str
    revision: int
    deadline: float
    text: str
    boundary: int
    turn: int
    candidate: BackchannelCandidate | None = None


class TranscriptBackchannelProcessor(FrameProcessor):
    """Own one classifier and one session's transcript/playback state.

    The context provider is synchronous and returns a snapshot-friendly sequence
    of messages. The processor copies it before inference. Owners can trim or
    redact history in that provider. Classifier setup/cleanup belongs here;
    provide a fresh classifier per pipeline.
    """

    def __init__(
        self,
        *,
        classifier: BaseClassifier,
        context_provider: ContextProvider,
        candidates: Sequence[BackchannelCandidate],
        clips: ClipLibrary,
        cooldown_s: float = 5.0,
        punctuation: str = "、,。.",
        skip_description: str = DEFAULT_SKIP_DESCRIPTION,
        classification_timeout_s: float = 1.0,
        min_classification_interval_s: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        for name, value in (
            ("cooldown_s", cooldown_s),
            ("min_classification_interval_s", min_classification_interval_s),
            ("classification_timeout_s", classification_timeout_s),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if classification_timeout_s == 0:
            raise ValueError("classification_timeout_s must be positive")
        if not punctuation or not skip_description.strip():
            raise ValueError("punctuation and skip_description must be non-empty")
        self._classifier = classifier
        self._context_provider = context_provider
        self._candidates = {c.id: c for c in candidates}
        self._clips = clips
        self._cooldown_s = cooldown_s
        self._timeout = classification_timeout_s
        self._interval = min_classification_interval_s
        self._tracker = PunctuationTracker(punctuation)
        self._question = ChoiceQuestion(
            instructions=(
                "Choose the listening response that fits this moment in the ongoing user speech, "
                "using the conversation, current transcript and previous backchannels. "
                "A backchannel briefly acknowledges the speaker and invites them to continue."
            ),
            options={
                "skip": skip_description,
                **{c.id: {"text": c.text, "meaning": c.description} for c in candidates},
            },
        )
        self.enabled = True
        self._active = True
        self._revision = 0
        self._turn = 0
        self._finals: list[str] = []
        self._last_partial = ""
        self._bot_speaking = False
        self._reply_pending = False
        self._last_played: float | None = None
        self._last_classified = float("-inf")
        self._history: deque[dict[str, Any]] = deque(maxlen=32)
        self._task: asyncio.Task | None = None
        self._pending: _Request | None = None
        self._emitted: _Request | None = None
        self._playing_own = False
        self._setup_done = False
        self._closed = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self._classifier.setup(self.task_manager)
            self._setup_done = True

            @self._classifier.event_handler("on_metrics")
            async def metrics(classifier, data):
                await self.push_frame(MetricsFrame(data=data))

        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._closed = True
            self._invalidate()
            await self._cancel_classification()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._active = False
            self._invalidate()
        elif isinstance(frame, UserStartedSpeakingFrame):
            if not self._active:
                self._turn += 1
                self._tracker.reset()
                self._finals.clear()
                self._last_partial = ""
                self._active = True
                self._invalidate()
            self._reply_pending = False
        elif isinstance(frame, InterruptionFrame):
            self._invalidate()
            self._emitted = None
            self._reply_pending = False
        elif direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
            if frame.text:
                self._finals.append(frame.text)
            self._last_partial = ""
            self._tracker.reset()
            self._invalidate()
        elif direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, InterimTranscriptionFrame
        ):
            self._last_partial = frame.text
            boundary, revised = self._tracker.update(frame.text)
            if revised or boundary is not None:
                self._invalidate()
            if boundary is not None:
                self._maybe_classify(frame.text, boundary)
        await self.push_frame(frame, direction)

    def _invalidate(self):
        self._revision += 1
        self._pending = None

    def _eligible(self) -> bool:
        if self._emitted and time.monotonic() >= self._emitted.deadline + 1.0:
            self._emitted = None
        return (
            self.enabled
            and self._active
            and not self._closed
            and self._clips.ready
            and not self._bot_speaking
            and not self._reply_pending
            and self._emitted is None
            and (
                self._last_played is None
                or time.monotonic() - self._last_played >= self._cooldown_s
            )
        )

    def _maybe_classify(self, text: str, boundary: int):
        now = time.monotonic()
        if not self._eligible() or (self._task and not self._task.done()):
            return
        if now - self._last_classified < self._interval:
            return
        self._last_classified = now
        request = _Request(
            id=uuid.uuid4().hex,
            revision=self._revision,
            deadline=now + self._timeout,
            text=text,
            boundary=boundary,
            turn=self._turn,
        )
        try:
            state = {
                "conversation": copy.deepcopy(list(self._context_provider())),
                "current_utterance": {
                    "final_segments": list(self._finals),
                    "interim": text,
                    "boundary_content_offset": boundary,
                    "turn_id": self._turn,
                },
                "backchannels": [
                    {**event, "seconds_ago": now - event["started_at_monotonic"]}
                    for event in self._history
                ],
                "seconds_since_last_backchannel": (
                    None if self._last_played is None else now - self._last_played
                ),
            }
        except Exception:
            logger.exception("Backchannel: context provider failed")
            return
        self._pending = request
        self._task = self.create_task(self._classify(request, state))

    async def _classify(self, request: _Request, state: dict[str, Any]):
        try:
            async with asyncio.timeout(self._timeout):
                results = await self._classifier.choice(state, {"backchannel": self._question})
            choice = results["backchannel"].choice
            logger.debug("Backchannel: classifier selected {}", choice)
            if choice == "skip":
                return
            candidate = self._candidates.get(choice)
            if candidate is None:
                logger.warning("Backchannel: classifier returned an unknown candidate ID")
                return
            request.candidate = candidate
            if self.allow_play(request.id):
                await self.push_frame(PlayCachedClipFrame(candidate.text, request_id=request.id))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Backchannel: classification skipped ({})", type(exc).__name__)

    def allow_play(self, request_id: str) -> bool:
        """Recheck the decision at the player, immediately before audio emission."""
        request = self._pending
        return bool(
            request
            and request.id == request_id
            and request.revision == self._revision
            and time.monotonic() < request.deadline
            and self._eligible()
        )

    def on_clip_emitted(self, request_id: str):
        """Reserve attribution until the transport reports speaking or interruption."""
        if self._pending and self._pending.id == request_id:
            self._emitted = self._pending
            self._pending = None

    def observe_output(self, frame: Frame, direction: FrameDirection):
        """Called by the player for real response and transport feedback frames."""
        if isinstance(frame, (LLMFullResponseStartFrame, TTSStartedFrame)):
            self._reply_pending = True
            self._invalidate()
        elif isinstance(frame, BotStartedSpeakingFrame) and direction == FrameDirection.UPSTREAM:
            self._bot_speaking = True
            request = self._emitted
            self._emitted = None
            # Attribution is bounded; a lost clip cannot shadow a future real reply.
            self._playing_own = bool(request and time.monotonic() < request.deadline + 1.0)
            if self._playing_own:
                frame.metadata["backchannel"] = True
                now = time.monotonic()
                self._last_played = now
                self._history.append(
                    {
                        "id": request.candidate.id,
                        "text": request.candidate.text,
                        "started_at_monotonic": now,
                        "turn_id": request.turn,
                        "interim": request.text,
                        "boundary_content_offset": request.boundary,
                    }
                )
            else:
                self._active = False
                self._invalidate()
        elif isinstance(frame, BotStoppedSpeakingFrame) and direction == FrameDirection.UPSTREAM:
            if self._playing_own:
                frame.metadata["backchannel"] = True
            else:
                self._reply_pending = False
            self._playing_own = False
            self._bot_speaking = False
            self._emitted = None
        elif isinstance(frame, InterruptionFrame):
            self._emitted = None
            self._playing_own = False
            self._bot_speaking = False
            self._reply_pending = False
            self._invalidate()

    async def _cancel_classification(self):
        if self._task and not self._task.done():
            await self.cancel_task(self._task)
        self._task = None

    async def cleanup(self):
        self._closed = True
        await self._cancel_classification()
        if self._setup_done:
            await self._classifier.cleanup()
            self._setup_done = False
        await super().cleanup()
