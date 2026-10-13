"""Public shorthand and detailed backchannel candidates."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BackchannelCandidate:
    """A cached utterance and optional classifier guidance.

    ``text`` is the cache/TTS key. An omitted ID is derived from its UTF-8 text,
    independent of list order and process. ``skip`` is reserved for silence.
    """

    text: str
    description: str | None = None
    id: str | None = None

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("candidate text must be a non-empty string")
        if self.description is not None and (
            not isinstance(self.description, str) or not self.description.strip()
        ):
            raise ValueError("candidate description must be a non-empty string")
        if self.id is None:
            digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
            object.__setattr__(self, "id", f"backchannel_{digest}")
        if not isinstance(self.id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.id):
            raise ValueError(
                "candidate id must contain only letters, digits, underscores or hyphens"
            )
        if self.id == "skip":
            raise ValueError("candidate id 'skip' is reserved")


def normalize_candidates(
    candidates: Sequence[str | BackchannelCandidate],
) -> tuple[BackchannelCandidate, ...]:
    """Resolve shorthand once and reject ambiguous inventories."""
    if isinstance(candidates, str) or not candidates:
        raise ValueError("candidates must be a non-empty sequence, e.g. ['はい']")
    resolved = tuple(
        BackchannelCandidate(text=item) if isinstance(item, str) else item for item in candidates
    )
    ids: set[str] = set()
    texts: set[str] = set()
    for item in resolved:
        if not isinstance(item, BackchannelCandidate):
            raise TypeError("candidates must contain strings or BackchannelCandidate objects")
        if item.id in ids:
            raise ValueError(f"duplicate candidate id: {item.id}")
        if item.text in texts:
            raise ValueError(f"duplicate candidate text: {item.text}")
        ids.add(item.id)
        texts.add(item.text)
    return resolved
