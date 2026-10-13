<div align="center">
  <a href="https://github.com/pipecat-ai/pipecat">
    <img alt="Pipecat" width="220px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/pipecat.png">
  </a>
  <p><em>A community integration for <a href="https://github.com/pipecat-ai/pipecat">Pipecat</a></em></p>
</div>

# pipecat-backchannel

[![PyPI](https://img.shields.io/pypi/v/pipecat-backchannel?cacheSeconds=3600)](https://pypi.org/project/pipecat-backchannel)
[![Python](https://img.shields.io/pypi/pyversions/pipecat-backchannel?cacheSeconds=3600)](https://pypi.org/project/pipecat-backchannel)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue)](LICENSE)
[![Pipecat](https://img.shields.io/badge/Pipecat-1.12.0-brightgreen)](https://github.com/pipecat-ai/pipecat)

Your Pipecat agent listens back: quiet "mhm"s and "yeah"s while the user is
still talking.

## Demo

https://github.com/user-attachments/assets/6c38d03b-4eb6-42bd-8f0d-47a1cfdb1bfc

## Why

Listen to two people on a phone call. While one tells a story, the other keeps
feeding back small signals: "mhm", "yeah". Linguists call these
**backchannels**. They mean _keep going, I'm with you_, and the speaker talks
straight through them.

Talk to a voice agent and you get silence until you finish. Silence on a call
reads as a dropped connection, so you hesitate, trail off, or ask "hello?".

This library adds the listening sounds. The agent keeps the floor with you the
whole time: a backchannel takes no turn and leaves no trace in your LLM
context.

## Install

```bash
uv add pipecat-backchannel
```

## Usage

```python
from pipecat_backchannel import Backchannel

backchannel = Backchannel()

pipeline = Pipeline(backchannel([
    transport.input(),
    stt,
    context_aggregator.user(),
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
]))
```

One wrapper call is the whole setup. The default configuration works without
an API key, a voice ID, or a sample rate, and places every processor for you.

The first run records the clips through **your own TTS**, so they come out in
your bot's voice, and caches them in `.clip_cache/`. Every later run starts
from the cache. Delete that directory after changing voice.

## Transcript mode

Use interim-transcript punctuation and a Pipecat classifier to decide both
whether to backchannel and which clip to play. Requires Pipecat 1.12.0 or newer
and an STT service producing cumulative interim transcripts with punctuation.
This mode creates no backchannel VAD, Smart Turn analyzer, or heuristic selector.
The default `mode="pause"` keeps the existing audio-based behavior.
This release raises the package's minimum Pipecat version from 1.6.0 to 1.12.0,
including for pause mode. Existing pause-mode arguments and its 2.5-second
default cooldown are preserved; NLTK is an explicit dependency for its selector.

```python
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat_backchannel import Backchannel

# Shared configuration and clip cache, built once for a voice/output rate.
backchannel = Backchannel(
    mode="transcript",
    candidates=["はい"],
    cooldown_s=1.5,  # omit for the transcript-mode default of 5 seconds
)

# Inside the session: supply its own classifier and conversation context.
# llm is an existing LLMService supporting run_inference().
pipeline = Pipeline(backchannel(
    [transport.input(), stt, user, llm, tts, transport.output(), assistant],
    classifier=LLMClassifier(llm=llm),
    context_provider=context.get_messages,
))
```

String candidates get stable IDs derived from their text. The configured TTS
records each unique utterance into the normal clip cache. Adding an utterance is
just `candidates=["はい", "なるほど", "そうなんですね"]`. For explicit semantics or
an application-owned ID, mix strings with detailed candidates:

```python
from pipecat_backchannel import BackchannelCandidate

backchannel = Backchannel(
    mode="transcript",
    candidates=[
        BackchannelCandidate(
            text="はい",
            description="短く受け止め、話の続きを促す",
            id="backchannel_hai",
        ),
        "なるほど",
    ],
    skip_description="今は相手の話の続きを聞くのが自然",
)
```

Only `text` is required. Candidate IDs and texts must be unique; `skip` is
reserved. A silent `skip` choice is always added. Its description is optional
and configurable; it has no audio and consumes no playback cooldown.

The classifier receives a copied conversation snapshot, finalized segments of
the current utterance, its latest interim and boundary position, the last 32
played backchannels with their turn/position and monotonic start timestamps,
and seconds since the last backchannel (`None` before the first). A synchronous
`context_provider` can trim or redact messages before returning them.

Timing controls:

- `punctuation="、,。."`: consideration occurs when the boundary count exceeds
  the maximum seen in the current STT segment. Consecutive punctuation counts as
  one boundary. Removing and restoring punctuation or editing words at the same
  count does not retrigger consideration. Multiple additional boundaries produce
  one consideration, at the latest. Counts advance even during cooldown or inference
  and reset on a final transcript or a new user turn. This deliberately approximate
  rule can count punctuation inserted into earlier text as a new opportunity, and
  miss a moved boundary when the total count stays unchanged.
- `cooldown_s=5.0`: measured from the transport's `BotStartedSpeakingFrame`,
  attributed to an admitted clip. Expiry waits for the next new boundary.
- `min_classification_interval_s=0.5`: bounds API call frequency even for `skip`.
- `classification_timeout_s=1.0`: total decision lifetime, including time spent
  reaching the player. Timeout, provider failure and stale results produce silence.

Classification runs outside the frame-processing path, one request at a time.
New boundaries arriving during inference are consumed and make the old result
stale; the next boundary can trigger another request. Plain continuation after
an unchanged boundary remains eligible. Final frames commit the STT segment and
invalidate pending results; they do not trigger classification themselves.
User-turn completion, real replies, and `enabled=False` prevent playback.

The player rechecks each queued decision before emitting audio. Clip requests
that never reach playback do not enter history. Transport feedback has no clip
ID, so attribution uses a bounded reservation of an admitted clip while the
output is idle, cleared on interruption. These timestamps describe server-side
playback notification, not when a remote listener physically hears the sound.
Speaking feedback attributed to a clip carries `frame.metadata["backchannel"] =
True` upstream, for custom turn strategies that distinguish acknowledgments from
real bot replies. The audio continues to use `append_to_context=False`.

Each pipeline owns its classifier's setup/cleanup; construct a fresh classifier
per session. An injected LLM service keeps its existing application/pipeline
lifecycle. `LLMClassifier` uses the provider extras already installed for that
LLM; `JevClassifier` additionally needs Pipecat's `jev` extra and its credentials.
Japanese accuracy and timing depend on the STT punctuation and classifier backend;
measure them on real conversations before choosing a model or a tighter timeout.

## Record before the first call

The first session waits a few seconds while the recorder fills the clip cache.
Move that wait to app startup if you want to avoid delay for first caller:

```python
backchannel = Backchannel()          # one per process

async def run_bot(...):
    pipeline = Pipeline(backchannel([... tts ...]))

if __name__ == "__main__":
    asyncio.run(backchannel.prewarm(tts=make_tts()))   # before the server opens
    main()
```

## Tuning

```python
Backchannel(
    volume=0.6,                              # how loud, relative to the recording
    params=BackchannelParams(
        fire_probability=0.8,                # chance of speaking up at a good moment
        cooldown_s=2.5,                      # minimum gap between backchannels
        min_speech_before_eligible_s=0.7,    # ignore pauses right after a turn starts
    ),
)
```

`fire_probability` and `cooldown_s` set how present the listener feels. Raise
the first and lower the second for a more active one. Keep some slack: a bot
that reacts to each eligible pause sounds mechanical, because human listeners
skip some of them.

`volume` sits below 1.0 by default so clips play under the user's voice.

Set `loguru` to `DEBUG` to see each fire/skip decision and its reason.

### Going quiet mid-session

Some stretches of a call want a silent listener: the user dictating a card
number, or the bot walking through a form where each pause belongs to a
field, and an "mhm" would read as confirmation.

The component that decides when to fire is the `BackchannelProcessor`, one
per pipeline. `backchannel([...])` returns your processor list with it
inserted, so keep that list, build the `Pipeline` from it, and pull the gate
out of it:

```python
from pipecat_backchannel.processor import BackchannelProcessor

processors = backchannel([transport.input(), stt, llm, tts, transport.output()])
pipeline = Pipeline(processors)

gate = next(p for p in processors if isinstance(p, BackchannelProcessor))
```

Then flip it from any event handler, as many times as the call needs:

```python
gate.enabled = False   # stop firing; every frame still passes through
...
gate.enabled = True    # resume
```

`enabled` is a plain attribute, safe to set at any moment. While it is
`False` the processor keeps passing audio through and keeps tracking speech,
so nothing downstream notices and re-enabling picks up mid-turn. It fires no
clips and runs no end-of-turn classification in that state. The pipeline, the
LLM context, and the other sessions' gates stay untouched.

## Custom clips

The library groups clips by conversational function: continuer, agreement,
hesitation, surprise.

```python
Backchannel(clip_groups={"continue": ["Mhm.", "Right."], "affirm": ["Yeah.", "Yep."]})
```

Two rules for a custom inventory:

- **Give each group at least two clips.** A one-clip group repeats itself.
- **Get variety from punctuation, and keep the vocabulary small.** Real
  backchannels draw on a handful of sounds: _uh-huh, yeah, mm-hmm, right,
  okay, oh, huh, hm_. A written phrase like "absolutely" reads fine on the
  page, then sounds like an answer when spoken over someone mid-sentence.
  Punctuation is the handle your TTS gives you: `"Mhm."` falls, `"Mhm,"` stays
  up, `"Hm..."` draws out.

Pass `synthesizer=` to use a different voice, or `cache=` to supply your own
recorded `.wav` files. See `pipecat_backchannel.synth` and `.cache`.

## Run the example

```bash
cp examples/.env.template examples/.env   # DEEPGRAM_API_KEY, CARTESIA_API_KEY, CEREBRAS_API_KEY
uv run examples/bot.py
```

Open `http://localhost:7860/client`, talk in mid-length sentences, and pause
mid-thought. A quiet "mhm" lands on clause-complete pauses. Word-search pauses
("we need the, uh...") and finished questions get silence, and the questions
get a real reply.

## Limitations

- **Pause-mode timing.** The audio-based mode reacts to pauses. Transcript mode
  can react during speech, but depends on interim punctuation and classifier latency.
- **About 2× audio inference in pause mode.** It runs a second VAD and
  turn-detector on purpose: the pipeline's copies are tuned for turn-taking,
  and pause timing needs the opposite settings. A handful of concurrent
  sessions runs fine; measure before a large deployment.
- **English defaults in pause mode.** The default clips and selector regexes are
  English. Transcript mode accepts caller-defined utterances and descriptions.

## Roadmap

- [ ] **Fire during speech.** Real listeners backchannel mid-utterance, cued
      by pitch and rhythm rather than silence. That takes Voice Activity
      Projection; [MaAI](https://github.com/MaAI-Kyoto/MaAI) ships a usable
      model. It reads a different signal, so it slots in beside the pause
      gate rather than replacing it, and it is the largest quality jump left.
- [ ] **Match the speaker's prosody.** The selector picks a clip at random
      within its group. Choosing the one whose pitch and energy fit the last
      utterance would likely want a small prediction model at fire time,
      trained to map the tail of the user's speech to a clip. Playback stays
      free — the clips already sit on disk — so the only new cost is that one
      lightweight inference.

## License

BSD 2-Clause © Taras Maister, matching Pipecat's own license. Not affiliated
with Daily or the Pipecat core team.
