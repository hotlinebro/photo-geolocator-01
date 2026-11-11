# Photo geolocator — make.fun entry

An agent that says where each of 25 street-level photos was taken, for the
make.fun competition `photo-geolocator-01`.

## How it works

`main.py` (Python 3.13, standard library only, plus the kit's official
Python SDK vendored byte for byte under `vendor/`) runs one trial as:

1. GET the photo list from the env at `MAKEFUN_ENV_URL` (`/photos`).
2. For each photo, in id order, download the JPEG and ask
   `claude-sonnet-5-5` — through the trial's model gateway at
   `ANTHROPIC_BASE_URL`, on the entrant's own pushed API key — where it was
   taken, with a structured reasoning prompt: the model works through the
   visible clues (signs and language, driving side, road markings, plates,
   architecture, vegetation, sun and shadows, landmarks) before giving one
   final `{"lat", "lon"}` line.
3. POST all 25 answers to the env (`/answers`) in id order, then exit.

## Why it beats the house baseline

The baseline (the kit's starter) asks the same model for coordinates with a
plain prompt and no reasoning, and leaves a photo unanswered when its call
is refused or its answer holds no coordinates. This entry does two things
better:

- **A reasoning prompt.** The model analyses the clues step by step before
  answering, which is what makes a vision model place a photo well instead
  of guessing.
- **No photo left unanswered.** A call the gateway refuses (no key pushed,
  budget spent, request cap) is final, and so is an answer with no
  coordinates — either way the photo still gets an answer: a fixed prior in
  central Europe (50 N, 10 E), a dense Mapillary region. An unanswered
  photo counts 20,015 km, the maximum error; the prior is far better.

## Determinism

One model call per photo, sequentially, in id order. Every request is a
function of the photo's bytes and the fixed prompt; there is no wall clock
and no unseeded randomness in anything sent, so the replay sends the same
calls and leaves the same answers. The SDK retries only transient provider
errors with a fixed backoff, which never changes a request's content.

## Model calls and cost

One `claude-sonnet-5-5` call per photo, `max_tokens` 1024 — the same shape
as the house baseline (25 calls, about $0.15 a trial), so it fits the
trial's $1.00 budget. The calls run on the entrant's own Anthropic key,
pushed to the competition before submitting; with no key pushed every call
is refused and every photo falls back to the prior above, and the entry
still posts all 25 answers.

## Files

- `main.py` — the agent.
- `vendor/makefun_agent.py` — the kit's official Python SDK, copied byte
  for byte (excluded from the entry size caps, and not counted against
  authorship).
- `LICENSE` — MIT.

No trained weights, embeddings, or table of photos and their places: every
location comes from the model reading the photo's pixels, or from the fixed
prior above.
