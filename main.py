#!/usr/bin/env python3
"""photo-geolocator-01: a vision-model geolocation agent.

The trial shows the agent 25 street-level photos (OpenStreetView-5M, metadata
stripped) through a small env API and scores the latitude/longitude it posts
for each. This entry:

  1. GETs the photo list from MAKEFUN_ENV_URL (/photos).
  2. For each photo, in id order, downloads the JPEG and asks
     claude-sonnet-5-5 (through the trial's model gateway at
     ANTHROPIC_BASE_URL, on the entrant's own pushed API key) where it was
     taken, with a structured reasoning prompt: the model works through the
     visible clues (signs and language, driving side, road markings, plates,
     architecture, vegetation, sun and shadows, landmarks) before giving one
     final {"lat", "lon"} line. A reasoning prompt beats the house baseline,
     which asks the same model for coordinates with no reasoning at all.
  3. POSTs all 25 answers to the env (/answers) in id order, then exits.

Model calls are optional in this competition, and a call the gateway refuses
(no key pushed, budget spent, request cap) is final, so every photo whose
model call fails or whose answer holds no coordinates still gets an answer:
a fixed prior (central Europe, 50 N 10 E, a dense Mapillary region), which
is far better than leaving it unanswered (20,015 km, the maximum error).
Nothing is left unanswered.

Determinism: the agent makes one model call per photo, sequentially, in id
order. Every request is a function of the photo's bytes and the fixed prompt
above; there is no wall clock and no unseeded randomness in anything sent,
so the replay sends the same calls and leaves the same answers. The model
client (the kit's official Python SDK, vendored byte for byte under vendor/)
retries only transient provider errors with a fixed backoff, which never
changes a request's content.

Standard library only, plus the vendored official SDK. No network beyond the
env and the model gateway. No trained weights, embeddings, or table of
photos and their places: every location comes from the model reading the
photo's pixels, or from the fixed prior above.
"""

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_agent as af  # noqa: E402  (official SDK, vendored byte for byte)

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 1024

# One structured reasoning prompt, the same for every photo. The model
# reasons briefly over the clues, then ends with exactly one JSON line.
PROMPT = (
    "You are an expert geolocation analyst. Look carefully at this "
    "street-level photo and work out where on Earth it was taken. Reason "
    "briefly, in a few short lines, over the visible clues: (1) any text, "
    "signs, or script and what language it is; (2) which side of the road "
    "traffic drives on, and the road markings; (3) license plates, vehicle "
    "types, and street furniture; (4) architecture, building materials, and "
    "roof shapes; (5) vegetation, terrain, and climate; (6) the sun and "
    "shadows, and what hemisphere and rough latitude they suggest; (7) any "
    "landmark, flag, or other distinctive feature. Then give your single "
    "best estimate of the location. End your reply with exactly one line of "
    "JSON and nothing after it: "
    '{"lat": <decimal degrees north, -90 to 90>, '
    '"lon": <decimal degrees east, -180 to 180>}.'
)

# Deterministic fallback for a photo the model could not place: a fixed
# prior in central Europe, a dense Mapillary / OpenStreetView-5M region.
# Far better than leaving the photo unanswered (20,015 km).
FALLBACK_LAT = 50.0
FALLBACK_LON = 10.0

ENV_TIMEOUT_S = 60
ENV_RETRIES = 2


def log(*args):
    print("geolocator:", *args, file=sys.stderr, flush=True)


def _http(method, url, body=None, headers=None, timeout=ENV_TIMEOUT_S):
    """One HTTP exchange against the env. Returns (status, bytes)."""
    req = urllib.request.Request(url, data=body, method=method,
                                 headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as err:
        try:
            return err.code, err.read()
        finally:
            err.close()


def env_get(env_url, path):
    """GET a JSON route of the env, retrying transport errors."""
    for attempt in range(ENV_RETRIES + 1):
        try:
            status, raw = _http("GET", env_url + path,
                                headers={"Accept": "application/json"})
        except (urllib.error.URLError, OSError) as err:
            if attempt < ENV_RETRIES:
                time.sleep(1.0)
                continue
            raise RuntimeError(f"GET {path} failed: {err}") from None
        if status != 200:
            raise RuntimeError(
                f"GET {path} answered {status}: {raw[:200]!r}")
        return json.loads(raw.decode("utf-8"))
    raise RuntimeError(f"GET {path} failed")


def env_get_bytes(env_url, path):
    """GET a photo of the env, retrying transport errors."""
    for attempt in range(ENV_RETRIES + 1):
        try:
            status, raw = _http("GET", env_url + path,
                                headers={"Accept": "image/jpeg"})
        except (urllib.error.URLError, OSError) as err:
            if attempt < ENV_RETRIES:
                time.sleep(1.0)
                continue
            raise RuntimeError(f"GET {path} failed: {err}") from None
        if status != 200:
            raise RuntimeError(
                f"GET {path} answered {status}: {raw[:200]!r}")
        return raw
    raise RuntimeError(f"GET {path} failed")


def env_post_answers(env_url, answers):
    """POST the answers to the env, retrying transport errors only."""
    payload = json.dumps({"answers": answers}, separators=(",", ":")).encode("utf-8")
    for attempt in range(ENV_RETRIES + 1):
        try:
            status, raw = _http("POST", env_url + "/answers", body=payload,
                                headers={"Content-Type": "application/json",
                                         "Accept": "application/json"})
        except (urllib.error.URLError, OSError) as err:
            if attempt < ENV_RETRIES:
                time.sleep(1.0)
                continue
            raise RuntimeError(f"POST /answers failed: {err}") from None
        if status != 200:
            raise RuntimeError(
                f"POST /answers answered {status}: {raw[:300]!r}")
        return json.loads(raw.decode("utf-8"))
    raise RuntimeError("POST /answers failed")


def _is_lat(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and -90.0 <= v <= 90.0


def _is_lon(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and -180.0 <= v <= 180.0


def coordinates(text):
    """The latitude and longitude an answer gives.

    Takes the LAST JSON object in the text with lat/lon keys (the final
    answer line the prompt asks for), so numbers mentioned in the reasoning
    never win. Falls back to the last two in-range numbers in the text.
    Returns (lat, lon) rounded to the microdegree, or None.
    """
    found = None
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(match.group(0))
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        lat = obj.get("lat", obj.get("latitude"))
        lon = obj.get("lon", obj.get("lng", obj.get("longitude")))
        if _is_lat(lat) and _is_lon(lon):
            found = (round(float(lat), 6), round(float(lon), 6))
    if found is not None:
        return found
    nums = [float(m.group(0)) for m in
            re.finditer(r"-?\d+(?:\.\d+)?", text)]
    for i in range(len(nums) - 2, -1, -1):
        if _is_lat(nums[i]) and _is_lon(nums[i + 1]):
            return (round(nums[i], 6), round(nums[i + 1], 6))
    return None


def ask(client, jpeg):
    """One model call for one photo. Returns the model's text, or None when
    the gateway refused the call (final) or the provider errored."""
    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(jpeg).decode("ascii")}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    }
    try:
        reply = client.messages(body)
    except Exception as err:  # LlmRefusalError / LlmError: final or failed
        log(f"model call refused or failed: {err}")
        return None
    parts = []
    for block in reply.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def main():
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env_url:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    anthropic_url = os.environ.get(
        "ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = af.Anthropic(af.LlmEndpoint(anthropic_url, anthropic_key))

    listed = env_get(env_url, "/photos")
    photos = sorted((p for p in listed.get("photos", [])
                     if not p.get("answered")),
                    key=lambda p: p["id"])
    log(f"{len(photos)} photos to place")

    answers = []
    placed = 0
    fell_back = 0
    for photo in photos:
        photo_id = photo["id"]
        jpeg = env_get_bytes(env_url, photo["url"])
        lat_lon = None
        text = ask(client, jpeg)
        if text is not None:
            lat_lon = coordinates(text)
            if lat_lon is None:
                log(f"{photo_id}: no coordinates in the answer: "
                    f"{text[:200]!r}")
        if lat_lon is None:
            lat_lon = (FALLBACK_LAT, FALLBACK_LON)
            fell_back += 1
            log(f"{photo_id}: fallback {lat_lon[0]}, {lat_lon[1]}")
        else:
            placed += 1
            log(f"{photo_id}: {lat_lon[0]}, {lat_lon[1]}")
        answers.append({"id": photo_id,
                        "lat": lat_lon[0], "lon": lat_lon[1]})

    if not answers:
        log("no photos to answer")
        return 0
    result = env_post_answers(env_url, answers)
    log(f"posted {len(answers)} answers "
        f"(model placed {placed}, fallback {fell_back}): {result}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as err:  # never crash without a word
        log(f"fatal: {err}")
        sys.exit(1)
