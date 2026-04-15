# Faster Qwen3-TTS — HTTP API Reference

This document describes the REST and OpenAI-compatible endpoints exposed by the demo server ([`demo/server.py`](../demo/server.py)). Run the server with:

```text
python demo/server.py --host 0.0.0.0 --port 7860
```

(or Docker as in your compose setup).

---

## Overview

| Item | Value |
|------|--------|
| **Default base URL** | `http://<host>:7860` |
| **CORS** | All origins (`*`) |
| **Max text length** | **1000** characters (`MAX_TEXT_CHARS`) on generate and OpenAI `input` |
| **Max reference upload** | **10 MiB** (`MAX_AUDIO_BYTES`) |
| **Concurrency** | One global lock serializes synthesis; `/generate/stream` may send a `queued` event while waiting |

---

## Native demo API

### `GET /`

Returns the web UI (`index.html`). Not used for programmatic TTS.

---

### `GET /status`

Server capabilities and current model state.

**200 — JSON fields**

| Field | Type | Description |
|-------|------|-------------|
| `loaded` | boolean | Whether a model is active |
| `model` | string \| null | Active Hugging Face model id |
| `loading` | boolean | A load is in progress |
| `available_models` | string[] | Models allowed (see `ACTIVE_MODELS` env) |
| `model_type` | string \| null | Active model family (clone / custom / design) |
| `speakers` | string[] | Speaker ids when a CustomVoice model is loaded |
| `transcription_available` | boolean | Whether `POST /transcribe` is available |
| `preset_refs` | object[] | `{ id, label, ref_text }` per server preset (no audio) |
| `queue_depth` | number | Waiters / lock pressure |
| `cached_models` | string[] | Models currently in the GPU LRU cache |

---

### `GET /preset_ref/{preset_id}`

Fetch one preset’s metadata and reference audio (base64).

**Path**

- `preset_id` — e.g. `aurelia_ruri`, `ref_audio`, `ref_audio_2`, `ref_audio_3` (depends on which preset files exist on the server).

**200 — JSON**

| Field | Description |
|-------|-------------|
| `id`, `label`, `filename`, `ref_text` | Preset metadata |
| `audio_b64` | Full preset file, base64-encoded |

**404** — `{"detail":"Preset not found"}`

---

### `POST /load`

Load or switch the active TTS model on the GPU (CUDA graphs, warmup).

**Content-Type:** `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `model_id` | yes | Full Hugging Face id, e.g. `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |

**200 — JSON**

- `{"status":"loaded","model":"<id>"}` — loaded after GPU work
- `{"status":"already_loaded","model":"<id>"}` — model was already in cache (instant switch)

**Notes**

- If `MODEL_CACHE_SIZE` is exceeded, the least-recently-used cached model may be evicted.
- Loading holds the same lock as generation to avoid OOM races.

**Typical full model ids** (when not restricted by `ACTIVE_MODELS`):

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`

---

### `POST /generate` (non-streaming)

Synthesize one utterance; response is JSON with a base64 WAV.

**Content-Type:** `multipart/form-data`

| Field | Default | Description |
|-------|---------|-------------|
| `text` | (required) | Input text, max 1000 chars |
| `language` | `English` | e.g. English, Chinese, French, German, Spanish, Auto |
| `mode` | `voice_clone` | One of: `voice_clone`, `custom`, `voice_design` (must match loaded model) |
| `ref_text` | empty | Reference transcript (advanced clone / ICL) |
| `speaker` | empty | Required for `custom`: built-in speaker id |
| `instruct` | empty | Style instructions for `custom` or `voice_design` |
| `xvec_only` | `true` | `true` = simple clone (embedding only); `false` = advanced ICL with `ref_text` |
| `temperature` | `0.9` | Talker sampling |
| `top_k` | `50` | Talker sampling |
| `repetition_penalty` | `1.05` | Talker sampling |
| `ref_preset` | empty | Preset id from `/status` → `preset_refs` |
| `ref_audio` | file | Optional upload; used if `ref_preset` is not set / invalid |

**200 — JSON**

```json
{
  "audio_b64": "<base64: full WAV, PCM16, mono>",
  "sample_rate": 12000,
  "metrics": {
    "total_ms": 1234,
    "audio_duration_s": 2.5,
    "rtf": 3.2
  }
}
```

**400** — Model not loaded, text too long, or bad request.

---

### `POST /generate/stream` (Server-Sent Events)

Same form fields as `POST /generate`, plus:

| Field | Default | Description |
|-------|---------|-------------|
| `chunk_size` | `8` | Codec steps per yielded chunk (~`chunk_size / 12` seconds of audio per chunk at 12 Hz) |

**Content-Type response:** `text/event-stream`

Each event is a single line followed by a blank line:

```text
data: <JSON>
```

**Event `type` values**

1. **`queued`** (optional) — `{"type":"queued","position":<int>}` — approximate queue position ahead of you.
2. **`chunk`** — Audio delta as a **complete mini-WAV** (PCM16) in base64:

   - `audio_b64`, `sample_rate`, `ttfa_ms`, `voice_clone_ms`, `rtf`, `total_audio_s`, `elapsed_ms`

3. **`done`** — `ttfa_ms`, `voice_clone_ms`, `rtf`, `total_audio_s`, `total_ms`

4. **`error`** — `message`, `detail` (may include stack trace)

**Client parsing**

- Buffer the byte stream; split on newlines; keep the last partial line.
- Only handle lines that start with `data: ` (note the space).
- JSON-parse the substring after `data: `.

---

### `POST /transcribe`

Transcribe uploaded audio (nano-parakeet), 16 kHz internally.

**Content-Type:** `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `audio` | yes | Audio file field |

**200** — `{"text":"<transcript>"}`

**503** — Transcription model not loaded.

**400** — File too large.

---

## OpenAI-compatible API

Use these endpoints when integrating tools that expect OpenAI’s speech API (e.g. Open WebUI: set **API base URL** to `http://<host>:7860/v1` so requests hit `/v1/audio/speech`).

### `GET /v1/models`

**200** — OpenAI-style model list:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
      "object": "model",
      "created": 1710000000,
      "owned_by": "qwen"
    }
  ]
}
```

Each `id` is an entry from `available_models` / `AVAILABLE_MODELS` on the server.

---

### `POST /v1/audio/speech`

**Content-Type:** `application/json`

**Body**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Full id or alias (below); server may auto-load/switch model |
| `input` | string | required | Text to speak; max 1000 chars |
| `voice` | string | server default | Preset map or custom **speaker** id (see below) |
| `response_format` | string | `mp3` | `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` |
| `speed` | number | `1.0` | **Not implemented** — ignored |
| `stream` | boolean | `false` | If `true`, response body is streamed raw bytes (not SSE) |

**Model aliases** (case-insensitive `model` value → full id)

| Alias | Resolved model |
|-------|----------------|
| `0.6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| `1.7b` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `0.6b-custom` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| `1.7b-custom` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| `1.7b-design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| `tts-1`, `tts-1-hd` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |

If the resolved id is not in `AVAILABLE_MODELS`, the server falls back to the **first** available model.

**Voice → clone preset** (when the active model is **not** custom / speaker-based)

| `voice` | Preset key |
|---------|------------|
| `alloy`, `onyx`, `nova` | `aurelia_ruri` |
| `echo` | `ref_audio` |
| `fable` | `ref_audio_2` |
| `shimmer` | `ref_audio_3` |
| `aurelia`, `aurelia_ruri` | `aurelia_ruri` |

Unknown `voice` strings are looked up as preset keys, then the server falls back to `ref_audio` if needed.

When the active model type is **custom**, `voice` is passed through as the **`speaker`** id for `generate_custom_voice*` (not the preset table above).

**Response**

- **Non-streaming:** HTTP 200 with a **single** response body (still returned as a streaming-type response in FastAPI) — full audio file bytes.
- **Streaming:** HTTP 200, body is **raw concatenated chunks** (no `data:` lines). Internal `chunk_size` is fixed at **8** for OpenAI streaming.

**Content-Type** (non-streaming and streaming use the same mapping)

| `response_format` | Content-Type |
|-------------------|----------------|
| `mp3` | `audio/mpeg` |
| `opus` | `audio/ogg` |
| `aac` | `audio/aac` |
| `flac` | `audio/flac` |
| `wav` | `audio/wav` |
| `pcm` | `audio/pcm` (raw s16le mono, no WAV header) |

**Authentication**

This demo does **not** validate API keys. Clients that require a header may send any `Authorization: Bearer <token>`.

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `ACTIVE_MODELS` | Comma-separated list of allowed Hugging Face model ids |
| `MODEL_CACHE_SIZE` | Max models kept in GPU cache (default **2**) |
| `ASSET_DIR` | Writable directory for downloaded preset assets (default `/tmp/faster-qwen3-tts-assets`) |
| `PORT` | Listen port (default **7860**) |

---

## Quick integration guide

| Goal | Endpoint | Tip |
|------|----------|-----|
| Open WebUI / OpenAI-style client | `POST http://<host>:7860/v1/audio/speech` | Base URL must end with `/v1` if the client appends `/audio/speech` |
| Lowest-latency WAV chunks | `POST /generate/stream` | Parse SSE; decode each `audio_b64` as WAV |
| One-shot WAV in JSON | `POST /generate` | Decode `audio_b64` once |
| Check server + presets | `GET /status` | |
| Switch model | `POST /load` with `model_id` | Call before generate if UI did not load a model |

---

## Example: `curl` (non-streaming generate)

PowerShell-friendly (two requests):

```powershell
curl.exe -s -X POST "http://localhost:7860/load" -F "model_id=Qwen/Qwen3-TTS-12Hz-1.7B-Base"
curl.exe -s -X POST "http://localhost:7860/generate" -F "text=Hello world." -F "mode=voice_clone" -F "language=English" -F "ref_preset=aurelia_ruri"
```

Adjust `ref_preset` / `ref_audio` / `mode` to match your loaded model and voice setup.

---

## Example: OpenAI speech (JSON)

```powershell
curl.exe -s -X POST "http://localhost:7860/v1/audio/speech" -H "Content-Type: application/json" -d "{\"model\":\"tts-1\",\"input\":\"Hello from the API.\",\"voice\":\"alloy\",\"response_format\":\"wav\"}" --output speech.wav
```

---

*Generated from the demo server implementation. If behavior diverges, trust [`demo/server.py`](../demo/server.py) as the source of truth.*
