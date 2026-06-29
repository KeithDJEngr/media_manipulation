# Jarvis Server Architecture Review

## Overview

This is a **speech-to-speech conversational AI pipeline** consisting of:
1. **Browser client** (`ProjectInterface/index.html`) -- captures mic audio, sends to server, receives transcripts and TTS audio
2. **Central WebSocket server** (`Server/jarvis_server.py`) -- 1048 lines, routes messages between all endpoints
3. **VAD service** (`STT/vad_service.py`) -- Silero VAD speech detection
4. **STT service** (`STT/stt_service.py`) -- NVIDIA Parakeet transcription
5. **LLM service** (`LLM/llm_service.py`) -- External OpenAI-compatible API client
6. **TTS service** (`TTS/tts_service.py`) -- Qwen3-TTS generation

All services connect to the central server via WebSocket. The browser connects only to `/vad`.

---

## WebSocket Endpoints

### `WS /` (Root)
- **Handler:** `handler()` -> `handle_client()` (jarvis_server.py:945-990)
- **Direction:** Browser -> Server (control messages), Server -> Browser (HTML + broadcasts)
- **Purpose:** Serves the HTML UI and handles HTTP upgrade for the main WebSocket connection. The browser opens its *only* WebSocket here for `/vad` (not `/`).
- **Browser connects:** Yes (HTTP request to serve the HTML page)
- **Backend services:** No

### `WS /vad` (Browser -> Server)
- **Handler:** `handle_vad()` (jarvis_server.py:417-462)
- **Direction:** Browser sends raw Int16 audio bytes -> Server forwards as base64 JSON to VAD service
- **Purpose:** Receives raw microphone audio chunks from the browser. Base64-encodes them and forwards to the VAD service on `/vad_service`. Increments `vad_chunk_id` counter. Has an audio heartbeat task that warns if no chunks received for 10-15s.
- **Browser connects:** Yes -- this is the browser's sole WebSocket connection
- **Backend services:** No (VAD service connects to `/vad_service`, not `/vad`)

### `WS /vad_service` (VAD Service -> Server)
- **Handler:** `handle_vad_service()` (jarvis_server.py:464-523)
- **Direction:** VAD service sends JSON control messages -> Server routes to STT and browser
- **Purpose:** Receives VAD output from the standalone `vad_service.py` process:
  - `audio_chunk`: Forwards to STT service
  - `vad_result` (speech=false): Forwards to STT service AND broadcasts `user_end` to browser
- **Browser connects:** No
- **Backend services:** `STT/vad_service.py` connects here via `wss://{host}:{port}/vad_service`

### `WS /stt` (STT Service -> Server)
- **Handler:** `handle_stt()` (jarvis_server.py:525-653)
- **Direction:** STT service sends JSON messages -> Server broadcasts to browser and forwards to LLM
- **Purpose:** Receives STT output from `stt_service.py`:
  - `partial_transcript`: Broadcasts `user_partial` to browser, updates conversation_history, forwards `user_input` (partial) to LLM service
  - `final_transcript`: Broadcasts `user_transcript` to browser, updates conversation_history, forwards `user_input` to LLM service
  - `wake_word`: Broadcasts to browser
- **Browser connects:** No
- **Backend services:** `STT/stt_service.py` connects here via `wss://{host}:{port}/stt`

### `WS /llm` (Browser -> Server, essentially unused)
- **Handler:** `handle_llm()` (jarvis_server.py:741-801)
- **Direction:** Browser sends `user_input` -> receives `llm_start`/`llm_token`/`llm_end`
- **Purpose:** Handles client connections to the `/llm` endpoint. Processes the exact same message types (`llm_start`, `llm_token`, `llm_end`, `user_input`) as `handle_llm_service`. `user_input` is forwarded to the LLM service.
- **Browser connects:** Declared as `TTS_WS_URL` in index.html (line 172) but never actually opened as a WebSocket
- **Backend services:** No
- **Actual usage:** The browser never connects to `/llm`. All LLM responses flow through the `/vad` connection. This endpoint is **dead code**.

### `WS /llm_service` (LLM Backend Service -> Server)
- **Handler:** `handle_llm_service()` (jarvis_server.py:813-894)
- **Direction:** LLM service sends `llm_start`/`llm_token`/`llm_end` -> receives `user_input`/`set_system_prompt`/`set_settings`
- **Purpose:** Receives LLM streaming output from `llm_service.py` and broadcasts to browser. On `llm_end`, forwards `tts_input` to TTS service. Maintains conversation history append. Has heartbeat tracking for service liveness.
- **Browser connects:** No
- **Backend services:** `LLM/llm_service.py` connects here via `wss://{host}:{port}/llm_service`

### `WS /tts` (TTS Service -> Server)
- **Handler:** `handle_tts()` (jarvis_server.py:896-943)
- **Direction:** TTS service sends `audio_chunk`/`audio_end` -> receives `tts_input`/`stop_generation`/`set_voice`
- **Purpose:** Receives TTS audio output and broadcasts to browser as `llm_audio`. On `audio_end`, broadcasts `audio_complete`.
- **Browser connects:** Declared as `TTS_WS_URL` in index.html but never opened as a WebSocket
- **Backend services:** `TTS/tts_service.py` connects here via `wss://{host}:{port}/tts`

---

## ConnectionManager State

The `ConnectionManager` class (jarvis_server.py:21-224) maintains:

| Attribute | Purpose |
|-----------|---------|
| `client_ws` | Main browser WebSocket (registered via `/`) |
| `vad_ws` | VAD service WebSocket |
| `vad_browser_ws` | Browser WebSocket connected to `/vad` |
| `stt_ws` | STT service WebSocket |
| `llm_client_ws` | Browser WebSocket connected to `/llm` |
| `llm_service_ws` | LLM service WebSocket connected to `/llm_service` |
| `llm_ws` | Legacy alias: points to whichever of the above connected last |
| `tts_ws` | TTS service WebSocket |
| `conversation_history` | Array of {role, content} messages, truncated to 20 entries |
| `service_status` | Dict tracking "offline"/"idle"/"active" per service |
| `service_active` | Boolean flags per service |

---

## Data Flow Pipeline

```
Browser (index.html)
    |  WS /vad (raw Int16 audio binary)
    v
Server: handle_vad()
    |  WS /vad_service (base64 audio_chunk JSON)
    v
VAD Service (vad_service.py, Silero VAD)
    |  WS /vad_service (audio_chunk JSON + vad_result JSON)
    v
Server: handle_vad_service()
    |---> WS /stt (audio_chunk JSON)
    |         v
    |      STT Service (stt_service.py, Parakeet)
    |         |  WS /stt (partial_transcript + final_transcript JSON)
    |         v
    |      Server: handle_stt()
    |         |---> Browser: user_partial / user_transcript (via /vad)
    |         |---> WS /llm_service (user_input JSON with history)
    |                    v
    |                 LLM Service (llm_service.py, external API)
    |                    |  WS /llm_service (llm_start + llm_token + llm_end JSON)
    |                    v
    |                 Server: handle_llm_service()
    |                    |---> Browser: llm_transcript / llm_end (via /vad)
    |                    |---> WS /tts (tts_input JSON)
    |                               v
    |                            TTS Service (tts_service.py, Qwen3-TTS)
    |                               |  WS /tts (audio_chunk + audio_end JSON)
    |                               v
    |                            Server: handle_tts()
    |                               |---> Browser: llm_audio / audio_complete (via /vad)
    |                               v
    |                            Browser plays audio via Web Audio API
```

---

## Duplication Analysis

### VAD: NOT duplicated (intentional separation)

- `/vad` is browser-facing (receives raw audio bytes)
- `/vad_service` is backend-facing (receives JSON from VAD service)
- Two separate handlers because the message formats differ (binary audio vs JSON)
- This is correct architecture -- different consumers need different interfaces

### LLM: TRUE duplication

- `/llm` (`handle_llm()`) and `/llm_service` (`handle_llm_service()`) handle **identical message types**:
  - `llm_start` -> broadcast to browser
  - `llm_token` -> accumulate + broadcast partial transcript
  - `llm_end` -> broadcast final + forward to TTS + append to conversation history
  - `user_input` -> forward to the other endpoint
- `handle_llm_service` additionally handles `heartbeat` messages and has `handler_cancel_scope` cleanup
- A shared helper `handle_llm_response_messages()` was already created (lines 655-738) but `handle_llm` still has its own duplicate inline implementation (lines 741-811)
- **Neither handler is actively used by the browser** -- the browser connects only to `/vad`
- The actual LLM streaming messages flow through `/llm_service` (handled by `handle_llm_service()`)
- `handle_llm()` processes the same messages but from the `/llm` endpoint, which the browser never uses

### Legacy alias chain

- `manager.llm_ws` points to whichever of `llm_client_ws` or `llm_service_ws` connected last
- `forward_message()` and `broadcast_to_service()` route via `manager.llm_ws`
- `set_settings` forwarding checks `llm_service_ws` first, falls back to `llm_ws`
- This creates ambiguity: is `llm_ws` pointing to the browser connection or the service connection?

### Unused endpoints in browser

- `TTS_WS_URL` is defined in index.html but never opened as a WebSocket
- The `/llm` endpoint is never opened by the browser either
- All browser communication flows through the single `/vad` connection

---

## Issues Summary

1. **`/llm` endpoint is dead code** -- the browser never connects to it. Both `handle_llm()` and `handle_llm_service()` do the same thing but `/llm` is never used.
2. **`/tts` endpoint is dead code** -- the browser declares `TTS_WS_URL` but never opens it. TTS audio arrives via the `/vad` broadcast path.
3. **`llm_ws` legacy alias ambiguity** -- unclear which connection it references at any given time.
4. **`handle_llm()` still has duplicate logic** -- the `handle_llm_response_messages()` helper exists but `handle_llm()` doesn't use it.
5. **`handle_llm()` vs `handle_llm_service()`** -- nearly identical code blocks (llm_start/token/end handling) duplicated across two functions.
6. **`handle_vad()` vs `handle_vad_service()`** -- correctly separated but could be unified under a single handler with connection-type detection.
