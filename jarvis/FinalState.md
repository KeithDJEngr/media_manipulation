# Jarvis Server Consolidation Plan

## Goal

Simplify the WebSocket architecture by removing dead endpoints, consolidating duplicate handlers, and eliminating legacy alias ambiguity -- all while preserving the current external behavior (same WebSocket paths, same message formats, same browser connection).

---

## Changes

### 1. Remove `/llm` endpoint and `handle_llm()`

The browser never connects to `/llm`. All LLM streaming messages flow through `/llm_service`. Remove:
- `handle_llm()` function (jarvis_server.py:741-811) -- ~71 lines of duplicate code
- Route `elif path == "/llm"` (jarvis_server.py:983)
- `llm_client_ws` attribute from ConnectionManager (line 29) -- the browser never connects to `/llm`
- `register_llm(websocket, is_service=False)` branch in ConnectionManager (lines 100-108)
- `unregister_llm()` elif branch for `llm_client_ws` (lines 118-119)
- `set_service_status("llm", "idle")` call in the non-service registration branch

**Impact:** None -- no consumer connects to `/llm`. The browser receives all LLM output via the `/vad` broadcast path.

### 2. Consolidate `handle_llm_service()` using `handle_llm_response_messages()`

The `handle_llm_response_messages()` helper (lines 655-738) already exists but `handle_llm_service()` ignores it. Refactor:

```python
async def handle_llm_service(websocket):
    """Handle LLM service connection to /llm_service endpoint."""
    await manager.register_llm(websocket, is_service=True)
    await websocket.send(json.dumps({"type": "start"}))
    logger.info("LLM service ready")
    
    accumulated_llm_text = ""
    current_turn_id = 0
    last_heartbeat = {}
    handler_cancel_scope = []
    heartbeat_task = await create_heartbeat_task(last_heartbeat, handler_cancel_scope)
    
    try:
        while True:
            result = await handle_llm_response_messages(
                websocket, accumulated_llm_text,
                [current_turn_id], last_heartbeat, handler_cancel_scope,
                "LLM-SERVICE"
            )
            accumulated_llm_text, current_turn_id = result
    except Exception as e:
        logger.error(f"LLM service error: {e}")
    finally:
        for t in handler_cancel_scope:
            t.cancel()
        await manager.set_service_active("llm", False)
        await manager.unregister_llm(websocket)
```

**Impact:** Eliminates ~70 lines of duplicated `llm_start`/`llm_token`/`llm_end`/`user_input` logic from `handle_llm_service()`.

### 3. Rename `llm_ws` to `llm_service_ws` everywhere

Remove the legacy alias `llm_ws` and use `llm_service_ws` consistently:

**Find and replace:**
- `self.llm_ws` -> `self.llm_service_ws` in ConnectionManager initialization (line 31)
- `self.llm_ws = websocket` -> `self.llm_service_ws = websocket` in `register_llm()` (line 106) -- remove the `if is_service` guard since there's only one LLM connection now
- `self.llm_ws = websocket` -> `self.llm_service_ws = websocket` in `register_llm()` fallback (line 108) -- remove `elif` since only service connects
- `if self.llm_ws == websocket:` in `unregister_llm()` (line 121) -> `if self.llm_service_ws == websocket:`
- All references in forwarding logic: `manager.llm_ws` -> `manager.llm_service_ws`
  - `broadcast_to_service()` (line 157)
  - `forward_message()` (line 176)
  - `handle_client_message()` reset (line 343)
  - `handle_client_message()` set_settings fallback (line 380)
  - `handle_client_message()` set_settings final send (line 391)
  - `handle_stt()` partial_transcript (line 591)
  - `handle_stt()` final_transcript (line 626)

**Impact:** Removes ambiguity. `llm_service_ws` is the only LLM connection.

### 4. Simplify ConnectionManager.register_llm()

After removing `llm_client_ws`, the registration simplifies to:

```python
async def register_llm(self, websocket):
    self.llm_service_ws = websocket
    self.service_status["llm"] = "idle"
    logger.info("LLM service connected")
    await self.set_service_status("llm", "idle")

async def unregister_llm(self, websocket):
    if self.llm_service_ws == websocket:
        self.llm_service_ws = None
        self.service_active["llm"] = False
        self.service_status["llm"] = "offline"
        logger.info("LLM service disconnected")
        await self.set_service_status("llm", "offline")
```

Remove the `is_service` parameter. Update the call in `handle_llm_service()` to `await manager.register_llm(websocket)`.

### 5. Keep `/vad` and `/vad_service` as-is

These are correctly separated:
- `/vad` receives binary audio from browser (different format than JSON)
- `/vad_service` receives JSON from the VAD service

No consolidation needed. The naming (`/vad` vs `/vad_service`) clearly distinguishes browser-facing from backend-facing.

### 6. Document that `/llm` is deprecated

Update the logging line (jarvis_server.py:1037) from:
```python
logger.info(f"WebSocket endpoints: /vad, /stt, /llm, /tts, /")
```
to:
```python
logger.info(f"WebSocket endpoints: /vad, /stt, /llm_service, /tts, /")
```

### 7. No changes needed to browser client

The browser already connects only to `/vad` and receives all output via broadcasts. No client-side changes required.

### 8. No changes needed to backend services

All services connect to their expected endpoints:
- `vad_service.py` -> `/vad_service` (unchanged)
- `stt_service.py` -> `/stt` (unchanged)
- `llm_service.py` -> `/llm_service` (unchanged)
- `tts_service.py` -> `/tts` (unchanged)

---

## Summary of Net Changes

| Item | Lines Removed | Lines Added | Notes |
|------|---------------|-------------|-------|
| Remove `handle_llm()` | ~71 | 0 | Dead endpoint handler |
| Remove `llm_client_ws` and related | ~15 | ~5 | Simplified registration |
| Consolidate `handle_llm_service()` | ~60 | ~10 | Uses existing helper |
| Rename `llm_ws` -> `llm_service_ws` | 0 | ~10 | Find/replace references |
| Remove route entry | 1 | 0 | No `/llm` route needed |
| **Total** | **~147** | **~25** | **Net ~122 lines removed** |

---

## After: Endpoints

| Endpoint | Handler | Consumer | Purpose |
|----------|---------|----------|---------|
| `/` | `handler()` -> HTTP serve | Browser | Serve HTML UI |
| `/vad` | `handle_vad()` | Browser | Receive raw audio, forward to VAD service |
| `/vad_service` | `handle_vad_service()` | VAD service | Receive VAD output, route to STT + browser |
| `/stt` | `handle_stt()` | STT service | Receive transcripts, broadcast to browser, forward to LLM |
| `/llm_service` | `handle_llm_service()` (consolidated) | LLM service | Receive LLM streaming, broadcast to browser, forward to TTS |
| `/tts` | `handle_tts()` | TTS service | Receive TTS audio, broadcast to browser |

---

## Migration Notes

- **Backward compatible:** No WebSocket path changes. External consumers see identical API.
- **`/llm` path returns `1008 Path not found`:** Any client that was connecting to `/llm` will be closed with this code. Verify no clients use it.
- **`llm_ws` alias gone:** Any external reference to `manager.llm_ws` must use `manager.llm_service_ws`.
