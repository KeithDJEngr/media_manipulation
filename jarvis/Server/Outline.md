# Server

## Overview
Python WebSocket server using `uv` virtual environment. Serves the webpage and handles all WebSocket connections for VAD, STT, LLM, and TTS.

## Setup
```bash
./start.sh
```

Or manually:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r Server/requirements.txt
.venv/bin/python Server/server.py
```

## Endpoints
- `GET /` - Serve the main webpage (ProjectInterface/index.html)
- `WS /vad` - VAD service WebSocket
- `WS /stt` - STT service WebSocket
- `WS /llm` - LLM service WebSocket
- `WS /tts` - TTS service WebSocket
- `WS /` - Browser client WebSocket

## Configuration
- `HOST` - Bind address (default: 0.0.0.0)
- `PORT` - Port number (default: 8765)

## Dependencies
- `websockets>=12.0` - WebSocket server library
