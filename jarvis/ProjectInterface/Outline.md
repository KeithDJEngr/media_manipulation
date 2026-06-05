# ProjectInterface

## Overview
A split-pane webpage for the speech-to-speech interface. Left pane shows user interaction, right pane shows LLM responses.

## Layout
- **Left Pane (User)**: Start/Stop button, user transcription display
- **Right Pane (LLM)**: LLM transcription display with streaming cursor

## Audio Flow
1. Microphone audio captured at 16kHz mono
2. Audio chunks (1024 samples) sent via WebSocket to VAD service
3. User transcriptions received via WebSocket and displayed
4. LLM transcriptions received via WebSocket and displayed with streaming cursor
5. LLM audio chunks received via WebSocket and played back
6. LLM audio stops immediately when user speaks again (interrupt)

## WebSocket Protocol
All WebSocket communication uses a single connection at `ws://localhost:8765/vad` with JSON messages:

### Server → Client Messages:
- `user_transcript` - Final user transcription with `text` and `final: true`
- `user_partial` - Partial user transcription with `text`
- `llm_transcript` - LLM transcription (partial with `final: false`, final with `final: true`)
- `llm_audio` - Base64-encoded audio chunk with `audio` field
- `user_start` - User started speaking
- `user_end` - User finished speaking, LLM processing
- `llm_start` - LLM started responding
- `llm_end` - LLM finished responding
- `interrupt` - User interrupted LLM, clear LLM output

## Files
- `index.html` - Main webpage with audio capture, streaming, and playback
