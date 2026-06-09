# ProjectInterface

## Overview
An upper thin pane with a start/stop button (for listening/stopping audio output) and a reset button (clearing the conversation). Underneath that is a scrollable pane for the conversation. It should be laid out like a text message - user on left and then the llm response down and to the right. If more speech is detected, the llm response is interrupted and the user response is transcribed beneath it to the right. It should always be user on the left, llm on the right and the next message showing up below.

Specifically, do the following:
Start/Stop button, Reset button
User/LLM response

## Layout
- **Upper Control Pane**: Start/Stop button (left) and Reset button (right), with status indicators for user and LLM states
- **Scrollable Conversation Pane**: Chat-style message bubbles with user messages aligned left (blue-tinted) and LLM messages aligned right (green-tinted), separated by dividers between turns

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
- `user_transcript` - Final user transcription with `text`, `final: true`, and `chunk_id`
- `user_partial` - Partial user transcription with `text` and `chunk_id`
- `llm_transcript` - LLM transcription with `text`, `partial: true` (streaming), and `turn_id`
- `llm_audio` - Base64-encoded audio chunk with `audio`, `turn_id`, and `audio_seq` fields
- `user_start` - User started speaking
- `user_end` - User finished speaking, LLM processing
- `llm_start` - LLM started responding with `turn_id`
- `llm_end` - LLM finished responding with `turn_id`
- `interrupt` - User interrupted LLM, clear LLM output
- `audio_complete` - Audio generation complete with `turn_id`
- `reset_complete` - Conversation reset complete

### Message Deduplication:
- User partial and final transcripts share the same `chunk_id` - UI updates the same message for both
- LLM tokens are accumulated per `turn_id` - UI maintains single message per turn
- Audio chunks use `turn_id` and `audio_seq` for proper ordering across turns

## Features
- **Scrollable conversation history**: Thin styled scrollbar for reviewing past messages
- **Message pairing**: User message + LLM response form a visual turn, separated by dividers
- **Streaming cursor**: Blinking cursor appears during LLM generation
- **Turn-based audio**: Audio queues are cleared on new turn to prevent out-of-order playback
- **Short transcription filter**: Transcriptions under 3 words are skipped to avoid filler words triggering LLM responses
- **Empty LLM response handling**: LLM messages with no content are removed from the UI

## Files
- `index.html` - Main webpage with audio capture, streaming, and playback
