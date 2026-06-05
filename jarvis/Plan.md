# Minimal Delay Speech-to-Speech Pipeline

## Architecture: Overlapping Stages (No Stage Waits for Previous to Finish)

```
Audio In → [VAD] → [STT streaming] → [LLM streaming] → [TTS streaming] → Audio Out
              ↓          ↓                ↓                ↓
          small       partial         partial          partial
         chunks     transcripts      tokens          audio
```

## Stage-by-Stage Optimization

### 1. VAD (Silero VAD) - Target: < 50ms
- Use small analysis windows (20-30ms chunks, 480-720 samples at 16kHz)
- Use `speech_tokenizer` with minimal delay mode
- Set short speaking timeout (e.g., 500ms) to detect endpoint quickly
- Use `return_seconds` instead of frame indices for direct sample mapping
- **Critical**: Start STT on the first speech segment immediately, don't wait for silence

### 2. STT (Parakeet-TDT Streaming) - Target: < 300ms latency
- Feed audio chunks to STT as they arrive from VAD (don't buffer)
- Use Parakeet's streaming/incremental mode if available
- Emit partial transcripts as soon as each chunk processes (~50-100ms per chunk)
- Use smaller context window, cache previous chunk embeddings for speed
- On VAD silence endpoint, send final chunk + "END" signal to LLM

### 3. LLM (Qwen3.6 35B A3B) - Target: < 50ms/token first token
- **Stream partial STT transcript to LLM** as it arrives (not waiting for sentence complete)
- Use a system prompt that handles partial/incomplete input gracefully
- Enable KV cache and continuous batching
- Use speculative decoding if possible (smaller draft model)
- **First token latency (TTFT)** is critical - optimize with:
  - Smaller context window (trim old conversation history)
  - Prefix caching for repeated prompt patterns
  - FlashAttention-2 for inference speed
- Stream generated tokens to TTS immediately (don't wait for full response)

### 4. TTS (Qwen3-TTS) - Target: < 200ms audio generation
- Start generating audio for first TTS tokens while LLM is still generating
- Use chunked TTS inference (process token chunks, not full sentence)
- Pre-warm TTS model on startup
- Use faster TTS variant (smaller model or distilled version)
- Overlap TTS audio output with remaining LLM generation

## Pipeline Synchronization

```
Time →

VAD:   [chunk1][chunk2][chunk3]...[silence detect]
STT:      [partial1][partial2][partial3]...[final]
LLM:          [first token...more tokens...done]
TTS:             [audio chunk1][audio chunk2][output]
```

### Key Design Decisions:

1. **STT → LLM boundary**: Send partial transcript to LLM with a marker like `[partial]text[partial]` so LLM knows it may not be complete. Only send `[end_turn]` when VAD confirms silence.

2. **LLM → TTS boundary**: LLM streams tokens to TTS. TTS processes tokens in small batches (e.g., every 4-8 tokens) to start producing audio quickly.

3. **Ring buffer for audio output**: Buffer ~200-500ms of generated audio to smooth out TTS chunk timing variations and provide seamless playback.

4. **Parallel warmup**: Pre-load VAD, STT, LLM, and TTS models on startup. Keep models in VRAM.

5. **Conversation management**: Keep conversation history trimmed to last N turns or token budget. Archive old context to CPU RAM if needed.

## Estimated Total Latency

| Stage | First Latency | Steady State |
|-------|---------------|--------------|
| VAD endpoint | ~200ms (silence detection) | ~200ms |
| STT first result | ~100ms | ~100ms |
| LLM first token | ~200-400ms (model dependent) | ~50ms/token |
| TTS first audio | ~100ms | ~50ms/chunk |
| **Per-turn total** | | **~600-1200ms** |

The bottleneck will be LLM generation speed. For conversational feel, aim for TTFT under 400ms and fast token streaming so audio starts within ~1 second of user finishing speech.
