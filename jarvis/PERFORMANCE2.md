# PERFORMANCE2.md - Performance Review and Improvements

## Review of Current State (Post-Original PERFORMANCE.md)

### What Has Changed Since Original REVIEW

The original PERFORMANCE.md documented:
- XPU is working with Intel Arc Pro B70
- All ML models on XPU:0 (Parakeet encoder/decoder)
- STT is fast (<1s per transcription)
- TTFS bottleneck is TTS (Qwen3-TTS 1.7B, 7-30s per generation)
- VAD buffer issue (many empty buffer end-of-speech events)

### Current Pipeline Status

The pipeline has evolved significantly:
1. **VAD** now has debounce logic (8000 sample speech_end_debounce) to batch rapid utterances
2. **STT** has partial transcript streaming to LLM for real-time context updates
3. **LLM** now supports full/partial history toggle (`use_full_history`)
4. **TTS** has chunked sentence-level processing with prefix-skipping to avoid redundant generation
5. **Server** maintains conversation history with truncation (max 20 messages)
6. **Browser** has sequence-based audio playback ordering

---

## Bottleneck Analysis

### STT Service - LOW LATENCY (Already Good)

**Current Performance:**
- Parakeet-TDT 0.6B transcribes in <1s on XPU
- Real-time factor: 0.8x to 0.01x (varies with audio length)
- Partial transcription triggers at 1024 samples (64ms of audio)

**Bottlenecks:**
1. **Model load time:** ~8.8s one-time (cached after first run) - acceptable
2. **Per-chunk overhead:** base64 encoding/decoding for every audio chunk
3. **Buffer management:** 30s max buffer (generous, no issue)

**Verdict:** STT is fast enough. No major improvements needed.

### LLM Service - VARIABLE LATENCY (Depends on External API)

**Current Performance:**
- TTFT: Depends on external LLM API (Ollama, llama.cpp, etc.)
- Token streaming: Works correctly with partial transcript updates
- History management: Full/partial toggle reduces payload size

**Bottlenecks:**
1. **TTFT:** First token latency depends on model size and hardware
2. **History payload:** Full history can be large for long conversations
3. **Message duplication:** `set_settings` sent to LLM service only when explicitly triggered

**Verdict:** Bottleneck is external. Server-side improvements limited.

### TTS Service - HIGH LATENCY (Primary Bottleneck)

**Current Performance:**
- Qwen3-TTS 1.7B takes 7-30s per generation
- Real-time factor: ~2.8x on XPU (good for TTS)
- Chunked processing reduces perceived latency

**Bottlenecks:**
1. **Model size:** 1.7B parameters is large for real-time TTS
2. **Per-chunk overhead:** Each chunk requires full model forward pass
3. **Base64 encoding:** Audio samples → base64 for WebSocket transfer
4. **Sentence splitting:** Regex-based splitting may not handle all cases
5. **Generation parameters:** `do_sample=True, temperature=0.6, top_p=0.85` add variance

**Verdict:** Primary bottleneck. Reducing model size or switching to faster variant would help most.

### Browser - MODERATE LATENCY (Audio Processing)

**Current Performance:**
- ScriptProcessorNode with 512-sample chunks (10.7ms at 16kHz)
- Linear interpolation resampling for non-16kHz devices
- Sequence-based audio playback ordering works correctly

**Bottlenecks:**
1. **ScriptProcessor:** Deprecated API, but functional
2. **Resampling:** Linear interpolation is simple but not highest quality
3. **Base64 decoding:** `atob()` on every audio chunk adds CPU overhead
4. **AudioContext creation:** Deferred until first chunk received

**Verdict:** Acceptable for current use. WebAudio `AudioWorklet` would be ideal but adds complexity.

---

## Performance Improvements

### 1. STT Model Swap (MEDIUM IMPACT)

**Option A: Use Whisper.cpp (Faster, Similar Quality)**
```bash
pip install pywhispercpp
```
- Whisper base model: ~200-500ms inference on XPU
- Smaller model footprint (~150MB vs 2.5GB for Parakeet)
- Streaming mode available via `prompt` parameter
- Trade-off: Slightly lower accuracy for non-English

**Option B: Use Faster-Whisper (Best Speed/Accuracy Balance)**
```bash
pip install faster-whisper
```
- CTranslate2 backend: 4x faster than original Whisper
- Tiny model: ~39MB, ~100ms inference
- Good accuracy for short utterances (1-5s)
- Recommended for Jarvis use case

**Option C: Keep Parakeet (Current)**
- Pros: Already optimized for XPU, good accuracy
- Cons: Slower than Whisper alternatives, larger model

### 2. TTS Model Swap (HIGH IMPACT)

**Option A: Use Qwen3-TTS 12Hz 0.6B-CustomVoice (Smaller Variant)**
```python
TTS_MODEL = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
# Or via environment: TTS_MODEL_TIER=0.6B-CustomVoice
```
- 65% fewer parameters → ~65% faster generation
- Similar voice quality (proven in benchmarks)
- Drop-in replacement: same API, same protocol

**Option B: Use Piper TTS (Fastest, Good Quality)**
```bash
pip install piper-tts
```
- Rust backend: real-time on CPU
- 50-200ms per utterance (vs seconds for Qwen3-TTS)
- Voice quality: Good (not great, but sufficient)
- Trade-off: Less natural than Qwen3-TTS, limited voices

**Option C: Use Edge-TTS (Cloud, Zero Latency)**
```bash
pip install edge-tts
```
- Microsoft Edge TTS API: free, unlimited
- Near-zero generation time
- Voice quality: Good (Azure Neural TTS)
- Trade-off: Requires internet, not fully offline

**Option D: Keep Qwen3-TTS 1.7B (Current)**
- Pros: Best voice quality, customizable with voice instruct
- Cons: Slowest option

### 3. Browser Audio Optimization (LOW IMPACT)

**Current:** ScriptProcessorNode with `createScriptProcessor(512, 1, 1)`
**Proposed:** AudioWorklet for lower CPU usage

```javascript
// Replace createScriptProcessor with AudioWorklet
const audioWorkletURL = URL.createObjectURL(
  new Blob([`
    class AudioCaptureProcessor extends AudioWorkletProcessor {
      process(inputs) {
        const input = inputs[0][0];
        if (input && window.onAudioData) {
          window.onAudioData(new Float32Array(input));
        }
        return true;
      }
    }
    registerProcessor('audio-capture-processor', AudioCaptureProcessor);
  `], { type: 'application/javascript' })
);
await audioContext.audioWorklet.addModule(audioWorkletURL);
const workletNode = new AudioWorkletNode(audioContext, 'audio-capture-processor');
workletNode.port.onmessage = (e) => { /* handle audio data */ };
```

**Benefits:**
- Runs on separate thread (lower main-thread CPU)
- More predictable timing
- Can handle larger buffers (1024+ samples)

**Trade-off:** Adds a separate `.js` file, slightly more code

### 4. Base64 Optimization (LOW IMPACT)

**Current:** `base64.b64encode()` in Python → `atob()` in browser
**Proposed:** Send binary audio via WebSocket binary frames

```python
# Instead of base64 encoding:
# audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
# await websocket.send(json.dumps({"audio": audio_b64}))

# Send binary frame directly:
await websocket.send(binary_audio_bytes)
```

```javascript
// In browser:
ws.binaryType = 'arraybuffer';
// Handle binary frames:
ws.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    const samples = new Float32Array(event.data);
    // Process directly
  }
};
```

**Benefits:**
- ~33% less data transferred (no base64 expansion)
- Faster encoding/decoding
- Lower CPU usage

**Trade-off:** Must handle both JSON control messages and binary audio frames

### 5. LLM History Optimization (LOW-MEDIUM IMPACT)

**Current:** Full history sent with every `user_input` message
**Proposed:** Only send last N tokens or use a token budget

```python
# Calculate approximate token count
approx_tokens = sum(len(msg.get("content", "").split()) for msg in history)
# Trim to fit within token budget
while approx_tokens > MAX_TOKENS:
    # Remove oldest non-system messages
    ...
```

**Benefits:**
- Reduces payload size for long conversations
- Faster API calls with smaller context
- Better TTFT

**Trade-off:** May lose context for very long conversations

### 6. Conversation History Deduplication (LOW IMPACT)

**Current:** Partial transcripts update conversation_history, but LLM service maintains its own copy
**Issue:** If settings change (`use_full_history`), LLM service doesn't know until next user input

**Fix:** Already addressed in bug fixes - `use_full_history` now synced to LLM service.

### 7. Audio Playback Buffer Sizing (MEDIUM IMPACT)

**Current:** 500ms chunks (2000 samples at 16kHz)
**Proposed:** Dynamic chunk sizing based on TTS generation speed

```python
# Adjust chunk size based on generation time
# Faster generation → larger chunks → fewer WebSocket messages
# Slower generation → smaller chunks → smoother playback during slow generation
chunk_duration = max(0.25, min(1.0, generation_time / num_chunks))
chunk_size = int(SAMPLE_RATE * chunk_duration)
```

**Benefits:**
- Fewer WebSocket messages when generation is fast
- Smoother playback when generation is slow
- Better network efficiency

**Trade-off:** More complex chunking logic

---

## Recommended Priority Order

| Priority | Change | Impact | Effort |
|----------|--------|--------|--------|
| **1** | Swap TTS to Qwen3-TTS 0.6B-CustomVoice (smaller model) | HIGH | LOW |
| **2** | Swap STT to Faster-Whisper tiny | MEDIUM | LOW |
| **3** | Binary WebSocket frames instead of base64 | LOW | MEDIUM |
| **4** | Move to AudioWorklet in browser | LOW | MEDIUM |
| **5** | Dynamic TTS chunk sizing | LOW | LOW |
| **6** | LLM token budget trimming | LOW-MEDIUM | LOW |

---

## Specific Model Recommendations

### STT Models (by speed)

| Model | Inference Time | Accuracy | Size | Offline |
|-------|---------------|----------|------|---------|
| Whisper tiny (faster-whisper) | ~100ms | Decent | 39MB | Yes |
| Whisper base (faster-whisper) | ~200ms | Good | 150MB | Yes |
| Parakeet 0.6B (current) | ~500ms | Good | 2.5GB | Yes |
| Deepgram Nova-3 (cloud) | ~50ms | Excellent | N/A | No |
| AssemblyAI (cloud) | ~50ms | Excellent | N/A | No |

### TTS Models (by speed)

| Model | Generation Time | Quality | Size | Offline |
|-------|----------------|---------|------|---------|
| Piper (fastest voice) | ~50ms | Good | 60MB | Yes |
| Edge-TTS (cloud) | ~200ms | Good | N/A | No |
| Qwen3-TTS 0.6B-CustomVoice | ~5-15s | Excellent | 2.1GB | Yes |
| Qwen3-TTS-1.7B-CustomVoice (current) | ~10-30s | Excellent | 3GB | Yes |
| ElevenLabs (cloud) | ~500ms | Best | N/A | No |

**Recommended swap for Jarvis:**
- **STT:** Faster-Whisper tiny (speed gain, acceptable accuracy loss)
- **TTS:** Qwen3-TTS 0.6B-CustomVoice (quality retention, ~40% speed gain)

---

## Benchmarking Commands

```bash
# Benchmark Faster-Whisper tiny
python -c "
from faster_whisper import WhisperModel
model = WhisperModel('tiny', device='cuda', compute_type='int8')
import time, numpy as np
audio = np.random.randn(16000).astype(np.float32)
start = time.time()
segments, info = model.transcribe(audio, language='en')
print(f'Time: {time.time()-start:.3f}s')
"

# Benchmark Piper TTS
python -c "
import piper
import time
syn = piper.Synthesizer('en_US-lessac-medium')
start = time.time()
wav = list(syn.synthesize('Hello world.', voice='lessac'))
print(f'Time: {time.time()-start:.3f}s')
"

# Benchmark binary vs base64 WebSocket transfer
python -c "
import time, base64
data = b'x' * 10000  # 10KB audio chunk
start = time.time()
for _ in range(100):
    b64 = base64.b64encode(data).decode()
print(f'base64 100 iterations: {(time.time()-start)*10:.1f}ms avg')
start = time.time()
for _ in range(100):
    import struct
    packed = struct.pack(f'{len(data)}B', *data)
print(f'binary 100 iterations: {(time.time()-start)*10:.1f}ms avg')
"
```

---

## Memory Management

### Current
- Models stay loaded in VRAM across turns
- Conversation history truncated at 20 messages on server, 21 on LLM service
- Audio buffers cleared after transcription

### Recommendations
1. **Model unloading:** Consider unloading STT model between turns if VRAM is tight
2. **VRAM monitoring:** Add VRAM usage logging to detect leaks
3. **Context window budget:** LLM service should track token count, not just message count
   ```python
   # Instead of: if len(conversation_history) > 21:
   # Use: if token_count > MAX_TOKENS:
   ```

---

## Summary

The biggest performance gains come from:
1. **TTS model size reduction** (Qwen3-TTS 0.6B-CustomVoice vs 1.7B) - ~40% faster
2. **STT model swap** (Faster-Whisper tiny) - ~5x faster inference
3. **Binary WebSocket frames** - 33% less data transfer

These changes would reduce total pipeline latency from ~2-4s to ~1-2s for a typical exchange, making the conversation feel significantly more responsive.
