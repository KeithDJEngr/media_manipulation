# EXTENSIONS.md

## Swappable Components

The Jarvis speech-to-speech pipeline is designed so that each component (VAD, STT, LLM, TTS) can be swapped independently with minimal changes. This document describes how to replace each component.

### Architecture

```
Browser → [VAD] → [STT] → [LLM] → [TTS] → Browser
```

Each component communicates via WebSocket messages. To swap a component, you only need to:
1. Write a new service that connects to the server's WebSocket endpoint
2. Follow the message protocol (described below)
3. Start the service (it auto-registers with the server)

### Shared Message Protocol

All services connect to `wss://host:port/{service}` (or `/llm_service` for LLM backend).

**Common message types:**
- `{"type": "start"}` — Send immediately after connecting to acknowledge readiness
- `{"type": "service_status", "service": "vad|stt|llm|tts", "status": "connected|disconnected|error"}` — Broadcast service health to the UI

---

## VAD (Voice Activity Detection)

**Current:** `STT/vad_service.py` — Silero VAD ONNX

**Server endpoint:** `/vad` (receives audio from browser), `/vad_service` (receives VAD output)

### Protocol

**Server → your service (on `/vad_service`):**
```json
{"type": "audio_chunk", "audio": "<base64 int16>", "chunk_id": 1}
{"type": "start"}  // Acknowledgment
```

**Your service → server:**
```json
{"type": "vad_result", "speech": true, "chunk_id": 1}
{"type": "vad_result", "silence": true, "chunk_id": 1}
{"type": "audio_chunk", "audio": "<base64 int16>", "chunk_id": 1}
```

### To swap VAD

1. Copy `STT/vad_service.py` to a new file (e.g., `STT/my_vad.py`)
2. Replace the model loading in `load_vad_model()` with your VAD implementation
3. Replace `VADProcessor.process_audio()` to return messages matching the protocol above
4. Update `start.sh` to launch your service instead:
   ```bash
   SERVER_HOST=${HOST} SERVER_PORT=${PORT} nohup .venv/bin/python STT/my_vad.py > /tmp/jarvis_vad.log 2>&1 &
   ```

### Alternative VAD options

| Library | Install | Notes |
|---------|---------|-------|
| [WebRTCVAD](https://pypi.org/project/webrtcvad/) | `pip install webrtcvad` | Lightweight, no model downloads, less accurate |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | `pip install pyannote.audio` | State-of-the-art, requires pytorch |
| [Silero VAD (PyTorch)](https://github.com/snakers4/silero-vad) | `pip install silero-vad` | Same model as current, PyTorch backend instead of ONNX |

**WebRTCVAD example** (drop-in replacement for `VADProcessor`):
```python
import webrtcvad

class MyVADProcessor:
    def __init__(self, sensitivity=3):
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(sensitivity)  # 0-3, higher = more aggressive
        self.is_speaking = False
        self.speech_buffer = []
        self.chunk_id = 0

    def process_audio(self, audio_bytes):
        messages = []
        self.chunk_id += 1
        # WebRTCVAD expects 10ms frames at 16kHz = 160 samples
        frames = [audio_bytes[i:i+160] for i in range(0, len(audio_bytes), 160)]
        for frame in frames:
            is_speech = self.vad.is_speech(frame, 16000)
            if is_speech and not self.is_speaking:
                self.is_speaking = True
                messages.append({"type": "vad_result", "speech": True, "chunk_id": self.chunk_id})
                self.speech_buffer = []
            elif not is_speech and self.is_speaking:
                self.is_speaking = False
                messages.append({"type": "vad_result", "silence": True, "chunk_id": self.chunk_id})
                if self.speech_buffer:
                    full = b''.join(self.speech_buffer)
                    messages.append({"type": "audio_chunk", "audio": base64.b64encode(full).decode(), "chunk_id": self.chunk_id})
                    self.speech_buffer = []
            elif is_speech:
                self.speech_buffer.append(frame)
        return messages
```

---

## STT (Speech-to-Text)

**Current:** `STT/stt_service.py` — NVIDIA Parakeet-TDT 0.6B

**Server endpoint:** `/stt`

### Protocol

**Server → your service:**
```json
{"type": "audio_chunk", "audio": "<base64 int16>", "chunk_id": 1}
{"type": "end_of_speech", "chunk_id": 1}
```

**Your service → server:**
```json
{"type": "partial_transcript", "text": "partial text", "chunk_id": 1}
{"type": "final_transcript", "text": "final text", "final": true, "chunk_id": 1}
```

### To swap STT

1. Copy `STT/stt_service.py` to `STT/my_stt.py`
2. Replace `ParakeetSTT` class with your STT implementation
3. The `add_audio_chunk()`, `transcribe()`, and `transcribe_partial()` methods must match the interface
4. Update `start.sh` to launch your service

### Alternative STT options

| Model | Install | Latency | Accuracy | Notes |
|-------|---------|---------|----------|-------|
| [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) via pywhispercpp | `pip install pywhispercpp` | Fast | Good | C++ backend, CPU/GPU |
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | `pip install faster-whisper` | Fast | Good | CTranslate2 backend |
| [Whisper Large-v3](https://huggingface.co/openai/whisper-large-v3-turbo) | transformers | Medium | Best | HuggingFace, needs GPU |
| [Deepgram Nova-3](https://developers.deepgram.com/) | API | Fastest | Best | Cloud API, $0.006/min |
| [AssemblyAI](https://www.assemblyai.com/) | API | Fastest | Best | Cloud API, $0.00025/min |
| [Vosk](https://github.com/alphacep/vosk-api) | `pip install vosk` | Fast | Decent | Offline, small models |

**Whisper.cpp example** (minimal STT service):
```python
# pip install pywhispercpp
import pywhispercpp
import numpy as np
import base64

class WhisperSTT:
    def __init__(self, model="base"):
        self.model = pywhispercpp.WhisperModel(model)
        self.buffer = np.array([], dtype=np.float32)

    def add_audio_chunk(self, audio_bytes):
        int16_arr = np.frombuffer(audio_bytes, dtype=np.int16)
        float32_arr = int16_arr.astype(np.float32) / 32768.0
        self.buffer = np.concatenate([self.buffer, float32_arr])

    def transcribe(self):
        if len(self.buffer) < 0.5:  # Need at least 0.5s
            return ""
        text = self.model.detect_language(self.buffer)[1]
        self.buffer = np.array([], dtype=np.float32)
        return text
```

---

## LLM (Language Model)

**Current:** `LLM/llm_service.py` — External OpenAI-compatible API (llama.cpp server, Ollama, etc.)

**Server endpoints:** `/llm` (browser-facing), `/llm_service` (backend-facing)

### Protocol

**Server → your service (on `/llm_service`):**
```json
{"type": "user_input", "text": "user said", "history": [...]}
{"type": "reset_history"}
```

**Your service → server (on `/llm_service`):**
```json
{"type": "llm_start"}
{"type": "llm_token", "text": "token", "partial": true}
{"type": "llm_end", "text": "full response"}
```

### To swap LLM

1. Copy `LLM/llm_service.py` to `LLM/my_llm.py`
2. Replace the HTTP streaming logic in `handle_llm()` with your LLM client
3. Must send `llm_start`, `llm_token`, `llm_end` messages back to the server
4. Update `start.sh` to launch your service

### Alternative LLM options

| Backend | Install | Local? | Notes |
|---------|---------|--------|-------|
| [Ollama](https://ollama.com/) | `curl -fsSL https://ollama.com/install.sh \| sh` | Yes | Easiest local LLM |
| [llama.cpp server](https://github.com/ggerganov/llama.cpp) | `make server` | Yes | OpenAI-compatible API |
| [vLLM](https://docs.vllm.ai/) | `pip install vllm` | Yes | High-throughput serving |
| [LM Studio](https://lmstudio.ai/) | Download | Yes | GUI, OpenAI-compatible API |
| [OpenAI API](https://platform.openai.com/) | API key | Cloud | GPT-4o, fastest first token |
| [Anthropic Claude](https://anthropic.com/) | API key | Cloud | Claude 3.5 Sonnet |
| [Google Gemini](https://ai.google.dev/) | API key | Cloud | Gemini 1.5 Pro |

**Ollama example** (change the API_URL in start.sh or env):
```python
# Current llm_service.py already supports Ollama
# Set: LLM_API_URL=http://localhost:11434/v1/chat/completions
# Ollama's /v1/chat/completions is OpenAI-compatible
```

---

## TTS (Text-to-Speech)

**Current:** `TTS/tts_service.py` — Qwen3-TTS 1.7B

**Server endpoint:** `/tts`

### Protocol

**Server → your service:**
```json
{"type": "tts_input", "text": "text to speak", "partial": false, "turn_id": 1}
{"type": "stop_generation"}
```

**Your service → server:**
```json
{"type": "audio_chunk", "audio": "<base64 float32>", "audio_seq": 0, "turn_id": 1}
{"type": "audio_end", "turn_id": 1}
```

### To swap TTS

1. Copy `TTS/tts_service.py` to `TTS/my_tts.py`
2. Replace `QwenTTSProcessor` class with your TTS implementation
3. The `generate_audio_chunks(text)` generator must yield dicts matching the protocol above
4. Update `start.sh` to launch your service

### Alternative TTS options

| Model | Install | Voice Quality | Speed | Notes |
|-------|---------|---------------|-------|-------|
| [Coqui TTS](https://github.com/coqui-ai/TTS) | `pip install TTS` | Good | Fast | 1000+ voices |
| [Bark](https://github.com/suno-ai/bark) | `pip install soundfile` | Great | Slow | Suno AI, generates music/sfx too |
| [XTTS](https://github.com/coqui-ai/TTS) | Coqui TTS | Great | Medium | Voice cloning |
| [Piper](https://github.com/rhasspel/piper) | `pip install piper-tts` | Good | Very Fast | Rust backend, offline |
| [ElevenLabs](https://elevenlabs.io/) | API key | Best | Medium | Cloud, best quality |
| [PlayHT](https://play.ht/) | API key | Great | Medium | Cloud |
| [gTTS](https://github.com/pndajies/gTTS) | `pip install gTTS` | Okay | Slow | Google Translate TTS |
| [MaryTTS](http://mary.dfki.de/) | Docker | Okay | Medium | Self-hosted |

**Piper TTS example** (minimal TTS service):
```python
# pip install piper-tts
import piper
import base64
import io

class PiperTTS:
    def __init__(self, voice="en_US-lessac-medium"):
        self.synthesizer = piper.Synthesizer(voice)

    def generate_audio_chunks(self, text):
        wav_data = bytes()
        for _, wav_chunk in self.synthesizer.synthesize_chunk(text):
            wav_data += wav_chunk
        # Send as one chunk (split into 500ms pieces if needed)
        audio = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_bytes = audio.tobytes()
        yield {
            "type": "audio_chunk",
            "audio": base64.b64encode(audio_bytes).decode(),
            "audio_seq": 0,
        }
        yield {"type": "audio_end"}
```

---

## Quick Swap Summary

To swap any component in under 5 minutes:

1. **Identify the protocol** — messages your service receives and sends (shown above)
2. **Copy the current service file** — `cp STT/stt_service.py STT/my_stt.py`
3. **Replace the core class** — keep the async WebSocket handling, swap the model logic
4. **Update start.sh** — change the launch line for that service
5. **Restart** — `./start.sh`

Each component is fully independent. Swapping one doesn't affect the others.
