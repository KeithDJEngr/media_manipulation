# STT Performance Investigation

## Device Placement (as of latest check)

### XPU Status
- **XPU available:** `True` (intel-compute-runtime installed via `pacman -S intel-compute-runtime`)
- **XPU device count:** `1` (Intel Arc Pro B70, Battlemage G31)
- **Kernel driver:** `xe` module loaded

### Parakeet-TDT Model (`nvidia/parakeet-tdt-0.6b-v3`)
All 651 parameters loaded on **XPU:0**:
- **Encoder:** 636 params on `xpu:0`
- **Decoder:** 9 params on `xpu:0`
- **Preprocessor:** 0 params (DSP operations, no learnable params)

### Whisper Model (`openai/whisper-tiny`)
All parameters loaded on **XPU:0**:
- **Encoder:** All params on `xpu:0`
- **Decoder:** All params on `xpu:0`

### torchaudio
- **Version:** `2.11.0+cpu` (CPU variant, no CUDA dependency)
- **Library:** `_torchaudio.abi3.so` no longer linked against `libcudart.so.13`
- **Role:** Audio I/O (loading/saving), not inference

## Benchmark Results

### Parakeet-TDT Transcription Timing (XPU)

| Audio Length | Generation Time | Real-time Factor |
|-------------|----------------|------------------|
| 1s | 0.80s | 0.8x real-time |
| 2s | 0.23s | 0.1x real-time |
| 3s | 0.78s | 0.3x real-time |
| 5s | 0.06s | 0.01x real-time |
| 10s | 0.36s | 0.04x real-time |
| 15s | 0.52s | 0.03x real-time |
| 20s | 0.46s | 0.02x real-time |
| 30s | 0.58s | 0.02x real-time |

### Parakeet-TDT Model Load Time
- **Load time:** ~8.8 seconds (one-time, cached after first run)

## What's Running Where

| Component | Device | Notes |
|-----------|--------|-------|
| Parakeet encoder | XPU:0 | Full model on XPU |
| Parakeet decoder | XPU:0 | Full model on XPU |
| Whisper encoder | XPU:0 | Via `device_map='xpu'` |
| Whisper decoder | XPU:0 | Via `device_map='xpu'` |
| torchaudio | CPU | Audio I/O only, no CUDA |
| numpy/pandas | CPU | Standard CPU operations |

**Conclusion:** All ML inference is on XPU. torchaudio is CPU-only but only used for audio file I/O, not inference.

## Bottleneck Analysis

### STT Service (Fast)
- **Transcription time:** <1s for all tested lengths
- **Log evidence:** Only 2 transcriptions in entire log, both near-instant
- **VAD buffer issue:** Many "End of speech, buffer size: 0" messages indicate VAD fires end-of-speech before STT accumulates audio

### TTS Service (Slow)
- **Per-generation time:** 7-30 seconds depending on text length (tested: "Hello world." = 7.1s, BBC Merlin paragraph = 30.3s)
- **Real-time factor:** 2.8x real-time for 1.7B model on XPU (good)
- **Wasted generations:** Previously, every partial LLM token triggered a full TTS generation. Fix added prefix-skipping to avoid regenerating for shorter partials.

## Commands Run

```bash
# Check XPU availability
.venv/bin/python -c "import torch; import torch.xpu; print('XPU:', torch.xpu.is_available()); print('Count:', torch.xpu.device_count())"

# Check Parakeet device placement
.venv/bin/python -c "
from nano_parakeet import from_pretrained
model = from_pretrained('nvidia/parakeet-tdt-0.6b-v3', device='xpu')
for name, p in model.named_parameters():
    print(f'{name}: {p.device}')
"

# Check Whisper device placement
.venv/bin/python -c "
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny', device_map='xpu')
for name, p in model.model.encoder.named_parameters():
    print(f'encoder {name}: {p.device}')
for name, p in model.model.decoder.named_parameters():
    print(f'decoder {name}: {p.device}')
"

# Benchmark Parakeet transcription
.venv/bin/python -c "
from nano_parakeet import from_pretrained
import torch
import numpy as np
import time
model = from_pretrained('nvidia/parakeet-tdt-0.6b-v3', device='xpu')
audio = torch.from_numpy(np.random.randn(16000*3)).to('xpu')
start = time.time()
model.transcribe_audio(audio)
print(f'Time: {time.time()-start:.2f}s')
"

# Check torchaudio library
ldd .venv/lib/python3.12/site-packages/torchaudio/lib/_torchaudio.abi3.so

# Install intel-compute-runtime (required for XPU)
sudo pacman -S intel-compute-runtime
```

## Key Findings

1. **XPU is working** - intel-compute-runtime was installed, XPU device count is 1
2. **All ML models on XPU** - Parakeet encoder/decoder and Whisper encoder/decoder all on XPU:0
3. **STT is fast** - Parakeet transcribes in <1s on XPU
4. **torchaudio is CPU** - But only used for I/O, not inference
5. **VAD buffer issue** - Many empty buffer end-of-speech events suggest VAD fires before STT accumulates enough audio
