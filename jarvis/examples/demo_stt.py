#!/usr/bin/env python3
"""Standalone STT demo - records from microphone, transcribes speech.

Usage:
    python examples/demo_stt.py

This demo shows how STT works independently:
1. Records audio from microphone
2. Uses VAD to detect speech endpoints
3. Transcribes spoken audio to text

Choose your STT backend:
  - Parakeet-TDT (current Jarvis default)
  - Whisper.cpp (fast, C++ backend)
  - Faster-Whisper (CTranslate2 backend)
  - Vosk (offline, small models)
"""

import asyncio
import base64
import json
import logging
import sys
import wave

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def demo_parakeet():
    """Demo using NVIDIA Parakeet-TDT (current Jarvis default)."""
    try:
        from nano_parakeet import from_pretrained
    except ImportError:
        logger.error("nano-parakeet not installed.")
        logger.error("Install: pip install nano-parakeet sentencepiece")
        return

    print("\nLoading Parakeet-TDT model... (first run downloads ~200MB)")
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        print(f"Using device: {device}")
    except ImportError:
        print("PyTorch not available, using CPU")

    model = from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3",
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.eval()
    print("Model loaded. Start speaking...")

    import pyaudio

    py = pyaudio.PyAudio()
    stream = py.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=512,
    )

    audio_buffer = np.array([], dtype=np.float32)
    is_speaking = False
    silence_frames = 0

    print("\nListening... (Ctrl+C to stop)\n")

    try:
        while True:
            data = stream.read(512, exception_on_overflow=False)
            frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.concatenate([audio_buffer, frame])

            # Simple energy-based VAD for demo
            energy = np.mean(np.abs(frame))
            if energy > 0.01:
                if not is_speaking:
                    is_speaking = True
                    print("[Speech STARTED]")
                silence_frames = 0
            else:
                if is_speaking:
                    silence_frames += 1
                    if silence_frames > 30:  # ~1.5s silence
                        is_speaking = False
                        # Transcribe
                        if len(audio_buffer) > 16000:  # At least 1 second
                            tensor = torch.from_numpy(audio_buffer).to(device)
                            token_ids = model.transcribe_audio(tensor)
                            text = model.sp.DecodeIds(token_ids).strip()
                            if text:
                                print(f"Transcript: {text}\n")
                        audio_buffer = np.array([], dtype=np.float32)
                        silence_frames = 0

    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        py.terminate()


async def demo_whisper_cpp():
    """Demo using Whisper.cpp via pywhispercpp."""
    try:
        import pywhispercpp
    except ImportError:
        logger.error("pywhispercpp not installed.")
        logger.error("Install: pip install pywhispercpp")
        return

    print("\nLoading Whisper.cpp model... (first run downloads model)")
    model = pywhispercpp.WhisperModel("base")
    print("Model loaded. Start speaking...")

    import pyaudio

    py = pyaudio.PyAudio()
    stream = py.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1600,  # 100ms chunks
    )

    audio_buffer = []

    print("\nListening... (Ctrl+C to stop)\n")

    try:
        while True:
            data = stream.read(1600, exception_on_overflow=False)
            audio_buffer.append(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)

            # Check for silence (every 5 chunks)
            if len(audio_buffer) % 10 == 0:
                full = np.concatenate(audio_buffer)
                energy = np.mean(np.abs(full))
                if energy < 0.01 and len(audio_buffer) > 20:  # ~2s of silence
                    full_audio = np.concatenate(audio_buffer)
                    if len(full_audio) > 16000:  # At least 1 second
                        try:
                            lang, text = model.detect_language(full_audio)
                            if text.strip():
                                print(f"[{lang}] Transcript: {text}\n")
                        except Exception as e:
                            logger.error(f"Transcription error: {e}")
                    audio_buffer = []

    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        py.terminate()


async def demo_whisper_fast():
    """Demo using Faster-Whisper (CTranslate2)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error("faster-whisper not installed.")
        logger.error("Install: pip install faster-whisper")
        return

    print("\nLoading Faster-Whisper model... (first run downloads)")
    model_size = "tiny"  # Use "base" or "small" for better accuracy
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    model = WhisperModel(model_size, device=device, compute_type="int8")
    print("Model loaded. Start speaking...")

    import pyaudio

    py = pyaudio.PyAudio()
    stream = py.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=3200,
    )

    audio_buffer = []
    silence_frames = 0

    print("\nListening... (Ctrl+C to stop)\n")

    try:
        while True:
            data = stream.read(3200, exception_on_overflow=False)
            frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer.append(frame)

            energy = np.mean(np.abs(frame))
            if energy > 0.01:
                silence_frames = 0
            else:
                silence_frames += 1
                if silence_frames > 20:  # ~1.3s silence
                    full_audio = np.concatenate(audio_buffer)
                    if len(full_audio) > 16000:
                        segments, info = model.transcribe(full_audio, language="auto")
                        text = " ".join(seg.text for seg in segments)
                        if text.strip():
                            print(f"Transcript: {text}\n")
                    audio_buffer = []
                    silence_frames = 0

    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        py.terminate()


async def demo_vosk():
    """Demo using Vosk (offline, small models)."""
    try:
        from vosk import Model, SpkRecognizer
    except ImportError:
        logger.error("vosk not installed.")
        logger.error("Install: pip install vosk")
        return

    print("\nLoading Vosk model...")
    model = Model("vosk-model-small-en-0.15")  # ~40MB small model
    print("Model loaded. Start speaking...")

    import pyaudio

    py = pyaudio.PyAudio()
    stream = py.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=4000,
    )

    recognizer = SpkRecognizer(model)
    recognizer.Start()

    print("\nListening... (Ctrl+C to stop)\n")

    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                if result.get("text", "").strip():
                    print(f"Transcript: {result['text']}\n")

    except KeyboardInterrupt:
        print("\nDemo stopped.")
        print(f"Partial result: {recognizer.FinalResult()}")
    finally:
        stream.stop_stream()
        stream.close()
        py.terminate()


async def demo_from_file():
    """Demo: transcribe an existing audio file."""
    from nano_parakeet import from_pretrained
    import wave

    print("\nSelect STT backend:")
    print("  1) Parakeet-TDT")
    print("  2) Whisper.cpp")
    print("  3) Faster-Whisper")
    choice = input("Select [1/2/3]: ").strip()

    audio_file = input("Path to WAV file (16kHz mono): ").strip()
    if not audio_file:
        audio_file = "test_audio.wav"
        print(f"Using default: {audio_file}")

    # Read audio file
    with wave.open(audio_file, "rb") as wf:
        samples = wf.readframes(wf.getnframes())
        audio = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0

    print(f"Audio loaded: {len(audio)/16000:.1f}s")

    if choice == "1":
        import torch
        model = from_pretrained("nvidia/parakeet-tdt-0.6b-v3", device="cpu")
        tensor = torch.from_numpy(audio).unsqueeze(0)
        token_ids = model.transcribe_audio(tensor)
        text = model.sp.DecodeIds(token_ids).strip()
        print(f"\nTranscript: {text}")

    elif choice == "2":
        import pywhispercpp
        model = pywhispercpp.WhisperModel("base")
        lang, text = model.detect_language(audio)
        print(f"\n[{lang}] Transcript: {text}")

    elif choice == "3":
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="cpu")
        segments, info = model.transcribe(audio, language="auto")
        text = " ".join(seg.text for seg in segments)
        print(f"\nTranscript: {text}")


async def main():
    print("=" * 50)
    print("  Jarvis STT Demo")
    print("=" * 50)
    print("\nChoose demo mode:")
    print("  1) Live microphone transcription")
    print("  2) Transcribe audio file")
    print()

    choice = input("Select [1/2]: ").strip()

    if choice == "1":
        print("\nSelect STT backend:")
        print("  1) Parakeet-TDT (current Jarvis default)")
        print("  2) Whisper.cpp (fast, lightweight)")
        print("  3) Faster-Whisper (best offline)")
        print("  4) Vosk (smallest model, ~40MB)")
        print()
        backend = input("Select [1/2/3/4]: ").strip()

        backends = {
            "1": demo_parakeet,
            "2": demo_whisper_cpp,
            "3": demo_whisper_fast,
            "4": demo_vosk,
        }
        fn = backends.get(backend, demo_parakeet)
        await fn()

    elif choice == "2":
        await demo_from_file()


if __name__ == "__main__":
    asyncio.run(main())
