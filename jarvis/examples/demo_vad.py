#!/usr/bin/env python3
"""Standalone VAD demo - records from microphone, detects speech endpoints.

Usage:
    python examples/demo_vad.py

Requirements:
    pip install webrtcvad   (or use Silero VAD: pip install silero-vad)

This demo shows how VAD works independently - it records audio from your
microphone and prints speech/silence detection events.
"""

import asyncio
import base64
import json
import logging
import sys
import wave

import numpy as np
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SIZE = 512  # Silero VAD expects 512 samples at 16kHz


async def demo_silero_vad():
    """Demo using Silero VAD ONNX."""
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch")
        return

    # Load Silero VAD
    logger.info("Loading Silero VAD model...")
    model, _ = torch.hub.load(
        "snakers4/silero-vad",
        "silero_vad",
        onnx=True,
        trust_repo=True,
    )
    logger.info("Silero VAD loaded. Start speaking...")

    # Connect to server for VAD processing
    server_host = "127.0.0.1"
    server_port = "8765"
    vad_url = f"wss://{server_host}:{server_port}/vad"

    try:
        async with websockets.connect(vad_url) as ws:
            await ws.send(json.dumps({"type": "start"}))
            logger.info("Connected to server VAD endpoint")
            logger.info("Press Ctrl+C to stop\n")

            async for msg in ws:
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("type") == "vad_result":
                        if data.get("speech"):
                            print(f"[{data['chunk_id']}] Speech started")
                        elif data.get("silence"):
                            print(f"[{data['chunk_id']}] Silence detected")

    except ConnectionRefusedError:
        logger.error("Server not running. Start with: ./start.sh")
        logger.error("Running standalone demo mode...")
        await demo_standalone_vad(model)


async def demo_standalone_vad(model):
    """Standalone VAD demo without server."""
    import pyaudio

    py = pyaudio.PyAudio()
    stream = py.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=WINDOW_SIZE,
    )

    is_speaking = False
    speech_buffer = []

    try:
        while True:
            data = stream.read(WINDOW_SIZE, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            tensor = torch.from_numpy(audio)
            prob = model(tensor, SAMPLE_RATE).item()

            if prob >= 0.5:
                if not is_speaking:
                    is_speaking = True
                    speech_buffer = []
                    print(f"[Speech STARTED] prob={prob:.2f}")
                speech_buffer.append(audio)
            else:
                if is_speaking:
                    is_speaking = False
                    if speech_buffer:
                        full = np.concatenate(speech_buffer)
                        # Save speech segment
                        fname = f"speech_{len(speech_buffer)}.wav"
                        audio_int16 = (full * 32767).astype(np.int16)
                        with wave.open(fname, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(audio_int16.tobytes())
                        print(f"[Speech ENDED] Saved to {fname}")
                        speech_buffer = []
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        py.terminate()


async def demo_webrtcvad():
    """Demo using WebRTCVAD (lighter weight, no PyTorch needed)."""
    try:
        import webrtcvad
    except ImportError:
        logger.error("webrtcvad not installed. Run: pip install webrtcvad")
        return

    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Most aggressive

    import pyaudio

    py = pyaudio.PyAudio()
    stream = py.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=160,  # 10ms frames
    )

    is_speaking = False
    speech_samples = 0

    try:
        while True:
            data = stream.read(160, exception_on_overflow=False)
            is_speech = vad.is_speech(data, SAMPLE_RATE)

            if is_speech and not is_speaking:
                is_speaking = True
                print(f"[Speech STARTED]")
            elif not is_speech and is_speaking:
                is_speaking = False
                print(f"[Speech ENDED] ({speech_samples} samples)")
                speech_samples = 0

            if is_speaking:
                speech_samples += 160

    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        py.terminate()


async def main():
    print("=" * 50)
    print("  JarVIs VAD Demo")
    print("=" * 50)
    print("\nChoose VAD backend:")
    print("  1) Silero VAD ONNX (requires PyTorch)")
    print("  2) WebRTCVAD (lightweight)")
    print("  3) Connect to Jarvis server")
    print()

    choice = input("Select [1/2/3]: ").strip()

    if choice == "1":
        await demo_silero_vad()
    elif choice == "2":
        await demo_webrtcvad()
    elif choice == "3":
        await demo_silero_vad()
    else:
        await demo_webrtcvad()


if __name__ == "__main__":
    asyncio.run(main())
