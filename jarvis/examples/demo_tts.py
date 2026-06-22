#!/usr/bin/env python3
"""Standalone TTS demo - generates audio from text.

Usage:
    python examples/demo_tts.py

This demo shows how TTS works independently:
1. Takes text input (interactive or from command line)
2. Generates audio using a TTS model
3. Saves to WAV file and/or plays back

Choose your TTS backend:
  - Qwen3-TTS (current Jarvis default)
  - Coqui TTS (1000+ voices)
  - Piper (fast, offline)
  - gTTS (Google Translate TTS)
"""

import asyncio
import base64
import json
import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def demo_qwen_tts():
    """Demo using Qwen3-TTS (current Jarvis default)."""
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        logger.error("qwen_tts not installed.")
        logger.error("Install: pip install qwen-tts diffusers onnx")
        return

    print("\nLoading Qwen3-TTS model... (first run downloads ~4GB)")

    # Try XPU first, then CUDA, then CPU
    device = "cpu"
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        elif torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    print(f"Using device: {device}")

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        attn_implementation="sdpa",
    )
    print("Model loaded. Enter text to synthesize (or 'quit' to exit):")

    while True:
        text = input("\nText> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        try:
            print("Generating audio...")
            wavs, sr = model.generate_custom_voice(
                text=text,
                speaker="eric",
                instruct="Speak in a calm, natural conversational tone.",
                language="Auto",
                non_streaming_mode=True,
                do_sample=True,
                temperature=0.7,
            )

            output_file = "demo_output.wav"
            import soundfile as sf
            sf.write(output_file, wavs[0], sr)
            print(f"Saved to {output_file} ({len(wavs[0])/sr:.1f}s)")

        except Exception as e:
            logger.error(f"TTS error: {e}")


async def demo_coqui_tts():
    """Demo using Coqui TTS."""
    try:
        from TTS.api import TTS
    except ImportError:
        logger.error("TTS (Coqui) not installed.")
        logger.error("Install: pip install TTS")
        return

    print("\nLoading Coqui TTS model...")

    # List available models
    models = TTS().list_models()
    print(f"\nAvailable models: {len(models)}")
    print("Recommended models:")
    recommended = [m for m in models if "tts_models/multilingual" in m or "vctk" in m]
    for m in recommended[:5]:
        print(f"  - {m}")

    model_name = input("\nSelect model (press Enter for multi-speaker): ").strip()
    if not model_name:
        # Use a default multi-speaker model
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

    tts = TTS(model_name=model_name)
    print("Model loaded. Enter text to synthesize (or 'quit' to exit):")

    while True:
        text = input("\nText> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        try:
            output_file = f"demo_output_{text[:20].replace(' ', '_')}.wav"
            tts.tts(text=text, file=output_file)
            print(f"Saved to {output_file}")

        except Exception as e:
            logger.error(f"TTS error: {e}")


async def demo_piper_tts():
    """Demo using Piper TTS (fast, offline)."""
    try:
        import piper
    except ImportError:
        logger.error("piper-tts not installed.")
        logger.error("Install: pip install piper-tts")
        return

    print("\nLoading Piper TTS model...")
    # Available voices: https://github.com/rhasspel/piper/releases
    voice = input("Voice name (press Enter for lessac): ").strip() or "en_US-lessac-medium"

    synthesizer = piper.Synthesizer(voice)
    print(f"Using voice: {voice}")
    print("Enter text to synthesize (or 'quit' to exit):")

    while True:
        text = input("\nText> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        try:
            import wave

            output_file = f"demo_output_{text[:20].replace(' ', '_')}.wav"
            wav_data = bytes()
            for _, wav_chunk in synthesizer.synthesize_chunk(text):
                wav_data += wav_chunk

            with wave.open(output_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(synthesizer.sample_rate)
                wf.writeframes(wav_data)

            print(f"Saved to {output_file}")

        except Exception as e:
            logger.error(f"TTS error: {e}")


async def demo_gtts():
    """Demo using Google Translate TTS (free, online)."""
    try:
        from gtts import gTTS
    except ImportError:
        logger.error("gTTS not installed.")
        logger.error("Install: pip install gTTS")
        return

    print("\nUsing Google Translate TTS (requires internet)")
    print("Enter text to synthesize (or 'quit' to exit):")

    languages = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "ru": "Russian",
    }

    while True:
        text = input("\nText> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        try:
            lang_code = input(
                f"Language [en] (options: {', '.join(languages.keys())}): "
            ).strip() or "en"

            tts = gTTS(text=text, lang=lang_code, slow=False)
            output_file = f"demo_output_{text[:20].replace(' ', '_')}.mp3"
            tts.save(output_file)
            print(f"Saved to {output_file}")

        except Exception as e:
            logger.error(f"TTS error: {e}")


async def demo_from_command_line():
    """Demo: generate audio from command line argument."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python demo_tts.py \"text to speak\"")
        return

    text = " ".join(sys.argv[1:])

    print(f"\nGenerating audio for: {text[:100]}...")

    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        output_file = "demo_output.mp3"
        tts.save(output_file)
        print(f"Saved to {output_file}")
    except ImportError:
        print("gTTS not installed. Run: pip install gTTS")


async def main():
    print("=" * 50)
    print("  Jarvis TTS Demo")
    print("=" * 50)

    if len(sys.argv) >= 2:
        await demo_from_command_line()
        return

    print("\nChoose TTS backend:")
    print("  1) Qwen3-TTS (current Jarvis default, best quality)")
    print("  2) Coqui TTS (1000+ voices, XTTS v2)")
    print("  3) Piper (fast, offline, lightweight)")
    print("  4) gTTS (free, Google Translate, requires internet)")
    print()

    choice = input("Select [1/2/3/4]: ").strip()

    backends = {
        "1": demo_qwen_tts,
        "2": demo_coqui_tts,
        "3": demo_piper_tts,
        "4": demo_gtts,
    }
    fn = backends.get(choice, demo_qwen_tts)
    await fn()


if __name__ == "__main__":
    asyncio.run(main())
