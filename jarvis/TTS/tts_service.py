#!/usr/bin/env python3
"""TTS (Text-to-Speech) service using Qwen3-TTS with chunked streaming.

Connects to the Jarvis server at ws://localhost:8765/tts and generates
audio from LLM token chunks. Implements Part 4 of the pipeline:
- Start generating audio for first TTS tokens while LLM is still generating
- Process token chunks in sentence-sized batches (not full sentences)
- Pre-warm TTS model on startup
- Overlap TTS audio output with remaining LLM generation

Protocol (server -> client):
  - tts_input: Text to synthesize with optional partial flag

TTS output sent to server:
  - audio_chunk: Base64-encoded float32 audio samples at 16kHz mono
  - audio_end: Generation complete
"""

import asyncio
import base64
import json
import logging
import os
import re
import ssl

import numpy as np
import torch
import websockets
from qwen_tts import Qwen3TTSModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# TTS configuration
SAMPLE_RATE = 16000
TTS_MODEL = os.getenv(
    "TTS_MODEL",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
)
TTS_DEVICE = os.getenv("TTS_DEVICE", "cpu")
TTS_DTYPE = os.getenv("TTS_DTYPE", "float32")
TTS_VOICE_INSTRUCT = os.getenv(
    "TTS_VOICE_INSTRUCT",
    "Speak in a calm, natural conversational tone.",
)
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "Auto")
TTS_CHUNK_WORDS = int(os.getenv("TTS_CHUNK_WORDS", "10"))  # Process every N words
TTS_MAX_CHUNK_DURATION = float(os.getenv("TTS_MAX_CHUNK_DURATION", "3.0"))  # Max seconds before forcing chunk

# Sentence boundary regex
SENTENCE_RE = re.compile(r'[.!?]\s+|[.!?]$')


class QwenTTSProcessor:
    """Qwen3-TTS processor with chunked streaming support.

    Accumulates text from partial LLM tokens, processes sentence-sized
    chunks for low-latency audio generation, and handles interruptions
    when the user speaks again.
    """

    def __init__(self, model_path, device, dtype_str, voice_instruct, language):
        self.is_generating = False
        self.interrupt_event = asyncio.Event()
        self.text_buffer = ""
        self.current_chunk_text = ""
        self.processed_words = 0

        # Load and pre-warm model
        logger.info(f"Loading Qwen3-TTS model: {model_path}")
        dtype = _parse_dtype(dtype_str)
        if "cpu" in device:
            logger.info("oneDNN optimizations enabled by default on CPU")
        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation="eager",
        )
        logger.info("Model loaded")

        self.voice_instruct = voice_instruct
        self.language = language

        # Pre-warm with a short utterance
        logger.info("Pre-warming model...")
        try:
            wavs, sr = self.model.generate_voice_design(
                text="Hello.",
                instruct=voice_instruct,
                language=language,
                non_streaming_mode=True,
                do_sample=False,
            )
            logger.info("Model pre-warmed")
        except Exception as e:
            logger.warning(f"Pre-warm failed (model may still work): {e}")

    def accumulate_text(self, text, partial=True):
        """Accumulate text from LLM tokens.

        Args:
            text: Text chunk from LLM (accumulated so far)
            partial: Whether more tokens are coming

        Yields:
            dict with audio_chunk data when a chunk is ready
        """
        if not text:
            return

        if partial:
            self.current_chunk_text = text
            current_words = len(text.split())
            new_words = current_words - self.processed_words
            if new_words <= 0:
                return

            self.processed_words = current_words

            # Check if we should process this chunk
            should_process = (
                self._has_sentence_boundary(text) or
                new_words >= TTS_CHUNK_WORDS
            )

            if should_process:
                yield self._process_chunk(self.current_chunk_text)
                self.processed_words = current_words
        else:
            # Final token - process remaining text
            self.current_chunk_text = text
            yield self._process_chunk(self.current_chunk_text)
            self.processed_words = 0

    def reset_buffer(self):
        """Reset for new response."""
        self.text_buffer = ""
        self.current_chunk_text = ""
        self.processed_words = 0
        self.interrupt_event.clear()

    def interrupt(self):
        """Signal interruption (user started speaking again)."""
        self.interrupt_event.set()

    def _has_sentence_boundary(self, text):
        """Check if text contains a sentence boundary."""
        return bool(SENTENCE_RE.search(text))

    def _process_chunk(self, text):
        """Process a text chunk and yield audio chunks.

        Args:
            text: Text chunk to synthesize

        Yields:
            dict with audio_chunk data
        """
        import time as _time

        if not text or not text.strip():
            return

        logger.info(f"TTS synthesizing chunk: \"{text[:80]}...\"")
        gen_start = _time.time()

        try:
            wavs, sr = self.model.generate_voice_design(
                text=text.strip(),
                instruct=self.voice_instruct,
                language=self.language,
                non_streaming_mode=True,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
            gen_time = _time.time() - gen_start
            logger.info(f"Model generation done in {gen_time:.1f}s, audio len={len(wavs[0])} samples")

            if not wavs or len(wavs) == 0:
                logger.warning("No audio generated")
                return

            # Convert audio to base64-encoded chunks for streaming
            audio = wavs[0]  # numpy array of float32 samples

            # Resample if needed
            if sr != SAMPLE_RATE:
                logger.info(f"Resampling from {sr} to {SAMPLE_RATE}")
                audio = _resample(audio, sr, SAMPLE_RATE)

            # Split into smaller chunks for smooth playback
            chunk_size = int(SAMPLE_RATE * 0.5)  # 500ms chunks
            num_samples = len(audio)

            for start in range(0, num_samples, chunk_size):
                end = min(start + chunk_size, num_samples)
                chunk = audio[start:end]

                # Normalize to prevent clipping
                peak = np.max(np.abs(chunk))
                if peak > 1.0:
                    chunk = chunk / peak

                # Encode as base64 raw bytes
                audio_bytes = chunk.astype(np.float32).tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

                yield {
                    "type": "audio_chunk",
                    "audio": audio_b64,
                    "chunk_id": start // chunk_size,
                    "sample_rate": SAMPLE_RATE,
                }

        except Exception as e:
            logger.error(f"TTS generation error: {e}")

    def generate_audio_chunks(self, text):
        """Generator that yields audio chunks from text.

        Handles interruption - stops generating when interrupted.
        """
        for result in self._process_chunk(text):
            if self.interrupt_event.is_set():
                logger.info("TTS interrupted, stopping generation")
                yield {
                    "type": "audio_chunk",
                    "audio": "",
                    "chunk_id": -1,
                    "interrupted": True,
                }
                return
            yield result


def _parse_dtype(dtype_str):
    """Parse dtype string to torch.dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def _resample(audio, from_sr, to_sr):
    """Simple resampling using librosa (lightweight)."""
    try:
        import librosa
        return librosa.resample(audio, orig_sr=from_sr, target_sr=to_sr)
    except ImportError:
        logger.warning("librosa not available, using numpy resampling")
        num_samples = int(len(audio) * to_sr / from_sr)
        indices = np.linspace(0, len(audio) - 1, num_samples)
        indices = np.clip(indices, 0, len(audio) - 1)
        return np.interp(indices, np.arange(len(audio)), audio)


async def handle_tts(websocket, tts_processor):
    """Handle the TTS WebSocket connection."""
    try:
        # Wait for client message
        try:
            msg = await asyncio.wait_for(websocket.recv(), timeout=10)
            if isinstance(msg, bytes):
                data = json.loads(msg)
            else:
                data = json.loads(msg)

            if data.get("type") == "start":
                logger.info("TTS service ready")
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for TTS start message")
            return

        # Process incoming messages
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    msg = json.loads(message)
                else:
                    msg = json.loads(message)

                msg_type = msg.get("type")

                if msg_type == "tts_input":
                    text = msg.get("text", "")
                    partial = msg.get("partial", True)

                    if not text.strip():
                        continue

                    logger.info(f"TTS received text (partial={partial}): \"{text[:80]}...\"")

                    # Reset buffer on new response
                    if not partial:
                        tts_processor.reset_buffer()

                    async def process_tts_input():
                        local_chunk_id = 0

                        def generate_chunks():
                            try:
                                for chunk in tts_processor.generate_audio_chunks(text):
                                    chunk_queue.put_nowait(chunk)
                            except Exception as e:
                                logger.error(f"Generation error: {e}")
                            generation_complete.set()

                        gen_task = asyncio.create_task(asyncio.to_thread(generate_chunks))

                        while not generation_complete.is_set():
                            try:
                                chunk = await asyncio.wait_for(chunk_queue.get(), timeout=2.0)
                                if chunk.get("interrupted"):
                                    break
                                if local_chunk_id > 0:
                                    chunk["chunk_id"] = local_chunk_id
                                local_chunk_id += 1
                                await websocket.send(json.dumps(chunk))
                            except asyncio.TimeoutError:
                                continue

                        # Wait for generation to fully complete
                        await gen_task

                        # Drain remaining chunks
                        while not chunk_queue.empty():
                            try:
                                chunk = chunk_queue.get_nowait()
                                if chunk.get("interrupted"):
                                    break
                                if local_chunk_id > 0:
                                    chunk["chunk_id"] = local_chunk_id
                                local_chunk_id += 1
                                await websocket.send(json.dumps(chunk))
                            except Exception:
                                break

                        await websocket.send(json.dumps({
                            "type": "audio_end",
                            "total_chunks": local_chunk_id,
                        }))
                        logger.info(f"TTS generation complete ({local_chunk_id} chunks)")

                    chunk_queue: asyncio.Queue = asyncio.Queue()
                    generation_complete = asyncio.Event()
                    task = asyncio.create_task(process_tts_input())
                    task.add_done_callback(
                        lambda t: logger.error(f"TTS task error: {t.exception()}") if t.exception() else None
                    )

                elif msg_type == "stop_generation":
                    tts_processor.interrupt()
                    await websocket.send(json.dumps({
                        "type": "audio_end",
                        "interrupted": True,
                    }))
                    logger.info("TTS generation stopped (interrupt)")

                elif msg_type == "set_voice":
                    voice = msg.get("voice", "male")
                    tts_processor.voice_instruct = msg.get("instruct", TTS_VOICE_INSTRUCT)
                    logger.info(f"TTS voice/instruct changed")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from server: {e}")
            except Exception as e:
                logger.error(f"TTS processing error: {e}")
                raise

    except websockets.ConnectionClosed as e:
        logger.info(f"TTS connection closed: {e}")
    except Exception as e:
        logger.error(f"TTS connection error: {e}")


async def main():
    server_host = os.getenv("SERVER_HOST", "127.0.0.1")
    server_port = int(os.getenv("SERVER_PORT", "8765"))
    tts_url = f"wss://{server_host}:{server_port}/tts"

    # Create TTS processor (loads and pre-warms model)
    logger.info(f"Initializing Qwen3-TTS with model: {TTS_MODEL}")
    tts_processor = QwenTTSProcessor(
        model_path=TTS_MODEL,
        device=TTS_DEVICE,
        dtype_str=TTS_DTYPE,
        voice_instruct=TTS_VOICE_INSTRUCT,
        language=TTS_LANGUAGE,
    )

    # Connect to server
    logger.info(f"Connecting to TTS endpoint at {tts_url}")
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    try:
        async with websockets.connect(tts_url, ssl=ssl_context) as websocket:
            await websocket.send(json.dumps({"type": "start"}))
            logger.info("TTS connected to server")
            await handle_tts(websocket, tts_processor)
    except ConnectionRefusedError:
        logger.error(
            f"Could not connect to server at {tts_url}. "
            f"Make sure the server is running on port {server_port}."
        )
    except Exception as e:
        logger.error(f"Failed to connect: {e}")


if __name__ == "__main__":
    asyncio.run(main())
