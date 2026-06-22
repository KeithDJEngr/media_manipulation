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

# Guard Level Zero GPU enumeration before torch import (same fix as STT service)
_ze_mask_before = os.environ.get("ZE_AFFINITY_MASK")
os.environ["ZE_AFFINITY_MASK"] = "0"

import numpy as np
import torch
import websockets
import soundfile as sf
from qwen_tts import Qwen3TTSModel

if _ze_mask_before is not None:
    os.environ["ZE_AFFINITY_MASK"] = _ze_mask_before
else:
    os.environ.pop("ZE_AFFINITY_MASK", None)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# TTS configuration
SAMPLE_RATE = 16000
TTS_MODEL = os.getenv(
    "TTS_MODEL",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
)
TTS_DEVICE = os.getenv("TTS_DEVICE", "xpu")
TTS_DTYPE = os.getenv("TTS_DTYPE", "float32")
TTS_VOICE_INSTRUCT = os.getenv(
    "TTS_VOICE_INSTRUCT",
    "Speak in a calm, natural conversational tone.",
)
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "Auto")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "eric")
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

    def __init__(self, model_path, device, dtype_str, voice_instruct, language, speaker):
        self.is_generating = False
        self.interrupt_event = asyncio.Event()
        self.text_buffer = ""
        self.current_chunk_text = ""
        self.processed_words = 0
        self.speaker = speaker
        self.last_generated_text = ""
        self.last_generated_len = 0
        self.current_turn_id = None
        self.model_path = model_path
        self.device = device
        self.dtype_str = dtype_str
        self.voice_instruct = voice_instruct
        self.language = language
        self._model_loaded = False
        self._load_model()

    def _load_model(self):
        """Load or reload the TTS model."""
        device = self.device
        if device == "xpu":
            try:
                import torch.xpu
                if torch.xpu.device_count() == 0:
                    logger.warning("XPU device count is zero, falling back to CPU")
                    device = "cpu"
            except Exception:
                logger.warning("Failed to detect XPU, falling back to CPU")
                device = "cpu"
        self.device = device

        logger.info(f"Loading Qwen3-TTS model: {self.model_path} on {device}")
        dtype = _parse_dtype(self.dtype_str)
        if "cpu" in device:
            logger.info("oneDNN optimizations enabled by default on CPU")
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_path,
                device_map=device,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            self._model_loaded = True
            logger.info("Model loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model_loaded = False
            raise

    def reload_model(self):
        """Reload the model (used after DEVICE_LOST error)."""
        logger.info("Reloading TTS model after device error...")
        # Free GPU memory
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._model_loaded = False
        self._load_model()
        # Re-pre-warm
        logger.info("Pre-warming model after reload...")
        try:
            wavs, sr = self.model.generate_custom_voice(
                text="Hello.",
                speaker=self.speaker,
                instruct=self.voice_instruct,
                language=self.language,
                non_streaming_mode=True,
                do_sample=False,
            )
            logger.info("Model re-warmed")
        except Exception as e:
            logger.warning(f"Re-warm failed: {e}")

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
            current_words = len(text.split())
            new_words = current_words - self.processed_words
            if new_words <= 0:
                return

            # Extract only the new words that haven't been processed yet
            words = text.split()
            new_text = " ".join(words[self.processed_words:])
            self.current_chunk_text = new_text
            self.processed_words = current_words

            # Check if we should process this chunk
            should_process = (
                self._has_sentence_boundary(text) or
                new_words >= TTS_CHUNK_WORDS
            )

            if should_process:
                yield self._process_chunk(self.current_chunk_text)
                self.current_chunk_text = ""
                self.processed_words = current_words
        else:
            # Final token - process remaining text
            current_words = len(text.split())
            new_words = current_words - self.processed_words
            if new_words > 0:
                words = text.split()
                self.current_chunk_text = " ".join(words[self.processed_words:])
                yield self._process_chunk(self.current_chunk_text)
            self.processed_words = 0
            self.current_chunk_text = ""

    def reset_buffer(self, turn_id=None):
        """Reset for new response."""
        self.text_buffer = ""
        self.current_chunk_text = ""
        self.processed_words = 0
        self.last_generated_text = ""
        self.last_generated_len = 0
        self.interrupt_event.clear()
        self.current_turn_id = turn_id

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
            wavs, sr = self.model.generate_custom_voice(
                text=text.strip(),
                speaker=self.speaker,
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
            error_str = str(e)
            if "DEVICE_LOST" in error_str or "level_zero" in error_str or "device lost" in error_str.lower():
                logger.error(f"GPU device lost error detected: {e}")
                try:
                    self.reload_model()
                except Exception as reload_err:
                    logger.error(f"Failed to reload model: {reload_err}")
                    return
                logger.info("Attempting regeneration after model reload...")
                try:
                    wavs, sr = self.model.generate_custom_voice(
                        text=text.strip(),
                        speaker=self.speaker,
                        instruct=self.voice_instruct,
                        language=self.language,
                        non_streaming_mode=True,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                    )
                    gen_time = _time.time() - gen_start
                    logger.info(f"Model regeneration done in {gen_time:.1f}s, audio len={len(wavs[0])} samples")
                except Exception as e2:
                    logger.error(f"TTS generation error after reload: {e2}")
                    return

                if not wavs or len(wavs) == 0:
                    logger.warning("No audio generated after reload")
                    return

                audio = wavs[0]
                if sr != SAMPLE_RATE:
                    logger.info(f"Resampling from {sr} to {SAMPLE_RATE}")
                    audio = _resample(audio, sr, SAMPLE_RATE)

                chunk_size = int(SAMPLE_RATE * 0.5)
                num_samples = len(audio)
                for start in range(0, num_samples, chunk_size):
                    end = min(start + chunk_size, num_samples)
                    chunk = audio[start:end]
                    peak = np.max(np.abs(chunk))
                    if peak > 1.0:
                        chunk = chunk / peak
                    audio_bytes = chunk.astype(np.float32).tobytes()
                    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                    yield {
                        "type": "audio_chunk",
                        "audio": audio_b64,
                        "chunk_id": start // chunk_size,
                        "sample_rate": SAMPLE_RATE,
                    }
            else:
                logger.error(f"TTS generation error: {e}")

    def split_into_sentences(self, text):
        """Split text into sentences using sentence boundary detection.
        
        Preserves sentence-ending punctuation. Returns list of sentences.
        """
        stripped = text.strip()
        if not stripped:
            return []
        
        # Split on sentence boundaries while preserving the delimiter
        sentences = re.split(r'(?<=[.!?])\s+', stripped)
        
        # Filter empty sentences
        result = [s.strip() for s in sentences if s.strip()]
        return result

    def generate_audio_chunks(self, text):
        """Generator that yields audio chunks from text.

        Skips redundant generation - if the same text has already been generated
        (from a previous partial message), skips to avoid regenerating audio.
        Also skips shorter texts that are prefixes of already-generated text,
        since they will be replaced by the longer version anyway.
        """
        stripped = text.strip()
        if not stripped:
            return
        if stripped == self.last_generated_text:
            return
        # Skip if this text is shorter than what we've already generated
        # and is a prefix of it (will be replaced by the longer version)
        if len(stripped) < self.last_generated_len and self.last_generated_text.startswith(stripped):
            return
        self.last_generated_text = stripped
        self.last_generated_len = len(stripped)

        for result in self._process_chunk(stripped):
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

    def generate_sentences_in_order(self, text):
        """Generate audio for each sentence in order.
        
        Splits text into sentences and generates audio for each one
        sequentially (in order), yielding audio chunks as they're ready.
        This ensures sentences play in the same order they were generated.
        """
        stripped = text.strip()
        if not stripped:
            return
        
        sentences = self.split_into_sentences(stripped)
        if not sentences:
            # No clear sentence boundaries, process as single chunk
            for chunk in self.generate_audio_chunks(stripped):
                yield chunk
            return
        
        logger.info(f"Split text into {len(sentences)} sentences for TTS")
        
        for i, sentence in enumerate(sentences):
            if self.interrupt_event.is_set():
                logger.info("TTS interrupted during sentence processing")
                yield {
                    "type": "audio_chunk",
                    "audio": "",
                    "chunk_id": -1,
                    "interrupted": True,
                }
                return
            
            logger.info(f"Generating audio for sentence {i+1}/{len(sentences)}: \"{sentence[:60]}...\"")
            
            for chunk in self.generate_audio_chunks(sentence):
                if chunk.get("interrupted"):
                    yield chunk
                    return
                chunk["sentence_index"] = i
                chunk["total_sentences"] = len(sentences)
                yield chunk


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
    async def send_heartbeats():
        while True:
            await asyncio.sleep(5)
            try:
                await websocket.send(json.dumps({"type": "heartbeat"}))
            except Exception:
                break

    heartbeat_task = asyncio.create_task(send_heartbeats())
    
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

        # Global audio sequence counter for ordering chunks across responses
        global_audio_seq = 0

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
                    turn_id = msg.get("turn_id")

                    if not text.strip():
                        continue

                    logger.info(f"TTS received text (partial={partial}, turn_id={turn_id}): \"{text[:80]}...\"")

                    # Reset buffer on new complete response
                    if not partial:
                        tts_processor.reset_buffer(turn_id)

                    async def process_tts_input():
                        local_chunk_id = 0

                        def generate_chunks():
                            try:
                                if partial:
                                    # Partial token - accumulate and process incrementally
                                    for chunk in tts_processor.accumulate_text(text, partial=True):
                                        if isinstance(chunk, dict):
                                            logger.info(f"Queuing audio: {chunk}")
                                            chunk_queue.put_nowait(chunk)
                                else:
                                    # Complete response - process sentence by sentence in order
                                    for chunk in tts_processor.generate_sentences_in_order(text):
                                        logger.info(f"Queuing audio: {chunk}")
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
                                chunk["audio_seq"] = global_audio_seq
                                global_audio_seq += 1
                                chunk["turn_id"] = tts_processor.current_turn_id
                                await websocket.send(json.dumps(chunk))
                                logger.info(f"sending audio: {chunk}")
                                local_chunk_id += 1
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
                                chunk["audio_seq"] = global_audio_seq
                                global_audio_seq += 1
                                chunk["turn_id"] = tts_processor.current_turn_id
                                await websocket.send(json.dumps(chunk))
                                local_chunk_id += 1
                            except Exception:
                                break

                        await websocket.send(json.dumps({
                            "type": "audio_end",
                            "total_chunks": local_chunk_id,
                            "turn_id": tts_processor.current_turn_id,
                        }))
                        if local_chunk_id > 0:
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
                        "turn_id": tts_processor.current_turn_id,
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
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


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
        speaker=TTS_SPEAKER,
    )

    # Connect to server
    logger.info(f"Connecting to TTS endpoint at {tts_url}")
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    while True:
        try:
            async with websockets.connect(tts_url, ssl=ssl_context) as websocket:
                await websocket.send(json.dumps({"type": "start"}))
                await websocket.send(json.dumps({
                    "type": "service_status",
                    "service": "tts",
                    "status": "connected",
                }))
                logger.info("TTS connected to server")
                await handle_tts(websocket, tts_processor)
                await websocket.send(json.dumps({
                    "type": "service_status",
                    "service": "tts",
                    "status": "disconnected",
                }))
        except websockets.ConnectionClosed:
            logger.info("TTS connection closed, reconnecting...")
        except ConnectionRefusedError:
            logger.error(
                f"Could not connect to server at {tts_url}. "
                f"Make sure the server is running on port {server_port}."
            )
            await asyncio.sleep(3)
            continue
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            await asyncio.sleep(3)
            continue


if __name__ == "__main__":
    asyncio.run(main())
