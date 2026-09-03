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
  - audio_end: Sent per input message (one per sentence boundary).
    "final": true marks the turn-final end (the one that clears the
    server's speaking state); "interrupted": true marks a barge-in stop
"""

import asyncio
import base64
import json
import logging
import os
import re
import ssl
import time as _time

# Guard Level Zero GPU enumeration before torch import (same fix as STT service)
_ze_mask_before = os.environ.get("ZE_AFFINITY_MASK")
os.environ["ZE_AFFINITY_MASK"] = "0"

import numpy as np
import torch
import websockets
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from kokoro import KPipeline

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
# Available TTS models (fastest → best quality):
#   Qwen3-TTS-12Hz-0.6B-Base    - Fastest, ~3-8s generation, good quality
#   Qwen3-TTS-12Hz-0.6B-CustomVoice - Fast, ~5-15s generation, excellent quality
#   Qwen3-TTS-12Hz-1.7B-CustomVoice - Slower, ~10-30s generation, best quality
TTS_MODEL_TIER = os.getenv("TTS_MODEL_TIER", "1.7B")
TTS_MODEL_MAP = {
    "Kokoro": "Kokoro",
    "0.6B-Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "0.6B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}
#TTS_MODEL = os.getenv("TTS_MODEL", TTS_MODEL_MAP.get(TTS_MODEL_TIER, TTS_MODEL_MAP["1.7B"]))
TTS_MODEL = "Kokoro"
TTS_DEVICE = os.getenv("TTS_DEVICE", "xpu")
TTS_DTYPE = os.getenv("TTS_DTYPE", "float32")
TTS_VOICE_INSTRUCT = os.getenv(
    "TTS_VOICE_INSTRUCT",
    "Speak in a calm, natural conversational tone.",
)
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "Auto")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "eric")
TTS_CHUNK_WORDS = int(os.getenv("TTS_CHUNK_WORDS", "30"))  # Process every N words (increased for fewer TTS passes)
TTS_MAX_CHUNK_DURATION = float(os.getenv("TTS_MAX_CHUNK_DURATION", "4.0"))  # Max seconds before forcing chunk
TTS_MIN_CHUNK_WORDS = int(os.getenv("TTS_MIN_CHUNK_WORDS", "8"))  # Minimum words before processing a chunk

# Sentence boundary regex. A terminal period/exclamation/question mark
# counts as a boundary when followed by:
#   - whitespace (the common case), or
#   - end of text, or
#   - a capitalized word (uppercase + lowercase) immediately preceded by
#     two word characters — this catches no-space concatenations from
#     streamed token joins ("end.This" -> "end." + "This") while keeping
#     "U.S." / "U.S.has" and numbers ("3.14") intact: an abbreviation's
#     period is preceded by a single letter or another period, so no
#     boundary is found there.
# Also used for flushing: a partial LLM message flushes to TTS as soon
# as it contains a complete sentence, so audio can start without waiting
# for the full response.
SENTENCE_RE = re.compile(r'[.!?](?=\s|$|(?<=\w\w[.!?])[A-Z][a-z])')

class TTSHandler:
    """TTS handling with chunked streaming support.

    Accumulates text from partial LLM tokens, processes sentence-sized
    chunks for low-latency audio generation, and handles interruptions
    when the user speaks again.
    """

    """
    Methods to keep in TTSHandler
    accumulate_text
    reset_buffer
    generate_sentences_in_order
    current_turn_id
    interrupt

    split_into_sentences
    generate_audio_chunks

    new:
    set_speaker
    set_voice_instruct


    Methods that are for TTSProcessor
    _process_chunk

    """

    def __init__(self, tts_processor):
        self.tts_processor = tts_processor

        self.processed_words = 0
        self.current_turn_id = None
        self.current_chunk_text = ""
        self.last_generated_text = ""
        self.last_generated_len = 0

        self.is_generating = False
        self.text_buffer = ""

    def set_speaker(self,speaker):
        self.tts_processor.speaker=speaker
        self.tts_processor.reload_model()

    def set_voice_instruct(self,voice_instruct):
        self.tts_processor.voice_instruct=voice_instruct
        self.tts_processor.reload_model()

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
            self.current_chunk_text += (" " if self.current_chunk_text else "") + new_text
            self.processed_words = current_words

            # Check if we should process this chunk. The boundary check
            # looks at the newly accumulated text, not the whole message:
            # the server forwards the full accumulated text every time, so
            # testing it would find the first sentence's boundary in every
            # partial and flush on every message.
            should_process = (
                self._has_sentence_boundary(self.current_chunk_text) or
                len(self.current_chunk_text.split()) >= TTS_CHUNK_WORDS or
                len(self.current_chunk_text) >= 150
            )

            if should_process:
                chunk_to_process = self.current_chunk_text
                self.current_chunk_text = ""
                self.processed_words = current_words
                # Must iterate the processor: a bare
                # `yield self.tts_processor._process_chunk(text)` hands the
                # worker an un-iterated generator object, which its dict
                # filter silently drops - no audio, no error. Routing
                # through generate_audio_chunks also applies dedup and the
                # interrupt check, as the final flush does.
                yield from self.generate_audio_chunks(chunk_to_process)
        else:
            # Final message - flush any text not yet covered by a
            # sentence-boundary partial. Only words not already processed are
            # appended, so sentences already spoken via partials are skipped.
            current_words = len(text.split())
            new_words = current_words - self.processed_words
            if new_words > 0:
                words = text.split()
                self.current_chunk_text += (" " if self.current_chunk_text else "") + " ".join(words[self.processed_words:])
            self.processed_words = 0
            if self.current_chunk_text.strip():
                remainder = self.current_chunk_text
                self.current_chunk_text = ""
                # Speak the remainder sentence by sentence so audio can start
                # immediately. Dedup is off here: the word accounting above
                # already guarantees only unspoken text reaches this flush.
                for sentence in self.split_into_sentences(remainder):
                    yield from self.generate_audio_chunks(sentence, dedup=False)

    def _has_sentence_boundary(self, text):
        """Check if text contains a sentence boundary."""
        return bool(SENTENCE_RE.search(text))

    def reset_buffer(self, turn_id=None):
        """Reset for new response."""
        self.text_buffer = ""
        self.current_chunk_text = ""
        self.processed_words = 0
        self.last_generated_text = ""
        self.last_generated_len = 0
        self.tts_processor.interrupt_event.clear()
        self.current_turn_id = turn_id
        # Clear so a new turn is never deduplicated against the previous
        # turn's generated prefix (e.g. "Yes." ending both turns).

    def interrupt(self):
        """Signal interruption (user started speaking again)."""
        self.tts_processor.interrupt_event.set()

    def split_into_sentences(self, text):
        """Split text into sentences using sentence boundary detection.

        Preserves sentence-ending punctuation. Returns list of sentences.
        """
        stripped = text.strip()
        if not stripped:
            return []

        # Split on sentence boundaries: whitespace, end of text, or a
        # capitalized word immediately preceded by two word chars follows
        # terminal punctuation (zero-width, so delimiters are not
        # consumed; see SENTENCE_RE).
        sentences = re.split(r'(?<=[.!?])(?=\s|$|(?<=\w\w[.!?])[A-Z][a-z])', stripped)

        # Filter empty sentences
        result = [s.strip() for s in sentences if s.strip()]
        return result

    def generate_audio_chunks(self, text, dedup=True):
        """Generator that yields audio chunks from text.

        With dedup enabled, skips redundant generation - if the same text has
        already been generated (from a previous partial message), it is skipped
        to avoid regenerating audio, as are shorter texts that are prefixes of
        already-generated text (they will be replaced by the longer version).
        Pass dedup=False when the caller has already de-duplicated by word
        accounting (the final flush), so every sentence is spoken.
        """
        stripped = text.strip()
        if not stripped:
            return
        if dedup:
            if stripped == self.last_generated_text:
                return
            # Skip if this text is shorter than what we've already generated
            # and is a prefix of it (will be replaced by the longer version)
            if len(stripped) < self.last_generated_len and self.last_generated_text.startswith(stripped):
                return
        self.last_generated_text = stripped
        self.last_generated_len = len(stripped)

        for result in self.tts_processor._process_chunk(stripped):
            if self.tts_processor.interrupt_event.is_set():
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
            if self.tts_processor.interrupt_event.is_set():
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



#class TTSProcessor:
#    """
#    Attempting to handle TTS Processors simpler but for now just template to copy to any new TTSProcessor
#    """
#
#    def __init__(self):


class KokoroTTSProcessor:
    """
    __init__(self)
    _load_model(self)
    _reload_model(self)
    _interrupt(self)
    _process_chunk(self,text)
    """

    def __init__(self,device="xpu",speaker="af_sarah",speed=1.0,model_path=None,voice_instruct=None,language="en"):
        self.interrupt_event = asyncio.Event()
        self.speaker = speaker
        self.speed = speed
        self.model_path = model_path
        self.device = device
        self.voice_instruct = voice_instruct
        self.language = language
        self._model_loaded = False
        self._load_model()

    def _load_model(self):
        self._reload_model()

    def reload_model(self):
        # Public alias used by TTSHandler.set_speaker (and the DEVICE_LOST
        # recovery path); re-creates the pipeline and re-warms it.
        self._reload_model()

    def _reload_model(self):
        # Initialize pipeline and pre-warm it (see _prewarm).
        self.pipeline = KPipeline(lang_code='a', device=self.device)
        self._prewarm()

    def _prewarm(self):
        """Run one throwaway generation so the first real sentence doesn't
        pay the XPU cold-start cost.

        The first pipeline() call on a fresh process takes ~4.5 s on the
        XPU vs ~1.7-2.3 s for subsequent calls (measured per utterance).
        Without this, the first audio of a turn lands after llm_end and
        partial streaming provides no benefit. KPipeline is lazy:
        inference runs when the returned generator is iterated, so the
        throwaway call must be drained.
        """
        logger.info("Pre-warming Kokoro pipeline (throwaway generation)...")
        t0 = _time.time()
        try:
            for _ in self.pipeline("Hello.", voice=self.speaker,
                                    speed=self.speed, split_pattern=r'\n+'):
                pass
            logger.info(f"Kokoro pre-warm done in {_time.time() - t0:.1f}s")
        except Exception as e:
            logger.warning(f"Kokoro pre-warm failed (first synthesis may be slow): {e}")

    def interrupt(self):
        self.interrupt_event.set()
        # TODO: make this actually implementing in the processing

    def _process_chunk(self,text):
        if not text or not text.strip():
            return

        # Kokoro doesn't handle certain characters well. Remove them.
        remove_characters_list=["**","\n"]
        for n in remove_characters_list:
            if text.find(n) > -1:
                text=text.replace(n,"")

        logger.info(f"TTS synthesizing chunk: \"{text[:80]}...\"")
        gen_start = _time.time()

        try:
            # Generate generator object (uses default voice 'af_sarah')
            generator = self.pipeline(text, voice=self.speaker, speed=self.speed, split_pattern=r'\n+')

            gen_time = _time.time() - gen_start
            logger.info(f"Model generation done in {gen_time:.1f}s")


            for i, (gs, ps, audio) in enumerate(generator):
                audio_np = audio.cpu().numpy().astype(np.float32)
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32)
                sr = 24000
                if sr != SAMPLE_RATE:
                    logger.info(f"Resampling kokoro audio from {sr} to {SAMPLE_RATE}")
                    audio_np = _resample(audio_np, sr, SAMPLE_RATE)
                chunk_size = int(SAMPLE_RATE * 0.5)
                num_samples = len(audio_np)
                for start in range(0, num_samples, chunk_size):
                    end = min(start + chunk_size, num_samples)
                    chunk = audio_np[start:end]
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
        except Exception as e:
            logger.error(f"STT _process_chunk failed with error ({e})")


class QwenTTSProcessor:
    """Qwen3-TTS processor with chunked streaming support.

    Accumulates text from partial LLM tokens, processes sentence-sized
    chunks for low-latency audio generation, and handles interruptions
    when the user speaks again.
    """

    def __init__(self, model_path, device, dtype_str, voice_instruct, language, speaker):
        self.interrupt_event = asyncio.Event()
        self.text_buffer = ""
        self.current_chunk_text = ""
        self.processed_words = 0
        self.speaker = speaker
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
        """Reload the model."""
        logger.info("Reloading TTS model...")
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

    def interrupt(self):
        """Signal interruption (user started speaking again)."""
        self.interrupt_event.set()

    def _process_chunk(self, text):
        """Process a text chunk and yield audio chunks.

        Args:
            text: Text chunk to synthesize

        Yields:
            dict with audio_chunk data
        """
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
                top_p=0.85,
                temperature=0.6,
                max_new_tokens=2048,
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
                        top_p=0.85,
                        temperature=0.6,
                        max_new_tokens=2048,
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


async def handle_tts(websocket, tts_handler):
    """Handle the TTS WebSocket connection."""
    async def send_heartbeats():
        while True:
            await asyncio.sleep(5)
            try:
                await websocket.send(json.dumps({"type": "heartbeat"}))
            except Exception:
                break

    heartbeat_task = asyncio.create_task(send_heartbeats())
    
    # Audio sequence counter. Reset for every TTS input message: the client
    # re-baselines its per-turn seq offset on every audio_end, and a
    # streamed response now sends one audio_end per input message.
    global_audio_seq = 0

    # TTS inputs are queued and handled by a single worker task so audio
    # chunks are always sent in order and an interrupt cancels the whole
    # in-flight generation chain (queued messages are dropped).
    tts_input_queue: asyncio.Queue = asyncio.Queue()
    tts_worker = None

    def log_tts_worker_error(t):
        if t.cancelled():
            return
        exc = t.exception()
        if exc:
            logger.error(f"TTS worker error: {exc}")

    async def process_tts_worker():
        """Process queued TTS input messages strictly in order.

        Each message gets its own audio_seq counter and sends its own
        audio_end, so a streamed response delivers several audio streams
        in sequence. Only the turn-final input (partial=False) carries
        "final": true in its audio_end; the server ends the speaking
        turn on that one (or on an interrupt), not on every per-sentence
        end.
        """
        nonlocal global_audio_seq
        while True:
            text, partial, turn_id = await tts_input_queue.get()
            # New response: reset dedup state (and any stale interrupt
            # event) so a previous, possibly interrupted, turn cannot
            # swallow this text.
            if turn_id is not None and turn_id != tts_handler.current_turn_id:
                tts_handler.reset_buffer(turn_id)
            # Interrupted while queued (user barge-in): drop it.
            if tts_handler.tts_processor.interrupt_event.is_set():
                continue
            global_audio_seq = 0

            local_chunk_id = 0
            chunk_queue: asyncio.Queue = asyncio.Queue()
            generation_complete = asyncio.Event()

            def generate_chunks():
                try:
                    if partial:
                        # Partial token - accumulate and process incrementally
                        for chunk in tts_handler.accumulate_text(text, partial=True):
                            if isinstance(chunk, dict):
                                logger.info("Partial Queuing audio")
                                chunk_queue.put_nowait(chunk)
                                logger.info("... Queued audio")
                    else:
                        # Complete response - flush the remainder (words
                        # already spoken via partials are deduplicated)
                        for chunk in tts_handler.accumulate_text(text, partial=False):
                            logger.info("Complete Queuing audio")
                            chunk_queue.put_nowait(chunk)
                            logger.info("... Queued audio")
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                finally:
                    generation_complete.set()
                    # Sentinel so the pump below wakes up the moment
                    # generation ends; otherwise a generation that yields no
                    # chunks (e.g. fully de-duplicated) stalls the pump for
                    # a full queue timeout (2.0 s) per message.
                    chunk_queue.put_nowait(None)

            gen_task = asyncio.create_task(asyncio.to_thread(generate_chunks))

            while not generation_complete.is_set():
                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    # No chunk within the window. If generation has since
                    # finished, the loop condition exits; otherwise a slow
                    # synthesis is in flight and we simply keep waiting.
                    continue
                if chunk is None:
                    # Generation-finished sentinel: break so the drain
                    # below picks up any chunks queued after the last yield.
                    break
                if chunk.get("interrupted"):
                    break
                chunk["audio_seq"] = global_audio_seq
                global_audio_seq += 1
                chunk["turn_id"] = tts_handler.current_turn_id
                await websocket.send(json.dumps(chunk))
                logger.info(f"sent audio chunk seq={global_audio_seq - 1}")
                local_chunk_id += 1

            # Wait for generation to fully complete
            await gen_task

            # Drain remaining chunks
            while not chunk_queue.empty():
                try:
                    chunk = chunk_queue.get_nowait()
                    if chunk is None or chunk.get("interrupted"):
                        break
                    chunk["audio_seq"] = global_audio_seq
                    global_audio_seq += 1
                    chunk["turn_id"] = tts_handler.current_turn_id
                    await websocket.send(json.dumps(chunk))
                    local_chunk_id += 1
                except Exception:
                    break

            await websocket.send(json.dumps({
                "type": "audio_end",
                "total_chunks": local_chunk_id,
                "turn_id": tts_handler.current_turn_id,
                "final": not partial,
            }))
            if local_chunk_id > 0:
                logger.info(f"TTS generation complete ({local_chunk_id} chunks)")

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
                    turn_id = msg.get("turn_id")

                    if not text.strip():
                        continue

                    logger.info(f"TTS received text (partial={partial}, turn_id={turn_id}): \"{text[:80]}...\"")

                    if tts_worker is None or tts_worker.done():
                        tts_worker = asyncio.create_task(process_tts_worker())
                        tts_worker.add_done_callback(log_tts_worker_error)
                    await tts_input_queue.put((text, partial, turn_id))

                elif msg_type == "stop_generation" or msg_type == "stop":
                    tts_handler.interrupt()
                    if tts_worker is not None and not tts_worker.done():
                        tts_worker.cancel()
                    while tts_input_queue.qsize():
                        try:
                            tts_input_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    await websocket.send(json.dumps({
                        "type": "audio_end",
                        "interrupted": True,
                        "final": True,
                        "turn_id": tts_handler.current_turn_id,
                    }))
                    logger.info("TTS generation stopped (interrupt)")

                elif msg_type == "set_voice":
                    speaker = msg.get("speaker", msg.get("voice"))
                    instruct = msg.get("instruct")
                    if speaker:
                        tts_handler.set_speaker(speaker)
                    if instruct:
                        tts_handler.tts_processor.set_voice_instruct = instruct
                    logger.info(f"TTS voice/instruct changed: speaker={speaker}, instruct={instruct[:50]}")

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
        if tts_worker is not None and not tts_worker.done():
            tts_worker.cancel()
            try:
                await tts_worker
            except (asyncio.CancelledError, Exception):
                pass
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
    if TTS_MODEL.find("Kokoro") > -1:
        tts_processor = KokoroTTSProcessor(device="xpu",speaker="af_sarah",speed=1.3,model_path=None,voice_instruct=None,language="en")
    elif TTS_MODEL.find("Qwen") > -1:
        tts_processor = QwenTTSProcessor(
            model_path=TTS_MODEL,
            device=TTS_DEVICE,
            dtype_str=TTS_DTYPE,
            voice_instruct=TTS_VOICE_INSTRUCT,
            language=TTS_LANGUAGE,
            speaker=TTS_SPEAKER,
           )
    else:
        logger.error("Unknown TTS model specified")
        raise

    tts_handler = TTSHandler(
        tts_processor=tts_processor
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
                await handle_tts(websocket, tts_handler)
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
