#!/usr/bin/env python3
"""STT (Speech-to-Text) service using NVIDIA Parakeet-TDT.

Connects to the Jarvis server at ws://localhost:8765/stt and transcribes
audio chunks from the VAD service into text.

Protocol (client -> server):
  - start: STT service is ready

Protocol (server -> client):
  - audio_chunk: Base64-encoded audio with chunk_id

STT output sent to server (and forwarded to LLM/browser):
  - partial_transcript: Ongoing transcription with text and chunk_id
  - final_transcript: Completed transcription with text and chunk_id
"""

import asyncio
import base64
import json
import logging
import os
import ssl
import sys
from collections import deque

import numpy as np
import torch
import websockets
from nano_parakeet import from_pretrained

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Device preference: use CUDA if available, XPU (Intel GPU) next, fall back to CPU
def _xpu_is_available():
    if not hasattr(torch, "xpu"):
        return False
    orig_mask = os.environ.get("ZE_AFFINITY_MASK")
    # Level Zero fails with multiple Intel GPUs; restrict to GPU 0
    os.environ["ZE_AFFINITY_MASK"] = "0"
    try:
        return torch.xpu.is_available()
    except RuntimeError:
        return False
    finally:
        if orig_mask is not None:
            os.environ["ZE_AFFINITY_MASK"] = orig_mask
        else:
            os.environ.pop("ZE_AFFINITY_MASK", None)


def _detect_device(preferred):
    if preferred in ("auto", None, ""):
        if torch.cuda.is_available():
            return "cuda"
        if _xpu_is_available():
            return "xpu"
        return "cpu"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "xpu" and _xpu_is_available():
        return "xpu"
    if preferred == "cpu":
        return "cpu"
    # Unrecognized device name — fall back to first available
    if torch.cuda.is_available():
        return "cuda"
    if _xpu_is_available():
        return "xpu"
    return "cpu"

DEVICE = _detect_device(os.getenv("STT_DEVICE", "auto"))
SAMPLE_RATE = 16000


class ParakeetSTT:
    """Wraps Parakeet-TDT for streaming transcription with buffering."""

    def __init__(self, model_name="nvidia/parakeet-tdt-0.6b-v3", device=None, dtype=None):
        """
        Args:
            model_name: Hugging Face model name or local path
            device: 'cuda', 'cpu', or None for auto-detect
            dtype: torch dtype, or None for auto (float16 on CUDA)
        """
        if device is None:
            device = DEVICE

        self.device = device
        self.dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)

        logger.info(f"Loading Parakeet-TDT model '{model_name}' on {device}...")
        self.model = from_pretrained(
            model_name,
            device=device,
            dtype=self.dtype,
        )
        self.model.eval()
        logger.info("Parakeet-TDT model loaded")

        # Audio buffer for accumulation
        self.audio_buffer = np.array([], dtype=np.float32)
        self.samples_since_last_partial = 0
        self.max_buffer_samples = int(30 * SAMPLE_RATE)  # 30 seconds max buffer

    def add_audio_chunk(self, audio_bytes):
        """Add an audio chunk to the buffer.

        Args:
            audio_bytes: Raw Int16 audio bytes (16kHz mono)
        """
        int16_arr = np.frombuffer(audio_bytes, dtype=np.int16)
        float32_arr = int16_arr.astype(np.float32) / 32768.0
        self.audio_buffer = np.concatenate([self.audio_buffer, float32_arr])
        if len(self.audio_buffer) > self.max_buffer_samples:
            logger.warning(f"STT buffer exceeded {self.max_buffer_samples} samples, clearing")
            self.audio_buffer = np.array([], dtype=np.float32)
            self.samples_since_last_partial = 0

    def add_float32_chunk(self, audio_float):
        """Add a float32 audio chunk to the buffer.

        Args:
            audio_float: Float32 numpy array in [-1.0, 1.0]
        """
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_float])
        if len(self.audio_buffer) > self.max_buffer_samples:
            logger.warning(f"STT buffer exceeded {self.max_buffer_samples} samples, clearing")
            self.audio_buffer = np.array([], dtype=np.float32)
            self.samples_since_last_partial = 0

    def transcribe(self):
        """Transcribe the accumulated audio buffer and clear it.

        Returns:
            str: Transcribed text, or empty string if no audio
        """
        if len(self.audio_buffer) < 512:  # Less than 32ms of audio
            return ""

        audio_tensor = torch.from_numpy(self.audio_buffer).to(self.device)
        token_ids = self.model.transcribe_audio(audio_tensor)

        # Decode tokens to text
        text = self.model.sp.DecodeIds(token_ids).strip()

        # Clear buffer after transcription
        self.audio_buffer = np.array([], dtype=np.float32)

        return text

    def transcribe_partial(self):
        """Transcribe the accumulated audio buffer without clearing it.

        Returns:
            str: Partial transcribed text, or empty string if no audio
        """
        if len(self.audio_buffer) < 512:  # Less than 32ms of audio
            return ""

        audio_tensor = torch.from_numpy(self.audio_buffer).to(self.device)
        token_ids = self.model.transcribe_audio(audio_tensor)

        # Decode tokens to text
        return self.model.sp.DecodeIds(token_ids).strip()

    def clear_buffer(self):
        """Clear the audio buffer without transcribing."""
        self.audio_buffer = np.array([], dtype=np.float32)


async def handle_stt(websocket, stt_service):
    """Handle the STT WebSocket connection."""
    async def send_heartbeats():
        while True:
            await asyncio.sleep(5)
            try:
                await websocket.send(json.dumps({"type": "heartbeat"}))
            except Exception:
                break

    heartbeat_task = asyncio.create_task(send_heartbeats())
    
    waiting_for_audio_after_eos = False
    last_chunk_time = asyncio.get_event_loop().time()
    STT_CHUNK_TIMEOUT = 3.0

    try:
        # Wait for client message
        try:
            msg = await asyncio.wait_for(websocket.recv(), timeout=10)
            if isinstance(msg, bytes):
                data = json.loads(msg)
            else:
                data = json.loads(msg)

            if data.get("type") == "start":
                logger.info("STT service ready")
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for STT start message")
            return

        # Process incoming messages
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    msg = json.loads(message)
                else:
                    msg = json.loads(message)

                msg_type = msg.get("type")
                logger.info(f'msg.get("type"): {msg.get("type")}, msg.get("speech"): {msg.get("speech")}')

                # Check for timeout while waiting for audio after speech end
                if waiting_for_audio_after_eos:
                    elapsed = asyncio.get_event_loop().time() - last_chunk_time
                    if elapsed >= STT_CHUNK_TIMEOUT and len(stt_service.audio_buffer) > 0:
                        logger.info(f"STT timeout waiting for audio: {elapsed:.1f}s, buffer: {len(stt_service.audio_buffer)} samples")
                        waiting_for_audio_after_eos = False
                        text = stt_service.transcribe()
                        if text:
                            await websocket.send(json.dumps({
                                "type": "final_transcript",
                                "text": text,
                                "final": True,
                                "chunk_id": 0,
                            }))
                            logger.info(f"STT sent final_transcript (timeout flush)")
                        stt_service.clear_buffer()
                        continue

                if msg_type == "audio_chunk":
                    audio_b64 = msg.get("audio")
                    if not audio_b64:
                        continue
                    audio_bytes = base64.b64decode(audio_b64)
                    stt_service.add_audio_chunk(audio_bytes)
                    last_chunk_time = asyncio.get_event_loop().time()
                    stt_service.samples_since_last_partial += len(audio_bytes) // 2
                    logger.info(f"Added audio chunk, buffer size: {len(stt_service.audio_buffer)}, samples since last partial: {stt_service.samples_since_last_partial}")
                    if stt_service.samples_since_last_partial >= 1024:
                        logger.info("Triggering transcribe_partial on buffer size: %d", len(stt_service.audio_buffer))
                        partial_text = stt_service.transcribe_partial()
                        logger.info("transcribe_partial returned: '%s'", partial_text)
                        chunk_id = msg.get("chunk_id", 0)
                        await websocket.send(json.dumps({
                            "type": "partial_transcript",
                            "text": partial_text,
                            "chunk_id": chunk_id,
                        }))
                        logger.info(f"Sent partial_transcript: '{partial_text}'")
                        if waiting_for_audio_after_eos:
                            waiting_for_audio_after_eos = False
                            if partial_text:
                                await websocket.send(json.dumps({
                                    "type": "final_transcript",
                                    "text": partial_text,
                                    "final": True,
                                    "chunk_id": chunk_id,
                                }))
                                logger.info(f"STT sent final_transcript to server (deferred from end_of_speech)")
                                stt_service.clear_buffer()
                            else:
                                # Buffer has data but partial returned empty - check if buffer is getting too large
                                if len(stt_service.audio_buffer) >= 5120:
                                    logger.info(f"Buffer too large ({len(stt_service.audio_buffer)} samples) while waiting, sending empty final")
                                    await websocket.send(json.dumps({
                                        "type": "final_transcript",
                                        "text": "",
                                        "final": True,
                                        "chunk_id": chunk_id,
                                    }))
                                    stt_service.clear_buffer()
                                else:
                                    stt_service.samples_since_last_partial = 0
                        else:
                            stt_service.samples_since_last_partial = 0

                elif msg_type == "vad_result" and msg.get("speech") == False:
                    logger.info(f"End of speech, buffer size: {len(stt_service.audio_buffer)}")
                    if len(stt_service.audio_buffer) == 0:
                        waiting_for_audio_after_eos = True
                        logger.info("End of speech with empty buffer, waiting for audio chunk")
                    else:
                        waiting_for_audio_after_eos = True
                        # Transcribe accumulated audio
                        text = stt_service.transcribe()
                        if text:
                            chunk_id = msg.get("chunk_id", 0)
                            logger.info(f"STT transcription: \"{text}\"")
                            # Send final transcript to LLM
                            await websocket.send(json.dumps({
                                "type": "final_transcript",
                                "text": text,
                                "final": True,
                                "chunk_id": chunk_id,
                            }))
                            logger.info(f"STT sent final_transcript to server")
                        # Clear partial text on end of speech
                        stt_service.clear_buffer()

                elif msg_type == "partial":
                    # Partial transcription request (for streaming)
                    text = stt_service.transcribe_partial()
                    if text:
                        await websocket.send(json.dumps({
                            "type": "partial_transcript",
                            "text": text,
                            "chunk_id": msg.get("chunk_id", 0),
                        }))

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from server: {e}")
            except Exception as e:
                logger.error(f"STT processing error: {e}")
                raise

    except websockets.ConnectionClosed as e:
        logger.info(f"STT connection closed: {e}")
    except Exception as e:
        logger.error(f"STT connection error: {e}")
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


async def websocket_send_status(url, ssl_ctx, service, status):
    """Quick helper to send a service_status message to the server."""
    try:
        async with websockets.connect(url, ssl=ssl_ctx) as ws:
            await ws.send(json.dumps({"type": "start"}))
            await ws.send(json.dumps({
                "type": "service_status",
                "service": service,
                "status": status,
            }))
    except Exception:
        pass


async def main():
    server_host = os.getenv("SERVER_HOST", "127.0.0.1")
    server_port = int(os.getenv("SERVER_PORT", "8765"))
    stt_url = f"wss://{server_host}:{server_port}/stt"

    # Create STT service
    stt_model_name = os.getenv("STT_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    stt_service = ParakeetSTT(
        model_name=stt_model_name,
        device=DEVICE,
    )

    # Connect to server with reconnection
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    while True:
        try:
            async with websockets.connect(stt_url, ssl=ssl_context) as websocket:
                await websocket.send(json.dumps({"type": "start"}))
                await websocket.send(json.dumps({
                    "type": "service_status",
                    "service": "stt",
                    "status": "connected",
                }))
                logger.info("STT connected to server")
                try:
                    await handle_stt(websocket, stt_service)
                except websockets.ConnectionClosed:
                    await websocket.send(json.dumps({
                        "type": "service_status",
                        "service": "stt",
                        "status": "disconnected",
                    }))
                    logger.info("STT connection closed, reconnecting...")
        except ConnectionRefusedError:
            await websocket_send_status(stt_url, ssl_context, "stt", "disconnected")
            logger.error(
                f"Could not connect to server at {stt_url}. "
                f"Make sure the server is running on port {server_port}."
            )
        except Exception as e:
            await websocket_send_status(stt_url, ssl_context, "stt", "error")
            logger.error(f"STT connection error: {e}")
        
        logger.info("STT service reconnecting in 3 seconds...")
        await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())
       
