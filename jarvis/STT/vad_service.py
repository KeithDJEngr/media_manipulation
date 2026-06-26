#!/usr/bin/env python3
"""VAD (Voice Activity Detection) service.

Connects to the Jarvis server at ws://localhost:8765/vad and runs Silero VAD
on incoming audio chunks to detect speech vs silence endpoints.

Protocol (server -> client):
  - start: VAD service is ready and connected

Protocol (client -> server):
  - start: Acknowledge connection is ready
  - vad_result: speech/silence detection result

Silero VAD ONNX input requirements:
  - 512 samples at 16kHz (32ms) minimum window
  - float32 audio in [-1.0, 1.0] range
"""

import asyncio
import base64
import json
import logging
import os
import ssl
import sys
from pathlib import Path

import numpy as np
import torch
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Silero VAD model cache directory
CACHE_DIR = Path.home() / ".cache" / "torch" / "hub"


async def load_vad_model():
    """Load Silero VAD ONNX model via torch hub."""
    # Add Silero VAD source to path so torch hub can find it
    vad_src = CACHE_DIR / "hub" / "snakers4_silero-vad_master" / "src"
    if vad_src.exists():
        sys.path.insert(0, str(vad_src))

    # Load model (downloads to torch hub cache on first run)
    try:
        model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            onnx=True,
            force_reload=False,
            trust_repo=True,
        )
        logger.info("Silero VAD ONNX model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load Silero VAD model: {e}")
        raise


def int16_to_float32(audio_bytes):
    """Convert raw Int16 audio bytes to float32 numpy array in [-1.0, 1.0]."""
    int16_arr = np.frombuffer(audio_bytes, dtype=np.int16)
    float32_arr = int16_arr.astype(np.float32) / 32768.0
    return float32_arr


class VADProcessor:
    """Processes audio with Silero VAD and tracks speech/silence state."""

    # Silero VAD ONNX expects 512 samples at 16kHz
    WINDOW_SIZE = 512
    SAMPLE_RATE = 16000

    def __init__(self, model, threshold=0.5, silence_threshold_samples=1600, silence_threshold_samples_grace=4000, speech_end_debounce_samples=8000):
        """
        Args:
            model: Silero VAD ONNX model
            threshold: Speech probability threshold (0.0-1.0)
            silence_threshold_samples: Number of silent samples before declaring end of speech
                                       (1600 samples = 100ms at 16kHz)
            silence_threshold_samples_grace: Silent samples at start before silence counting begins
                                             (~250ms grace period for VAD state warmup)
            speech_end_debounce_samples: Additional silence samples to wait before confirming speech end
                                        (8000 samples = 0.5s debounce to batch rapid utterances)
        """
        self.model = model
        self.threshold = threshold
        self.silence_threshold_samples = silence_threshold_samples
        self.silence_threshold_samples_grace = silence_threshold_samples_grace
        self.speech_end_debounce_samples = speech_end_debounce_samples

        logger.info(f"Starting the vad with: ")
        logger.info(f"  WINDOW_SIZE: {self.WINDOW_SIZE}")
        logger.info(f"  SAMPLE_RATE: {self.SAMPLE_RATE}")
        logger.info(f"  THRESHOLD: {self.threshold}")
        logger.info(f"  SILENCE_THRESHOLD: {self.silence_threshold_samples}")
        logger.info(f"  SILENCE_GRACE: {self.silence_threshold_samples_grace}")
        logger.info(f"  DEBOUNCE: {self.speech_end_debounce_samples}")
        logger.info(f"  Resulting silence gap for speech end: {self.silence_threshold_samples/self.SAMPLE_RATE} sec")


        # State tracking
        self.is_speaking = False
        self.silence_sample_count = 0
        self.chunk_id = 0
        self.speech_ended = False

        # Audio buffer for speech segments (accumulates audio during speaking)
        self.speech_buffer = []
        self.speech_sample_count = 0

        # Debounce state for speech end batching
        self.debounce_silence_count = 0
        self.pending_speech_end = False

        # Internal processing state
        self._prev_prob = None

    def process_audio(self, audio_bytes):
        """Process a chunk of raw Int16 audio audio and return list of messages.

        Args:
            audio_bytes: Raw Int16 audio bytes (16kHz mono)

        Returns:
            List of message dicts to send to the server
        """
        messages = []
        self.chunk_id += 1

        # Convert to float32
        audio_float = int16_to_float32(audio_bytes)

        # Process in windows
        step = self.WINDOW_SIZE  # Non-overlapping windows
        windows = []
        offset = 0
        while offset + self.WINDOW_SIZE <= len(audio_float):
            windows.append(audio_float[offset : offset + self.WINDOW_SIZE])
            offset += step

        # Pad last window if needed
        if len(windows) == 0:
            # Audio chunk is smaller than window size
            padded = np.zeros(self.WINDOW_SIZE, dtype=np.float32)
            padded[: len(audio_float)] = audio_float
            windows = [padded]

        for window in windows:
            # Run VAD inference
            audio_tensor = torch.from_numpy(window)
            prob = self.model(audio_tensor, self.SAMPLE_RATE)

            # Ensure prob is a Python float
            if hasattr(prob, "item"):
                prob = prob.item()

            # Round for stability
            prob = round(prob, 4)

            is_speech = prob >= self.threshold

            if is_speech:
                # Speech detected
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_ended = False
                    logger.info("Detected start of speech")
                    messages.append(
                        {
                            "type": "vad_result",
                            "speech": True,
                            "chunk_id": self.chunk_id,
                        }
                    )

                # Add window to speech buffer
                self.speech_buffer.append(window)
                self.speech_sample_count += self.WINDOW_SIZE
                self.silence_sample_count = 0

            elif self.is_speaking:
                # Silence detected during speech
                self.silence_sample_count += self.WINDOW_SIZE

                # Check if silence duration exceeds threshold
                if self.silence_sample_count >= self.silence_threshold_samples:
                    # Check if we already have a pending speech end (debounce)
                    if self.pending_speech_end:
                        # Still in debounce - check if silence exceeds debounce threshold
                        if self.debounce_silence_count >= self.speech_end_debounce_samples:
                            # Debounce complete - confirm speech end
                            if self.speech_buffer:
                                full_audio = np.concatenate(self.speech_buffer)
                                messages.append(
                                    {
                                        "type": "audio_chunk",
                                        "audio": base64.b64encode(
                                            self.float32_to_int16_bytes(full_audio)
                                        ).decode("ascii"),
                                        "chunk_id": self.chunk_id,
                                    }
                                )
                                self.speech_buffer = []
                                self.speech_sample_count = 0

                            self.is_speaking = False
                            self.speech_ended = True
                            self.pending_speech_end = False
                            self.debounce_silence_count = 0
                            logger.info("Detected end of speech (after debounce)")
                            messages.append(
                                {
                                    "type": "vad_result",
                                    "speech": False,
                                    "chunk_id": self.chunk_id,
                                }
                            )
                        else:
                            # Still in debounce - accumulate silence count
                            self.debounce_silence_count += self.WINDOW_SIZE
                            # Send window individually during debounce
                            messages.append(
                                {
                                    "type": "audio_chunk",
                                    "audio": base64.b64encode(
                                        self.float32_to_int16_bytes(window)
                                    ).decode("ascii"),
                                    "chunk_id": self.chunk_id,
                                }
                            )
                    else:
                        # First silence threshold hit - start debounce
                        self.pending_speech_end = True
                        self.debounce_silence_count = 0
                        # Send accumulated speech buffer so far
                        if self.speech_buffer:
                            full_audio = np.concatenate(self.speech_buffer)
                            messages.append(
                                {
                                    "type": "audio_chunk",
                                    "audio": base64.b64encode(
                                        self.float32_to_int16_bytes(full_audio)
                                    ).decode("ascii"),
                                    "chunk_id": self.chunk_id,
                                }
                            )
                            self.speech_buffer = []
                            self.speech_sample_count = 0
                        # Don't send speech_end yet - waiting for debounce
                        logger.info("Silence threshold hit, starting debounce")

                else:
                    # Reset debounce if speech resumes during debounce period
                    if self.pending_speech_end:
                        self.pending_speech_end = False
                        self.debounce_silence_count = 0
                        logger.info("Speech resumed during debounce")

                    # Still accumulating speech - send window individually
                    messages.append(
                        {
                            "type": "audio_chunk",
                            "audio": base64.b64encode(
                                self.float32_to_int16_bytes(window)
                            ).decode("ascii"),
                            "chunk_id": self.chunk_id,
                        }
                    )

            else:
                # Silence during silence period - only send ack if we haven't sent end-of-speech yet
                if self.silence_sample_count < self.silence_threshold_samples_grace:
                    self.silence_sample_count += self.WINDOW_SIZE
                else:
                    self.silence_sample_count = self.silence_threshold_samples_grace

                if not self.speech_ended:
                    messages.append(
                        {
                            "type": "vad_result",
                            "speech": False,
                            "chunk_id": self.chunk_id,
                        }
                    )

        return messages

    @staticmethod
    def warmup(model, window_size=512, sample_rate=16000, num_frames=10):
        """Prime the Silero VAD model's hidden states with silence frames.
        
        Silero VAD uses an LSTM internally. The hidden states start at zero,
        making the first few predictions unreliable. Warmup ensures the model
        is ready for the first real audio chunk.
        """
        silence = np.zeros(window_size, dtype=np.float32)
        for _ in range(num_frames):
            tensor = torch.from_numpy(silence)
            model(tensor, sample_rate)

    @staticmethod
    def float32_to_int16_bytes(audio):
        """Convert float32 array back to Int16 bytes for base64 encoding."""
        clipped = np.clip(audio, -1.0, 1.0)
        int16_data = (clipped * 32767).astype(np.int16)
        return int16_data.tobytes()


async def handle_vad_connection(websocket, vad_processor):
    """Handle the VAD WebSocket connection to the server."""
    # Send start immediately to signal VAD is ready
    await websocket.send(json.dumps({"type": "start"}))
    logger.info("VAD connected to server, started sending audio")

    async def send_heartbeats():
        while True:
            await asyncio.sleep(5)
            try:
                await websocket.send(json.dumps({"type": "heartbeat"}))
            except Exception:
                break

    heartbeat_task = asyncio.create_task(send_heartbeats())

    try:
        async for message in websocket:
            try:
                # Check if it's a JSON control message
                if isinstance(message, str):
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "start":
                        logger.info("Server acknowledged VAD start")
                        continue

                    if msg_type == "audio_chunk":
                        # base64-encoded audio from browser via server
                        audio_b64 = data.get("audio", "")
                        audio_bytes = base64.b64decode(audio_b64)
                        #logger.info(f"Received audio_chunk chunk_id={data.get('chunk_id')} size={len(audio_bytes)}")
                        vad_messages = vad_processor.process_audio(audio_bytes)
                        
                        if len(vad_messages) > 0:
                            logger.info(f"VAD processed, sending {len(vad_messages)} messages")
                            for msg in vad_messages:
                                await websocket.send(json.dumps(msg))
                            continue

                    #logger.info(f"Unknown VAD message type: {msg_type}")
                    continue

                # Binary message: raw audio Int16 chunk (from browser)
                vad_messages = vad_processor.process_audio(message)
                for msg in vad_messages:
                    await websocket.send(json.dumps(msg))

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from server: {e}")
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                raise

    except websockets.ConnectionClosed as e:
        logger.info(f"VAD connection closed: {e}")
    except Exception as e:
        logger.error(f"VAD connection error: {e}")
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


async def main():
    server_host = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "8765"))
    vad_url = f"wss://{server_host}:{server_port}/vad_service"

    # Load Silero VAD model
    logger.info("Loading Silero VAD model...")
    model = await load_vad_model()

    # Create VAD processor with tuned parameters
    vad_processor = VADProcessor(
        model, threshold=0.6, silence_threshold_samples=9000,
        silence_threshold_samples_grace=4000, speech_end_debounce_samples=8000
    )

    # Prime the VAD model's hidden states so the first audio predictions are reliable
    VADProcessor.warmup(model)
    logger.info("VAD model warmed up")

    # Connect to the server
    logger.info(f"Connecting to server at {vad_url}")
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    while True:
        try:
            async with websockets.connect(vad_url, ssl=ssl_context) as websocket:
                await websocket.send(json.dumps({"type": "start"}))
                await websocket.send(json.dumps({
                    "type": "service_status",
                    "service": "vad",
                    "status": "connected",
                }))
                logger.info("VAD connected to server")
                await handle_vad_connection(websocket, vad_processor)
                await websocket.send(json.dumps({
                    "type": "service_status",
                    "service": "vad",
                    "status": "disconnected",
                }))
        except websockets.ConnectionClosed:
            logger.info("VAD connection closed, reconnecting...")
        except ConnectionRefusedError:
            logger.error(
                f"Could not connect to server at {vad_url}. "
                "Make sure the server is running on port {server_port}."
            )
            await asyncio.sleep(3)
            continue
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            await asyncio.sleep(3)
            continue


if __name__ == "__main__":
    asyncio.run(main())
