"""Pytest configuration for jarvis tests."""
import asyncio
import json
import numpy as np
import sys
import pytest
from unittest.mock import MagicMock, patch


class MockQwen3TTSModel:
    """Mock Qwen3TTSModel for TTS tests."""

    def __init__(self, audio_data=None, sample_rate=16000):
        self.audio_data = audio_data or np.random.randn(16000).astype(np.float32)
        self.sample_rate = sample_rate
        self.call_count = 0

    def generate_custom_voice(
        self, text, speaker, instruct, language, non_streaming_mode=True, do_sample=False, **kwargs
    ):
        self.call_count += 1
        return self.audio_data, self.sample_rate


# Mock heavy imports before they're loaded
sys.modules["nano_parakeet"] = MagicMock()
sys.modules["qwen_tts"] = MagicMock()
sys.modules["diffusers"] = MagicMock()
sys.modules["onnxruntime"] = MagicMock()
sys.modules["onnxruntime-gpu"] = MagicMock()
sys.modules["openvino"] = MagicMock()
sys.modules["intel_extension_for_pytorch"] = MagicMock()
sys.modules["soundfile"] = MagicMock()

# Set up qwen_tts mock to return MockQwen3TTSModel from from_pretrained
def mock_from_pretrained(*args, **kwargs):
    return MockQwen3TTSModel()

sys.modules["qwen_tts"].Qwen3TTSModel.from_pretrained = mock_from_pretrained


class MockWebSocket:
    """Minimal mock of websockets.WebSocketCommonProtocol for testing."""

    def __init__(self):
        self.state = MagicMock()
        self.state.name = "OPEN"
        self._messages_sent = []
        self._message_queue = asyncio.Queue()
        self._closed = False

    async def send(self, data):
        if isinstance(data, str):
            self._messages_sent.append(json.loads(data))
        else:
            self._messages_sent.append(data)

    def add_incoming(self, msg_dict):
        self._message_queue.put_nowait(json.dumps(msg_dict))

    async def recv_text(self):
        msg = await self._message_queue.get()
        return msg

    async def recv(self):
        return await self.recv_text()

    @property
    def messages_sent(self):
        return self._messages_sent


def pytest_configure(config):
    """Configure pytest-asyncio mode."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an asyncio test"
    )


@pytest.fixture
def mock_websocket():
    """Provide a mock WebSocket connection."""
    return MockWebSocket()


class MockSTTModel:
    """Mock Parakeet-TDT model for STT tests."""

    def __init__(self, response_text="Hello world", partial_response="Hello wo"):
        self.response_text = response_text
        self.partial_response = partial_response
        self.call_count = 0
        self.sp = MagicMock()
        self.sp.DecodeIds.return_value = response_text

    def transcribe_audio(self, tensor):
        # Return response_text by default, partial_response only on every 5th call
        if self.call_count % 5 == 0:  # Every 5th call returns partial
            self.sp.DecodeIds.return_value = self.partial_response
        else:
            self.sp.DecodeIds.return_value = self.response_text
        self.call_count += 1
        return [1, 2, 3, 4, 5]  # Mock token IDs


@pytest.fixture
def mock_stt_model():
    """Provide a mock STT model."""
    model = MockSTTModel()
    model.call_count = 1  # Start at 1 so first call returns response_text
    return model


@pytest.fixture
def mock_tts_model():
    """Provide a mock TTS model."""
    return MockQwen3TTSModel()


@pytest.fixture(autouse=True)
def ensure_qwen_mock():
    """Ensure qwen_tts mock is set up before each test."""
    sys.modules["qwen_tts"] = MagicMock()
    sys.modules["qwen_tts"].Qwen3TTSModel.from_pretrained = lambda *args, **kwargs: MockQwen3TTSModel()
    yield
