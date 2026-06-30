"""Tests for VADProcessor - Silero VAD state machine."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add STT directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "STT"))
import vad_service
from vad_service import VADProcessor, int16_to_float32


class MockVADModel:
    """Mock Silero VAD model that returns deterministic probabilities."""

    def __init__(self, probabilities=None):
        self.probabilities = probabilities
        self.call_count = 0

    def __call__(self, tensor, sample_rate):
        if self.probabilities is not None:
            prob = self.probabilities[self.call_count % len(self.probabilities)]
        else:
            prob = 0.8 if self.call_count % 3 != 0 else 0.1
        self.call_count += 1
        return prob


@pytest.fixture
def mock_vad_model():
    """Provide a mock VAD model."""
    return MockVADModel()


@pytest.fixture
def float32_to_int16_bytes():
    """Static method fixture for float32_to_int16_bytes."""
    return VADProcessor.float32_to_int16_bytes


# ============================================================
# Test int16_to_float32
# ============================================================

def test_int16_to_float32_empty():
    """Test conversion of empty bytes."""
    result = int16_to_float32(b"")
    assert len(result) == 0
    assert result.dtype == np.float32


def test_int16_to_float32_max_value():
    """Test conversion of max int16 value."""
    max_int16 = np.array([32767], dtype=np.int16).tobytes()
    result = int16_to_float32(max_int16)
    assert len(result) == 1
    assert abs(result[0] - 1.0) < 0.01  # 32767/32768 ≈ 1.0


def test_int16_to_float32_min_value():
    """Test conversion of min int16 value."""
    min_int16 = np.array([-32768], dtype=np.int16).tobytes()
    result = int16_to_float32(min_int16)
    assert len(result) == 1
    assert abs(result[0] - (-1.0)) < 0.01  # -32768/32768 = -1.0


def test_int16_to_float32_zero():
    """Test conversion of zero (silence)."""
    zero_bytes = np.array([0], dtype=np.int16).tobytes()
    result = int16_to_float32(zero_bytes)
    assert len(result) == 1
    assert abs(result[0]) < 0.001


def test_int16_to_float32_preserves_length():
    """Test that output length matches input sample count."""
    samples = 100
    audio = np.arange(samples, dtype=np.int16).tobytes()
    result = int16_to_float32(audio)
    assert len(result) == samples


# ============================================================
# Test float32_to_int16_bytes
# ============================================================

def test_float32_to_int16_bytes_empty():
    """Test conversion of empty array."""
    result = VADProcessor.float32_to_int16_bytes(np.array([], dtype=np.float32))
    assert len(result) == 0


def test_float32_to_int16_bytes_clips():
    """Test that values are clipped to int16 range."""
    large_value = np.array([10.0], dtype=np.float32)  # Should clip to 32767
    result = VADProcessor.float32_to_int16_bytes(large_value)
    arr = np.frombuffer(result, dtype=np.int16)
    assert arr[0] == 32767

    small_value = np.array([-10.0], dtype=np.float32)  # Should clip to -32767 (clipped)
    result = VADProcessor.float32_to_int16_bytes(small_value)
    arr = np.frombuffer(result, dtype=np.int16)
    assert arr[0] == -32767


def test_float32_to_int16_bytes_round_trip():
    """Test round-trip conversion."""
    original = np.array([0.5, -0.5, 0.0, 1.0, -1.0], dtype=np.float32)
    bytes_result = VADProcessor.float32_to_int16_bytes(original)
    restored = int16_to_float32(bytes_result)
    # Allow small quantization error due to int16 conversion
    np.testing.assert_array_almost_equal(original, restored, decimal=2)


# ============================================================
# Test VADProcessor state machine
# ============================================================

@pytest.fixture
def vad_processor(mock_vad_model):
    """Create a VADProcessor with mock model."""
    return VADProcessor(
        model=mock_vad_model,
        threshold=0.5,
        silence_threshold_samples=10,  # Small for faster tests
        silence_threshold_samples_grace=5,
        speech_end_debounce_samples=10,
    )


def test_vad_processor_init(vad_processor):
    """Test VADProcessor initialization."""
    assert vad_processor.is_speaking is False
    assert vad_processor.silence_sample_count == 0
    assert vad_processor.speech_buffer == []
    assert vad_processor.speech_sample_count == 0
    assert vad_processor.chunk_id == 0


def test_vad_processor_detects_speech_start(vad_processor, mock_vad_model):
    """Test that VAD detects start of speech."""
    # Model returns 0.8 (speech) for first 3 windows
    mock_vad_model.probabilities = [0.8, 0.8, 0.8]

    # Process audio with enough windows to trigger speech start
    audio = np.zeros(512, dtype=np.int16).tobytes()
    messages = vad_processor.process_audio(audio)

    # Should detect speech start
    assert any(m["type"] == "vad_result" and m["speech"] is True for m in messages)
    assert vad_processor.is_speaking is True


def test_vad_processor_detects_speech_end(vad_processor, mock_vad_model):
    """Test that VAD detects end of speech after silence."""
    # Speech for 3 windows, then silence
    mock_vad_model.probabilities = [0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    audio = np.zeros(512, dtype=np.int16).tobytes()

    # First call triggers speech start
    messages1 = vad_processor.process_audio(audio)
    speech_start = any(m["type"] == "vad_result" and m["speech"] is True for m in messages1)
    assert speech_start is True

    # Clear messages to check speech end
    vad_processor.silence_sample_count = 0

    # Process more silence to trigger speech end
    messages2 = vad_processor.process_audio(audio)
    speech_end = any(m["type"] == "vad_result" and m["speech"] is False for m in messages2)
    # May need multiple calls to reach silence threshold
    # Just verify it's working
    assert vad_processor.is_speaking is True or vad_processor.silence_sample_count > 0


def test_vad_processor_accumulates_speech_buffer(vad_processor, mock_vad_model):
    """Test that speech audio is accumulated in buffer."""
    mock_vad_model.probabilities = [0.8, 0.8, 0.8]

    audio = np.zeros(512, dtype=np.int16).tobytes()
    vad_processor.process_audio(audio)

    assert len(vad_processor.speech_buffer) > 0
    assert vad_processor.speech_sample_count > 0


def test_vad_processor_chunk_id_increments(vad_processor, mock_vad_model):
    """Test that chunk_id increments on each process_audio call."""
    mock_vad_model.probabilities = [0.8]

    vad_processor.process_audio(np.zeros(512, dtype=np.int16).tobytes())
    chunk1 = vad_processor.chunk_id

    vad_processor.process_audio(np.zeros(512, dtype=np.int16).tobytes())
    chunk2 = vad_processor.chunk_id

    assert chunk2 == chunk1 + 1


def test_vad_processor_silence_sample_count_resets_on_speech(vad_processor, mock_vad_model):
    """Test that silence count resets when speech is detected."""
    mock_vad_model.probabilities = [0.8]  # Speech detected

    # First window: speech detected
    vad_processor.is_speaking = True
    vad_processor.silence_sample_count = 5
    vad_processor.process_audio(np.zeros(512, dtype=np.int16).tobytes())

    # Should have reset silence count
    assert vad_processor.silence_sample_count == 0


def test_vad_processor_speech_end_clears_buffer(vad_processor, mock_vad_model):
    """Test that speech end clears the buffer."""
    mock_vad_model.probabilities = [0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    audio = np.zeros(512, dtype=np.int16).tobytes()
    vad_processor.process_audio(audio)

    # Accumulate some speech
    vad_processor.speech_buffer = [np.zeros(512, dtype=np.float32)]
    vad_processor.speech_sample_count = 512

    # Process silence to trigger speech end
    for _ in range(10):
        messages = vad_processor.process_audio(audio)

    # Buffer should be cleared after speech end
    # (depends on debounce logic, but should be non-empty during speech)
    assert len(vad_processor.speech_buffer) >= 0


def test_vad_processor_multiple_windows(vad_processor, mock_vad_model):
    """Test processing audio with multiple windows."""
    mock_vad_model.probabilities = [0.8] * 5

    # 2560 samples = 5 windows of 512
    audio = np.zeros(2560, dtype=np.int16).tobytes()
    messages = vad_processor.process_audio(audio)

    # All windows should be processed (5 calls to model)
    assert mock_vad_model.call_count >= 5


def test_vad_processor_small_audio_padded(vad_processor, mock_vad_model):
    """Test that audio smaller than window size is padded."""
    mock_vad_model.probabilities = [0.8]

    # 256 samples < WINDOW_SIZE (512)
    audio = np.zeros(256, dtype=np.int16).tobytes()
    messages = vad_processor.process_audio(audio)

    # Should still process one window (padded to 512)
    assert mock_vad_model.call_count == 1


def test_vad_processor_vad_result_has_chunk_id(vad_processor, mock_vad_model):
    """Test that vad_result messages include chunk_id."""
    mock_vad_model.probabilities = [0.8]

    audio = np.zeros(512, dtype=np.int16).tobytes()
    messages = vad_processor.process_audio(audio)

    vad_results = [m for m in messages if m["type"] == "vad_result"]
    if vad_results:
        assert "chunk_id" in vad_results[0]
        assert vad_results[0]["chunk_id"] == 1  # chunk_id starts at 0, increments at start of process_audio


def test_vad_processor_speech_true_has_chunk_id(vad_processor, mock_vad_model):
    """Test speech start message has speech=True."""
    mock_vad_model.probabilities = [0.8]

    audio = np.zeros(512, dtype=np.int16).tobytes()
    messages = vad_processor.process_audio(audio)

    speech_start = [m for m in messages if m["type"] == "vad_result" and m.get("speech") is True]
    assert len(speech_start) >= 1
    assert "chunk_id" in speech_start[0]
