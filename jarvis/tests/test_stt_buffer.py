"""Tests for ParakeetSTT - audio buffer management."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add STT directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "STT"))
from stt_service import ParakeetSTT


@pytest.fixture
def stt_service(mock_stt_model):
    """Create a ParakeetSTT instance with mock model."""
    service = ParakeetSTT(model_name="mock", device="cpu", dtype=np.float32)
    service.model = mock_stt_model
    return service


def test_stt_buffer_init_empty(stt_service):
    """Test initial buffer is empty."""
    assert len(stt_service.audio_buffer) == 0
    assert stt_service.samples_since_last_partial == 0
    assert stt_service.is_muted is False


def test_stt_add_audio_chunk(stt_service):
    """Test adding raw Int16 audio bytes to buffer."""
    samples = 1600
    audio = np.zeros(samples, dtype=np.int16).tobytes()
    stt_service.add_audio_chunk(audio)

    assert len(stt_service.audio_buffer) == samples
    assert stt_service.audio_buffer.dtype == np.float32


def test_stt_add_float32_chunk(stt_service):
    """Test adding float32 numpy array to buffer."""
    samples = 1600
    audio = np.random.randn(samples).astype(np.float32)
    stt_service.add_float32_chunk(audio)

    assert len(stt_service.audio_buffer) == samples


def test_stt_add_multiple_chunks_accumulates(stt_service):
    """Test that multiple chunks accumulate in buffer."""
    chunk_size = 1600
    audio = np.zeros(chunk_size, dtype=np.int16).tobytes()

    stt_service.add_audio_chunk(audio)
    stt_service.add_audio_chunk(audio)
    stt_service.add_audio_chunk(audio)

    assert len(stt_service.audio_buffer) == chunk_size * 3


def test_stt_clear_buffer(stt_service):
    """Test that clear_buffer empties the buffer."""
    stt_service.audio_buffer = np.random.randn(1000).astype(np.float32)
    stt_service.clear_buffer()

    assert len(stt_service.audio_buffer) == 0


def test_stt_buffer_overflow_clears(stt_service):
    """Test that buffer clears when exceeding max_buffer_samples."""
    old_max = stt_service.max_buffer_samples
    stt_service.max_buffer_samples = 3200  # Small limit for test

    # Add chunks that exceed the limit
    chunk = np.zeros(2000, dtype=np.int16).tobytes()
    stt_service.add_audio_chunk(chunk)
    stt_service.add_audio_chunk(chunk)
    # Total = 4000 > 3200, should trigger overflow

    assert len(stt_service.audio_buffer) == 0
    assert stt_service.samples_since_last_partial == 0

    stt_service.max_buffer_samples = old_max


def test_stt_transcribe_with_empty_buffer(stt_service):
    """Test transcribe with empty buffer returns empty string."""
    result = stt_service.transcribe()
    assert result == ""


def test_stt_transcribe_with_small_buffer(stt_service):
    """Test transcribe with buffer smaller than 512 samples returns empty."""
    stt_service.audio_buffer = np.random.randn(100).astype(np.float32)
    result = stt_service.transcribe()
    assert result == ""


def test_stt_transcribe_calls_model(stt_service, mock_stt_model):
    """Test that transcribe calls the model's transcribe_audio."""
    stt_service.audio_buffer = np.random.randn(5120).astype(np.float32)
    result = stt_service.transcribe()

    assert mock_stt_model.call_count >= 1
    assert result == mock_stt_model.response_text


def test_stt_transcribe_clears_buffer(stt_service, mock_stt_model):
    """Test that transcribe clears the buffer after transcription."""
    stt_service.audio_buffer = np.random.randn(5120).astype(np.float32)
    stt_service.transcribe()

    assert len(stt_service.audio_buffer) == 0


def test_stt_transcribe_partial_returns_text(stt_service, mock_stt_model):
    """Test that transcribe_partial returns transcribed text without clearing buffer."""
    stt_service.audio_buffer = np.random.randn(5120).astype(np.float32)
    stt_service.transcribe()  # Call transcribe first to increment call_count
    stt_service.audio_buffer = np.random.randn(5120).astype(np.float32)  # Refill buffer
    result = stt_service.transcribe_partial()

    assert result == mock_stt_model.partial_response
    # Buffer should NOT be cleared
    assert len(stt_service.audio_buffer) > 0


def test_stt_transcribe_partial_calls_model(stt_service, mock_stt_model):
    """Test that transcribe_partial calls the model."""
    stt_service.audio_buffer = np.random.randn(5120).astype(np.float32)
    stt_service.transcribe_partial()

    assert mock_stt_model.call_count >= 1


def test_stt_transcribe_partial_does_not_clear_buffer(stt_service, mock_stt_model):
    """Test that transcribe_partial does not clear the buffer."""
    stt_service.audio_buffer = np.random.randn(5120).astype(np.float32)
    original_len = len(stt_service.audio_buffer)
    stt_service.transcribe_partial()

    assert len(stt_service.audio_buffer) == original_len


def test_stt_check_wake_word_mute(stt_service):
    """Test wake word detection for 'mute'."""
    assert stt_service.check_wake_word("mute the microphone") == "mute"
    assert stt_service.check_wake_word("Mute the mic") == "mute"
    assert stt_service.check_wake_word("mute.") == "mute"
    assert stt_service.check_wake_word("mute!") == "mute"


def test_stt_check_wake_word_unmute(stt_service):
    """Test wake word detection for 'unmute'."""
    assert stt_service.check_wake_word("unmute the microphone") == "unmute"
    assert stt_service.check_wake_word("Unmute please") == "unmute"


def test_stt_check_wake_word_none(stt_service):
    """Test that non-wake-word text returns None."""
    assert stt_service.check_wake_word("hello world") is None
    assert stt_service.check_wake_word("what is the weather") is None
    assert stt_service.check_wake_word("") is None


def test_stt_check_wake_word_mute_in_sentence(stt_service):
    """Test that 'mute' only matches as first word."""
    assert stt_service.check_wake_word("this is mute") is None
    assert stt_service.check_wake_word("the mute button") is None


def test_stt_is_muted_flag(stt_service):
    """Test is_muted flag can be set."""
    assert stt_service.is_muted is False
    stt_service.is_muted = True
    assert stt_service.is_muted is True


def test_stt_float32_chunk_accumulates(stt_service):
    """Test that add_float32_chunk properly accumulates."""
    chunk1 = np.random.randn(1000).astype(np.float32)
    chunk2 = np.random.randn(1000).astype(np.float32)

    stt_service.add_float32_chunk(chunk1)
    stt_service.add_float32_chunk(chunk2)

    assert len(stt_service.audio_buffer) == 2000


def test_stt_add_float32_chunk_overflow(stt_service):
    """Test overflow detection for float32 chunks."""
    old_max = stt_service.max_buffer_samples
    stt_service.max_buffer_samples = 1000

    chunk = np.random.randn(600).astype(np.float32)
    stt_service.add_float32_chunk(chunk)
    stt_service.add_float32_chunk(chunk)  # Total 1200 > 1000

    assert len(stt_service.audio_buffer) == 0
    stt_service.max_buffer_samples = old_max
