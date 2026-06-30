"""Tests for STT wake word detection and partial transcript handling."""
import pytest
import sys
from pathlib import Path

# Add STT directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "STT"))
from stt_service import ParakeetSTT


@pytest.fixture
def stt_service():
    """Create a minimal ParakeetSTT for wake word tests."""
    return ParakeetSTT(model_name="mock", device="cpu", dtype=None)


def test_wake_word_mute_lowercase(stt_service):
    """Test 'mute' in lowercase is detected."""
    assert stt_service.check_wake_word("mute") == "mute"
    assert stt_service.check_wake_word("mute the microphone") == "mute"
    assert stt_service.check_wake_word("mute, please") == "mute"


def test_wake_word_mute_uppercase(stt_service):
    """Test 'Mute' in uppercase is detected."""
    assert stt_service.check_wake_word("Mute") == "mute"
    assert stt_service.check_wake_word("MUTE") == "mute"
    assert stt_service.check_wake_word("Mute the mic") == "mute"


def test_wake_word_unmute_lowercase(stt_service):
    """Test 'unmute' in lowercase is detected."""
    assert stt_service.check_wake_word("unmute") == "unmute"
    assert stt_service.check_wake_word("unmute the microphone") == "unmute"
    assert stt_service.check_wake_word("unmute, please") == "unmute"


def test_wake_word_unmute_uppercase(stt_service):
    """Test 'Unmute' in uppercase is detected."""
    assert stt_service.check_wake_word("Unmute") == "unmute"
    assert stt_service.check_wake_word("UNMUTE") == "unmute"
    assert stt_service.check_wake_word("Unmute the mic") == "unmute"


def test_wake_word_mute_with_punctuation(stt_service):
    """Test 'mute' with trailing punctuation is detected."""
    assert stt_service.check_wake_word("mute!") == "mute"
    assert stt_service.check_wake_word("mute.") == "mute"
    assert stt_service.check_wake_word("mute?") == "mute"
    assert stt_service.check_wake_word("mute,") == "mute"
    assert stt_service.check_wake_word("mute;") == "mute"
    assert stt_service.check_wake_word("mute:") == "mute"


def test_wake_word_unmute_with_punctuation(stt_service):
    """Test 'unmute' with trailing punctuation is detected."""
    assert stt_service.check_wake_word("unmute!") == "unmute"
    assert stt_service.check_wake_word("unmute.") == "unmute"
    assert stt_service.check_wake_word("unmute?") == "unmute"


def test_wake_word_mute_not_in_middle(stt_service):
    """Test 'mute' in the middle of text is NOT detected."""
    assert stt_service.check_wake_word("mute button") == "mute"
    assert stt_service.check_wake_word("the mute button") is None
    assert stt_service.check_wake_word("press mute to mute") is None


def test_wake_word_unmute_not_in_middle(stt_service):
    """Test 'unmute' in the middle of text is NOT detected."""
    assert stt_service.check_wake_word("unmute button") == "unmute"
    assert stt_service.check_wake_word("the unmute button") is None


def test_wake_word_empty_string(stt_service):
    """Test empty string returns None."""
    assert stt_service.check_wake_word("") is None


def test_wake_word_whitespace_only(stt_service):
    """Test whitespace-only string returns None."""
    assert stt_service.check_wake_word("   ") is None
    assert stt_service.check_wake_word("\t\n") is None


def test_wake_word_random_text(stt_service):
    """Test random text returns None."""
    assert stt_service.check_wake_word("hello world") is None
    assert stt_service.check_wake_word("the weather is nice") is None
    assert stt_service.check_wake_word("what is the meaning of life") is None


def test_wake_word_similar_words(stt_service):
    """Test words similar to mute/unmute but not exact match."""
    assert stt_service.check_wake_word("muting") is None
    assert stt_service.check_wake_word("unmuting") is None
    assert stt_service.check_wake_word("mutated") is None
    assert stt_service.check_wake_word("unmuted") is None


def test_wake_word_with_leading_whitespace(stt_service):
    """Test that leading whitespace is stripped."""
    assert stt_service.check_wake_word("  mute") == "mute"
    assert stt_service.check_wake_word("  unmute") == "unmute"


def test_wake_word_mixed_case(stt_service):
    """Test mixed case 'mute'/'unmute' is detected."""
    assert stt_service.check_wake_word("MuTe") == "mute"
    assert stt_service.check_wake_word("UnMuTe") == "unmute"
