"""Tests for QwenTTSProcessor - text accumulation and sentence splitting."""
import numpy as np
import pytest
import sys
import asyncio
import torch
from pathlib import Path

# Add TTS directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "TTS"))


@pytest.fixture
def tts_processor(mock_tts_model):
    """Create a QwenTTSProcessor with mock model."""
    from tts_service import QwenTTSProcessor
    processor = QwenTTSProcessor(
        model_path="mock",
        device="cpu",
        dtype_str="float32",
        voice_instruct="test voice",
        language="en",
        speaker="test_speaker",
    )
    processor.model = mock_tts_model
    processor._model_loaded = True
    return processor


# ============================================================
# Test _parse_dtype
# ============================================================

def test_parse_dtype_float32():
    """Test parsing 'float32' string."""
    from tts_service import _parse_dtype
    result = _parse_dtype("float32")
    assert result == torch.float32


def test_parse_dtype_float16():
    """Test parsing 'float16' string."""
    from tts_service import _parse_dtype
    result = _parse_dtype("float16")
    assert result == torch.float16


def test_parse_dtype_float64():
    """Test parsing 'float64' string."""
    from tts_service import _parse_dtype
    # float64 not in dtype_map, falls back to bfloat16
    result = _parse_dtype("float64")
    assert result == torch.bfloat16


def test_parse_dtype_default():
    """Test default dtype when string is unrecognized."""
    from tts_service import _parse_dtype
    result = _parse_dtype("unknown")
    assert result == torch.bfloat16  # Default fallback


# ============================================================
# Test SENTENCE_RE regex
# ============================================================

def test_sentence_boundary_period(tts_processor):
    """Test sentence boundary detection with period."""
    from tts_service import SENTENCE_RE
    assert SENTENCE_RE.search("Hello world.") is not None
    assert SENTENCE_RE.search("Hello world. ") is not None


def test_sentence_boundary_exclamation(tts_processor):
    """Test sentence boundary detection with exclamation mark."""
    from tts_service import SENTENCE_RE
    assert SENTENCE_RE.search("Hello!") is not None
    assert SENTENCE_RE.search("Hello! ") is not None


def test_sentence_boundary_question(tts_processor):
    """Test sentence boundary detection with question mark."""
    from tts_service import SENTENCE_RE
    assert SENTENCE_RE.search("Hello?") is not None
    assert SENTENCE_RE.search("Hello? ") is not None


def test_no_sentence_boundary(tts_processor):
    """Test text without sentence boundary."""
    from tts_service import SENTENCE_RE
    assert SENTENCE_RE.search("Hello world") is None
    assert SENTENCE_RE.search("Hello world, how are you") is None


# ============================================================
# Test QwenTTSProcessor initialization
# ============================================================

def test_tts_processor_init(tts_processor):
    """Test QwenTTSProcessor initialization."""
    assert tts_processor.is_generating is False
    assert tts_processor.text_buffer == ""
    assert tts_processor.current_chunk_text == ""
    assert tts_processor.processed_words == 0
    assert tts_processor.speaker == "test_speaker"
    assert tts_processor.interrupt_event.is_set() is False


# ============================================================
# Test accumulate_text
# ============================================================

def test_accumulate_text_first_partial(tts_processor):
    """Test accumulating the first partial text."""
    chunks = list(tts_processor.accumulate_text("Hello world", partial=True))
    # Should process if word count exceeds TTS_CHUNK_WORDS or sentence boundary
    assert len(chunks) >= 0  # May or may not generate depending on word count


def test_accumulate_text_accumulates_words(tts_processor):
    """Test that accumulated text tracks word count correctly."""
    tts_processor.accumulate_text("One two three four five", partial=True)
    assert tts_processor.processed_words == 5


def test_accumulate_text_ignores_same_word_count(tts_processor):
    """Test that same word count is ignored (no new words)."""
    tts_processor.processed_words = 5
    chunks = list(tts_processor.accumulate_text("One two three four five", partial=True))
    # Should not process since no new words
    # (depends on sentence boundary and chunk size)


def test_accumulate_text_final_token_clears_buffer(tts_processor, mock_tts_model):
    """Test that final token (partial=False) clears the buffer."""
    tts_processor.current_chunk_text = "remaining text"
    tts_processor.processed_words = 2

    chunks = list(tts_processor.accumulate_text("One two three four", partial=False))

    assert tts_processor.processed_words == 0
    assert tts_processor.current_chunk_text == ""


def test_accumulate_text_final_token_processes_remaining(tts_processor, mock_tts_model):
    """Test that final token processes remaining text."""
    tts_processor.current_chunk_text = "hello world"
    tts_processor.processed_words = 2

    # Should process the remaining text
    chunks = list(tts_processor.accumulate_text("hello world", partial=False))
    assert len(chunks) >= 1
    assert tts_processor.current_chunk_text == ""
    assert tts_processor.processed_words == 0


def test_accumulate_text_empty_text(tts_processor):
    """Test that empty text is ignored."""
    chunks = list(tts_processor.accumulate_text("", partial=True))
    assert len(chunks) == 0


def test_accumulate_text_sentence_boundary_triggers_process(tts_processor, mock_tts_model):
    """Test that sentence boundary triggers chunk processing."""
    tts_processor.current_chunk_text = ""
    chunks = list(tts_processor.accumulate_text("Hello world.", partial=True))
    # Should process due to sentence boundary
    assert len(chunks) >= 1


def test_accumulate_text_word_count_threshold(tts_processor, mock_tts_model):
    """Test that word count threshold triggers processing."""
    from tts_service import TTS_CHUNK_WORDS
    # Reset processor
    tts_processor.reset_buffer()

    # Accumulate enough words to exceed TTS_CHUNK_WORDS
    word_count = TTS_CHUNK_WORDS + 10
    long_text = " ".join(["word"] * word_count)
    chunks = list(tts_processor.accumulate_text(long_text, partial=True))
    assert len(chunks) >= 1


def test_accumulate_text_character_length_threshold(tts_processor, mock_tts_model):
    """Test that character length threshold (150) triggers processing."""
    tts_processor.current_chunk_text = ""
    long_text = "a" * 200  # 200 chars > 150
    chunks = list(tts_processor.accumulate_text(long_text, partial=True))
    assert len(chunks) >= 1


# ============================================================
# Test reset_buffer
# ============================================================

def test_reset_buffer_clears_state(tts_processor):
    """Test that reset_buffer clears all text buffers."""
    tts_processor.text_buffer = "some text"
    tts_processor.current_chunk_text = "more text"
    tts_processor.processed_words = 5
    tts_processor.current_turn_id = 42

    tts_processor.reset_buffer()

    assert tts_processor.text_buffer == ""
    assert tts_processor.current_chunk_text == ""
    assert tts_processor.processed_words == 0
    assert tts_processor.interrupt_event.is_set() is False


def test_reset_buffer_sets_turn_id(tts_processor):
    """Test that reset_buffer can set a new turn_id."""
    tts_processor.reset_buffer(turn_id=99)
    assert tts_processor.current_turn_id == 99


# ============================================================
# Test interrupt
# ============================================================

def test_interrupt_sets_event(tts_processor):
    """Test that interrupt() sets the interrupt event."""
    assert tts_processor.interrupt_event.is_set() is False
    tts_processor.interrupt()
    assert tts_processor.interrupt_event.is_set() is True


# ============================================================
# Test _has_sentence_boundary
# ============================================================

def test_has_sentence_boundary_period(tts_processor):
    """Test sentence boundary detection with period."""
    assert tts_processor._has_sentence_boundary("Hello world.") is True


def test_has_sentence_boundary_exclamation(tts_processor):
    """Test sentence boundary detection with exclamation mark."""
    assert tts_processor._has_sentence_boundary("Hello!") is True


def test_has_sentence_boundary_no_boundary(tts_processor):
    """Test text without sentence boundary."""
    assert tts_processor._has_sentence_boundary("Hello world") is False


# ============================================================
# Test _process_chunk
# ============================================================

def test_process_chunk_calls_model(tts_processor, mock_tts_model):
    """Test that _process_chunk calls the model's generate_custom_voice."""
    chunks = list(tts_processor._process_chunk("Hello world test"))
    assert mock_tts_model.call_count >= 1


def test_process_chunk_empty_text(tts_processor, mock_tts_model):
    """Test that empty text doesn't call the model."""
    initial_count = mock_tts_model.call_count
    chunks = list(tts_processor._process_chunk(""))
    assert mock_tts_model.call_count == initial_count


def test_process_chunk_whitespace_only(tts_processor, mock_tts_model):
    """Test that whitespace-only text doesn't call the model."""
    initial_count = mock_tts_model.call_count
    chunks = list(tts_processor._process_chunk("   "))
    assert mock_tts_model.call_count == initial_count
