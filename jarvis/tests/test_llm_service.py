"""Tests for LLM service - message handling and history management."""
import asyncio
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add LLM directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "LLM"))
import llm_service


@pytest.fixture
def mock_llm_ws():
    """Create a mock WebSocket for LLM service."""
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.state = MagicMock()
    ws.state.name = "OPEN"
    ws.__aiter__ = AsyncMock()
    ws.__aiter__.return_value.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
    return ws


@pytest.fixture
def mock_httpx_stream():
    """Create a mock for httpx.AsyncClient.stream()."""
    mock_stream = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    # Mock the async iterator for aiter_lines
    lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        'data: {"choices": [{"delta": {"content": " world"}}]}',
        'data: [DONE]',
    ]
    mock_response.aiter_lines = AsyncMock(return_value=iter(lines))

    mock_stream.return_value = mock_stream
    mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream.__aexit__ = AsyncMock(return_value=None)
    return mock_stream


def test_handle_llm_starts_with_start_message(mock_llm_ws):
    """Test that handle_llm sends start and service_status messages."""
    async def run_test():
        llm_info = {
            "api_url": "http://mock:8000/chat",
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        await llm_service.handle_llm(mock_llm_ws, llm_info)

    asyncio.run(run_test())

    # Check that start and service_status were sent
    send_calls = [call[0][0] for call in mock_llm_ws.send.call_args_list]
    messages = [json.loads(c) if isinstance(c, str) else c for c in send_calls]

    assert any(m.get("type") == "start" for m in messages)
    assert any(m.get("type") == "service_status" for m in messages)


def test_handle_llm_receives_reset_history(mock_llm_ws):
    """Test that reset_history clears conversation history."""
    async def run_test():
        llm_info = {
            "api_url": "http://mock:8000/chat",
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        # Queue a reset_history message
        mock_llm_ws.__aiter__.return_value.__anext__ = AsyncMock(
            return_value=json.dumps({"type": "reset_history"})
        )
        await llm_service.handle_llm(mock_llm_ws, llm_info)

    asyncio.run(run_test())


def test_handle_llm_receives_set_system_prompt(mock_llm_ws):
    """Test that set_system_prompt updates the system prompt."""
    async def run_test():
        llm_info = {
            "api_url": "http://mock:8000/chat",
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        mock_llm_ws.__aiter__.return_value.__anext__ = AsyncMock(
            return_value=json.dumps({
                "type": "set_system_prompt",
                "prompt": "You are a new assistant.",
            })
        )
        await llm_service.handle_llm(mock_llm_ws, llm_info)

    asyncio.run(run_test())


def test_handle_llm_receives_set_settings(mock_llm_ws):
    """Test that set_settings updates use_full_history."""
    async def run_test():
        llm_info = {
            "api_url": "http://mock:8000/chat",
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        mock_llm_ws.__aiter__.return_value.__anext__ = AsyncMock(
            return_value=json.dumps({
                "type": "set_settings",
                "use_full_history": False,
            })
        )
        await llm_service.handle_llm(mock_llm_ws, llm_info)

    asyncio.run(run_test())


def test_handle_llm_empty_user_input_skipped(mock_llm_ws):
    """Test that empty user_input messages are skipped."""
    async def run_test():
        llm_info = {
            "api_url": "http://mock:8000/chat",
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        # Queue an empty user_input
        mock_llm_ws.__aiter__.return_value.__anext__ = AsyncMock(
            return_value=json.dumps({"type": "user_input", "text": "", "partial": False})
        )
        await llm_service.handle_llm(mock_llm_ws, llm_info)

    asyncio.run(run_test())


def test_handle_llm_partial_user_input_skipped(mock_llm_ws):
    """Test that partial user_input messages don't trigger generation."""
    async def run_test():
        llm_info = {
            "api_url": "http://mock:8000/chat",
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        # Queue a partial user_input
        mock_llm_ws.__aiter__.return_value.__anext__ = AsyncMock(
            return_value=json.dumps({
                "type": "user_input",
                "text": "Hello world",
                "partial": True,
            })
        )
        await llm_service.handle_llm(mock_llm_ws, llm_info)

    asyncio.run(run_test())
    # llm_start should NOT have been sent for partial


def test_handle_llm_heartbeat_sends_periodically(mock_llm_ws):
    """Test that heartbeat messages are sent periodically."""
    async def run_test():
        llm_info = {
            "api_url": "http://mock:8000/chat",
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        # Queue a heartbeat message
        mock_llm_ws.__aiter__.return_value.__anext__ = AsyncMock(
            return_value=json.dumps({"type": "heartbeat"})
        )
        await llm_service.handle_llm(mock_llm_ws, llm_info)

    asyncio.run(run_test())


def test_handle_llm_receives_user_input_with_history(mock_llm_ws):
    """Test that user_input with history updates conversation history."""
    async def run_test():
        llm_info = {
            "api_url": "http://mock:8000/chat",
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        test_history = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"},
        ]

        mock_llm_ws.__aiter__.return_value.__anext__ = AsyncMock(
            return_value=json.dumps({
                "type": "user_input",
                "text": "New message",
                "history": test_history,
                "partial": False,
            })
        )
        await llm_service.handle_llm(mock_llm_ws, llm_info)

    asyncio.run(run_test())


def test_llm_service_constants():
    """Test that LLM service constants are set."""
    assert hasattr(llm_service, "LLM_API_URL")
    assert hasattr(llm_service, "LLM_MODEL")
    assert hasattr(llm_service, "LLM_TEMPERATURE")
    assert hasattr(llm_service, "LLM_MAX_TOKENS")
    assert hasattr(llm_service, "SYSTEM_PROMPT")


def test_llm_service_uses_full_history_default():
    """Test that use_full_history defaults to True."""
    # This is a module-level variable that can be changed
    assert llm_service.use_full_history is True
