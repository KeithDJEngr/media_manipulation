"""Tests for ConnectionManager - the central hub for all WebSocket connections."""
import asyncio
import json
import pytest
import sys
from pathlib import Path

# Add Server directory to path so we can import jarvis_server
sys.path.insert(0, str(Path(__file__).parent.parent / "Server"))
from jarvis_server import ConnectionManager


class MockWS:
    """Minimal mock WebSocket with OPEN state."""

    def __init__(self):
        self.state = type("State", (), {"name": "OPEN"})()
        self.messages = []
        self._closed = False

    async def send(self, data):
        if isinstance(data, bytes):
            self.messages.append(data)
        else:
            self.messages.append(data)

    async def close(self):
        self._closed = True


@pytest.mark.asyncio
async def test_register_and_unregister_client():
    """Test client registration and unregistration."""
    manager = ConnectionManager()
    ws = MockWS()

    assert manager.client_ws is None
    await manager.register_client(ws)
    assert manager.client_ws is ws

    await manager.unregister_client(ws)
    assert manager.client_ws is None


@pytest.mark.asyncio
async def test_register_client_double_register():
    """Test that double-registering client replaces the old one."""
    manager = ConnectionManager()
    ws1 = MockWS()
    ws2 = MockWS()

    await manager.register_client(ws1)
    await manager.register_client(ws2)

    assert manager.client_ws is ws2
    assert manager.client_ws is not ws1


@pytest.mark.asyncio
async def test_unregister_wrong_websocket():
    """Test unregistering with the wrong WebSocket doesn't clear the client."""
    manager = ConnectionManager()
    ws1 = MockWS()
    ws2 = MockWS()

    await manager.register_client(ws1)
    await manager.unregister_client(ws2)

    assert manager.client_ws is ws1


@pytest.mark.asyncio
async def test_register_vad_service(mock_websocket):
    """Test VAD service registration broadcasts vad_connected."""
    manager = ConnectionManager()
    client_ws = MockWS()
    await manager.register_client(client_ws)

    await manager.register_vad(mock_websocket, is_browser=False)

    assert manager.vad_ws is mock_websocket
    assert manager.service_status["vad"] == "idle"

    # Check that vad_connected was broadcast to client
    sent = json.loads(client_ws.messages[0])
    assert sent["type"] == "vad_connected"


@pytest.mark.asyncio
async def test_register_vad_browser(mock_websocket):
    """Test browser VAD endpoint registration."""
    manager = ConnectionManager()
    await manager.register_vad(mock_websocket, is_browser=True)

    assert manager.vad_browser_ws is mock_websocket
    assert manager.vad_ws is None


@pytest.mark.asyncio
async def test_register_unregister_vad_service():
    """Test VAD service unregistration."""
    manager = ConnectionManager()
    ws = MockWS()

    await manager.register_vad(ws, is_browser=False)
    assert manager.vad_ws is ws

    await manager.unregister_vad(ws)
    assert manager.vad_ws is None
    assert manager.service_status["vad"] == "offline"


@pytest.mark.asyncio
async def test_register_stt_service(mock_websocket):
    """Test STT service registration."""
    manager = ConnectionManager()
    client_ws = MockWS()
    await manager.register_client(client_ws)

    await manager.register_stt(mock_websocket)

    assert manager.stt_ws is mock_websocket
    assert manager.service_status["stt"] == "idle"


@pytest.mark.asyncio
async def test_register_unregister_stt_service():
    """Test STT service unregistration."""
    manager = ConnectionManager()
    ws = MockWS()

    await manager.register_stt(ws)
    assert manager.stt_ws is ws

    await manager.unregister_stt(ws)
    assert manager.stt_ws is None
    assert manager.service_status["stt"] == "offline"


@pytest.mark.asyncio
async def test_register_llm_service(mock_websocket):
    """Test LLM service registration."""
    manager = ConnectionManager()
    client_ws = MockWS()
    await manager.register_client(client_ws)

    await manager.register_llm(mock_websocket)

    assert manager.llm_service_ws is mock_websocket
    assert manager.service_status["llm"] == "idle"


@pytest.mark.asyncio
async def test_register_unregister_llm_service():
    """Test LLM service unregistration."""
    manager = ConnectionManager()
    ws = MockWS()

    await manager.register_llm(ws)
    assert manager.llm_service_ws is ws

    await manager.unregister_llm(ws)
    assert manager.llm_service_ws is None
    assert manager.service_status["llm"] == "offline"


@pytest.mark.asyncio
async def test_register_tts_service(mock_websocket):
    """Test TTS service registration."""
    manager = ConnectionManager()
    client_ws = MockWS()
    await manager.register_client(client_ws)

    await manager.register_tts(mock_websocket)

    assert manager.tts_ws is mock_websocket
    assert manager.service_status["tts"] == "idle"


@pytest.mark.asyncio
async def test_register_unregister_tts_service():
    """Test TTS service unregistration."""
    manager = ConnectionManager()
    ws = MockWS()

    await manager.register_tts(ws)
    assert manager.tts_ws is ws

    await manager.unregister_tts(ws)
    assert manager.tts_ws is None
    assert manager.service_status["tts"] == "offline"


@pytest.mark.asyncio
async def test_broadcast_to_client_no_client():
    """Test broadcast_to_client when no client is connected."""
    manager = ConnectionManager()
    await manager.broadcast_to_client({"type": "test"})
    # Should not raise


@pytest.mark.asyncio
async def test_broadcast_to_client_sends_message(mock_websocket):
    """Test broadcast_to_client sends message to connected client."""
    manager = ConnectionManager()
    client_ws = MockWS()
    await manager.register_client(client_ws)

    await manager.broadcast_to_client({"type": "test_message", "data": "hello"})

    assert len(client_ws.messages) == 1
    sent = json.loads(client_ws.messages[0])
    assert sent["type"] == "test_message"
    assert sent["data"] == "hello"


@pytest.mark.asyncio
async def test_broadcast_to_service_no_service(mock_websocket):
    """Test broadcast_to_service when service is not registered."""
    manager = ConnectionManager()
    await manager.broadcast_to_service("stt", {"type": "test"})
    # Should not raise


@pytest.mark.asyncio
async def test_broadcast_to_service_sends_message(mock_websocket):
    """Test broadcast_to_service sends message to registered service."""
    manager = ConnectionManager()
    stt_ws = MockWS()
    await manager.register_stt(stt_ws)

    await manager.broadcast_to_service("stt", {"type": "partial", "text": "test"})

    assert len(stt_ws.messages) == 1
    sent = json.loads(stt_ws.messages[0])
    assert sent["type"] == "partial"
    assert sent["text"] == "test"


@pytest.mark.asyncio
async def test_set_service_status_broadcasts(mock_websocket):
    """Test set_service_status updates status and broadcasts."""
    manager = ConnectionManager()
    client_ws = MockWS()
    await manager.register_client(client_ws)

    await manager.set_service_status("vad", "active")

    assert manager.service_status["vad"] == "active"
    sent = json.loads(client_ws.messages[0])
    assert sent["type"] == "service_status"
    assert sent["service"] == "vad"
    assert sent["status"] == "active"


@pytest.mark.asyncio
async def test_set_service_active_activates(mock_websocket):
    """Test set_service_active sets active flag and broadcasts."""
    manager = ConnectionManager()
    client_ws = MockWS()
    await manager.register_client(client_ws)

    assert manager.service_active["stt"] is False
    await manager.set_service_active("stt", True)

    assert manager.service_active["stt"] is True
    assert manager.service_status["stt"] == "active"
    sent = json.loads(client_ws.messages[0])
    assert sent["type"] == "service_status"
    assert sent["status"] == "active"


@pytest.mark.asyncio
async def test_set_service_active_deactivates(mock_websocket):
    """Test set_service_active deactivates and sets idle."""
    manager = ConnectionManager()
    client_ws = MockWS()
    await manager.register_client(client_ws)

    await manager.set_service_active("stt", True)
    client_ws.messages.clear()

    await manager.set_service_active("stt", False)

    assert manager.service_active["stt"] is False
    assert manager.service_status["stt"] == "idle"
    sent = json.loads(client_ws.messages[0])
    assert sent["status"] == "idle"


@pytest.mark.asyncio
async def test_set_service_active_no_change(mock_websocket):
    """Test set_service_active doesn't broadcast if already at target state."""
    manager = ConnectionManager()
    client_ws = MockWS()
    await manager.register_client(client_ws)

    await manager.set_service_active("stt", True)
    client_ws.messages.clear()

    await manager.set_service_active("stt", True)

    assert len(client_ws.messages) == 0


@pytest.mark.asyncio
async def test_forward_message_to_client(mock_websocket):
    """Test forwarding message from service to client."""
    manager = ConnectionManager()
    client_ws = MockWS()
    await manager.register_client(client_ws)

    await manager.forward_message("stt", "client", {"type": "user_transcript", "text": "hi"})

    sent = json.loads(client_ws.messages[0])
    assert sent["type"] == "user_transcript"
    assert sent["text"] == "hi"


@pytest.mark.asyncio
async def test_forward_message_to_vad(mock_websocket):
    """Test forwarding message from client to VAD service."""
    manager = ConnectionManager()
    vad_ws = MockWS()
    await manager.register_vad(vad_ws, is_browser=False)

    await manager.forward_message("client", "vad", {"type": "audio_chunk"})

    sent = json.loads(vad_ws.messages[0])
    assert sent["type"] == "audio_chunk"


@pytest.mark.asyncio
async def test_forward_message_to_stt(mock_websocket):
    """Test forwarding message from client to STT service."""
    manager = ConnectionManager()
    stt_ws = MockWS()
    await manager.register_stt(stt_ws)

    await manager.forward_message("vad", "stt", {"type": "audio_chunk"})

    sent = json.loads(stt_ws.messages[0])
    assert sent["type"] == "audio_chunk"


@pytest.mark.asyncio
async def test_forward_message_to_llm(mock_websocket):
    """Test forwarding message from STT to LLM service."""
    manager = ConnectionManager()
    llm_ws = MockWS()
    await manager.register_llm(llm_ws)

    await manager.forward_message("stt", "llm", {"type": "user_input", "text": "hello"})

    sent = json.loads(llm_ws.messages[0])
    assert sent["type"] == "user_input"
    assert sent["text"] == "hello"


@pytest.mark.asyncio
async def test_forward_message_to_tts(mock_websocket):
    """Test forwarding message from LLM to TTS service."""
    manager = ConnectionManager()
    tts_ws = MockWS()
    await manager.register_tts(tts_ws)

    await manager.forward_message("llm", "tts", {"type": "tts_input", "text": "hello"})

    sent = json.loads(tts_ws.messages[0])
    assert sent["type"] == "tts_input"
    assert sent["text"] == "hello"


@pytest.mark.asyncio
async def test_service_status_defaults():
    """Test initial service status values."""
    manager = ConnectionManager()

    assert manager.service_status["vad"] == "offline"
    assert manager.service_status["stt"] == "offline"
    assert manager.service_status["llm"] == "offline"
    assert manager.service_status["tts"] == "offline"


@pytest.mark.asyncio
async def test_service_active_defaults():
    """Test initial service active values."""
    manager = ConnectionManager()

    assert manager.service_active["vad"] is False
    assert manager.service_active["stt"] is False
    assert manager.service_active["llm"] is False
    assert manager.service_active["tts"] is False


@pytest.mark.asyncio
async def test_conversation_history_defaults():
    """Test initial conversation history is empty."""
    manager = ConnectionManager()

    assert manager.conversation_history == []
    assert manager.max_history_messages == 20
    assert manager.use_full_history is True


@pytest.mark.asyncio
async def test_truncate_conversation_history_under_limit():
    """Test truncate does nothing when under limit."""
    manager = ConnectionManager()
    manager.conversation_history = [{"role": "user", "content": "hi"}]

    manager.truncate_conversation_history()
    assert len(manager.conversation_history) == 1


@pytest.mark.asyncio
async def test_truncate_conversation_history_over_limit():
    """Test truncate removes excess messages."""
    manager = ConnectionManager()
    manager.max_history_messages = 5

    for i in range(10):
        manager.conversation_history.append({"role": "user", "content": f"msg{i}"})

    manager.truncate_conversation_history()
    assert len(manager.conversation_history) == 5
    # Should keep the last 5
    for i, entry in enumerate(manager.conversation_history):
        assert entry["role"] == "user"
        assert entry["content"] == f"msg{i + 5}"
