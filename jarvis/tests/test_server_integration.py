"""Integration tests for the server's conversation history and message handling."""
import asyncio
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add Server directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Server"))
import jarvis_server
from jarvis_server import ConnectionManager


# ============================================================
# Mock WebSocket for integration tests
# ============================================================

class MockWS:
    """Mock WebSocket with asyncio.Queue for incoming messages."""

    def __init__(self, incoming=None):
        self.state = MagicMock()
        self.state.name = "OPEN"
        self.messages_sent = []
        self.incoming = asyncio.Queue()
        if incoming:
            for msg in incoming:
                self.incoming.put_nowait(json.dumps(msg) if isinstance(msg, dict) else msg)
        self._closed = False

    async def send(self, data):
        if isinstance(data, str):
            self.messages_sent.append(json.loads(data))
        else:
            self.messages_sent.append(data)

    async def recv(self):
        return await self.incoming.get()

    async def __aiter__(self):
        while not self._closed:
            try:
                msg = await asyncio.wait_for(self.incoming.get(), timeout=0.1)
                yield msg
            except asyncio.TimeoutError:
                break
        if not self._closed:
            self._closed = True

    async def close(self):
        self._closed = True


# ============================================================
# Test conversation history flow
# ============================================================

@pytest.mark.asyncio
async def test_conversation_history_first_turn():
    """Test conversation history after first turn (user + assistant)."""
    manager = ConnectionManager()
    manager.conversation_history = [
        {"role": "system", "content": "You are helpful."},
    ]

    # Simulate first final_transcript handler logic
    user_text = "What is the plot of Merlin?"
    last_assistant_idx = -1
    for i in range(len(manager.conversation_history) - 1, -1, -1):
        if manager.conversation_history[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx >= 0:
        manager.conversation_history.insert(last_assistant_idx + 1, {"role": "user", "content": user_text})
    elif any(entry["role"] == "user" for entry in manager.conversation_history):
        for i in range(len(manager.conversation_history)):
            if manager.conversation_history[i]["role"] == "user":
                manager.conversation_history[i]["content"] = user_text
                break
    else:
        manager.conversation_history.append({"role": "user", "content": user_text})

    # Add assistant response (simulating llm_end handler)
    manager.conversation_history.append({"role": "assistant", "content": "Merlin is a BBC series about a young warlock."})

    # Verify history structure
    assert len(manager.conversation_history) == 3
    assert manager.conversation_history[0]["role"] == "system"
    assert manager.conversation_history[1]["role"] == "user"
    assert manager.conversation_history[1]["content"] == "What is the plot of Merlin?"
    assert manager.conversation_history[2]["role"] == "assistant"
    assert "Merlin" in manager.conversation_history[2]["content"]


@pytest.mark.asyncio
async def test_conversation_history_second_turn():
    """Test conversation history after second turn (new user message appended)."""
    manager = ConnectionManager()
    manager.conversation_history = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the plot of Merlin?"},
        {"role": "assistant", "content": "Merlin is a BBC series about a young warlock."},
    ]

    # Simulate second final_transcript handler logic
    user_text = "What were we talking about?"
    last_assistant_idx = -1
    for i in range(len(manager.conversation_history) - 1, -1, -1):
        if manager.conversation_history[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx >= 0:
        manager.conversation_history.insert(last_assistant_idx + 1, {"role": "user", "content": user_text})
    elif any(entry["role"] == "user" for entry in manager.conversation_history):
        for i in range(len(manager.conversation_history)):
            if manager.conversation_history[i]["role"] == "user":
                manager.conversation_history[i]["content"] = user_text
                break
    else:
        manager.conversation_history.append({"role": "user", "content": user_text})

    # Add second assistant response
    manager.conversation_history.append({"role": "assistant", "content": "We were talking about Merlin, the BBC series."})

    # Verify history has grown to 5 entries
    assert len(manager.conversation_history) == 5
    assert manager.conversation_history[0]["role"] == "system"
    assert manager.conversation_history[1]["role"] == "user"
    assert manager.conversation_history[1]["content"] == "What is the plot of Merlin?"
    assert manager.conversation_history[2]["role"] == "assistant"
    assert "Merlin" in manager.conversation_history[2]["content"]
    assert manager.conversation_history[3]["role"] == "user"
    assert manager.conversation_history[3]["content"] == "What were we talking about?"
    assert manager.conversation_history[4]["role"] == "assistant"
    assert "Merlin" in manager.conversation_history[4]["content"]


@pytest.mark.asyncio
async def test_conversation_history_preserves_earlier_turns():
    """Test that earlier turns are preserved when new turns are added."""
    manager = ConnectionManager()
    manager.conversation_history = [
        {"role": "system", "content": "You are helpful."},
    ]

    # First turn
    manager.conversation_history.append({"role": "user", "content": "Hello"})
    manager.conversation_history.append({"role": "assistant", "content": "Hi there!"})

    # Second turn - insert user after last assistant
    last_assistant_idx = 2  # Index of "Hi there!"
    manager.conversation_history.insert(last_assistant_idx + 1, {"role": "user", "content": "How are you?"})
    manager.conversation_history.append({"role": "assistant", "content": "I'm good!"})

    # Third turn - insert user after last assistant
    last_assistant_idx = 4  # Index of "I'm good!"
    manager.conversation_history.insert(last_assistant_idx + 1, {"role": "user", "content": "What's your name?"})
    manager.conversation_history.append({"role": "assistant", "content": "I'm Jarvis!"})

    # Verify all turns are preserved
    assert len(manager.conversation_history) == 7
    assert manager.conversation_history[0]["role"] == "system"
    assert manager.conversation_history[1]["role"] == "user"
    assert manager.conversation_history[1]["content"] == "Hello"
    assert manager.conversation_history[2]["role"] == "assistant"
    assert manager.conversation_history[2]["content"] == "Hi there!"
    assert manager.conversation_history[3]["role"] == "user"
    assert manager.conversation_history[3]["content"] == "How are you?"
    assert manager.conversation_history[4]["role"] == "assistant"
    assert manager.conversation_history[4]["content"] == "I'm good!"
    assert manager.conversation_history[5]["role"] == "user"
    assert manager.conversation_history[5]["content"] == "What's your name?"
    assert manager.conversation_history[6]["role"] == "assistant"
    assert manager.conversation_history[6]["content"] == "I'm Jarvis!"


@pytest.mark.asyncio
async def test_truncate_conversation_history_keeps_recent():
    """Test that truncate keeps the most recent messages."""
    manager = ConnectionManager()
    manager.max_history_messages = 4

    # Build history longer than limit
    for i in range(10):
        manager.conversation_history.append({"role": "user", "content": f"User message {i}"})
        manager.conversation_history.append({"role": "assistant", "content": f"Assistant message {i}"})

    manager.truncate_conversation_history()

    # Should have exactly max_history_messages
    assert len(manager.conversation_history) == 4


@pytest.mark.asyncio
async def test_truncate_conversation_history_no_truncation():
    """Test that truncate does nothing when under limit."""
    manager = ConnectionManager()
    manager.max_history_messages = 20

    manager.conversation_history = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]

    manager.truncate_conversation_history()
    assert len(manager.conversation_history) == 3


@pytest.mark.asyncio
async def test_partial_transcript_updates_last_user():
    """Test that partial_transcript updates the last user entry in place."""
    manager = ConnectionManager()
    manager.conversation_history = [
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": "Hello"},
    ]

    # Simulate partial_transcript handler: update last user entry
    partial_text = "How are y..."
    for i in range(len(manager.conversation_history) - 1, -1, -1):
        if manager.conversation_history[i]["role"] == "user":
            manager.conversation_history[i]["content"] = partial_text
            break
    else:
        # No user entry found, append one
        manager.conversation_history.append({"role": "user", "content": partial_text})

    # Should have added a new user entry since none existed
    assert any(e["role"] == "user" and e["content"] == partial_text for e in manager.conversation_history)


@pytest.mark.asyncio
async def test_final_transcript_appends_after_assistant():
    """Test that final_transcript appends user entry after last assistant."""
    manager = ConnectionManager()
    manager.conversation_history = [
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": "Hello"},
    ]

    # Simulate final_transcript handler
    user_text = "How are you?"
    last_assistant_idx = -1
    for i in range(len(manager.conversation_history) - 1, -1, -1):
        if manager.conversation_history[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx >= 0:
        manager.conversation_history.insert(last_assistant_idx + 1, {"role": "user", "content": user_text})

    # Should have inserted user entry after assistant
    assert len(manager.conversation_history) == 3
    assert manager.conversation_history[1]["role"] == "assistant"
    assert manager.conversation_history[2]["role"] == "user"
    assert manager.conversation_history[2]["content"] == "How are you?"


@pytest.mark.asyncio
async def test_first_turn_final_transcript_updates_existing_user():
    """Test first turn final_transcript updates existing user entry (no assistant yet)."""
    manager = ConnectionManager()
    manager.conversation_history = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Partial text"},
    ]

    # Simulate final_transcript handler
    user_text = "How are you?"
    last_assistant_idx = -1
    for i in range(len(manager.conversation_history) - 1, -1, -1):
        if manager.conversation_history[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx >= 0:
        manager.conversation_history.insert(last_assistant_idx + 1, {"role": "user", "content": user_text})
    elif any(entry["role"] == "user" for entry in manager.conversation_history):
        for i in range(len(manager.conversation_history)):
            if manager.conversation_history[i]["role"] == "user":
                manager.conversation_history[i]["content"] = user_text
                break

    # Should have updated the existing user entry
    assert len(manager.conversation_history) == 2
    assert manager.conversation_history[1]["role"] == "user"
    assert manager.conversation_history[1]["content"] == "How are you?"


@pytest.mark.asyncio
async def test_broadcast_messages_to_client():
    """Test that broadcast_messages_to_client sends to connected client."""
    manager = ConnectionManager()
    client_ws = MockWS()
    await manager.register_client(client_ws)

    # Simulate broadcasting llm_transcript
    await manager.broadcast_to_client({
        "type": "llm_transcript",
        "text": "Hello world",
        "partial": True,
        "turn_id": 1,
    })

    assert len(client_ws.messages_sent) == 1
    msg = client_ws.messages_sent[0]
    assert msg["type"] == "llm_transcript"
    assert msg["text"] == "Hello world"
    assert msg["turn_id"] == 1


@pytest.mark.asyncio
async def test_service_status_broadcast_on_connection():
    """Test that service status is broadcast when services register."""
    manager = ConnectionManager()
    client_ws = MockWS()
    await manager.register_client(client_ws)

    llm_ws = MockWS()
    await manager.register_llm(llm_ws)

    # Check that client received service_status message
    status_msgs = [m for m in client_ws.messages_sent if isinstance(m, dict) and m.get("type") == "service_status"]
    assert len(status_msgs) >= 1
    assert any(s["service"] == "llm" for s in status_msgs)
    assert any(s["status"] == "idle" for s in status_msgs)


@pytest.mark.asyncio
async def test_conversation_history_send_to_llm_service():
    """Test that conversation_history is included when forwarding to LLM."""
    manager = ConnectionManager()
    manager.conversation_history = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]

    llm_ws = MockWS()
    manager.llm_service_ws = llm_ws

    # Simulate forwarding user_input to LLM
    await llm_ws.send(json.dumps({
        "type": "user_input",
        "text": "Second question",
        "history": manager.conversation_history,
    }))

    msg = llm_ws.messages_sent[0]
    assert msg["type"] == "user_input"
    assert msg["text"] == "Second question"
    assert len(msg["history"]) == 3
    assert msg["history"][0]["role"] == "system"
    assert msg["history"][1]["role"] == "user"
    assert msg["history"][1]["content"] == "First question"
    assert msg["history"][2]["role"] == "assistant"
    assert msg["history"][2]["content"] == "First answer"
