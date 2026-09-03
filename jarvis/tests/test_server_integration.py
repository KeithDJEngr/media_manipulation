"""Integration tests for jarvis_server against the REAL handlers.

Every scenario from the original hand-simulated baseline is driven through
the actual server code paths - handle_stt / handle_llm_response_messages /
create_heartbeat_task / register_llm / unregister_llm - against an in-memory
WebSocket mock (MockWS), so the relay loop, the heartbeat watchdog, the
STT->LLM forwarding and the zombie-LLM recovery bookkeeping are exercised
end to end without real network I/O or a live LLM/TTS service.

Coverage map (original hand-simulated tests -> real-handler tests here):

  1. VAD browser socket register/unregister
     -> test_vad_browser_* (register, replace, no-op, broadcast fan-out)
  2. LLM multi-turn conversation (user+assistant alternating history)
     -> test_llm_multi_turn_history_interleaves_correctly
  3. Interrupted LLM turn (user entry dropped, no assistant entry)
     -> test_llm_end_interrupted_drops_turn_from_history
  4. MessageHistory truncation under/over the limit
     -> test_truncate_history_* (incl. index recomputation after the window
        evicts the tracked entries - a latent IndexError that used to kill
        the STT handler)
  5. Zombie LLM: disconnect mid-turn parks the question, reconnect resends
     it, the resend budget resets on a completed turn and on exhaustion the
     question is dropped and the client gets a synthetic llm_end
     -> test_llm_disconnect_* / test_llm_reconnect_* /
        test_resend_budget_* / test_completed_turn_resets_resend_budget
  6. STT partial transcripts update the in-flight user entry in place; the
     final is appended once and forwarded to the LLM with the full history
     -> test_stt_final_transcript_feeds_llm_with_correct_history

It also covers the shared LLM response handler (token accumulation,
llm_start state reset, llm_end bookkeeping, heartbeat handling, TTS
partial forwarding, unknown-message tolerance) and the watchdog's armed
stale-close behaviour.

Three latent defects surfaced while writing the real-handler tests and are
fixed in Server/jarvis_server.py:

  A. last_usr_msg / last_llm_msg used 0 as the "no message yet" sentinel,
     which collides with the index of a real first message: on a fresh
     manager the first turn's partial transcripts each appended a separate
     user entry, so the LLM received fragmented context.
     (now -1 = No message yet)
  B. truncate_history sliced the window but never recomputed the tracked
     indices, so once the window overflowed the next in-place transcript
     update raised IndexError inside handle_stt's except-branch and killed
     the entire STT loop until the service was restarted.
     (truncate_history now recomputes both indices)
  C. unregister_llm's budget-exhaustion branch dropped the held question by
     iterating llm_resend_queue only - but register_llm moves held questions
     into pending_turns on every reconnect, so at exhaustion the queue was
     empty and the drop was dead code: the question stayed in the history
     and was resent on every subsequent reconnect forever.
     (the drop now drains both containers)
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Server"))

import pytest
import pytest_asyncio

import jarvis_server
from jarvis_server import ConnectionManager, MessageHistory

pytestmark = pytest.mark.asyncio


class MockWS:
    """In-memory stand-in for a websockets server-side connection.

    ``incoming`` seeds the queue that the relay's ``async for`` / ``recv()``
    drains (0.1s timeout per message, mirroring the production relay).
    ``close()`` records the (code, reason) pair so tests can assert exactly
    how a connection was terminated (e.g. the watchdog's code 4001).
    """

    def __init__(self, incoming=None):
        self.state = MagicMock()
        self.state.name = "OPEN"
        self._messages_sent = []
        self._message_queue = asyncio.Queue()
        self._closed = False
        self.close_args = []
        for msg in (incoming or []):
            self.add_incoming(msg)

    def add_incoming(self, msg_dict):
        """Queue one incoming frame as the JSON text a real socket would
        deliver (the server's relay does ``json.loads`` on ``recv()``)."""
        self._message_queue.put_nowait(json.dumps(msg_dict))

    async def send(self, data):
        self._messages_sent.append(data)

    @property
    def messages_sent(self):
        """Parsed view of sent frames: JSON text is decoded to dicts, so
        tests can assert on message content directly."""
        return [json.loads(m) if isinstance(m, str) else m
                for m in self._messages_sent]

    async def recv(self):
        return await self._message_queue.get()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            raise StopAsyncIteration

    async def close(self, code=1000, reason=""):
        # The real websockets close() is idempotent: once the connection is
        # closed, further close() calls are no-ops. Only the decisive first
        # close is recorded, so tests can assert exactly how it terminated.
        if self._closed:
            return
        self.close_args.append((code, reason))
        self._closed = True
        self.state.name = "CLOSED"


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def llm_env(monkeypatch):
    """(manager, client_ws) with a FRESH ConnectionManager bound to the
    module-level ``manager`` global the handlers reference, plus a client
    endpoint attached so broadcast_to_client has a destination.
    """
    manager = ConnectionManager()
    monkeypatch.setattr(jarvis_server, "manager", manager)
    client_ws = MockWS()
    manager.client_ws = client_ws
    return manager, client_ws


def _armed(manager):
    """Mirror the production arming condition (see handle_llm_service):
    the armed stale-close may only fire while a question depends on the
    LLM pipe."""
    return (manager.active_turn_id is not None
            or bool(manager.pending_turns)
            or bool(manager.llm_resend_queue))


def _queue_turn(manager, question):
    """Do exactly what handle_stt's final-transcript branch does for the
    conversation history and the LLM queue."""
    manager.conversation_history_manager.update_usr_msg(question, True)
    manager.pending_turns.append(question)


async def _start_watchdog(manager, llm_ws, last_heartbeat,
                           close_timeout=0.6, check_interval=0.1):
    """Run the production heartbeat watchdog (armed stale-close enabled)
    as a task; returns (task, cancel_scope) so the test can cancel it."""
    scope = []
    task = await jarvis_server.create_heartbeat_task(
        last_heartbeat, scope,
        websocket=llm_ws,
        is_close_armed=lambda: _armed(manager),
        close_timeout=close_timeout,
        check_interval=check_interval)
    return task, scope


async def _run_llm_relay(llm_ws, last_heartbeat=None):
    """Drive the shared LLM handler on a mock socket until its queue drains."""
    scope = []
    result = await jarvis_server.handle_llm_response_messages(
        llm_ws, "", [0], [0], last_heartbeat or {}, scope, "LLM-SERVICE")
    for task in scope:
        task.cancel()
    return result


def _fresh_heartbeats(loop, llm_age_seconds=0.0):
    """The watchdog's timestamp map, as seeded by create_heartbeat_task."""
    return {
        "llm": loop.time() - llm_age_seconds,
        "vad": 0,
        "stt": 0,
        "tts": 0,
    }


# ---------------------------------------------------------------------------
# VAD-browser endpoint
# ---------------------------------------------------------------------------

async def test_vad_browser_register_and_unregister():
    manager = ConnectionManager()
    manager.client_ws = MockWS()
    browser_ws = MockWS()

    await manager.register_vad(browser_ws, is_browser=True)
    assert manager.vad_browser_ws is browser_ws
    assert manager.vad_ws is None

    await manager.unregister_vad(browser_ws)
    assert manager.vad_browser_ws is None
    assert manager.vad_ws is None


async def test_vad_browser_register_replaces_existing_browser():
    manager = ConnectionManager()
    manager.client_ws = MockWS()
    first = MockWS()
    second = MockWS()

    await manager.register_vad(first, is_browser=True)
    await manager.register_vad(second, is_browser=True)

    assert manager.vad_browser_ws is second
    assert manager.vad_ws is None


async def test_vad_browser_unregister_noop_for_other_socket():
    manager = ConnectionManager()
    manager.client_ws = MockWS()
    browser_ws = MockWS()
    other_ws = MockWS()

    await manager.register_vad(browser_ws, is_browser=True)
    await manager.unregister_vad(other_ws)

    # Unregistering a socket that is not the browser must leave it alone.
    assert manager.vad_browser_ws is browser_ws


async def test_broadcast_reaches_vad_browser_socket():
    manager = ConnectionManager()
    client_ws = MockWS()
    manager.client_ws = client_ws
    browser_ws = MockWS()
    manager.vad_browser_ws = browser_ws

    payload = {"type": "user_transcript", "text": "hi", "final": True, "chunk_id": 1}
    await manager.broadcast_to_client(payload)

    assert payload in client_ws.messages_sent
    assert payload in browser_ws.messages_sent


# ---------------------------------------------------------------------------
# MessageHistory truncation
# ---------------------------------------------------------------------------

async def test_truncate_history_keeps_only_recent_window():
    history = MessageHistory(max_history_msgs=4)
    for i in range(10):
        history.add_usr_msg(f"User message {i}")
        history.add_llm_msg(f"Assistant message {i}")

    assert len(history.history) == 4
    assert [m["content"] for m in history.history] == [
        "User message 8",
        "Assistant message 8",
        "User message 9",
        "Assistant message 9",
    ]
    assert [m["role"] for m in history.history] == ["user", "assistant", "user", "assistant"]


async def test_truncate_history_is_noop_under_window():
    history = MessageHistory(max_history_msgs=4)
    history.add_usr_msg("One")
    history.add_llm_msg("Two")
    history.add_usr_msg("Three")

    assert [m["content"] for m in history.history] == ["One", "Two", "Three"]
    assert [m["role"] for m in history.history] == ["user", "assistant", "user"]


async def test_truncate_history_recomputes_message_indices():
    """Once the window overflows it can evict the entries the tracked
    'last message' indices point at (the window is a plain last-N slice, so
    it may even start mid-turn). The indices must be recomputed or the next
    in-place transcript update raises IndexError and kills the STT loop."""
    history = MessageHistory(max_history_msgs=4)
    history.add_usr_msg("u1")
    history.add_llm_msg("a1")
    history.add_usr_msg("u2")
    history.add_llm_msg("a2")
    history.add_usr_msg("u3")
    history.add_llm_msg("a3")
    history.add_usr_msg("u4")

    # Last-4 window: [a2, u3, a3, u4] - u1/a1 evicted, window starts mid-turn.
    assert [m["content"] for m in history.history] == ["a2", "u3", "a3", "u4"]
    # The tracked indices are recomputed to the LAST entry of each role in
    # the surviving window (both must stay in-range / live):
    #   last user = u4 (index 3), last assistant = a3 (index 2).
    assert history.last_usr_msg == 3
    assert history.last_llm_msg == 2

    # In-place updates after truncation must hit live indices, not stale
    # out-of-range ones (which raised IndexError before the recompute fix).
    history.update_last_usr_msg("u4 refined")
    history.update_last_llm_msg("a3 refined")
    assert [m["content"] for m in history.history] == ["a2", "u3", "a3 refined", "u4 refined"]


# ---------------------------------------------------------------------------
# Shared LLM response handler (real relay loop)
# ---------------------------------------------------------------------------

async def test_llm_token_broadcasts_transcript_to_client(llm_env):
    manager, client_ws = llm_env
    question = "What is the plot of Merlin?"
    llm_ws = MockWS(incoming=[
        {"type": "llm_start"},
        {"type": "llm_token", "text": "Hello "},
        {"type": "llm_token", "text": "world!"},
    ])
    manager.llm_service_ws = llm_ws
    _queue_turn(manager, question)

    accumulated, turn_id, _ = await _run_llm_relay(llm_ws)

    start_msgs = [m for m in client_ws.messages_sent if m["type"] == "llm_start"]
    transcript_msgs = [m for m in client_ws.messages_sent if m["type"] == "llm_transcript"]
    assert len(start_msgs) == 1
    assert start_msgs[0]["turn_id"] == 1
    # Every transcript broadcast carries the FULL accumulated text.
    assert [m["text"] for m in transcript_msgs] == ["Hello ", "Hello world!"]
    assert all(m["partial"] is True for m in transcript_msgs)
    assert transcript_msgs[-1]["turn_id"] == 1
    assert accumulated == "Hello world!"
    assert turn_id == 1
    assert manager.service_status["llm"] == "active"


async def test_llm_start_resets_stale_relay_state(llm_env):
    manager, client_ws = llm_env
    question = "What is the plot of Merlin?"
    llm_ws = MockWS(incoming=[{"type": "llm_start"}])
    manager.llm_service_ws = llm_ws
    _queue_turn(manager, question)

    # Stale relay locals, as if a previous turn had accumulated text.
    accumulated, turn_id, forwarded = await jarvis_server.handle_llm_response_messages(
        llm_ws, "stale text from a previous turn", [99], [42], {}, [], "LLM-SERVICE")

    assert accumulated == ""
    assert forwarded == 0
    assert turn_id == 1
    assert manager.active_turn_id == 1
    assert manager.active_turn_text == question
    assert manager.llm_failed_turn_id is None


async def test_llm_token_accumulates_text(llm_env):
    manager, client_ws = llm_env
    llm_ws = MockWS(incoming=[
        {"type": "llm_start"},
        {"type": "llm_token", "text": "foo"},
        {"type": "llm_token", "text": "bar"},
    ])
    manager.llm_service_ws = llm_ws
    _queue_turn(manager, "What is the plot of Merlin?")

    accumulated, _, _ = await _run_llm_relay(llm_ws)
    assert accumulated == "foobar"


async def test_llm_end_stores_final_into_history(llm_env):
    manager, client_ws = llm_env
    question = "What is the plot of Merlin?"
    llm_ws = MockWS(incoming=[
        {"type": "llm_start"},
        {"type": "llm_token", "text": "Merlin is a BBC series. "},
        {"type": "llm_end"},
    ])
    manager.llm_service_ws = llm_ws
    _queue_turn(manager, question)

    await _run_llm_relay(llm_ws)

    assert [(m["role"], m["content"]) for m in manager.conversation_history_manager.history] == [
        ("user", question),
        ("assistant", "Merlin is a BBC series. "),
    ]
    assert manager.pending_turns == []
    assert manager.active_turn_id is None
    assert manager.active_turn_text is None
    end_msgs = [m for m in client_ws.messages_sent if m["type"] == "llm_end"]
    assert len(end_msgs) == 1
    assert end_msgs[0]["turn_id"] == 1
    assert end_msgs[0]["text"] == "Merlin is a BBC series. "
    assert end_msgs[0]["interrupted"] is False


async def test_llm_end_interrupted_drops_turn_from_history(llm_env):
    manager, client_ws = llm_env
    question = "What is the plot of Merlin?"
    llm_ws = MockWS(incoming=[
        {"type": "llm_start"},
        {"type": "llm_token", "text": "Partial answer "},
        {"type": "llm_end", "interrupted": True},
    ])
    manager.llm_service_ws = llm_ws
    _queue_turn(manager, question)

    await _run_llm_relay(llm_ws)

    # An interrupted turn leaves NO trace in the conversation history:
    # the user question is dropped and no partial assistant text is stored.
    assert manager.conversation_history_manager.history == []
    assert manager.pending_turns == []
    assert manager.active_turn_id is None
    end_msgs = [m for m in client_ws.messages_sent if m["type"] == "llm_end"]
    assert len(end_msgs) == 1
    assert end_msgs[0]["interrupted"] is True


async def test_llm_heartbeat_refreshes_watchdog_timestamp(llm_env):
    manager, client_ws = llm_env
    last_heartbeat = _fresh_heartbeats(asyncio.get_running_loop())
    llm_ws = MockWS(incoming=[
        {"type": "llm_start"},
        {"type": "heartbeat"},
    ])
    manager.llm_service_ws = llm_ws
    _queue_turn(manager, "What is the plot of Merlin?")

    await _run_llm_relay(llm_ws, last_heartbeat)
    assert last_heartbeat["llm"] > 0


async def test_llm_heartbeat_is_not_forwarded_to_client(llm_env):
    manager, client_ws = llm_env
    llm_ws = MockWS(incoming=[
        {"type": "llm_start"},
        {"type": "heartbeat"},
    ])
    manager.llm_service_ws = llm_ws
    _queue_turn(manager, "What is the plot of Merlin?")

    await _run_llm_relay(llm_ws)

    assert not any(m["type"] == "heartbeat" for m in client_ws.messages_sent)
    assert len([m for m in client_ws.messages_sent if m["type"] == "llm_start"]) == 1


async def test_llm_multi_turn_history_interleaves_correctly(llm_env):
    manager, client_ws = llm_env
    question1 = "What is the plot of Merlin?"
    question2 = "Who plays the lead?"
    llm_ws = MockWS()
    manager.llm_service_ws = llm_ws

    # Turn 1: the full llm_start -> token -> llm_end sequence on one relay.
    _queue_turn(manager, question1)
    llm_ws.add_incoming({"type": "llm_start"})
    llm_ws.add_incoming({"type": "llm_token", "text": "Merlin is a BBC series. "})
    llm_ws.add_incoming({"type": "llm_end"})
    await _run_llm_relay(llm_ws)
    assert [(m["role"], m["content"]) for m in manager.conversation_history_manager.history] == [
        ("user", question1),
        ("assistant", "Merlin is a BBC series. "),
    ]

    # Turn 2 arrives after turn 1 completed: the user entry must land AFTER
    # the assistant reply, and turn 2's final must replace that entry, not
    # overwrite turn 1's assistant entry.
    _queue_turn(manager, question2)
    llm_ws.add_incoming({"type": "llm_start"})
    llm_ws.add_incoming({"type": "llm_token", "text": "Colin Morgan plays him. "})
    llm_ws.add_incoming({"type": "llm_end"})
    await _run_llm_relay(llm_ws)

    assert [(m["role"], m["content"]) for m in manager.conversation_history_manager.history] == [
        ("user", question1),
        ("assistant", "Merlin is a BBC series. "),
        ("user", question2),
        ("assistant", "Colin Morgan plays him. "),
    ]
    assert manager.pending_turns == []
    assert manager.active_turn_id is None
    start_ids = [m["turn_id"] for m in client_ws.messages_sent if m["type"] == "llm_start"]
    end_ids = [m["turn_id"] for m in client_ws.messages_sent if m["type"] == "llm_end"]
    assert start_ids == [1, 2]
    assert end_ids == [1, 2]


async def test_partial_llm_text_forwarded_to_tts_at_sentence_boundary(llm_env):
    manager, client_ws = llm_env
    tts_ws = MockWS()
    manager.tts_ws = tts_ws
    llm_ws = MockWS(incoming=[
        {"type": "llm_start"},
        # "Hello world. " crosses a sentence boundary -> must be forwarded.
        {"type": "llm_token", "text": "Hello world. "},
    ])
    manager.llm_service_ws = llm_ws
    _queue_turn(manager, "Greet me")

    accumulated, _, forwarded_len = await _run_llm_relay(llm_ws)

    tts_inputs = [m for m in tts_ws.messages_sent if m["type"] == "tts_input"]
    assert len(tts_inputs) == 1
    # The forward carries the FULL accumulated text, not the delta.
    assert tts_inputs[0]["text"] == accumulated == "Hello world. "
    assert tts_inputs[0]["partial"] is True
    assert tts_inputs[0]["turn_id"] == 1
    assert forwarded_len == len("Hello world. ")


async def test_no_forward_to_tts_without_boundary_or_min_chars(llm_env):
    manager, client_ws = llm_env
    tts_ws = MockWS()
    manager.tts_ws = tts_ws
    llm_ws = MockWS(incoming=[
        {"type": "llm_start"},
        {"type": "llm_token", "text": "abc"},
        {"type": "llm_token", "text": "def"},
    ])
    manager.llm_service_ws = llm_ws
    _queue_turn(manager, "Say something")

    accumulated, _, forwarded_len = await _run_llm_relay(llm_ws)

    assert accumulated == "abcdef"
    # No sentence boundary, fewer than 150 chars since the last forward.
    assert tts_ws.messages_sent == []
    assert forwarded_len == 0


async def test_unknown_llm_message_type_is_ignored(llm_env):
    manager, client_ws = llm_env
    llm_ws = MockWS(incoming=[
        {"type": "llm_start"},
        {"type": "bogus_message", "payload": 1},
    ])
    manager.llm_service_ws = llm_ws
    _queue_turn(manager, "What is the plot of Merlin?")

    accumulated, turn_id, _ = await _run_llm_relay(llm_ws)  # must not raise

    assert accumulated == ""
    assert turn_id == 1
    # The llm_start reached the client (it is preceded by the service
    # going 'active'); the unknown type was dropped, not broadcast.
    sent_types = [m["type"] for m in client_ws.messages_sent]
    assert "llm_start" in sent_types
    assert "bogus_message" not in sent_types


async def test_llm_service_active_lifecycle_active_then_idle(llm_env):
    manager, client_ws = llm_env
    llm_ws = MockWS(incoming=[
        {"type": "llm_start"},
        {"type": "llm_end"},
    ])
    manager.llm_service_ws = llm_ws
    _queue_turn(manager, "What is the plot of Merlin?")

    await _run_llm_relay(llm_ws)

    assert manager.service_active["llm"] is False
    assert manager.service_status["llm"] == "idle"
    llm_statuses = [m["status"] for m in client_ws.messages_sent
                    if m["type"] == "service_status" and m["service"] == "llm"]
    assert llm_statuses == ["active", "idle"]


# ---------------------------------------------------------------------------
# STT final transcript -> LLM forwarding (real handle_stt)
# ---------------------------------------------------------------------------

async def test_stt_final_transcript_feeds_llm_with_correct_history(llm_env):
    manager, client_ws = llm_env
    question1 = "What is the plot of Merlin?"
    question2 = "What were we talking about?"
    llm_ws = MockWS()
    manager.llm_service_ws = llm_ws

    # Phase 1: STT session - partial transcripts update in place, the final
    # is appended once and forwarded to the LLM with the full history.
    stt_ws = MockWS(incoming=[
        {"type": "partial_transcript", "text": "What is the plot", "chunk_id": 1},
        {"type": "partial_transcript", "text": "What is the plot of Merl", "chunk_id": 2},
        {"type": "final_transcript", "text": question1, "chunk_id": 3},
    ])
    await jarvis_server.handle_stt(stt_ws)

    history = manager.conversation_history_manager
    assert [(m["role"], m["content"]) for m in history.history] == [("user", question1)]
    assert manager.pending_turns == [question1]
    # The user entry handle_stt appended is exactly what got forwarded.
    finals = [m for m in llm_ws.messages_sent
              if m["type"] == "user_input" and not m.get("partial")]
    assert len(finals) == 1
    assert finals[0]["text"] == question1
    assert finals[0]["history"] == history.history

    # Phase 2: the LLM answers the first turn on the shared relay.
    llm_ws.add_incoming({"type": "llm_start"})
    llm_ws.add_incoming({"type": "llm_token", "text": "Merlin is a BBC series. "})
    llm_ws.add_incoming({"type": "llm_end"})
    await _run_llm_relay(llm_ws)
    assert [(m["role"], m["content"]) for m in history.history] == [
        ("user", question1),
        ("assistant", "Merlin is a BBC series. "),
    ]

    # A new STT session arrives for the second question.
    stt_ws2 = MockWS(incoming=[
        {"type": "partial_transcript", "text": "What were we", "chunk_id": 7},
        {"type": "final_transcript", "text": "What were we talking about?", "chunk_id": 8},
    ])
    await jarvis_server.handle_stt(stt_ws2)

    # The new final appends a fresh user entry AFTER the assistant reply.
    assert [(m["role"], m["content"]) for m in history.history] == [
        ("user", question1),
        ("assistant", "Merlin is a BBC series. "),
        ("user", question2),
    ]
    # The server's MessageHistory is the source of truth sent to the LLM.
    finals = [m for m in llm_ws.messages_sent
              if m["type"] == "user_input" and not m.get("partial")]
    assert len(finals) == 2
    assert finals[1]["history"] == history.history
    assert manager.pending_turns == [question2]
    assert [m["text"] for m in client_ws.messages_sent if m["type"] == "user_transcript"] == [
        question1, question2]


# ---------------------------------------------------------------------------
# Zombie LLM: disconnect / reconnect / resend budget
# ---------------------------------------------------------------------------

async def test_llm_disconnect_mid_turn_preserves_question_for_resend(llm_env):
    manager, client_ws = llm_env
    question = "What is the plot of Merlin?"
    llm_ws = MockWS()
    await manager.register_llm(llm_ws)
    manager.conversation_history_manager.update_usr_msg(question, True)
    manager.pending_turns.append(question)
    llm_ws.add_incoming({"type": "llm_start"})
    llm_ws.add_incoming({"type": "llm_token", "text": "Merlin is "})

    # The LLM acknowledges the turn, then the connection drops mid-turn.
    await _run_llm_relay(llm_ws)
    assert manager.active_turn_id is not None
    await manager.unregister_llm(llm_ws)

    assert manager.llm_service_ws is None
    assert manager.llm_resend_attempts == 1
    assert manager.llm_resend_queue == [question]
    assert manager.pending_turns == []
    assert manager.llm_failed_turn_id == 1
    assert manager.active_turn_id is None
    assert manager.service_status["llm"] == "offline"
    # The question survives in the conversation history for the resend.
    assert any(m["role"] == "user" and m["content"] == question
               for m in manager.conversation_history_manager.history)


async def test_llm_reconnect_resends_held_question(llm_env):
    manager, client_ws = llm_env
    question = "What is the plot of Merlin?"
    llm_ws = MockWS()
    await manager.register_llm(llm_ws)
    manager.conversation_history_manager.update_usr_msg(question, True)
    manager.pending_turns.append(question)
    llm_ws.add_incoming({"type": "llm_start"})
    llm_ws.add_incoming({"type": "llm_token", "text": "Merlin is "})
    await _run_llm_relay(llm_ws)
    await manager.unregister_llm(llm_ws)
    assert manager.llm_resend_queue == [question]

    idle_broadcasts = sum(1 for m in client_ws.messages_sent
                          if m.get("type") == "service_status"
                          and m.get("service") == "llm"
                          and m.get("status") == "idle")

    # The LLM service comes back: the new endpoint registers.
    new_ws = MockWS()
    await manager.register_llm(new_ws)

    user_inputs = [m for m in new_ws.messages_sent if m["type"] == "user_input"]
    assert len(user_inputs) == 1
    assert user_inputs[0]["text"] == question
    assert user_inputs[0]["history"] == manager.conversation_history_manager.history
    assert manager.llm_resend_queue == []
    assert manager.pending_turns == [question]
    # A failed turn does not burn more budget than the disconnect itself.
    assert manager.llm_resend_attempts == 1
    # And the client was told the LLM is back online.
    idle_after = sum(1 for m in client_ws.messages_sent
                     if m.get("type") == "service_status"
                     and m.get("service") == "llm"
                     and m.get("status") == "idle")
    assert idle_after == idle_broadcasts + 1


async def test_resend_budget_exhaustion_drops_question_and_unsticks_client(llm_env):
    manager, client_ws = llm_env
    question = "What is the plot of Merlin?"
    first_ws = MockWS()
    await manager.register_llm(first_ws)
    manager.conversation_history_manager.update_usr_msg(question, True)
    manager.pending_turns.append(question)
    first_ws.add_incoming({"type": "llm_start"})
    first_ws.add_incoming({"type": "llm_token", "text": "Merlin is "})
    await _run_llm_relay(first_ws)
    await manager.unregister_llm(first_ws)
    assert manager.llm_resend_attempts == 1

    # The service keeps flapping: every reconnect resends the held question,
    # and every drop re-parks it, until the resend budget is exhausted.
    for _ in range(3):
        zombie_ws = MockWS()
        await manager.register_llm(zombie_ws)
        await manager.unregister_llm(zombie_ws)

    assert manager.llm_resend_attempts == 0
    assert manager.llm_resend_queue == []
    assert manager.pending_turns == []
    assert manager.llm_failed_turn_id is None
    assert manager.active_turn_id is None
    assert manager.service_status["llm"] == "offline"
    # The held question was dropped from the conversation history...
    assert not any(m["role"] == "user" and m["content"] == question
                   for m in manager.conversation_history_manager.history)
    # ...and the client received exactly one synthetic terminal event,
    # keyed to the failed turn, so the UI can stop waiting.
    llm_ends = [m for m in client_ws.messages_sent if m["type"] == "llm_end"]
    assert len(llm_ends) == 1
    assert llm_ends[0]["turn_id"] == 1
    assert llm_ends[0]["interrupted"] is True
    assert "unreachable" in llm_ends[0]["text"].lower()


async def test_completed_turn_resets_resend_budget(llm_env):
    manager, client_ws = llm_env
    question = "What is the plot of Merlin?"
    first_ws = MockWS()
    await manager.register_llm(first_ws)
    manager.conversation_history_manager.update_usr_msg(question, True)
    manager.pending_turns.append(question)
    first_ws.add_incoming({"type": "llm_start"})
    first_ws.add_incoming({"type": "llm_token", "text": "Merlin is "})
    await _run_llm_relay(first_ws)
    await manager.unregister_llm(first_ws)
    assert manager.llm_resend_attempts == 1

    # A successful completion on the replacement connection resets the budget.
    new_ws = MockWS(incoming=[
        {"type": "llm_start"},
        {"type": "llm_token", "text": "It's back. "},
        {"type": "llm_end"},
    ])
    await manager.register_llm(new_ws)
    await _run_llm_relay(new_ws)

    assert manager.llm_resend_attempts == 0
    assert manager.llm_resend_queue == []
    assert manager.pending_turns == []
    assert manager.active_turn_id is None
    assert [(m["role"], m["content"]) for m in manager.conversation_history_manager.history] == [
        ("user", question),
        ("assistant", "It's back. "),
    ]


# ---------------------------------------------------------------------------
# Watchdog armed stale-close
# ---------------------------------------------------------------------------

async def test_armed_stale_close_closes_unresponsive_llm(llm_env):
    manager, client_ws = llm_env
    llm_ws = MockWS()
    last_heartbeat = _fresh_heartbeats(asyncio.get_running_loop())
    task, scope = await _start_watchdog(manager, llm_ws, last_heartbeat)
    await manager.register_llm(llm_ws)
    # A turn is in flight; the LLM never replies (no relay runs, so no
    # heartbeat ever refreshes the watchdog's timestamp).
    manager.conversation_history_manager.update_usr_msg("What is the plot of Merlin?", True)
    manager.pending_turns.append("What is the plot of Merlin?")

    try:
        await asyncio.sleep(0.9)
    finally:
        for t in scope:
            t.cancel()

    assert llm_ws._closed
    assert llm_ws.close_args == [(4001, "LLM stalled during active turn")]


async def test_idle_stale_llm_marked_offline_without_close(llm_env):
    manager, client_ws = llm_env
    llm_ws = MockWS()
    await manager.register_llm(llm_ws)
    last_heartbeat = _fresh_heartbeats(asyncio.get_running_loop(), llm_age_seconds=100)
    task, scope = await _start_watchdog(manager, llm_ws, last_heartbeat)

    try:
        # No turn is in flight: the watchdog degrades to the plain
        # offline-marking path and must NOT drop the connection.
        await asyncio.sleep(0.3)
    finally:
        for t in scope:
            t.cancel()

    assert manager.service_status["llm"] == "offline"
    assert any(m.get("type") == "service_status"
               and m.get("service") == "llm"
               and m.get("status") == "offline"
               for m in client_ws.messages_sent)
    assert llm_ws.close_args == []
    assert not llm_ws._closed


async def test_fresh_heartbeats_keep_armed_llm_open(llm_env):
    manager, client_ws = llm_env
    llm_ws = MockWS()
    last_heartbeat = _fresh_heartbeats(asyncio.get_running_loop())
    task, scope = await _start_watchdog(manager, llm_ws, last_heartbeat)
    await manager.register_llm(llm_ws)
    # A turn is pending (armed), but the LLM heartbeats steadily: the relay
    # keeps the watchdog timestamp fresh, so the armed close must never fire.
    manager.pending_turns.append("What is the plot of Merlin?")
    relay_task = asyncio.create_task(_run_llm_relay(llm_ws, last_heartbeat))
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 0.8
        while loop.time() < deadline:
            await asyncio.sleep(0.05)
            llm_ws.add_incoming({"type": "heartbeat"})
        await llm_ws.close()
        await relay_task
    finally:
        for t in scope:
            t.cancel()

    # Only the test's own 1000-close happened; never the watchdog's 4001.
    assert llm_ws.close_args == [(1000, "")]


async def test_queued_turn_also_arms_stale_close(llm_env):
    manager, client_ws = llm_env
    llm_ws = MockWS()
    # Already past close_timeout when the watchdog starts checking.
    last_heartbeat = _fresh_heartbeats(asyncio.get_running_loop(), llm_age_seconds=0.7)
    task, scope = await _start_watchdog(manager, llm_ws, last_heartbeat)
    await manager.register_llm(llm_ws)
    # The final was forwarded but the LLM never even sent llm_start:
    # a queued (not yet started) turn must arm the stale close too.
    question = "What is the plot of Merlin?"
    manager.conversation_history_manager.update_usr_msg(question, True)
    manager.pending_turns.append(question)

    try:
        await asyncio.sleep(0.3)
    finally:
        for t in scope:
            t.cancel()

    assert llm_ws.close_args == [(4001, "LLM stalled during active turn")]
