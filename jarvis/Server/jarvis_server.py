import asyncio
import base64
import json
import os
import re
import ssl
import logging
from pathlib import Path
from http import HTTPStatus
from websockets.http11 import Request, Response, Headers
from websockets.exceptions import InvalidHandshake
import websockets

HEARTBEAT_TIMEOUT = 15
# A LLM connection silent for this long WHILE A TURN IS ACTIVE OR PENDING
# gets closed (at 15s it is only marked offline / red dot). A healthy
# *idle* LLM connection is never closed by this - the close is armed only
# while a question is in flight.
LLM_STALE_CLOSE_TIMEOUT = 30
# Reconnect cycles allowed before an unanswered question is dropped from
# the conversation history and the client is told the turn failed.
# Reset to zero whenever any turn completes (llm_end): a working pipe
# re-earns the full budget, so a flapping-but-functional LLM cannot be
# starved of resends by earlier blips.
LLM_RESEND_MAX = 3
# Partial LLM text is forwarded to TTS as soon as a sentence boundary is
# crossed (or this many new chars accumulate without one), so audio
# synthesis can begin on the first sentence instead of waiting for the
# full response.
#
# A boundary is a terminal period/exclamation/question mark followed by
# whitespace, end of text, or (for no-space concatenations from streamed
# token joins, "end.This") a capitalized word immediately preceded by two
# word chars — the lookbehind keeps "U.S." and "3.14" intact.
TTS_PARTIAL_MIN_CHARS = 150
SENTENCE_BOUNDARY_RE = re.compile(r'[.!?](?=\s|$|(?<=\w\w[.!?])[A-Z][a-z])')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
HTML_FILE = BASE_DIR / "ProjectInterface" / "index.html"
STATIC_DIR = BASE_DIR / "ProjectInterface"

class MessageHistory:
    """Manages all the message history and sends it to the llm srv"""

    def __init__(self,max_history_msgs):
        self.max_history_msgs=max_history_msgs
        self.reset()

    def reset(self):
        self.history=[]
        self.last_usr_msg=-1
        self.last_llm_msg=-1
        self.receved_final_usr_msg=False
        self.receved_final_llm_msg=False

    def add_usr_msg(self,text):
        msg={'role':'user','content':text}
        self.last_usr_msg = len(self.history)
        self.history.append(msg)
        self.truncate_history()

    def add_llm_msg(self,text):
        msg={'role':'assistant','content':text}
        self.last_llm_msg = len(self.history)
        self.history.append(msg)
        self.truncate_history()

    def update_last_usr_msg(self,text):
        if self.last_usr_msg < 0:
            self.add_usr_msg(text)
        else:
            self.history[self.last_usr_msg]["content"] = text

    def update_last_llm_msg(self,text):
        if self.last_llm_msg < 0:
            self.add_llm_msg(text)
        else:
            self.history[self.last_llm_msg]["content"] = text

    def update_usr_msg(self,text,final=False):
        if self.receved_final_usr_msg:
            self.add_usr_msg(text)
        else:
            self.update_last_usr_msg(text)
        self.receved_final_usr_msg=final

    def update_llm_msg(self,text,final=False):
        if self.receved_final_llm_msg:
            self.add_llm_msg(text)
        else:
            self.update_last_llm_msg(text)
        self.receved_final_llm_msg=final

    def drop_last_usr_msg(self):
        """Remove a trailing user message (a partial transcript that got no final)
        and recompute last_usr_msg (-1 = no user message)."""
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()
        last_usr = -1
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]["role"] == "user":
                last_usr = i
                break
        self.last_usr_msg = last_usr

    def drop_usr_msg(self, text):
        """Remove the most recent user message matching `text` (a turn whose
        reply was interrupted or whose pipe dropped) and recompute
        last_usr_msg. Returns True if a message was removed."""
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]["role"] == "user" and self.history[i]["content"] == text:
                del self.history[i]
                break
        else:
            return False
        last_usr = -1
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]["role"] == "user":
                last_usr = i
                break
        self.last_usr_msg = last_usr
        return True

    def truncate_history(self):
        if len(self.history) > self.max_history_msgs:
            excess = len(self.history) - self.max_history_msgs
            logger.info(f"Truncating conversation history: removing {excess} messages (limit: {self.max_history_msgs})")
            self.history[:] = self.history[-self.max_history_msgs:]
            # The window may have evicted the entries last_usr_msg /
            # last_llm_msg point at: recompute so in-place transcript
            # updates keep addressing live indices (a stale index raises
            # IndexError on the next partial and kills the STT handler).
            self._recompute_msg_indices()

    def _recompute_msg_indices(self):
        last_usr = -1
        last_llm = -1
        for i in range(len(self.history) - 1, -1, -1):
            role = self.history[i]["role"]
            if role == "user" and last_usr < 0:
                last_usr = i
            if role == "assistant" and last_llm < 0:
                last_llm = i
            if last_usr < 0 and last_llm < 0:
                break
        self.last_usr_msg = last_usr
        self.last_llm_msg = last_llm



class ConnectionManager:
    """Manages all WebSocket connections and message routing."""
    
    def __init__(self):
        self.client_ws = None
        self.vad_ws = None
        self.vad_browser_ws = None
        self.stt_ws = None
        self.llm_service_ws = None
        self.tts_ws = None
        self.max_history_msgs = 20
        self.conversation_history_manager = MessageHistory(self.max_history_msgs)
        self.use_full_history = True
        # Turn bookkeeping for interrupt handling: finals forwarded to the
        # LLM service wait here until their turn ends, so an interrupted (or
        # dropped) turn can be matched to its user message and the dangling
        # question removed from the conversation history.
        self.pending_turns = []
        self.active_turn_id = None
        self.active_turn_text = None
        # Zombie-LLM recovery: questions that are in flight (pending_turns)
        # or already parked when the LLM pipe drops are held here and
        # re-sent when the LLM service reconnects (see register_llm /
        # unregister_llm).
        self.llm_resend_queue = []
        self.llm_resend_attempts = 0
        # Monotonic LLM turn id, kept across LLM reconnects. Per-connection
        # counters would restart at 1 on every reconnect and could collide
        # with the client's interruptedTurnId dead-turn guard, silently
        # dropping a re-sent turn's audio.
        self.llm_turn_counter = 0
        # Turn id that the client is still waiting on: set when the LLM
        # pipe drops with a turn in flight, so the budget-exhausted path
        # can send a synthetic llm_end matching the turn the client's
        # "Thinking..." row belongs to (the id survives every intermediate
        # park/reconnect cycle).
        self.llm_failed_turn_id = None
        self.service_status = {
            "vad": "offline",
            "stt": "offline",
            "llm": "offline",
            "tts": "offline",
        }
        self.service_active = {
            "vad": False,
            "stt": False,
            "llm": False,
            "tts": False,
        }
        # Turn whose TTS stream is known to have ended (a final or
        # interrupted audio_end, or a TTS disconnect mid-turn). A late
        # llm_end for that turn must not resurrect the tts active flag,
        # which would make the barge-in gate fire on every later user
        # utterance.
        self.last_tts_final_turn = None
        
    async def register_client(self, websocket):
        self.client_ws = websocket
        logger.info("Client connected")
        
    async def unregister_client(self, websocket):
        if self.client_ws == websocket:
            self.client_ws = None
        logger.info("Client disconnected")
        
    async def register_vad(self, websocket, is_browser=False):
        if is_browser:
            self.vad_browser_ws = websocket
            logger.info("Browser connected to VAD endpoint")
        else:
            if self.vad_ws is not None and self.vad_ws is not websocket:
                logger.warning(f"VAD service re-registered: replacing live connection (state={self.vad_ws.state.name}) - duplicate service suspected")
            self.vad_ws = websocket
            self.service_status["vad"] = "idle"
            logger.info("VAD service connected")
            await self.broadcast_to_client({"type": "vad_connected"})
            await self.set_service_status("vad", "idle")
        
    async def unregister_vad(self, websocket):
        if self.vad_ws == websocket:
            self.vad_ws = None
            self.service_active["vad"] = False
            self.service_status["vad"] = "offline"
            logger.info("VAD service disconnected")
            await self.set_service_status("vad", "offline")
        elif self.vad_browser_ws == websocket:
            self.vad_browser_ws = None
            logger.info("Browser disconnected from VAD endpoint")
        
    async def register_stt(self, websocket):
        if self.stt_ws is not None and self.stt_ws is not websocket:
            logger.warning(f"STT service re-registered: replacing live connection (state={self.stt_ws.state.name}) - duplicate service suspected")
        self.stt_ws = websocket
        self.service_status["stt"] = "idle"
        logger.info("STT service connected")
        await self.set_service_status("stt", "idle")
        
    async def unregister_stt(self, websocket):
        if self.stt_ws == websocket:
            self.stt_ws = None
            self.service_active["stt"] = False
            self.service_status["stt"] = "offline"
            logger.info("STT service disconnected")
            await self.set_service_status("stt", "offline")
        
    async def register_llm(self, websocket):
        if self.llm_service_ws is not None and self.llm_service_ws is not websocket:
            logger.warning(f"LLM service re-registered: replacing live connection (state={self.llm_service_ws.state.name}) - duplicate service suspected")
        self.llm_service_ws = websocket
        self.service_status["llm"] = "idle"
        logger.info("LLM service connected")
        await self.set_service_status("llm", "idle")
        # The previous LLM pipe dropped with a question in flight: re-send
        # the held questions (plus any finals that piled up in
        # pending_turns while the LLM was offline). Each user_input carries
        # the full conversation history, so the LLM service - which treats
        # the server as its history source of truth - needs no other state.
        if self.llm_resend_queue:
            resend_all = self.llm_resend_queue + list(self.pending_turns)
            self.llm_resend_queue = []
            self.pending_turns = list(resend_all)
            logger.info(f"LLM service reconnect: re-sending {len(resend_all)} held question(s) (re-send attempts so far {self.llm_resend_attempts}/{LLM_RESEND_MAX})")
            for text in resend_all:
                try:
                    await websocket.send(json.dumps({
                        "type": "user_input",
                        "text": text,
                        "history": self.conversation_history_manager.history,
                    }))
                except Exception:
                    # Pipe dropped again before the re-send completed; the
                    # handler loop will exit and unregister_llm parks the
                    # rest for the next reconnect attempt.
                    logger.warning("LLM re-send aborted: connection dropped again")
                    break
        
    async def unregister_llm(self, websocket):
        if self.llm_service_ws == websocket:
            self.llm_service_ws = None
            self.service_active["llm"] = False
            self.service_status["llm"] = "offline"
            # The LLM pipe dropped. If a question is in flight (active
            # turn) or queued - or already parked from an earlier blip -
            # it must NOT be dropped from the conversation history: park
            # it in llm_resend_queue and re-send it when the LLM service
            # reconnects. Only after LLM_RESEND_MAX consecutive failures
            # do we drop the question and tell the client the turn failed,
            # so the UI is never left stuck on "Thinking...".
            turn_in_flight = (self.active_turn_id is not None
                               or bool(self.pending_turns)
                               or bool(self.llm_resend_queue))
            if turn_in_flight:
                if self.llm_resend_attempts < LLM_RESEND_MAX:
                    self.llm_resend_attempts += 1
                    self.llm_resend_queue = self.llm_resend_queue + list(self.pending_turns)
                    self.pending_turns.clear()
                    # Remember which turn the client is stuck on before the
                    # bookkeeping below clears it (may be re-parked on later
                    # disconnects; the budget-exhausted drop reports it).
                    if self.active_turn_id is not None:
                        self.llm_failed_turn_id = self.active_turn_id
                    self.active_turn_id = None
                    self.active_turn_text = None
                    logger.info(f"LLM service dropped during active/pending turn - holding {len(self.llm_resend_queue)} question(s) for re-send (attempt {self.llm_resend_attempts}/{LLM_RESEND_MAX})")
                else:
                    # Budget exhausted: the LLM keeps dying before any
                    # answer lands. Drop the held questions and broadcast a
                    # synthetic llm_end so the client's stuck "Thinking..."
                    # state is cleared.
                    failed_turn_id = (self.llm_failed_turn_id
                                       if self.llm_failed_turn_id is not None
                                       else self.active_turn_id)
                    # Drain BOTH held-question containers: register_llm
                    # moved them into pending_turns on each reconnect, so
                    # the queue is usually empty when the budget expires.
                    for text in self.llm_resend_queue + self.pending_turns:
                        if self.conversation_history_manager.drop_usr_msg(text):
                            logger.info("LLM re-send budget exhausted; removed held user message from history")
                    self.llm_resend_queue.clear()
                    self.pending_turns.clear()
                    self.llm_resend_attempts = 0
                    self.active_turn_id = None
                    self.active_turn_text = None
                    self.llm_failed_turn_id = None
                    # The client is stuck on this turn's "Thinking..." row:
                    # broadcast a synthetic end so the UI unsticks and the
                    # user can ask again (the held question was dropped
                    # from the history above).
                    await manager.broadcast_to_client({
                        "type": "llm_end",
                        "turn_id": failed_turn_id,
                        "text": "Sorry - the language model was unreachable; your question was not answered.",
                        "interrupted": True,
                    })
            logger.info("LLM service disconnected")
            await self.set_service_status("llm", "offline")
        
    async def register_tts(self, websocket):
        if self.tts_ws is not None and self.tts_ws is not websocket:
            logger.warning(f"TTS service re-registered: replacing live connection (state={self.tts_ws.state.name}) - duplicate service suspected")
        self.tts_ws = websocket
        self.service_status["tts"] = "idle"
        logger.info("TTS service connected")
        await self.set_service_status("tts", "idle")
        
    async def unregister_tts(self, websocket):
        if self.tts_ws == websocket:
            self.tts_ws = None
            self.service_active["tts"] = False
            self.service_status["tts"] = "offline"
            # If the TTS pipe dropped mid-turn, the turn-final audio_end
            # will never arrive from this connection. Record the turn as
            # ended so a late llm_end (the LLM pipe may still be alive)
            # cannot resurrect the tts flag after TTS comes back.
            if self.active_turn_id is not None:
                self.last_tts_final_turn = self.active_turn_id
            logger.info("TTS service disconnected")
            await self.set_service_status("tts", "offline")
        
    async def broadcast_to_client(self, message):
        msg_type = message.get("type", "unknown")
        if msg_type in ("llm_start", "llm_transcript", "llm_end", "llm_audio", "user_transcript"):
            logger.info(f"[BROADCAST] type={msg_type} turn_id={message.get('turn_id', 'N/A')} textLen={len(message.get('text', ''))}")
        if self.client_ws and self.client_ws.state.name == "OPEN":
            try:
                await self.client_ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
        if self.vad_browser_ws and self.vad_browser_ws.state.name == "OPEN":
            try:
                await self.vad_browser_ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                
    async def broadcast_to_service(self, service, message):
        ws_map = {
            "vad": self.vad_ws,
            "stt": self.stt_ws,
            "llm": self.llm_service_ws,
            "tts": self.tts_ws
        }
        ws = ws_map.get(service)
        if ws and ws.state.name == "OPEN":
            try:
                await ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending to {service}: {e}")
                
    async def forward_message(self, from_service, to_service, message):
        """Forward message from one service to another."""
        if to_service == "client":
            await self.broadcast_to_client(message)
        elif to_service == "vad":
            await self.broadcast_to_service("vad", message)
        elif to_service == "stt":
            await self.broadcast_to_service("stt", message)
        elif to_service == "llm":
            await self.broadcast_to_service("llm", message)
        elif to_service == "tts":
            await self.broadcast_to_service("tts", message)

    async def set_service_status(self, service, status):
        """Update service status and broadcast to all clients."""
        self.service_status[service] = status
        await self.broadcast_to_client({
            "type": "service_status",
            "service": service,
            "status": status,
        })

    async def broadcast_to_vad(self, message):
        """Send message to the VAD service."""
        ws = self.vad_ws
        if ws and ws.state.name == "OPEN":
            try:
                await ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending to VAD: {e}")

    async def set_service_active(self, service, active):
        """Set a service's active state and broadcast if changed."""
        if self.service_active.get(service) == active:
            return
        self.service_active[service] = active
        if active:
            self.service_status[service] = "active"
        elif self.service_status.get(service) == "active":
            self.service_status[service] = "idle"
        await self.set_service_status(service, self.service_status[service])

    async def set_websocket_status(self, status):
        """Set the WebSocket (WEB) status indicator."""
        self.service_status["web"] = status
        await self.broadcast_to_client({
            "type": "service_status",
            "service": "web",
            "status": status,
        })


manager = ConnectionManager()


async def create_heartbeat_task(last_heartbeat, handler_cancel_scope, websocket=None, is_close_armed=None, close_timeout=LLM_STALE_CLOSE_TIMEOUT, check_interval=1.0):
    """Create a task that checks for heartbeat timeout per service.
    
    last_heartbeat: dict mapping service name -> last heartbeat unix timestamp
    handler_cancel_scope: list to store the task for cancellation
    websocket: the connection to close once it has been silent for
        close_timeout seconds (only the LLM service passes one)
    is_close_armed: 0-arg callable; the stale close fires only while it
        returns True (a turn is active/pending/queued), so a healthy *idle*
        LLM connection is never killed
    close_timeout: silence threshold for the armed close (LLM: 30s)
    check_interval: seconds between checks (1s in production; tests pass a
        smaller value for speed)
    """
    services = ["vad", "stt", "llm", "tts"]
    
    # Initialize all services as offline at handler start
    for svc in services:
        if svc not in last_heartbeat:
            last_heartbeat[svc] = 0

    async def check():
        while True:
            await asyncio.sleep(check_interval)
            now = asyncio.get_event_loop().time()
            for svc in services:
                last = last_heartbeat.get(svc, 0)
                if last > 0 and (now - last) > HEARTBEAT_TIMEOUT:
                    logger.info(f"Heartbeat timeout for {svc}, broadcasting offline")
                    manager.service_status[svc] = "offline"
                    await manager.set_service_status(svc, "offline")
                    # Other services reset to 0 so the offline mark fires
                    # once. The LLM keeps its original timestamp so the
                    # armed close below can measure close_timeout seconds of
                    # *total* silence (the repeated offline broadcast is a
                    # no-op for the client dots).
                    if svc != "llm":
                        last_heartbeat[svc] = 0

                if (svc == "llm"
                        and websocket is not None
                        and is_close_armed is not None
                        and last > 0 and (now - last) > close_timeout
                        and is_close_armed()):
                    logger.warning(f"LLM connection silent for {now - last:.0f}s with a turn in flight - closing (code 4001); held question(s) will be re-sent on reconnect")
                    try:
                        await websocket.close(code=4001, reason="LLM stalled during active turn")
                    except Exception as e:
                        logger.info(f"Stale LLM close failed: {e}")
                    break

    task = asyncio.create_task(check())
    handler_cancel_scope.append(task)


def serve_html_file(connection, request, html_file):
    """Serve HTML file for HTTP requests to / and static files from ProjectInterface."""
    content_types = {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css",
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".webp": "image/webp",
    }

    if request.path == "/":
        if html_file.exists():
            content_type = content_types.get(html_file.suffix, "application/octet-stream")
            content = html_file.read_bytes()
            headers = Headers()
            headers["Content-Type"] = content_type
            return Response(
                status_code=HTTPStatus.OK,
                reason_phrase="OK",
                headers=headers,
                body=content,
            )
        return None

    if request.path.startswith("/") and not request.path.startswith("/vad") and not request.path.startswith("/stt") and not request.path.startswith("/llm_service") and not request.path.startswith("/tts"):
        static_file = STATIC_DIR / request.path.lstrip("/")
        if static_file.exists() and static_file.is_file():
            content_type = content_types.get(static_file.suffix, "application/octet-stream")
            content = static_file.read_bytes()
            headers = Headers()
            headers["Content-Type"] = content_type
            return Response(
                status_code=HTTPStatus.OK,
                reason_phrase="OK",
                headers=headers,
                body=content,
            )
    return None

    #try:
    #    if request.path == "/":
    #        if html_file.exists():
    #            content_type = content_types.get(html_file.suffix, "application/octet-stream")
    #            content = html_file.read_bytes()
    #            headers = Headers()
    #            headers["Content-Type"] = content_type
    #            return Response(
    #                status_code=HTTPStatus.OK,
    #                reason_phrase="OK",
    #                headers=headers,
    #                body=content,
    #            )
    #        return None

    #    if request.path.startswith("/") and not request.path.startswith("/vad") and not request.path.startswith("/stt") and not request.path.startswith("/llm_service") and not request.path.startswith("/tts"):
    #        static_file = STATIC_DIR / request.path.lstrip("/")
    #        if static_file.exists() and static_file.is_file():
    #            content_type = content_types.get(static_file.suffix, "application/octet-stream")
    #            content = static_file.read_bytes()
    #            headers = Headers()
    #            headers["Content-Type"] = content_type
    #            return Response(
    #                status_code=HTTPStatus.OK,
    #                reason_phrase="OK",
    #                headers=headers,
    #                body=content,
    #            )

    #    return None
    #except InvalidHandshake:
    #    headers = Headers()
    #    headers["Connection"] = "close"
    #    logger.info(f"Rejected non-WebSocket connection to {request.path}")
    #    connection.handshake_exc = None
    #    return Response(
    #        status_code=HTTPStatus.UPGRADE_REQUIRED,
    #        reason_phrase="Upgrade Required",
    #        headers=headers,
    #        body=b"This is a WebSocket service.\n",
    #    )

async def handle_client(websocket):
    """Handle browser WebSocket connection."""
    await manager.register_client(websocket)
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_client_message(data, websocket)
            except json.JSONDecodeError:
                logger.error("Invalid JSON from client")
    except Exception as e:
        logger.error(f"Client error: {e}")
    finally:
        await manager.unregister_client(websocket)

async def handle_client_message(data, websocket):
    """Handle messages from the browser client."""
    msg_type = data.get("type")
    
    if msg_type == "start":
        await manager.broadcast_to_client({"type": "user_start"})
        if manager.vad_ws and manager.vad_ws.state.name == "OPEN":
            await manager.vad_ws.send(json.dumps({"type": "start"}))
            
    elif msg_type == "stop":
        await manager.broadcast_to_client({"type": "user_end"})
        if manager.vad_ws and manager.vad_ws.state.name == "OPEN":
            await manager.vad_ws.send(json.dumps({"type": "stop"}))
            
    elif msg_type == "interrupt":
        await manager.broadcast_to_client({"type": "interrupt"})
        logger.info(f"Client interrupt: active_turn_id={manager.active_turn_id} pending_turns={len(manager.pending_turns)}")
        if manager.tts_ws and manager.tts_ws.state.name == "OPEN":
            await manager.tts_ws.send(json.dumps({"type": "stop"}))
            logger.info("Client interrupt: stop sent to TTS")
        else:
            logger.warning("Client interrupt: no TTS connection registered - in-flight audio will NOT stop")
        if manager.llm_service_ws and manager.llm_service_ws.state.name == "OPEN":
            await manager.llm_service_ws.send(json.dumps({"type": "stop"}))
            logger.info("Client interrupt: stop sent to LLM")
        else:
            logger.warning("Client interrupt: no LLM connection registered")

    elif msg_type == "reset":
        manager.conversation_history_manager.reset()
        manager.pending_turns.clear()
        manager.active_turn_id = None
        manager.active_turn_text = None
        if manager.tts_ws and manager.tts_ws.state.name == "OPEN":
            await manager.tts_ws.send(json.dumps({"type": "stop_generation"}))
        if manager.llm_service_ws and manager.llm_service_ws.state.name == "OPEN":
            await manager.llm_service_ws.send(json.dumps({"type": "reset_history"}))
        await manager.broadcast_to_client({"type": "reset_complete"})
        logger.info("Conversation reset by user")

    elif msg_type == "set_settings":
        voice = data.get("voice")
        sample_rate = data.get("sample_rate")
        system_prompt = data.get("system_prompt")
        wake_word = data.get("wake_word")
        tts_instruct = data.get("tts_instruct")
        use_full_history = data.get("use_full_history")
        manager.use_full_history = use_full_history if use_full_history is not None else True
        logger.info(f"Message history mode: {'full' if manager.use_full_history else 'latest only'}")

        if voice:
            if manager.tts_ws and manager.tts_ws.state.name == "OPEN":
                await manager.tts_ws.send(json.dumps({
                    "type": "set_voice",
                    "speaker": voice,
                }))
            logger.info(f"Updated TTS voice to: {voice}")

        if tts_instruct:
            if manager.tts_ws and manager.tts_ws.state.name == "OPEN":
                await manager.tts_ws.send(json.dumps({
                    "type": "set_voice",
                    "instruct": tts_instruct,
                }))
            logger.info(f"Updated TTS instruct to: {tts_instruct}")

        if system_prompt:
            if manager.llm_service_ws and manager.llm_service_ws.state.name == "OPEN":
                await manager.llm_service_ws.send(json.dumps({
                    "type": "set_system_prompt",
                    "prompt": system_prompt,
                }))
            logger.info(f"Updated LLM system prompt")

        if wake_word is not None:
            logger.info(f"Wake word detection: {'enabled' if wake_word else 'disabled'}")

        if use_full_history is not None:
            ws_to_send = manager.llm_service_ws
            if ws_to_send and ws_to_send.state.name == "OPEN":
                await ws_to_send.send(json.dumps({
                    "type": "set_settings",
                    "use_full_history": use_full_history,
                }))
                logger.info(f"Sent use_full_history={use_full_history} to LLM service")

        logger.info("Settings updated")

    elif msg_type == "set_voice":
        if manager.tts_ws and manager.tts_ws.state.name == "OPEN":
            await manager.tts_ws.send(json.dumps({"type": "set_voice", "voice": data.get("voice", "eric")}))
        logger.info(f"Voice changed to: {data.get('voice', 'eric')}")

    elif msg_type == "set_sample_rate":
        logger.info(f"Sample rate set to: {data.get('sample_rate', 16000)}")

    elif msg_type == "set_system_prompt":
        if manager.llm_service_ws and manager.llm_service_ws.state.name == "OPEN":
            await manager.llm_service_ws.send(json.dumps({"type": "set_system_prompt", "prompt": data.get("prompt", "")}))
        logger.info("System prompt updated")

    elif msg_type == "set_wake_word":
        logger.info(f"Wake word: {data.get('enabled', False)}")

async def handle_vad(websocket):
    """Handle browser connection to /vad endpoint.
    
    Receives raw audio from the browser and forwards it to the VAD service.
    VAD output is processed by handle_vad_service (not here).
    """
    await manager.register_vad(websocket, is_browser=True)
    await manager.set_service_active("vad", True)
    vad_chunk_id = 0
    last_chunk_time = asyncio.get_event_loop().time()

    async def audio_heartbeat():
        while True:
            await asyncio.sleep(10)
            elapsed = asyncio.get_event_loop().time() - last_chunk_time
            if elapsed > 15 and vad_chunk_id > 0:
                logger.warning(f"VAD audio heartbeat: no chunks for {elapsed:.1f}s (last at chunk {vad_chunk_id})")
            elif elapsed > 10 and vad_chunk_id == 0:
                logger.warning(f"VAD audio heartbeat: {vad_chunk_id} chunks total, no chunks for {elapsed:.1f}s")

    heartbeat_task = asyncio.create_task(audio_heartbeat())

    try:
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Binary: raw Int16 audio from browser - forward to VAD service
                    vad_chunk_id += 1
                    #logger.info(f"Browser -> VAD: audio_chunk chunk_id={vad_chunk_id} size={len(message)} bytes")
                    await manager.set_service_active("vad", True)
                    await manager.broadcast_to_vad({
                        "type": "audio_chunk",
                        "audio": base64.b64encode(message).decode("ascii"),
                        "chunk_id": vad_chunk_id
                    })
                    last_chunk_time = asyncio.get_event_loop().time()
                else:
                    # The browser's only WebSocket is /vad, so its JSON
                    # commands (start/stop/interrupt/reset/set_settings)
                    # arrive here - without this branch they are dropped.
                    data = json.loads(message)
                    await handle_client_message(data, websocket)

            except json.JSONDecodeError:
                logger.error("Invalid JSON from VAD")
    except Exception as e:
        logger.error(f"VAD error: {e}")
    finally:
        heartbeat_task.cancel()
        logger.info(f"Browser VAD disconnected: received {vad_chunk_id} audio chunks total")
        await manager.set_service_active("vad", False)
        await manager.unregister_vad(websocket)

async def handle_vad_service(websocket):
    """Handle VAD service connection (separate from browser)."""
    await manager.register_vad(websocket, is_browser=False)
    
    last_heartbeat = {}
    handler_cancel_scope = []
    heartbeat_task = await create_heartbeat_task(last_heartbeat, handler_cancel_scope)
    
    try:
        async for message in websocket:
            try:
                # Only text messages expected from VAD service
                if isinstance(message, str):
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "start":
                        logger.info("VAD service acknowledged")
                        continue

                    if msg_type == "heartbeat":
                        import time
                        last_heartbeat["vad"] = asyncio.get_event_loop().time()
                        continue

                    if msg_type == "audio_chunk":
                        vad_chunk_id_vad = data.get("chunk_id", "?")
                        logger.info(f"VAD service -> Server: audio_chunk chunk_id={vad_chunk_id_vad} forwarding to STT")
                        await manager.set_service_active("vad", True)
                        await manager.set_service_active("stt", True)
                        await manager.broadcast_to_service("stt", {
                            "type": "audio_chunk",
                            "audio": data.get("audio"),
                            "chunk_id": data.get("chunk_id")
                        })
                    elif msg_type == "vad_result":
                        if data.get("speech") == False:
                            await manager.set_service_active("vad", True)
                            await manager.set_service_active("stt", True)
                            await manager.broadcast_to_service("stt", {
                                "type": "vad_result",
                                "speech": False,
                                "chunk_id": data.get("chunk_id")
                            })
                            await manager.broadcast_to_client({"type": "user_end"})
                        elif data.get("speech") == True:
                            # Barge-in: the user started speaking while the
                            # assistant is busy (LLM generating or TTS
                            # playing). The client only interrupts on its
                            # stop button, so without this the response
                            # keeps playing over the user's voice. Mirror
                            # the client interrupt path: stop TTS + LLM and
                            # tell the client to drop playback. The in-flight
                            # turn ends with llm_end(interrupted=True), which
                            # drops its user message from history.
                            if (manager.service_active.get("tts")
                                    or manager.service_active.get("llm")
                                    or manager.pending_turns):
                                await manager.broadcast_to_client({"type": "interrupt"})
                                tts_stop = "NOT sent (no TTS connection)"
                                if manager.tts_ws and manager.tts_ws.state.name == "OPEN":
                                    await manager.tts_ws.send(json.dumps({"type": "stop"}))
                                    tts_stop = "sent to TTS"
                                llm_stop = "NOT sent (no LLM connection)"
                                if manager.llm_service_ws and manager.llm_service_ws.state.name == "OPEN":
                                    await manager.llm_service_ws.send(json.dumps({"type": "stop"}))
                                    llm_stop = "sent to LLM"
                                logger.warning(f"Barge-in: user speech started during active turn - interrupting (turn_id={manager.active_turn_id} tts_active={manager.service_active.get('tts')} llm_active={manager.service_active.get('llm')} pending={len(manager.pending_turns)} TTS-stop={tts_stop} LLM-stop={llm_stop})")
                    else:
                        logger.info(f"Unknown VAD service message: {msg_type}")
                else:
                    logger.info(f"Unexpected binary message from VAD service")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from VAD service: {e}")
            except Exception as e:
                logger.error(f"VAD service error: {e}")
    finally:
        for t in handler_cancel_scope:
            t.cancel()
        await manager.set_service_active("vad", False)
        await manager.set_service_active("stt", False)
        await manager.unregister_vad(websocket)

async def handle_stt(websocket):
    """Handle STT service connection."""
    await manager.register_stt(websocket)
    
    last_heartbeat = {}
    handler_cancel_scope = []
    heartbeat_task = await create_heartbeat_task(last_heartbeat, handler_cancel_scope)
    stt_chunk_count = 0
    last_audio_time = asyncio.get_event_loop().time()

    async def stt_audio_heartbeat():
        while True:
            await asyncio.sleep(10)
            elapsed = asyncio.get_event_loop().time() - last_audio_time
            if elapsed > 15 and stt_chunk_count > 0:
                logger.warning(f"STT audio heartbeat: no VAD chunks for {elapsed:.1f}s (last at chunk {stt_chunk_count})")
            elif elapsed > 10 and stt_chunk_count == 0:
                logger.warning(f"STT audio heartbeat: {stt_chunk_count} chunks total, no VAD chunks for {elapsed:.1f}s")

    stt_heartbeat_task = asyncio.create_task(stt_audio_heartbeat())
    handler_cancel_scope.append(stt_heartbeat_task)
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "start":
                    await websocket.send(json.dumps({"type": "start"}))
                    continue
                    
                if msg_type == "heartbeat":
                    last_heartbeat["stt"] = asyncio.get_event_loop().time()
                    continue

                # Update audio heartbeat for any STT activity (not just audio_chunk)
                last_audio_time = asyncio.get_event_loop().time()
                if msg_type == "audio_chunk":
                    stt_chunk_count += 1
                    logger.info(f"STT -> Server: received audio_chunk chunk_id={data.get('chunk_id')} from VAD (total: {stt_chunk_count})")
                    
                if msg_type == "partial_transcript":
                    logger.debug(f"original manager.conversation_history: {manager.conversation_history_manager.history}")
                    await manager.set_service_active("stt", True)
                    partial_text = data.get("text", "")
                    logger.info("STT partial_transcript: '%s'", partial_text)
                    await manager.broadcast_to_client({
                        "type": "user_partial",
                        "text": data.get("text", ""),
                        "chunk_id": data.get("chunk_id")
                    })
                    if not partial_text.strip():
                        # Empty partial (noise/crosstalk): keep the client event flow,
                        # but don't create/update an empty user message and don't ping the LLM
                        continue
                    manager.conversation_history_manager.update_usr_msg(partial_text)
                    ws_to_send = manager.llm_service_ws
                    if ws_to_send and ws_to_send.state.name == "OPEN":
                        await ws_to_send.send(json.dumps({
                            "type": "user_input",
                            "text": partial_text,
                            "history": manager.conversation_history_manager.history,
                            "partial": True,
                        }))
                    
                elif msg_type == "final_transcript":
                    logger.debug(f"original manager.conversation_history: {manager.conversation_history_manager.history}")
                    await manager.set_service_active("stt", True)
                    user_text = data.get("text", "")
                    logger.info("STT final_transcript: '%s'", user_text)
                    if not user_text.strip():
                        # Noise/crosstalk produced no transcript. Don't add an empty user
                        # turn to the conversation and don't call the LLM (the empty user
                        # turns + wasted LLM calls were the source of the silent turns).
                        # Drop the dangling partial so the model never sees an unanswered fragment.
                        manager.conversation_history_manager.drop_last_usr_msg()
                        manager.conversation_history_manager.receved_final_usr_msg = True
                        logger.info("STT final transcript empty - ignored (no history update, no LLM call)")
                        await manager.broadcast_to_client({
                            "type": "user_transcript",
                            "text": "",
                            "final": True,
                            "chunk_id": data.get("chunk_id")
                        })
                        continue
                    await manager.set_service_active("llm", True)
                    manager.conversation_history_manager.update_usr_msg(user_text,True)
                    await manager.broadcast_to_client({
                        "type": "user_transcript",
                        "text": user_text,
                        "final": True,
                        "chunk_id": data.get("chunk_id")
                    })
                    logger.debug(f"STT conversation_history before forwarding to LLM: {json.dumps(manager.conversation_history_manager.history, ensure_ascii=False)}")
                    # Forward directly to LLM service (not through browser /llm endpoint)
                    ws_to_send = manager.llm_service_ws
                    if ws_to_send and ws_to_send.state.name == "OPEN":
                        await ws_to_send.send(json.dumps({
                            "type": "user_input",
                            "text": user_text,
                            "history": manager.conversation_history_manager.history,
                        }))
                        # Track this final until its turn ends (llm_end) so an
                        # interrupted/dropped turn can drop its user message.
                        manager.pending_turns.append(user_text)
                        logger.info(f"Forwarded user_input to LLM service (pending turns: {len(manager.pending_turns)})")

                elif msg_type == "wake_word":
                    await manager.broadcast_to_client({
                        "type": "wake_word",
                        "command": data.get("command"),
                    })
                    logger.info(f"Forwarded wake_word '{data.get('command')}' to browser")

                logger.debug(f"manager.conversation_history: {manager.conversation_history_manager.history}")

            except json.JSONDecodeError:
                logger.error("Invalid JSON from STT")
    except Exception as e:
        logger.error(f"STT error: {e}")
    finally:
        for t in handler_cancel_scope:
            t.cancel()
        logger.info(f"STT disconnected: processed {stt_chunk_count} audio chunks total")
        await manager.set_service_active("stt", False)
        await manager.unregister_stt(websocket)

async def handle_llm_response_messages(websocket, accumulated_text, turn_id_ref, forwarded_len_ref, last_heartbeat, handler_cancel_scope, source_label):
    """Shared handler for LLM streaming messages (llm_start, llm_token, llm_end).
    
    Called iteratively from handle_llm_service.
    Returns updated (accumulated_text, turn_id, forwarded_len) tuple.
    """
    turn_id = turn_id_ref[0]
    
    async for message in websocket:
        try:
            # Every received message proves the pipe is alive - refresh the
            # watchdog timestamp here (not just on heartbeat messages) so a
            # frozen LLM is detected by the silence that follows its last
            # message of any kind.
            if last_heartbeat:
                last_heartbeat["llm"] = asyncio.get_event_loop().time()
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "llm_start":
                await manager.set_service_active("llm", True)
                accumulated_text = ""
                turn_id_ref[0] = turn_id = manager.llm_turn_counter + 1
                manager.llm_turn_counter = turn_id
                forwarded_len_ref[0] = 0
                # The turn consumed the oldest forwarded final; remember it so
                # an interrupt (or a dropped pipe) can drop the matching user
                # message from the conversation history.
                manager.active_turn_id = turn_id
                manager.active_turn_text = manager.pending_turns[0] if manager.pending_turns else None
                # A new turn supersedes any earlier failed one: clear the
                # remembered id so a later budget-exhausted drop reports
                # this turn, not a stale one.
                manager.llm_failed_turn_id = None
                logger.info(f"[{source_label}] >>> llm_start turn_id={turn_id}")
                await manager.broadcast_to_client({"type": "llm_start", "turn_id": turn_id})
                
            elif msg_type == "heartbeat" and source_label == "LLM-SERVICE":
                # Timestamp already refreshed for every received message
                continue

            elif msg_type == "service_status":
                if data.get("status") == "disconnected":
                    await manager.unregister_llm(websocket)
                continue
            
            elif msg_type == "llm_token":
                await manager.set_service_active("llm", True)
                token_text = data.get("text", "")
                accumulated_text += token_text
                logger.info(f"[{source_label}] >>> llm_transcript turn_id={turn_id} tokenLen={len(token_text)} accumulatedLen={len(accumulated_text)}")
                await manager.broadcast_to_client({
                    "type": "llm_transcript",
                    "text": accumulated_text,
                    "partial": True,
                    "turn_id": turn_id
                })
                # Stream to TTS as soon as a new sentence (or a
                # boundary-free span of TTS_PARTIAL_MIN_CHARS) has been
                # produced since the last forward, so synthesis can start
                # while the LLM is still generating.
                new_since_forward = accumulated_text[forwarded_len_ref[0]:]
                if SENTENCE_BOUNDARY_RE.search(new_since_forward) or len(new_since_forward) >= TTS_PARTIAL_MIN_CHARS:
                    forwarded_len_ref[0] = len(accumulated_text)
                    await manager.forward_message("llm", "tts", {
                        "type": "tts_input",
                        "text": accumulated_text,
                        "partial": True,
                        "turn_id": turn_id,
                    })
                    logger.info(f"Forwarded partial to TTS (turn_id={turn_id}, len={len(accumulated_text)})")
                
            elif msg_type == "llm_end":
                interrupted = data.get("interrupted", False)
                if manager.pending_turns:
                    completed_text = manager.pending_turns.pop(0)
                else:
                    completed_text = None
                    logger.warning(f"[{source_label}] llm_end without a pending turn (bookkeeping desync) turn_id={turn_id}")
                manager.active_turn_id = None
                manager.active_turn_text = None
                # A completed turn proves the pipe works: hand the full
                # re-send budget back so a flapping-but-functional LLM
                # cannot be starved of resends by earlier blips.
                manager.llm_resend_attempts = 0
                await manager.set_service_active("llm", False)
                # Mark TTS active only if the turn completed (an
                # interrupted turn's stop path sends the final audio_end
                # itself) and the TTS pipe is actually connected (a dead
                # TTS can never deliver the final audio_end, so the flag
                # would stay stuck on and fire spurious barge-ins on
                # every later user utterance).
                if (not interrupted
                        and manager.tts_ws is not None
                        and turn_id != manager.last_tts_final_turn):
                    await manager.set_service_active("tts", True)
                logger.info(f"[{source_label}] >>> llm_end turn_id={turn_id} finalTextLen={len(accumulated_text)} interrupted={interrupted} text='{accumulated_text[:100]}'")
                await manager.broadcast_to_client({
                    "type": "llm_end",
                    "turn_id": turn_id,
                    "text": accumulated_text,
                    "interrupted": interrupted,
                })
                if interrupted and completed_text:
                    if manager.conversation_history_manager.drop_usr_msg(completed_text):
                        logger.info(f"Turn {turn_id} was interrupted - removed its user message from history")
                if interrupted and accumulated_text.strip():
                    logger.info(f"Turn {turn_id} was interrupted - final text (len={len(accumulated_text)}) NOT forwarded to TTS")
                if accumulated_text.strip() and not interrupted:
                    manager.conversation_history_manager.update_llm_msg(accumulated_text,True)
                if accumulated_text.strip() and not interrupted:
                    await manager.forward_message("llm", "tts", {
                        "type": "tts_input",
                        "text": accumulated_text,
                        "partial": False,
                        "turn_id": turn_id,
                    })
                    logger.info(f"Sent final LLM response to TTS: \"{accumulated_text[:80]}...\"")
                accumulated_text = ""
                
            elif msg_type == "user_input":
                await manager.set_service_active("llm", True)
                logger.info(f"LLM received input: \"{data.get('text', '')}\"")
                if manager.tts_ws and manager.tts_ws.state.name == "OPEN":
                    await manager.tts_ws.send(json.dumps({"type": "stop_generation"}))
            else:
                logger.info(f"Unknown LLM handler message type: {msg_type}")
                    
        except json.JSONDecodeError:
            if source_label == "LLM-SERVICE":
                logger.error("Invalid JSON from LLM service")
            else:
                logger.error("Invalid JSON from LLM")
    return accumulated_text, turn_id, forwarded_len_ref[0]


async def handle_llm_service(websocket):
    """Handle LLM service connection to /llm_service endpoint.
    
    LLM service sends llm_start, llm_token, llm_end messages.
    Receives user_input from the server.
    """
    await manager.register_llm(websocket)
    await websocket.send(json.dumps({"type": "start"}))
    logger.info("LLM service ready")
    
    accumulated_llm_text = ""
    current_turn_id = 0
    forwarded_len = 0
    last_heartbeat = {}
    handler_cancel_scope = []
    heartbeat_task = await create_heartbeat_task(
        last_heartbeat, handler_cancel_scope,
        websocket=websocket,
        # Close a stalled LLM only while a question depends on it: an
        # active turn, a queued final, or a question already parked from
        # an earlier disconnect. A healthy idle connection is never killed.
        is_close_armed=lambda: (manager.active_turn_id is not None
                                 or bool(manager.pending_turns)
                                 or bool(manager.llm_resend_queue)),
    )
    
    try:
        while True:
            result = await handle_llm_response_messages(
                websocket, accumulated_llm_text,
                [current_turn_id], [forwarded_len], last_heartbeat, handler_cancel_scope,
                "LLM-SERVICE"
            )
            accumulated_llm_text, current_turn_id, forwarded_len = result
            # websockets 16 terminates `async for` silently on a *normal*
            # close; without this check the loop would re-enter the handler
            # forever on a dead connection, spinning the event loop at
            # 100% CPU and wedging every endpoint.
            if websocket.state.name != "OPEN":
                logger.info("LLM service connection closed; handler exiting")
                break
    except Exception as e:
        logger.error(f"LLM service error: {e}")
    finally:
        for t in handler_cancel_scope:
            t.cancel()
        await manager.set_service_active("llm", False)
        await manager.unregister_llm(websocket)

async def handle_tts(websocket):
    """Handle TTS service connection."""
    await manager.register_tts(websocket)
    await websocket.send(json.dumps({"type": "start"}))
    await websocket.send(json.dumps({"type": "service_status", "service": "tts", "status": "idle"}))
    logger.info("TTS service ready")
    
    last_heartbeat = {}
    handler_cancel_scope = []
    heartbeat_task = await create_heartbeat_task(last_heartbeat, handler_cancel_scope)
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "heartbeat":
                    last_heartbeat["tts"] = asyncio.get_event_loop().time()
                    continue
                
                if msg_type == "audio_chunk":
                    #logger.info("TTS received and broadcasting audio chunk")
                    await manager.set_service_active("tts", True)
                    await manager.broadcast_to_client({
                        "type": "llm_audio",
                        "audio": data.get("audio", ""),
                        "turn_id": data.get("turn_id"),
                        "audio_seq": data.get("audio_seq"),
                    })
                    
                elif msg_type == "audio_end":
                    # TTS sends one audio_end per input message (one per
                    # sentence boundary). Only the turn-final one (or a
                    # barge-in stop) ends the assistant's speaking turn -
                    # clearing the flag on every per-sentence end let
                    # barge-in miss every later sentence of the same turn.
                    interrupted = bool(data.get("interrupted", False))
                    final = bool(data.get("final", False)) or interrupted
                    turn_id = data.get("turn_id")
                    logger.info(f"TTS received and broadcasting audio end (turn_id={turn_id} interrupted={interrupted} final={final})")
                    if final:
                        if turn_id is not None:
                            manager.last_tts_final_turn = turn_id
                        await manager.set_service_active("tts", False)
                    await manager.broadcast_to_client({
                        "type": "audio_complete",
                        "turn_id": turn_id,
                        "interrupted": interrupted,
                        "final": final,
                    })
                        
            except json.JSONDecodeError:
                logger.error("Invalid JSON from TTS")
    except Exception as e:
        logger.error(f"TTS error: {e}")
    finally:
        for t in handler_cancel_scope:
            t.cancel()
        await manager.set_service_active("tts", False)
        await manager.unregister_tts(websocket)

async def handler(websocket):
    """Main WebSocket handler that routes to appropriate handler."""
    path = websocket.request.path
    
    if path == "/":
        if not HTML_FILE.exists():
            await websocket.send("File not found")
            return
        
        try:
            content = HTML_FILE.read_bytes()
            if HTML_FILE.suffix == ".html":
                content_type = "text/html; charset=utf-8"
            elif HTML_FILE.suffix == ".css":
                content_type = "text/css"
            elif HTML_FILE.suffix in (".js", ".mjs"):
                content_type = "application/javascript"
            elif HTML_FILE.suffix == ".json":
                content_type = "application/json"
            elif HTML_FILE.suffix in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp"):
                content_type = f"image/{HTML_FILE.suffix[1:]}"
            else:
                content_type = "application/octet-stream"
            
            await websocket.send(content_type)
            await websocket.send(content)
            return
        except Exception as e:
            logger.error(f"Error serving {HTML_FILE}: {e}")
            await websocket.send("Internal server error")
            return
    
    if path == "/vad":
        await handle_vad(websocket)
    elif path == "/vad_service":
        await handle_vad_service(websocket)
    elif path == "/stt":
        await handle_stt(websocket)
    elif path == "/llm_service":
        await handle_llm_service(websocket)
    elif path == "/tts":
        await handle_tts(websocket)
    else:
        await websocket.close(1008, "Path not found")

async def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8765"))
    
    cert_dir = Path(__file__).parent.parent
    cert_file = cert_dir / "localhost+2.pem"
    key_file = cert_dir / "localhost+2-key.pem"
    
    ssl_context = None
    use_https = False
    
    if cert_file.exists() and key_file.exists():
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(str(cert_file), keyfile=str(key_file))
        use_https = True
        logger.info("SSL certificates found, enabling HTTPS")
    else:
        logger.warning("SSL certificates not found, running on HTTP")
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Serving HTML from: {HTML_FILE}")
    
    server = await websockets.serve(
        handler,
        host,
        port,
        ping_interval=30,
        ping_timeout=70,
        ssl=ssl_context,
        process_request=lambda conn, req: serve_html_file(conn, req, HTML_FILE)
    )
    
    protocol = "https" if use_https else "http"
    logger.info(f"Server running at {protocol}://{host}:{port}")
    
    if host == "0.0.0.0":
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            logger.info(f"Accessible from local network at {protocol}://{local_ip}:{port}")
        except Exception:
            pass
        logger.info(f"WebSocket endpoints: /vad, /stt, /llm_service, /tts, /")
        
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        finally:
            server.close()
            await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
