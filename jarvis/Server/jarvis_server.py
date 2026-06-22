import asyncio
import base64
import json
import os
import ssl
import logging
from pathlib import Path
from http import HTTPStatus
from websockets.http11 import Request, Response, Headers
import websockets

HEARTBEAT_TIMEOUT = 15

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
HTML_FILE = BASE_DIR / "ProjectInterface" / "index.html"
STATIC_DIR = BASE_DIR / "ProjectInterface"

class ConnectionManager:
    """Manages all WebSocket connections and message routing."""
    
    def __init__(self):
        self.client_ws = None
        self.vad_ws = None
        self.vad_browser_ws = None
        self.stt_ws = None
        self.llm_client_ws = None  # Browser client connected to /llm
        self.llm_service_ws = None  # LLM service connected to /llm_service
        self.llm_ws = None  # Legacy alias for backward compatibility
        self.tts_ws = None
        self.conversation_history = []
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
        
    async def register_llm(self, websocket, is_service=False):
        if is_service:
            self.llm_service_ws = websocket
            self.service_status["llm"] = "idle"
            logger.info("LLM service connected")
            await self.set_service_status("llm", "idle")
        else:
            self.llm_client_ws = websocket
            self.service_status["llm"] = "idle"
            logger.info("LLM client connected")
        # Keep legacy alias pointing to service ws for backward compat
        if is_service:
            self.llm_ws = websocket
        elif self.llm_ws is None:
            self.llm_ws = websocket
        
    async def unregister_llm(self, websocket):
        if self.llm_service_ws == websocket:
            self.llm_service_ws = None
            self.service_active["llm"] = False
            self.service_status["llm"] = "offline"
            logger.info("LLM service disconnected")
            await self.set_service_status("llm", "offline")
        elif self.llm_client_ws == websocket:
            self.llm_client_ws = None
            logger.info("LLM client disconnected")
        if self.llm_ws == websocket:
            self.llm_ws = None
        
    async def register_tts(self, websocket):
        self.tts_ws = websocket
        self.service_status["tts"] = "idle"
        logger.info("TTS service connected")
        await self.set_service_status("tts", "idle")
        
    async def unregister_tts(self, websocket):
        if self.tts_ws == websocket:
            self.tts_ws = None
            self.service_active["tts"] = False
            self.service_status["tts"] = "offline"
            logger.info("TTS service disconnected")
            await self.set_service_status("tts", "offline")
        
    async def broadcast_to_client(self, message):
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
            "llm": self.llm_ws,
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


async def create_heartbeat_task(last_heartbeat, handler_cancel_scope):
    """Create a task that checks for heartbeat timeout per service.
    
    last_heartbeat: dict mapping service name -> last heartbeat unix timestamp
    handler_cancel_scope: list to store the task for cancellation
    """
    services = ["vad", "stt", "llm", "tts"]
    
    # Initialize all services as offline at handler start
    for svc in services:
        if svc not in last_heartbeat:
            last_heartbeat[svc] = 0

    async def check():
        while True:
            await asyncio.sleep(1)
            now = asyncio.get_event_loop().time()
            for svc in services:
                last = last_heartbeat.get(svc, 0)
                if last > 0 and (now - last) > HEARTBEAT_TIMEOUT:
                    logger.info(f"Heartbeat timeout for {svc}, broadcasting offline")
                    manager.service_status[svc] = "offline"
                    await manager.set_service_status(svc, "offline")
                    last_heartbeat[svc] = 0
    
    task = asyncio.create_task(check())
    handler_cancel_scope.append(task)


manager = ConnectionManager()

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

    if request.path.startswith("/") and not request.path.startswith("/vad") and not request.path.startswith("/stt") and not request.path.startswith("/llm") and not request.path.startswith("/tts") and not request.path.startswith("/llm_service"):
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
        if manager.tts_ws and manager.tts_ws.state.name == "OPEN":
            await manager.tts_ws.send(json.dumps({"type": "stop"}))
        logger.info("TTS interrupted by user speech")

    elif msg_type == "reset":
        manager.conversation_history = []
        if manager.tts_ws and manager.tts_ws.state.name == "OPEN":
            await manager.tts_ws.send(json.dumps({"type": "stop_generation"}))
        if manager.llm_ws and manager.llm_ws.state.name == "OPEN":
            await manager.llm_ws.send(json.dumps({"type": "reset_history"}))
        await manager.broadcast_to_client({"type": "reset_complete"})
        logger.info("Conversation reset by user")

async def handle_vad(websocket):
    """Handle browser connection to /vad endpoint.
    
    Receives raw audio from the browser and forwards it to the VAD service.
    VAD output is processed by handle_vad_service (not here).
    """
    await manager.register_vad(websocket, is_browser=True)
    await manager.set_service_active("vad", True)
    vad_chunk_id = 0

    try:
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Binary: raw Int16 audio from browser - forward to VAD service
                    vad_chunk_id += 1
                    logger.info(f"Browser -> VAD: audio_chunk chunk_id={vad_chunk_id} size={len(message)} bytes")
                    await manager.set_service_active("vad", True)
                    await manager.broadcast_to_vad({
                        "type": "audio_chunk",
                        "audio": base64.b64encode(message).decode("ascii"),
                        "chunk_id": vad_chunk_id
                    })

            except json.JSONDecodeError:
                logger.error("Invalid JSON from VAD")
    except Exception as e:
        logger.error(f"VAD error: {e}")
    finally:
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
                        await manager.set_service_active("vad", True)
                        await manager.set_service_active("stt", True)
                        await manager.broadcast_to_service("stt", {
                            "type": "audio_chunk",
                            "audio": data.get("audio"),
                            "chunk_id": data.get("chunk_id")
                        })
                    elif msg_type == "vad_result":
                        if data.get("silence"):
                            await manager.set_service_active("vad", True)
                            await manager.set_service_active("stt", True)
                            await manager.broadcast_to_service("stt", {
                                "type": "end_of_speech",
                                "chunk_id": data.get("chunk_id")
                            })
                            await manager.broadcast_to_client({"type": "user_end"})
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
                    
                if msg_type == "partial_transcript":
                    await manager.set_service_active("stt", True)
                    logger.info("STT partial_transcript: '%s'", data.get("text", ""))
                    await manager.broadcast_to_client({
                        "type": "user_partial",
                        "text": data.get("text", ""),
                        "chunk_id": data.get("chunk_id")
                    })
                    
                elif msg_type == "final_transcript":
                    await manager.set_service_active("stt", True)
                    await manager.set_service_active("llm", True)
                    user_text = data.get("text", "")
                    logger.info("STT final_transcript: '%s'", user_text)
                    await manager.broadcast_to_client({
                        "type": "user_transcript",
                        "text": user_text,
                        "final": True,
                        "chunk_id": data.get("chunk_id")
                    })
                    manager.conversation_history.append({"role": "user", "content": user_text})
                    # Forward directly to LLM service (not through browser /llm endpoint)
                    ws_to_send = manager.llm_service_ws if manager.llm_service_ws else manager.llm_ws
                    if ws_to_send and ws_to_send.state.name == "OPEN":
                        await ws_to_send.send(json.dumps({
                            "type": "user_input",
                            "text": user_text,
                            "history": manager.conversation_history,
                        }))
                        logger.info(f"Forwarded user_input to LLM service")
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON from STT")
    except Exception as e:
        logger.error(f"STT error: {e}")
    finally:
        for t in handler_cancel_scope:
            t.cancel()
        await manager.set_service_active("stt", False)
        await manager.unregister_stt(websocket)

async def handle_llm(websocket):
    """Handle client connections to /llm endpoint.
    
    Clients send user_input and receive llm_transcript/llm_audio messages.
    user_input is forwarded to the LLM service on /llm_service.
    """
    await manager.register_llm(websocket, is_service=False)
    
    # Accumulate tokens for complete response
    accumulated_llm_text = ""
    current_turn_id = 0
    llm_service_connected = False

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                # Handle messages from LLM service (llm_start, llm_token, llm_end)
                if msg_type == "llm_start":
                    await manager.set_service_active("llm", True)
                    llm_service_connected = True
                    accumulated_llm_text = ""
                    current_turn_id += 1
                    logger.info(f"[LLM-HANDLER] Broadcasting llm_start turn_id={current_turn_id}")
                    await manager.broadcast_to_client({"type": "llm_start", "turn_id": current_turn_id})
                    
                elif msg_type == "llm_token":
                    await manager.set_service_active("llm", True)
                    token_text = data.get("text", "")
                    accumulated_llm_text += token_text
                    # Show transcript to browser (individual tokens for streaming display)
                    logger.info(f"[LLM-HANDLER] Broadcasting llm_transcript turn_id={current_turn_id} textLen={len(accumulated_llm_text)}")
                    await manager.broadcast_to_client({
                        "type": "llm_transcript",
                        "text": accumulated_llm_text,
                        "partial": True,
                        "turn_id": current_turn_id
                    })
                    
                elif msg_type == "llm_end":
                    await manager.set_service_active("llm", False)
                    await manager.set_service_active("tts", True)
                    logger.info(f"[LLM-HANDLER] Broadcasting llm_end turn_id={current_turn_id} textLen={len(accumulated_llm_text)}")
                    await manager.broadcast_to_client({
                        "type": "llm_end",
                        "turn_id": current_turn_id,
                        "text": accumulated_llm_text,
                    })
                    # Send complete accumulated text to TTS for sentence-level processing
                    if accumulated_llm_text.strip():
                        await manager.forward_message("llm", "tts", {
                            "type": "tts_input",
                            "text": accumulated_llm_text,
                            "partial": False,
                            "turn_id": current_turn_id,
                        })
                        logger.info(f"Sent complete LLM response to TTS: \"{accumulated_llm_text[:80]}...\"")
                    accumulated_llm_text = ""
                    
                # Handle messages from browser/client (user_input)
                elif msg_type == "user_input":
                    # Forward to LLM service for processing
                    ws_to_send = manager.llm_service_ws if manager.llm_service_ws else manager.llm_ws
                    if ws_to_send and ws_to_send.state.name == "OPEN":
                        await ws_to_send.send(json.dumps({
                            "type": "user_input",
                            "text": data.get("text", "")
                        }))
                        logger.info(f"Forwarded user_input to LLM service")
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON from LLM")
    except Exception as e:
        logger.error(f"LLM error: {e}")
    finally:
        await manager.set_service_active("llm", False)
        await manager.unregister_llm(websocket, is_service=False)

async def handle_llm_service(websocket):
    """Handle LLM service connection to /llm_service endpoint.
    
    LLM service sends llm_start, llm_token, llm_end messages.
    Receives user_input from the server.
    """
    await manager.register_llm(websocket, is_service=True)
    await websocket.send(json.dumps({"type": "start"}))
    logger.info("LLM service ready")
    
    accumulated_llm_text = ""
    current_turn_id = 0
    last_heartbeat = {}
    handler_cancel_scope = []
    heartbeat_task = await create_heartbeat_task(last_heartbeat, handler_cancel_scope)
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "llm_start":
                    await manager.set_service_active("llm", True)
                    accumulated_llm_text = ""
                    current_turn_id += 1
                    logger.info(f"[LLM-SERVICE] Broadcasting llm_start turn_id={current_turn_id}")
                    await manager.broadcast_to_client({"type": "llm_start", "turn_id": current_turn_id})
                    
                elif msg_type == "heartbeat":
                    last_heartbeat["llm"] = asyncio.get_event_loop().time()
                    continue
                
                elif msg_type == "llm_token":
                    await manager.set_service_active("llm", True)
                    token_text = data.get("text", "")
                    accumulated_llm_text += token_text
                    logger.info(f"[LLM-SERVICE] Broadcasting llm_transcript turn_id={current_turn_id} textLen={len(accumulated_llm_text)}")
                    await manager.broadcast_to_client({
                        "type": "llm_transcript",
                        "text": accumulated_llm_text,
                        "partial": True,
                        "turn_id": current_turn_id
                    })
                    
                elif msg_type == "llm_end":
                    await manager.set_service_active("llm", False)
                    await manager.set_service_active("tts", True)
                    logger.info(f"[LLM-SERVICE] Broadcasting llm_end turn_id={current_turn_id} textLen={len(accumulated_llm_text)}")
                    await manager.broadcast_to_client({
                        "type": "llm_end",
                        "turn_id": current_turn_id,
                        "text": accumulated_llm_text,
                    })
                    # Send complete accumulated text to TTS for sentence-level processing
                    if accumulated_llm_text.strip():
                        await manager.forward_message("llm", "tts", {
                            "type": "tts_input",
                            "text": accumulated_llm_text,
                            "partial": False,
                            "turn_id": current_turn_id,
                        })
                        logger.info(f"Sent complete LLM response to TTS: \"{accumulated_llm_text[:80]}...\"")
                    accumulated_llm_text = ""
                    
                elif msg_type == "user_input":
                    await manager.set_service_active("llm", True)
                    logger.info(f"LLM received input: \"{data.get('text', '')}\"")
                    if manager.tts_ws and manager.tts_ws.state.name == "OPEN":
                        await manager.tts_ws.send(json.dumps({"type": "stop_generation"}))
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from LLM service: {e}")
    except Exception as e:
        logger.error(f"LLM service error: {e}")
    finally:
        for t in handler_cancel_scope:
            t.cancel()
        await manager.set_service_active("llm", False)
        await manager.unregister_llm(websocket, is_service=True)

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
                    await manager.set_service_active("tts", True)
                    await manager.broadcast_to_client({
                        "type": "llm_audio",
                        "audio": data.get("audio", ""),
                        "turn_id": data.get("turn_id"),
                        "audio_seq": data.get("audio_seq"),
                    })
                    
                elif msg_type == "audio_end":
                    await manager.set_service_active("tts", False)
                    await manager.broadcast_to_client({
                        "type": "audio_complete",
                        "turn_id": data.get("turn_id"),
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
    elif path == "/llm":
        await handle_llm(websocket)
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
        ping_interval=20,
        ping_timeout=20,
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
        logger.info(f"WebSocket endpoints: /vad, /stt, /llm, /tts, /")
        
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        finally:
            server.close()
            await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
