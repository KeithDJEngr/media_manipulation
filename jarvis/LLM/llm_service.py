#!/usr/bin/env python3
"""LLM service that connects to an external LLM API.
Connects to the Jarvis server at ws://localhost:8765/llm and handles
chat completions via the OpenAI-compatible API at LLM_API_URL.
Protocol (server -> client):
  - user_input: User text from STT
LLM output sent to server (and forwarded to TTS/browser):
  - llm_start: LLM began generation
  - llm_token: Partial text token with partial=True (streaming)
  - llm_token: Final text with partial=False
  - llm_end: Generation complete
"""
import asyncio
import json
import logging
import os
import ssl
import sys
from collections import deque
import websockets
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# External LLM API configuration
# My custom setup
#LLM_API_URL = os.getenv("LLM_API_URL", "http://192.168.0.118:8001/chat/completions")
#LLM_API_URL = os.getenv("LLM_API_URL", "http://192.168.0.121:8001/chat/completions")
#LLM_MODEL = os.getenv("LLM_MODEL", "dummy")

# Basic setup
LLM_API_URL = os.getenv("LLM_API_URL", "http://192.168.0.118:8000/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
# Qwen3's thinking mode consumes the max_tokens budget in reasoning_content BEFORE
# any content token is emitted, so a 1024 budget exhausted entirely in thinking and
# returned an empty (silent) reply with finish_reason="length". 2048 leaves headroom
# for a full spoken answer with thinking disabled.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
# Disable Qwen3 thinking mode by default (see above). Set to 0 to re-enable thinking
# (in which case also raise LLM_MAX_TOKENS, e.g. 4096, or the budget runs out in thinking).
LLM_DISABLE_THINKING = os.getenv("LLM_DISABLE_THINKING", "1") != "0"

use_full_history=True

# Manage stopping llm generation
continue_generating=True

# System prompt for the LLM
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise assistant. Respond naturally to user messages. "
    "Keep responses concise but informative and conversational.",
)

async def handle_llm(websocket, llm_info):
    """Handle the LLM WebSocket connection."""
    pre_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    conversation_history = []

    async def send_heartbeats():
        while True:
            await asyncio.sleep(5)
            try:
                await websocket.send(json.dumps({"type": "heartbeat"}))
            except Exception:
                break

    heartbeat_task = asyncio.create_task(send_heartbeats())

    # Incoming messages are consumed by a dedicated pump task instead of this
    # loop reading the websocket directly. While a turn is generating, this
    # loop is blocked inside the LLM API stream and cannot see incoming
    # messages; the pump observes a "stop" the moment it arrives and flips
    # continue_generating immediately, so a barge-in (user interrupt / Stop
    # button) kills the in-flight stream at the next token boundary instead of
    # letting the whole response finish and leak into TTS. A None sentinel
    # means the pump saw the websocket close (or error) and ends this loop.
    close_reason = None
    incoming_queue = asyncio.Queue()

    async def pump_messages():
        """Read the websocket and enqueue parsed messages for the main loop."""
        nonlocal close_reason
        global continue_generating
        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    logger.error(f"LLM-SERVICE: invalid JSON from server: {str(raw)[:100]}")
                    continue
                if isinstance(msg, dict) and msg.get("type") == "stop" and continue_generating:
                    # Barge-in: flip immediately without waiting for the main
                    # loop, which may be blocked for the whole turn. The main
                    # loop's own stop branch (idempotent) handles dispatch.
                    logger.info("LLM-SERVICE: stop received - interrupting in-flight generation")
                    continue_generating = False
                await incoming_queue.put(msg)
        except websockets.ConnectionClosed as e:
            close_reason = getattr(e, "reason", None) or str(e)
        except Exception as e:
            logger.error(f"LLM-SERVICE: pump error: {e}")
        finally:
            await incoming_queue.put(None)

    pump_task = asyncio.create_task(pump_messages())

    global continue_generating
    try:
        # Signal ready immediately
        await websocket.send(json.dumps({"type": "start"}))
        await websocket.send(json.dumps({
            "type": "service_status",
            "service": "llm",
            "status": "connected",
        }))
        logger.info("LLM service ready")

        # Process incoming messages (parsed dicts from the pump task;
        # a None sentinel means the pump saw the websocket close or error)
        while True:
            try:
                msg = await incoming_queue.get()
                if msg is None:
                    break

                msg_type = msg.get("type")

                if msg_type == "set_system_prompt":
                    new_prompt = msg.get("prompt", "")
                    if new_prompt:
                        pre_history = [{"role": "system", "content": new_prompt}]
                        logger.info(f"System prompt updated to: {new_prompt[:80]}...")
                    continue

                if msg_type == "stop":
                    continue_generating=False
                    logger.info(f"Interrupted generation")
                    continue

                if msg_type == "reset_history":
                    logger.info(f"Resetting LLM conversation history")
                    conversation_history[:] = pre_history
                    continue

                if msg_type == "set_settings":
                    if "use_full_history" in msg:
                        global use_full_history
                        use_full_history = msg["use_full_history"]
                        logger.info(f"Message history mode set to: {'full' if use_full_history else 'latest only'}")
                    continue

                if msg_type == "user_input":
                    user_text = msg.get("text", "")
                    logger.info(f"LLM-SERVICE <<< user_input text='{user_text[:80]}...'")
                    if not user_text.strip():
                        continue
                        
                    is_partial = msg.get("partial", False)
                    
                    # Accept history from server if provided
                    if "history" in msg:
                        new_history = msg["history"]
                        logger.debug(f"LLM-SERVICE - got new history from server: {new_history[1:]}")
                        # Keep system prompt, replace rest
                        if len(new_history) > 0 and new_history[0].get("role") == "system":
                            conversation_history[:] = pre_history + new_history[1:]
                        else:
                            conversation_history[:] = pre_history + new_history
                    
                    logger.info(f"LLM-SERVICE <<< user_input (partial={is_partial}, text='{user_text[:80]}...')")
                    
                    # For partials, just update history without generating a response
                    if is_partial:
                        continue

                    logger.debug(f"LLM-SERVICE >>> conversation_history for LLM: {json.dumps(conversation_history, ensure_ascii=False, indent=2)}")
                    
                    # Send start signal
                    logger.info(f"LLM-SERVICE: sending llm_start (user_text='{user_text[:80]}...')")
                    await websocket.send(json.dumps({"type": "llm_start"}))
                    
                    last_token = None
                    llm_msg = ""

                    # Generate response and stream tokens
                    #llm_service.add_user_message(user_text)
                    #async for token in llm_service.generate_response(user_text):

                    if use_full_history:
                        messages = conversation_history.copy()
                    else:
                        # Only use system prompt and latest user/assistant exchange
                        messages = [conversation_history[0]]  # System prompt
                        # Find the last user message
                        for i in range(len(conversation_history) - 1, -1, -1):
                            if conversation_history[i]["role"] == "user":
                                messages.append(conversation_history[i])
                                # Include the next assistant message if it exists
                                if i + 1 < len(conversation_history) and conversation_history[i + 1]["role"] == "assistant":
                                    messages.append(conversation_history[i + 1])
                                break
                        logger.info(f"Using latest-only mode: {len(messages)} messages")

                    payload = {
                        "model": llm_info['model'],
                        "messages": messages,
                        "stream": True,
                        "temperature": llm_info.get('temperature', LLM_TEMPERATURE),
                        "max_tokens": llm_info.get('max_tokens', LLM_MAX_TOKENS),
                    }
                    if LLM_DISABLE_THINKING:
                        # Qwen3's llama.cpp server ignores the OpenAI-style "thinking" key
                        # (verified: reasoning_content still streams with it set) but honors
                        # the chat-template flag. Without enable_thinking=false the model
                        # spends the entire max_tokens budget in reasoning_content and
                        # returns an empty reply with finish_reason="length" -> silence.
                        # Both keys are sent: "thinking" for OpenAI-compatible backends
                        # that honor it, chat_template_kwargs for llama.cpp's Qwen3 template.
                        payload["thinking"] = {"type": "disabled"}
                        payload["chat_template_kwargs"] = {"enable_thinking": False}

                    continue_generating=True
                    try:
                        logger.info(f"Connecting to {llm_info['api_url']}...")
                        llm_msg=""

                        # Use AsyncClient with a timeout (handles connect, read, and write delays)
                        # 30s is the *between-reads* limit on the stream: a stalled
                        # upstream API is caught quickly and the existing error path
                        # below sends the spoken fallback + llm_end, so the user's
                        # turn resolves instead of hanging for 4 minutes.
                        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:

                            # CRITICAL CHANGE: Use .stream("POST", ...) instead of .post()
                            # This forces httpx to handle the response as a stream immediately.
                            token_count = 0
                            reasoning_chars = 0
                            finish_reason = None
                            interrupted = False
                            async with client.stream("POST", llm_info['api_url'], json=payload) as response:
                                if response.status_code != 200:
                                    logger.error(f"LLM API error: {response.status_code}")
                                    break

                                logger.info("Receiving Stream:\n")

                                # Read all data at once to avoid aiter_lines() dropping the last line(s)
                                # when the server closes the connection before fully flushing its buffer

                                #raw_data = await response.aread()
                                #text_data = raw_data.decode("utf-8", errors="replace")
                                #lines = text_data.split("\n")

                                #for line in lines:
                                async for line in response.aiter_lines():
                                    if not continue_generating:
                                        interrupted = True
                                        continue_generating=True
                                        break
                                    if not line:
                                        continue

                                    try:
                                        # Handle both Ollama ("data: {json}") and OpenAI ("{json}") formats
                                        data = line.strip()
                                        if data.startswith("data:"):
                                            data = data[5:].strip()
                                        if not data:
                                            continue
                                        if data == "[DONE]":
                                            # OpenAI-style end-of-stream sentinel (llama.cpp sends it too);
                                            # not JSON, so it must be handled before json.loads
                                            break

                                        json_data = json.loads(data)

                                        choices = json_data.get('choices', [])
                                        if choices:
                                            choice = choices[0]
                                            delta = choice.get('delta', {})
                                            fr = choice.get('finish_reason')
                                            if fr:
                                                finish_reason = fr

                                            # reasoning_content is the model's internal "thinking" (Qwen3).
                                            # Count it for diagnostics only - never stream it to the
                                            # client/TTS, it is not conversational text.
                                            reasoning = delta.get('reasoning_content') or ''
                                            if reasoning:
                                                reasoning_chars += len(reasoning)
                                                logger.debug(f"reasoning: {reasoning[:60]}")

                                            # delta.get('content') is None on the first chunk;
                                            # 'or' normalizes it to '' so it is skipped cleanly
                                            token = delta.get('content') or ''

                                            if token:
                                                llm_msg+=token
                                                token_count += 1
                                                logger.debug(f"content: {token}")
                                                try:
                                                    await websocket.send(json.dumps({"type": "llm_token", "text": token, "partial": True}))
                                                except websockets.ConnectionClosed:
                                                    logger.info("WebSocket closed during token streaming")
                                                    break

                                    except json.JSONDecodeError:
                                        continue

                                # Post-stream diagnostics: finish_reason="length" means the
                                # max_tokens budget ran out. With zero content tokens the whole
                                # budget was spent on thinking (Qwen3) and the user got silence.
                                if response.status_code == 200:
                                    if finish_reason == "length":
                                        if token_count == 0:
                                            logger.error(
                                                f"LLM produced NO content: max_tokens budget exhausted entirely "
                                                f"in reasoning_content ({reasoning_chars} chars, finish_reason='length'). "
                                                f"Set LLM_DISABLE_THINKING=1 (default) or raise LLM_MAX_TOKENS."
                                            )
                                        else:
                                            logger.warning(
                                                f"LLM response TRUNCATED at max_tokens "
                                                f"({token_count} content tokens, {reasoning_chars} reasoning chars, finish_reason='length')"
                                            )
                                    else:
                                        logger.info(f"Done. finish_reason={finish_reason}, content tokens={token_count}, reasoning chars={reasoning_chars}")

                    except httpx.ConnectError:
                        logger.error("Could not connect. Make sure the LLM server (llama-server) is running or check IP address.")
                    except Exception as e:
                        logger.error(f"An error occurred: {e}")


                    # TODO: clear when it's validated this works without
                    # Add to conversation history (user message already in history from server)
                    #conversation_history.append({"role": "assistant", "content": llm_msg})
                    # Trim history to keep last 20 messages (system prompt + 19 turns) to manage context window
                    #if len(conversation_history) > 21:
                    #    conversation_history = conversation_history[-19:]

                    # The model produced nothing: budget exhausted in thinking mode, an empty
                    # completion, or an API/connection failure. Never leave the user in
                    # silence - speak a fallback so the turn is audible, and it stays in the
                    # conversation history as context for the next turn.
                    if not llm_msg.strip() and not interrupted:
                        logger.error("LLM generated empty content - sending spoken fallback instead of silence")
                        llm_msg = "Sorry, I couldn't generate an answer to that. Could you say it again, please?"
                        try:
                            await websocket.send(json.dumps({"type": "llm_token", "text": llm_msg, "partial": True}))
                        except websockets.ConnectionClosed:
                            logger.info("WebSocket closed while sending fallback message")

                    # Send end signal with accumulated text
                    logger.info(f"LLM-SERVICE: sending llm_end (total_text_len={len(llm_msg)}, text='{llm_msg[:100]}...')")
                    try:
                        await websocket.send(json.dumps({
                            "type": "llm_end",
                            "text": llm_msg,
                            "interrupted": interrupted,
                        }))
                    except websockets.ConnectionClosed:
                        logger.info("WebSocket closed before sending llm_end")

            except Exception as e:
                logger.error(f"LLM processing error: {e}")
                raise
        # The pump terminates when the websocket closes; mirror the close
        # logging (and best-effort status) the old inline async-for path used
        # to produce here. The except handlers below still cover failures that
        # surface during a send.
        if close_reason is not None:
            if "1011" in close_reason or "keepalive" in close_reason.lower():
                logger.info(f"LLM connection closed (ping timeout): {close_reason}")
            else:
                logger.info(f"LLM connection closed: {close_reason}")
            try:
                await websocket.send(json.dumps({
                    "type": "service_status",
                    "service": "llm",
                    "status": "disconnected",
                }))
            except Exception:
                pass

    except websockets.ConnectionClosed as e:
        reason = getattr(e, 'reason', str(e)) if hasattr(e, 'reason') else str(e)
        # Only send status if we're still connected
        try:
            await websocket.send(json.dumps({
                "type": "service_status",
                "service": "llm",
                "status": "disconnected",
            }))
        except Exception:
            pass
        if "1011" in reason or "keepalive" in reason.lower():
            logger.info(f"LLM connection closed (ping timeout): {reason}")
        else:
            logger.info(f"LLM connection closed: {reason}")
    except Exception as e:
        logger.error(f"LLM connection error: {e}")
        try:
            await websocket.send(json.dumps({
                "type": "service_status",
                "service": "llm",
                "status": "error",
                "error": str(e)[:100],
            }))
        except Exception:
            pass
    finally:
        heartbeat_task.cancel()
        pump_task.cancel()
        try:
            await heartbeat_task
        except (asyncio.CancelledError, Exception):
            pass
        try:
            await pump_task
        except (asyncio.CancelledError, Exception):
            pass

async def main():
    server_host = os.getenv("SERVER_HOST", "127.0.0.1")
    server_port = int(os.getenv("SERVER_PORT", "8765"))
    service_llm_socket = f"wss://{server_host}:{server_port}/llm_service"

    # Create LLM service
    llm_info = {
            "api_url":LLM_API_URL,
            "model":LLM_MODEL,
            "temperature":LLM_TEMPERATURE,
            "max_tokens":LLM_MAX_TOKENS,
            }

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    while True:
        try:
            async with websockets.connect(service_llm_socket, ssl=ssl_context) as websocket:
                await websocket.send(json.dumps({"type": "start"}))
                await websocket.send(json.dumps({
                    "type": "service_status",
                    "service": "llm",
                    "status": "connected",
                }))
                logger.info("LLM connected to server")
                await handle_llm(websocket, llm_info)
                await websocket.send(json.dumps({
                    "type": "service_status",
                    "service": "llm",
                    "status": "disconnected",
                }))


        # Potentially use later
        #try:
        #    async with asyncio.timeout(10):
        #        async with websockets.connect(service_llm_socket, ssl=ssl_context) as websocket:
        #            await websocket.send(json.dumps({"type": "start"}))
        #            await websocket.send(json.dumps({
        #                "type": "service_status",
        #                "service": "llm",
        #                "status": "connected",
        #            }))
        #            logger.info("LLM connected to server")
        #            await handle_llm(websocket, llm_info)
        #except websockets.ConnectionClosed as e:
        #    logger.info(f"LLM connection closed: {e}, reconnecting...")
        #    try:
        #        async with asyncio.timeout(10):
        #            async with websockets.connect(service_llm_socket, ssl=ssl_context) as ws:
        #                await ws.send(json.dumps({
        #                    "type": "service_status",
        #                    "service": "llm",
        #                    "status": "disconnected",
        #                }))


        #    except Exception:
        #        pass
        except websockets.ConnectionClosed as e:
            logger.info(f"LLM connection closed: {e}, reconnecting...")
        except asyncio.TimeoutError:
            logger.error("LLM handshake timed out, server may be busy")
        except ConnectionRefusedError:
            logger.error(
                f"Could not connect to server at {service_llm_socket}. "
                f"Make sure the server is running on port {server_port}."
            )
        except Exception as e:
            logger.error(f"Failed to connect: {e}")

        logger.info(f"Reconnecting in 2 seconds...")
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
