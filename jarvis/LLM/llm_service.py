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
LLM_API_URL = os.getenv("LLM_API_URL", "http://192.168.0.121:8000/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# System prompt for the LLM
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise assistant. Respond naturally to user messages. "
    "Keep responses concise but informative and conversational.",
)


async def handle_llm(websocket, llm_info):
    """Handle the LLM WebSocket connection."""
    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    async def send_heartbeats():
        while True:
            await asyncio.sleep(5)
            try:
                await websocket.send(json.dumps({"type": "heartbeat"}))
            except Exception:
                break

    heartbeat_task = asyncio.create_task(send_heartbeats())
    
    try:
        # Signal ready immediately
        await websocket.send(json.dumps({"type": "start"}))
        await websocket.send(json.dumps({
            "type": "service_status",
            "service": "llm",
            "status": "connected",
        }))
        logger.info("LLM service ready")

        # Process incoming messages
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    msg = json.loads(message)
                else:
                    msg = json.loads(message)

                msg_type = msg.get("type")

                if msg_type == "reset_history":
                    logger.info("LLM history reset")
                    conversation_history = [
                        {"role": "system", "content": SYSTEM_PROMPT}
                    ]
                    continue

                if msg_type == "user_input":
                    user_text = msg.get("text", "")
                    if not user_text.strip():
                        continue
                        
                    is_partial = msg.get("partial", False)
                    
                    # Accept history from server if provided
                    if "history" in msg:
                        new_history = msg["history"]
                        # Keep system prompt, replace rest
                        if len(new_history) > 0 and new_history[0].get("role") == "system":
                            conversation_history = new_history.copy()
                        else:
                            conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
                            conversation_history.extend(new_history)
                    
                    logger.info(f"LLM-SERVICE <<< user_input (partial={is_partial}, text='{user_text[:80]}...')")
                    
                    # For partials, just update history without generating a response
                    if is_partial:
                        continue

                    logger.info(f"LLM-SERVICE >>> conversation_history for LLM: {json.dumps(conversation_history, ensure_ascii=False, indent=2)}")
                    
                    # Send start signal
                    logger.info(f"LLM-SERVICE: sending llm_start (user_text='{user_text[:80]}...')")
                    await websocket.send(json.dumps({"type": "llm_start"}))
                    
                    last_token = None
                    llm_msg = ""

                    # Generate response and stream tokens
                    #llm_service.add_user_message(user_text)
                    #async for token in llm_service.generate_response(user_text):













                    # Build messages list with full conversation history (user text already in history from server)
                    messages = conversation_history.copy()

                    payload = {
                        "model": llm_info['model'],
                        "messages": messages,
                        "stream": True,
                        "temperature": llm_info.get('temperature', LLM_TEMPERATURE),
                        "max_tokens": llm_info.get('max_tokens', LLM_MAX_TOKENS),
                    }

                    try:
                        logger.info(f"Connecting to {llm_info['api_url']}...")
                        llm_msg=""

                        # Use AsyncClient with a 60s timeout (handles connect, read, and write delays)
                        async with httpx.AsyncClient(timeout=60) as client:

                            # CRITICAL CHANGE: Use .stream("POST", ...) instead of .post()
                            # This forces httpx to handle the response as a stream immediately.
                            token_count = 0
                            async with client.stream("POST", llm_info['api_url'], json=payload) as response:
                                if response.status_code != 200:
                                    logger.info(f"Error: {response.status_code}")
                                    return

                                logger.info("Receiving Stream:\n")

                                # aiter_lines is the non-blocking equivalent of requests' iter_lines
                                async for line in response.aiter_lines():
                                    if not line:
                                        continue

                                    try:
                                        # Handle both Ollama ("data: {json}") and OpenAI ("{json}") formats
                                        data = line.strip()
                                        if data.startswith("data:"):
                                            data = data[5:].strip()

                                        json_data = json.loads(data)

                                        choices = json_data.get('choices', [])
                                        if choices:
                                            delta = choices[0].get('delta', {})

                                            # Support standard content or reasoning tags (common in newer models)
                                            token = delta.get('content', '') or delta.get('reasoning_content', '')

                                            if token:
                                                llm_msg+=token
                                                token_count += 1
                                                if token_count <= 3 or token_count % 20 == 0:
                                                    logger.info(f"LLM-SERVICE: token #{token_count}: '{token}' (accumulated so far: '{llm_msg[:80]}...')")
                                                await websocket.send(json.dumps({"type": "llm_token", "text": token, "partial": True}))

                                    except json.JSONDecodeError:
                                        continue

                                logger.info(f"\n\nDone. Total tokens streamed: {token_count}")

                    except httpx.ConnectError:
                        logger.info("Could not connect. Make sure Ollama is running or check IP address.")
                    except Exception as e:
                        logger.info(f"An error occurred: {e}")





                    #    await websocket.send(json.dumps({"type": "llm_token", "text": token, "partial": True}))
                    #    logger.info(f"LLM token: {token}")
                    #    last_token = token
                    #if last_token:
                    #    llm_service.history.append({"role": "assistant", "content": last_token})



                    # Add to conversation history (user message already in history from server)
                    conversation_history.append({"role": "assistant", "content": llm_msg})

                    # Trim history to keep last 20 messages (system prompt + 19 turns) to manage context window
                    if len(conversation_history) > 21:
                        conversation_history = conversation_history[:2] + conversation_history[-19:]

                    # Send end signal with accumulated text
                    logger.info(f"LLM-SERVICE: sending llm_end (total_text_len={len(llm_msg)}, text='{llm_msg[:100]}...')")
                    await websocket.send(json.dumps({
                        "type": "llm_end",
                        "text": llm_msg,
                    }))

                    # Log generation status
                    if llm_msg:
                        logger.info(f"LLM generation complete: {llm_msg[:100]}...")
                    else:
                        logger.info("LLM generation complete (empty response)")

                #elif msg_type == "reset":
                #    # Reset conversation history
                #    llm_service.history.clear()
                #    llm_service.history.append({"role": "system", "content": SYSTEM_PROMPT})
                #    logger.info("LLM conversation reset")
                #
                #elif msg_type == "set_system_prompt":
                #    new_prompt = msg.get("prompt", "")
                #    if new_prompt:
                #        llm_service.history[0] = {"role": "system", "content": new_prompt}
                #        logger.info("System prompt updated")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from server: {e}")
            except Exception as e:
                logger.error(f"LLM processing error: {e}")
                raise

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
        try:
            await heartbeat_task
        except asyncio.CancelledError:
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
                logger.info("LLM service handler completed, reconnecting...")
        except websockets.ConnectionClosed as e:
            logger.info(f"LLM connection closed: {e}, reconnecting...")
            try:
                async with websockets.connect(service_llm_socket, ssl=ssl_context) as ws:
                    await ws.send(json.dumps({
                        "type": "service_status",
                        "service": "llm",
                        "status": "disconnected",
                    }))
            except Exception:
                pass
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
