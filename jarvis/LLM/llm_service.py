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
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# System prompt for the LLM
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise assistant. Respond naturally to user messages. "
    "Keep responses concise but informative. Use proper punctuation.",
)


class LLMService:
    """Handles LLM API calls and conversation management."""

    def __init__(self, api_url, model, temperature, max_tokens):
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Conversation history
        self.history = deque(maxlen=20)  # Keep last 20 turns
        self.history.append({"role": "system", "content": SYSTEM_PROMPT})

    def add_user_message(self, text):
        """Add user message to history."""
        self.history.append({"role": "user", "content": text})

    async def generate_response(self, user_text):
        """Generate LLM response for user text.

        Yields text tokens as they become available.
        """
        self.add_user_message(user_text)

        messages = list(self.history)

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "stream": True,
                    },
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code != 200:
                    logger.error(f"LLM API error: {response.status_code} - {response.text}")
                    return

                # Handle streaming response
                text_buffer = ""
                chunk_count = 0

                # Check if response is streaming (SSE format) or single JSON
                first_line = ""
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if not first_line:
                        first_line = line

                    # Parse SSE format: "data: {...}"
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    text_buffer += content
                                    chunk_count += 1
                                    # Send partial token
                                    yield {
                                        "type": "llm_token",
                                        "text": text_buffer,
                                        "partial": True,
                                    }
                        except json.JSONDecodeError:
                            continue

                # If not streaming, parse single JSON response
                if chunk_count == 0:
                    try:
                        json_data = response.json()
                        choices = json_data.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            if content:
                                # Stream character by character for display
                                for i, char in enumerate(content):
                                    text_buffer += char
                                    yield {
                                        "type": "llm_token",
                                        "text": text_buffer,
                                        "partial": True,
                                    }
                                # Final token
                                yield {
                                    "type": "llm_token",
                                    "text": content,
                                    "partial": False,
                                }
                    except (json.JSONDecodeError, KeyError):
                        logger.error("Failed to parse LLM response")

                # Update history with assistant response
                if text_buffer:
                    self.history.append({"role": "assistant", "content": text_buffer})

            #except httpx.TimeoutError:
            #    logger.error("LLM API request timed out")
            except httpx.ConnectError as e:
                logger.error(f"LLM API connection error: {e}")
                logger.error(f"Make sure LLM is running at {self.api_url}")
            except Exception as e:
                logger.error(f"LLM API error: {e}")


async def handle_llm(websocket, llm_service):
    """Handle the LLM WebSocket connection."""
    try:
        # Signal ready immediately
        await websocket.send(json.dumps({"type": "start"}))
        logger.info("LLM service ready")

        # Process incoming messages
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    msg = json.loads(message)
                else:
                    msg = json.loads(message)

                msg_type = msg.get("type")

                if msg_type == "user_input":
                    user_text = msg.get("text", "")
                    if not user_text.strip():
                        continue

                    logger.info(f"LLM received input: \"{user_text}\"")

                    # Send start signal
                    await websocket.send(json.dumps({"type": "llm_start"}))

                    # Generate response and stream tokens
                    async for token_data in llm_service.generate_response(user_text):
                        logger.info(f"received {token_data}")
                        await websocket.send(json.dumps(token_data))

                    # Send end signal
                    await websocket.send(json.dumps({"type": "llm_end"}))
                    logger.info(f"LLM generation complete: {token_data}\n    {type(token_data)}")

                elif msg_type == "reset":
                    # Reset conversation history
                    llm_service.history.clear()
                    llm_service.history.append({"role": "system", "content": SYSTEM_PROMPT})
                    logger.info("LLM conversation reset")

                elif msg_type == "set_system_prompt":
                    new_prompt = msg.get("prompt", "")
                    if new_prompt:
                        llm_service.history[0] = {"role": "system", "content": new_prompt}
                        logger.info("System prompt updated")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from server: {e}")
            except Exception as e:
                logger.error(f"LLM processing error: {e}")
                raise

    except websockets.ConnectionClosed as e:
        logger.info(f"LLM connection closed: {e}")
    except Exception as e:
        logger.error(f"LLM connection error: {e}")


async def main():
    server_host = os.getenv("SERVER_HOST", "127.0.0.1")
    server_port = int(os.getenv("SERVER_PORT", "8765"))
    llm_url = f"wss://{server_host}:{server_port}/llm_service"

    # Create LLM service
    llm_service = LLMService(
        api_url=LLM_API_URL,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    # Connect to server
    logger.info(f"Connecting to LLM endpoint at {llm_url}")
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    try:
        async with websockets.connect(llm_url, ssl=ssl_context) as websocket:
            await websocket.send(json.dumps({"type": "start"}))
            logger.info("LLM connected to server")
            await handle_llm(websocket, llm_service)
    except ConnectionRefusedError:
        logger.error(
            f"Could not connect to server at {llm_url}. "
            f"Make sure the server is running on port {server_port}."
        )
    except Exception as e:
        logger.error(f"Failed to connect: {e}")


if __name__ == "__main__":
    asyncio.run(main())
