#!/usr/bin/env python3
"""Simple test script to stream a response from the LLM API."""

import json
import os
import time

import requests

LLM_API_URL = os.getenv("LLM_API_URL", "http://192.168.0.121:8000/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise assistant. Respond naturally to user messages. "
    "Keep responses concise but informative. Use proper punctuation.",
)

USER_MESSAGE = "What is the plot of Merlin?"


def main():
    print(f"Requesting: {USER_MESSAGE}")
    print(f"\nLLM: ", end="", flush=True)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_MESSAGE},
        ],
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "512")),
        "stream": True,
    }

    start_time = time.time()
    token_count = 0
    first_token_time = None
    raw_chunk_count = 0

    response = requests.post(LLM_API_URL, json=payload, stream=True)

    if response.status_code != 200:
        print(f"\nError: {response.status_code} - {response.text}")
        return

    for chunk_bytes in response.iter_content(chunk_size=4096):
        raw_chunk_count += 1
        elapsed = time.time() - start_time

        if raw_chunk_count <= 3:
            print(f"\n[+{elapsed:.3f}s] raw chunk #{raw_chunk_count}: {len(chunk_bytes)} bytes", end="", flush=True)

        data_str = chunk_bytes.decode("utf-8", errors="replace")
        for line in data_str.split("\n"):
            if not line or not line.startswith("data: "):
                continue
            data_str_inner = line[6:].strip()
            if data_str_inner == "[DONE]":
                continue
            try:
                data_obj = json.loads(data_str_inner)
                choices = data_obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "") or delta.get("reasoning_content", "")
                    if content:
                        token_count += 1
                        elapsed = time.time() - start_time
                        if first_token_time is None:
                            first_token_time = elapsed
                        print(f"\n[{elapsed:.3f}s] token #{token_count}: {repr(content[:60])}", end="", flush=True)
                        print(content, end="", flush=True)
            except json.JSONDecodeError:
                continue

    print(f"\n\n[SUCCESS: {time.time()-start_time:.3f}s] {token_count} tokens parsed, {raw_chunk_count} raw chunks received")
    if first_token_time:
        print(f"[first token at: {first_token_time:.3f}s]")


if __name__ == "__main__":
    main()
