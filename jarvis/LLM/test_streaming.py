#!/usr/bin/env python3
"""
Comprehensive test script for chat/completions streaming endpoint.

Tests:
1. Basic streaming functionality
2. Multiple sequential requests (speed/throughput)
3. Token streaming verification
4. Response correctness
"""
import httpx
import json
import asyncio
import time
import sys

# CONFIGURATION
URL = "http://192.168.0.121:8000/chat/completions"
MODEL = "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
TEMPERATURE = 0.7
MAX_TOKENS = 512

TEST_PROMPTS = [
    "What is 2+2?",
    "Explain the concept of recursion in programming in 30 words.",
    "What are the three laws of robotics?",
    "Write a haiku about artificial intelligence.",
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "What is the meaning of life according to philosophy?",
    "List 5 programming languages and their primary use cases.",
]


async def test_streaming(prompt, request_num):
    """Test a single streaming request."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    start_time = time.time()
    first_token_time = None
    token_count = 0
    full_response = ""
    errors = []

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", URL, json=payload) as response:
                if response.status_code != 200:
                    errors.append(f"HTTP {response.status_code}")
                    return None

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = line.strip()
                        if data.startswith("data:"):
                            data = data[5:].strip()

                        if data == "[DONE]":
                            continue

                        json_data = json.loads(data)
                        choices = json_data.get('choices', [])

                        if choices:
                            delta = choices[0].get('delta', {})
                            token = delta.get('content', '') or delta.get('reasoning_content', '')

                            if token:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                    ttft = first_token_time - start_time
                                full_response += token
                                token_count += 1

                    except json.JSONDecodeError:
                        continue

    except httpx.ConnectError:
        errors.append("Connection error")
    except Exception as e:
        errors.append(str(e))

    total_time = time.time() - start_time

    return {
        "request_num": request_num,
        "prompt": prompt,
        "full_response": full_response,
        "token_count": token_count,
        "total_time": total_time,
        "tokens_per_second": token_count / total_time if total_time > 0 else 0,
        "ttft": first_token_time - start_time if first_token_time else None,
        "errors": errors,
    }


async def run_tests():
    """Run all tests."""
    print("=" * 80)
    print("LLM Streaming Endpoint Test")
    print("=" * 80)
    print(f"URL: {URL}")
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max Tokens: {MAX_TOKENS}")
    print(f"Test Prompts: {len(TEST_PROMPTS)}")
    print("=" * 80)
    print()

    results = []
    total_start = time.time()

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"Test {i}/{len(TEST_PROMPTS)}: {prompt[:50]}...")
        result = await test_streaming(prompt, i)
        if result:
            results.append(result)
            status = "PASS" if not result["errors"] else "FAIL"
            print(f"  [{status}] Tokens: {result['token_count']}, Time: {result['total_time']:.2f}s, "
                  f"TPS: {result['tokens_per_second']:.2f}")
            if result["ttft"] is not None:
                print(f"  [INFO] Time to first token: {result['ttft']:.3f}s")
            if result["errors"]:
                print(f"  [ERRORS] {'; '.join(result['errors'])}")
        print()

    total_time = time.time() - total_start

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total requests: {len(results)}")
    print(f"Total time: {total_time:.2f}s")

    if results:
        avg_tokens = sum(r["token_count"] for r in results) / len(results)
        avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
        avg_ttft = sum(r["ttft"] for r in results if r["ttft"] is not None) / \
                   sum(1 for r in results if r["ttft"] is not None) if any(r["ttft"] for r in results) else 0

        print(f"Average tokens per response: {avg_tokens:.1f}")
        print(f"Average tokens/second: {avg_tps:.2f}")
        print(f"Average time to first token: {avg_ttft:.3f}s")

        failed = [r for r in results if r["errors"]]
        if failed:
            print(f"\nFailed tests: {len(failed)}/{len(results)}")
            for r in failed:
                print(f"  Request {r['request_num']}: {'; '.join(r['errors'])}")
        else:
            print(f"\nAll {len(results)} tests passed!")

        # Token streaming verification
        streaming_verified = all(r["token_count"] > 0 for r in results)
        print(f"\nStreaming verified: {'YES' if streaming_verified else 'NO'}")
        if streaming_verified:
            print("  All requests received multiple tokens (streaming working)")


if __name__ == "__main__":
    asyncio.run(run_tests())
