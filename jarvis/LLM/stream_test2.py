import httpx
import json
import asyncio

# CONFIGURATION
URL = "http://192.168.0.121:8000/chat/completions"
MODEL = "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
USER_PROMPT = "Explain streaming tokens in 50 words."

async def stream_response_async():
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": USER_PROMPT}
        ],
        "stream": True # The magic flag for the API
    }
    
    try:
        print(f"Connecting to {URL}...")
        
        # Use AsyncClient with a 60s timeout (handles connect, read, and write delays)
        async with httpx.AsyncClient(timeout=60) as client:
            
            # CRITICAL CHANGE: Use .stream("POST", ...) instead of .post()
            # This forces httpx to handle the response as a stream immediately.
            async with client.stream("POST", URL, json=payload) as response:
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    return

                print("Receiving Stream:\n")
                
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
                            # here you select to view the thinking or not
                            token = delta.get('content', '') # or delta.get('reasoning_content', '')
                            
                            if token:
                                print(token, end="", flush=True)
                                
                    except json.JSONDecodeError:
                        continue
                        
                print("\n\nDone.")

    except httpx.ConnectError:
        print("Could not connect. Make sure Ollama is running or check IP address.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(stream_response_async())
