import requests
import json
import time

# CONFIGURATION
# 1. If using Open WebUI, change this to http://localhost:3000/api/chat
# 2. If using Ollama directly, use localhost:11434
URL = "http://192.168.0.121:8008/chat/completions" 

# Ensure you have a model installed (e.g., 'ollama pull llama3')
MODEL = "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf" 
USER_PROMPT = "Explain streaming tokens in 50 words."


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



def stream_response():
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": USER_PROMPT}
        ],
        "stream": True # The magic flag
    }

    try:
        print(f"Connecting to {URL}...")
        
        # stream=True is essential here
        response = requests.post(URL, json=payload, stream=True)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return

        print("Receiving Stream:\n")
        
        # SSE (Server-Sent Events) format usually splits by newlines
        for line in response.iter_lines():
            
            # Skip empty lines which are common separators in SSE
            if not line: 
                continue
            
            #print(f"{time.time()}: {line}")
            line_str = line.decode('utf-8')
            
            # Strip SSE "data: " prefix if present
            if line_str.startswith('data: '):
                line_str = line_str[6:]
            
            try:
                json_data = json.loads(line_str)
                
                choices = json_data.get('choices', [])
                if choices:
                    delta = choices[0].get('delta', {})
                    token = delta.get('content', ''): # or delta.get('reasoning_content', '')
                    
                    # Print without newline, flush immediately to see live effect
                    if token:
                        print(token, end="", flush=True)
            
            except json.JSONDecodeError:
                print(f"json decode error with {line_str}")
                continue

        print("\n\nDone.")

    except ConnectionError:
        print("Could not connect. Make sure Ollama (or your backend) is running.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    stream_response()
