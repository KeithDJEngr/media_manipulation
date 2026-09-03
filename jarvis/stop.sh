#!/bin/bash
# Stop the Jarvis speech-to-speech pipeline
# Usage: ./stop.sh [host] [port]
#   host: Server bind address (default: 0.0.0.0 for LAN access, 127.0.0.1 for localhost only)
#   port: Server port (default: 8765)

echo "========================================"
echo "  Jarvis Speech-to-Speech Pipeline"
echo "========================================"
# Parse arguments (positional args override env vars, then defaults)
HOST=${1:-${HOST:-0.0.0.0}}
PORT=${2:-${PORT:-8765}}

echo "Server: http://${HOST}:${PORT}"
echo "STT Device: ${STT_DEVICE:-auto}"
echo "STT Model: ${STT_MODEL:-nvidia/parakeet-tdt-0.6b-v3}"
echo "LLM API: ${LLM_API_URL:-http://192.168.0.118:8001/chat/completions}"
echo ""

# Check if a server is already running
if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    # Try shutting down services
    # Only match the actual service processes (python running the scripts), not
    # editors/watchers whose command line merely mentions the script names.
    for pid in $(ps aux | grep -E "(python[0-9.]*|pip[0-9.]*).*(jarvis_server|vad_service|stt_service|llm_service|tts_service)\.py" | grep -v grep | awk '{print $2}'); do kill -9 $pid 2>/dev/null; done; sleep 1; ps aux | grep -E "(jarvis_server|vad_service|stt_service|llm_service|tts_service)" | grep -v grep || echo "All stopped"

    #echo "Server already running on port ${PORT}, stopping..."
    #pkill -f "jarvis_server.py" 2>/dev/null || true
    #pkill -f "llm_service.py" 2>/dev/null || true
    #pkill -f "tts_service.py" 2>/dev/null || true
    sleep 2
    if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
        echo "WARNING: port ${PORT} still in use after stop attempt:"
        ss -tlnp | grep ":${PORT} "
    else
        echo "Port ${PORT} free - all services stopped"
    fi
fi

