#!/bin/bash
# Start the Jarvis speech-to-speech pipeline
# Usage: ./start.sh [host] [port]
#   host: Server bind address (default: 0.0.0.0 for LAN access, 127.0.0.1 for localhost only)
#   port: Server port (default: 8765)
#
# Environment variables:
#   PORT: Server port (default: 8765)
#   HOST: Server bind address (default: 0.0.0.0)
#   STT_DEVICE: Device for STT model ('cpu' or 'cuda', auto-detects if not set)
#   TTS_DEVICE: Device for TTS model ('cpu' for Intel GPU/CPU, 'cuda:0' for NVIDIA GPU)
#   TTS_MODEL_TIER: TTS model size ('0.6B-Base' fastest, '0.6B-CustomVoice' balanced, '1.7B' best quality, default: 1.7B)
#   STT_MODEL: Parakeet model name (default: nvidia/parakeet-tdt-0.6b-v3)
#   LLM_API_URL: External LLM API URL (default: http://192.168.0.121:8000/chat/completions)
#   LLM_MODEL: LLM model name for API calls (default: dummy)

# Parse arguments
#HOST=${1:-${HOST:-0.0.0.0}}
IP=$(hostname -i | awk '{print $1}')
echo "IP: $IP"
HOST=${1:-${IP:-0.0.0.0}}
PORT=${2:-${PORT:-8765}}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
/bin/bash stop.sh $HOST $PORT || true

echo "========================================"
echo "  Jarvis Speech-to-Speech Pipeline"
echo "========================================"
echo "Server: https://${HOST}:${PORT}"
echo "STT Device: ${STT_DEVICE:-auto}"
echo "STT Model: ${STT_MODEL:-nvidia/parakeet-tdt-0.6b-v3}"
echo "LLM API: ${LLM_API_URL:-http://192.168.0.121:8000/chat/completions}"
echo ""

# Dependencies required
if ! command -v mkcert &> /dev/null; then
    echo "Installing mkcert..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y mkcert
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y mkcert
    elif command -v yum &> /dev/null; then
        sudo yum install -y mkcert
    elif command -v brew &> /dev/null; then
        brew install mkcert
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm mkcert
    else
        echo "WARNING: Could not install mkcert automatically."
        echo "Please install mkcert manually: https://github.com/FiloSottile/mkcert"
        echo "Continuing without SSL certificates..."
    fi
else
    echo "mkcert already installed"
fi

# Create venv if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

# Install dependencies
echo "Installing dependencies..."
uv pip install -q -r Server/requirements.txt
uv pip install -q -r STT/requirements.txt

# Trap to kill all services on exit
SERVER_PID=""
VAD_PID=""
STT_PID=""
LLM_PID=""
TTS_PID=""

cleanup() {
    echo ""
    echo "Stopping all services..."
    [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null || true
    [ -n "$VAD_PID" ] && kill $VAD_PID 2>/dev/null || true
    [ -n "$STT_PID" ] && kill $STT_PID 2>/dev/null || true
    [ -n "$LLM_PID" ] && kill $LLM_PID 2>/dev/null || true
    [ -n "$TTS_PID" ] && kill $TTS_PID 2>/dev/null || true
    sleep 1
    [ -n "$SERVER_PID" ] && kill -9 $SERVER_PID 2>/dev/null || true
    [ -n "$VAD_PID" ] && kill -9 $VAD_PID 2>/dev/null || true
    [ -n "$STT_PID" ] && kill -9 $STT_PID 2>/dev/null || true
    [ -n "$LLM_PID" ] && kill -9 $LLM_PID 2>/dev/null || true
    [ -n "$TTS_PID" ] && kill -9 $TTS_PID 2>/dev/null || true
    echo "All services stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM


# Start server
# Generate SSL cert for localhost and local IP
if [[ ! -e "localhost+2.pem" ]]; then
    mkcert -install
    CERT_ARGS=("localhost" "127.0.0.1" "::1")
    
    if [ -n "$IP" ]; then
        CERT_ARGS+=("$IP")
    fi
    mkcert "${CERT_ARGS[@]}"
fi


# Configure nginx/caddy to terminate TLS and proxy :8765
echo "Starting Jarvis server on port ${PORT}..."
SERVER_HOST=${HOST} SERVER_PORT=${PORT} nohup .venv/bin/python Server/jarvis_server.py > /tmp/jarvis_server.log 2>&1 &
SERVER_PID=$!
disown $SERVER_PID

# Wait for server to be ready
echo "Waiting for server to start..."
for i in $(seq 1 20); do
    if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
        echo "Server started on port ${PORT}"
        break
    fi
    sleep 1
done

if ! ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo "ERROR: Server failed to start. Check /tmp/jarvis_server.log"
    cat /tmp/jarvis_server.log
    exit 1
fi

# Start VAD service
echo "Starting VAD service..."
SERVER_HOST=${HOST} SERVER_PORT=${PORT} nohup .venv/bin/python STT/vad_service.py > /tmp/jarvis_vad.log 2>&1 &
VAD_PID=$!
disown $VAD_PID
sleep 1

# Start STT service (this will load the Parakeet model, which takes time)
echo "Starting STT service (loading Parakeet model)..."
STT_MODEL_DEFAULT=${STT_MODEL:-nvidia/parakeet-tdt-0.6b-v3}
STT_DEVICE_DEFAULT=${STT_DEVICE:-auto}
SERVER_HOST=${HOST} SERVER_PORT=${PORT} STT_DEVICE=${STT_DEVICE_DEFAULT} STT_MODEL=${STT_MODEL_DEFAULT} nohup .venv/bin/python STT/stt_service.py > /tmp/jarvis_stt.log 2>&1 &
STT_PID=$!
disown $STT_PID

# Start LLM service
echo "Starting LLM service..."
SERVER_HOST=${HOST} SERVER_PORT=${PORT} LLM_API_URL=${LLM_API_URL:-http://192.168.0.121:8000/chat/completions} LLM_MODEL=${LLM_MODEL:-dummy} nohup .venv/bin/python LLM/llm_service.py > /tmp/jarvis_llm.log 2>&1 &
LLM_PID=$!
disown $LLM_PID
sleep 2

# Start TTS service
echo "Starting TTS service..."
SERVER_HOST=${HOST} SERVER_PORT=${PORT} TTS_DEVICE=${TTS_DEVICE:-xpu} TTS_MODEL_TIER=${TTS_MODEL_TIER:-1.7B} nohup .venv/bin/python TTS/tts_service.py > /tmp/jarvis_tts.log 2>&1 &
TTS_PID=$!
disown $TTS_PID

echo ""
echo "========================================"
echo "  All services started!"
echo "========================================"
echo ""
echo "  Browser: https://${HOST}:${PORT}"
echo "  VAD:     wss://${HOST}:${PORT}/vad_service"
echo "  STT:     wss://${HOST}:${PORT}/stt"
echo "  LLM:     wss://${HOST}:${PORT}/llm"
echo "  TTS:     wss://${HOST}:${PORT}/tts"
echo ""
echo "Logs:"
echo "  Server: /tmp/jarvis_server.log"
echo "  VAD:    /tmp/jarvis_vad.log"
echo "  STT:    /tmp/jarvis_stt.log"
echo "  LLM:    /tmp/jarvis_llm.log"
echo "  TTS:    /tmp/jarvis_tts.log"
echo ""
echo ""

# Wait for all processes
echo "All services running. Press Ctrl+C to stop."
while true; do
    # Check if any service PID is still alive
    for pid in $SERVER_PID $VAD_PID $STT_PID $LLM_PID $TTS_PID; do
        if [ -n "$pid" ] && ! kill -0 $pid 2>/dev/null; then
            echo "Service with PID $pid exited, waiting for others..."
            sleep 1
            continue 2
        fi
    done
    sleep 1
done
