#!/bin/bash
sudo pacman -S intel-compute-runtime level-zero-loader ocl-icd clinfo opencl-headers sox -y
sudo usermod -aG render,video $USER

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
if [[ $SCRIPT_DIR == "" ]] ; then
    echo "SCRIPT_DIR:$SCRIPT_DIR"
    exit 1
fi
SCRIPT_NAME=$0

#git clone https://github.com/QwenLM/Qwen3-TTS.git ~/git/Qwen3-TTS
#$SCRIPT_DIR/../../.venv/bin/pip install -e ~/git/Qwen3-TTS/
$SCRIPT_DIR/../../.venv/bin/pip uninstall qwen-tts -y

sleep 5

# Clean out the problematic torch installations first
$SCRIPT_DIR/../../.venv/bin/pip uninstall -y torch torchvision torchaudio intel-extension-for-pytorch

# Tried torch==2.8.0 to match for intel-extension-for-pytorch==2.8.0 but failing compatible torchaudio
$SCRIPT_DIR/../../.venv/bin/pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# Install Qwen3-TTS
#$SCRIPT_DIR/../../.venv/bin/pip uninstall qwen-tts -y
#$SCRIPT_DIR/../../.venv/bin/pip uninstall qwen-tts -y

$SCRIPT_DIR/../../.venv/bin/pip uninstall flash-attn -y
$SCRIPT_DIR/../../.venv/bin/pip install -r $SCRIPT_DIR/requirements.txt
# Haven't got this working - just for intel, otherwise remove and remove torch version torch==2.8.0
#$SCRIPT_DIR/../../.venv/bin/pip install intel-extension-for-pytorch==2.8.0

# Download the custom quantized GGUF + ONNX repository
git clone https://huggingface.co/cgisky/qwen3-tts-custom-gguf ~/git/qwen3_gguf
cd ~/git/qwen3_gguf
# Initialize Git LFS just to be sure it's active in your environment
git lfs install
# Force pull the large files (.onnx, .gguf, etc.)
git lfs pull

echo "COMPLETED"
