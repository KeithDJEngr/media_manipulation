# Use:
# SYCL_UR_USE_LEVEL_ZERO_V2=:0 SYCL_PI_LEVEL_ZERO_USM_RESIDENT=:0 .venv/bin/python TTS/qwen3-tts/qwen3_run.py

import torch
# Havne't gotten this working yet
##import intel_extension_for_pytorch as ipex # Ensure IPEX is explicitly imported
## 1. Enable TensorFloat32 (TF32) on Intel GPU for faster matrix math
#torch.backends.cuda.matmul.allow_tf32 = True
## 2. Force TorchInductor to optimize aggressively for inference latency
## This makes the first compile take longer, but subsequent runs are maximized for speed

import soundfile as sf
from qwen_tts import Qwen3TTSModel  # Ensure your model variant matches the exact import

print("Initializing Qwen 3 TTS on Intel XPU...")

# 1. Load the model directly into Intel GPU memory using BF16
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="xpu",
    dtype=torch.bfloat16
)

# 2. Intel Core Optimization: Compile the model for massive inference speedups
# The first execution will take a moment to compile, but subsequent runs will be fast.
print("Compiling model via TorchInductor/Triton for Intel GPU...")
model.model = torch.compile(
    model.model,
    mode="reduce-overhead", # Optimize for B70, Minimizes Python framework overhead
    fullgraph=True          # Optimize for B70, Prevents graph breaks that drop back to Python
)

# Define your text prompt
text_prompt = "Arch Linux and Intel Battlemage B70 running Qwen 3 TTS is an incredibly fast combination, don't you think?"

print("Generating audio...")

# 3. Generate the speech (adjust parameters like voice/speaker depending on your model subtype)
# For Base/VoiceClone, pass your text; for CustomVoice, add the speaker tag.
#wavs, sr = model.generate(
#    text=text_prompt,
#    language="English"
#)

# Use generate_custom_voice with a preset speaker (e.g., "Ryan", "Vivian")
# TODO: look into stream
wavs, sr = model.generate_custom_voice(
    text=text_prompt,
    language="English",
    speaker="Ryan",
    use_cache=True
)


# Save the generated audio file
output_file = "b70_output.wav"

# 1. If wavs is a PyTorch Tensor, move to CPU and convert to NumPy
if hasattr(wavs, "cpu"):
    wavs = wavs.cpu().numpy()

# 2. Qwen TTS usually returns a batched shape [1, T] or nested list.
# Squeeze out the extra dimensions to make it a 1D audio array.
import numpy as np
wavs = np.squeeze(wavs)

# 3. Write to file (Format: file, data, samplerate)
# Double-check that 'sr' is an integer (like 24000 or 44100)
sf.write(output_file, wavs, int(sr))

print(f"Success! Audio saved to {output_file}")






## Voice cloning
#import torch
#import soundfile as sf
#from qwen_tts import Qwen3TTSModel  # Ensure your model variant matches the exact import
#
#print("Initializing Qwen 3 TTS on Intel XPU...")
#
## 1. Load the model directly into Intel GPU memory using BF16
#model = Qwen3TTSModel.from_pretrained(
#    "Qwen/Qwen3-TTS-12Hz-1.7B-BaseVoice",
#    device_map="xpu",
#    dtype=torch.bfloat16
#)
#
## 2. Intel Core Optimization: Compile the model for massive inference speedups
## The first execution will take a moment to compile, but subsequent runs will be fast.
#print("Compiling model via TorchInductor/Triton for Intel GPU...")
#model.model = torch.compile(model.model)
#
## Define your text prompt
#text_prompt = "Arch Linux and Intel Battlemage running Qwen 3 TTS is an incredibly fast combination."
#
#print("Generating audio...")
#
## 3. Generate the speech (adjust parameters like voice/speaker depending on your model subtype)
## For Base/VoiceClone, pass your text; for CustomVoice, add the speaker tag.
#wavs, sr = model.generate_voice_clone(
#    text=text_prompt,
#    language="English",
#    ref_audio="/path/to/your/3_second_voice_sample.wav",
#    ref_text="The exact text spoken in that short sample audio file."
#)
#
### Use generate_custom_voice with a preset speaker (e.g., "Ryan", "Vivian")
##wavs, sr = model.generate_custom_voice(
##    text=text_prompt,
##    language="English",
##    speaker="Ryan"
##)
#
#
## Save the generated audio file
#output_file = "b70_output.wav"
#sf.write(output_file, wavs, sr)
#print(f"Success! Audio saved to {output_file}")
