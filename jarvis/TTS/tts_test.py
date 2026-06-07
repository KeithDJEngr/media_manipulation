#import onnxruntime as ort
#import numpy as np
#import json
#import wave
#
## Load ONNX model with Vulkan
#session = ort.InferenceSession(
#    "./qwen-tts-onnx/model.onnx",
#    providers=["VulkanProvider"],
#    provider_options=[{"vulkan_device_id": 0}]  # 0 = first Arc GPU
#)
#
## Load config
#with open("./qwen-tts-onnx/config.json") as f:
#    config = json.load(f)
#
#def synthesize_chunk(text_chunk, stream=True):
#    # Tokenize & pad to model's expected length (usually ~16-32 audio frames)
#    inputs = {"input_ids": np.array([text_chunk], dtype=np.int64)}
#    
#    # Run forward pass
#    outputs = session.run(None, inputs)
#    audio_chunk = outputs[0]  # shape: (1, 1, chunk_len)
#    
#    if stream:
#        yield audio_chunk.flatten()
#
## Streaming usage
#full_audio = np.concatenate(list(synthesize_chunk("Hello world.", stream=True)))



#from transformers import AutoModelForACausalLM, AutoTokenizer
#improt soundfile as sf
#import torch
#
#device="xpu"
#
#model_name= "Qwen/

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

#model = Qwen3TTSModel.from_pretrained(
#    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
#    device_map="cuda:0",
#    dtype=torch.bfloat16,
#    attn_implementation="flash_attention_2",
#)
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    torch_dtype=torch.float16,
    attn_implementation="sdpa"  # Uses PyTorch's optimized native attention
)

# single inference
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
    speaker="Vivian",
    instruct="用特别愤怒的语气说", # Omit if not needed.
)
sf.write("output_custom_voice.wav", wavs[0], sr)

# batch inference
wavs, sr = model.generate_custom_voice(
    text=[
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
        "She said she would be here by noon."
    ],
    language=["Chinese", "English"],
    speaker=["Vivian", "Ryan"],
    instruct=["", "Very happy."]
)
sf.write("output_custom_voice_1.wav", wavs[0], sr)
sf.write("output_custom_voice_2.wav", wavs[1], sr)
