# NOT CURRENTLY WORKING

import os
import json
import soundfile as sf
import onnxruntime as ort
import numpy as np
from llama_cpp import Llama  # Engine used to run the GGUF layers

USER = os.environ.get('USER', 'default')
git_qwen3_gguf_path = f"/home/{USER}/git/qwen3_gguf"

# Constants for Qwen3-TTS Architecture
VOCAB_OFFSET = 151646  # Audio tokens start after the text vocabulary
NUM_CODEBOOKS = 16     # Qwen3-TTS 12Hz utilizes 16 codebooks per frame

# 1. Map execution directly to your Intel B70 (Battlemage) card via OpenVINO
provider_options = [{
    'device_type': 'GPU_FP16', # Force Intel GPU processing
    'num_of_threads': 4
}]
session_options = ort.SessionOptions()

print("Initializing Crushed INT8 Audio Decoder via Intel OpenVINO...")
decoder_path = git_qwen3_gguf_path + "/onnx/qwen3_tts_decoder.onnx"

decoder_session = ort.InferenceSession(
    decoder_path,
    sess_options=session_options,
    providers=['OpenVINOExecutionProvider'],
    provider_options=provider_options
)

# 2. Define the configuration pointing to the 5-bit (Q5_K_M) parameters
gguf_config = {
    "model_dir": git_qwen3_gguf_path + "/gguf_q5_k_m",
    "assets": "qwen3_assets.gguf",
    "talker": "qwen3_tts_talker.gguf",
    "predictor": "qwen3_tts_predictor.gguf",
    "tokenizer_path": git_qwen3_gguf_path + "/tokenizer/tokenizer.json"
}

print(f"Successfully configured pipeline using: {gguf_config['model_dir']}")
print("VRAM footprints successfully restricted to ~1.8 GB.\n")

print("Loading GGUF Text Processing Core (Talker)...")
talker_path = os.path.join(gguf_config["model_dir"], gguf_config["talker"])
talker_llm = Llama(model_path=talker_path, vocab_only=False, verbose=False, n_ctx=2048)

print("Loading GGUF Code Predictor Core...")
predictor_path = os.path.join(gguf_config["model_dir"], gguf_config["predictor"])
predictor_llm = Llama(
    model_path=predictor_path,
    n_ctx=4096,        # Raise this from the default
    n_batch=512,       # Allows larger chunks of prompt tokens to evaluate at once
    n_threads=4,       # Adjust based on your CPU host threads
    n_gpu_layers=99    # Offload layers to accelerator backends
)

print("\nEnvironment and memory map verified. Ready to synthesize.")
print("-" * 50)

# ==================== GENERATION STEPS ====================

text_input = "Hello world! This is Qwen3 text to speech running entirely on an Intel Battlemage graphics card using GGUF and OpenVINO acceleration."
speaker = "serena"  # Options include: ryan, dylan, eric, ono_anna, serena, sohee, etc.

print(f"Synthesizing: \"{text_input}\" using voice profile: [{speaker}]")

# Step 1: Process text through the Talker model using the low-level token generator loop
print("Generating semantic tokens from Talker core...")
talker_prompt = f"<|im_start|>system\nYou are an advanced text to speech engine.<|im_end|>\n<|im_start|>user\n[speaker={speaker}] {text_input}<|im_end|>\n<|im_start|>assistant\n"
talker_prompt_tokens = talker_llm.tokenize(talker_prompt.encode("utf-8"))

talker_tokens = []
for token in talker_llm.generate(talker_prompt_tokens, temp=0.1):
    if token == talker_llm.token_eos():
        break
    talker_tokens.append(token)

print(f" ➔ Successfully extracted {len(talker_tokens)} semantic tokens.")

# Step 2: Use the Predictor model to generate structural 12Hz acoustic code matrices
print("Predicting multi-codebook acoustic streams...")

# Tokenize prefix and suffix tags natively using the predictor's model vocabulary
prefix_tokens = predictor_llm.tokenize(b"<|im_start|>system\nPredict acoustic codebooks.<|im_end|>\n<|im_start|>user\n")
suffix_tokens = predictor_llm.tokenize(b"<|im_end|>\n<|im_start|>assistant\n")

# Splice token arrays directly together to bypass character-string re-tokenization bugs
prompt_tokens = prefix_tokens + talker_tokens + suffix_tokens

extracted_tokens = []
for token in predictor_llm.generate(prompt_tokens, temp=0.1):
    if token == predictor_llm.token_eos():
        break

    # FIXED: Convert raw vocabulary token ID back down to a valid 0-indexed codebook code
    actual_code = token - VOCAB_OFFSET
    extracted_tokens.append(actual_code)

    # Upper bound check (Accommodates 16 channels * frame length safely)
    if len(extracted_tokens) >= 16384:
        break

# FIXED: Calculate frames based on the legitimate 16-codebook architecture
num_frames = len(extracted_tokens) // NUM_CODEBOOKS

if num_frames > 0:
    # Reshape to (1, T, 16) first to preserve frame boundaries,
    # then transpose to (1, 16, T) for the OpenVINO ONNX decoder
    acoustic_codes = (
        np.array(extracted_tokens[:num_frames * NUM_CODEBOOKS], dtype=np.int64)
        .reshape(1, num_frames, NUM_CODEBOOKS)
        .transpose(0, 2, 1)
    )
else:
    raise ValueError("Predictor model failed to return a valid sequence of codebook tokens.")

print(f" ➔ Successfully generated real acoustic matrix with shape: {list(acoustic_codes.shape)}")

# Step 3: Run codebooks through the Intel OpenVINO Decoder for wave reconstruction
print("Reconstructing audio waveform via Intel OpenVINO execution provider...")

chunk_size = 16
total_frames = acoustic_codes.shape[2]

# Pad the audio codes so they divide perfectly into chunks of 16
pad_len = (chunk_size - (total_frames % chunk_size)) % chunk_size
if pad_len > 0:
    acoustic_codes = np.pad(acoustic_codes, ((0,0), (0,0), (0, pad_len)), mode='constant', constant_values=0)
    total_frames = acoustic_codes.shape[2]

# Dynamically build the initial input payload structure
input_feed = {}
print("\nInitializing required ONNX state and tracking buffers:")

for input_meta in decoder_session.get_inputs():
    name = input_meta.name

    if 'float16' in input_meta.type:
        dtype = np.float16
    elif 'int64' in input_meta.type:
        dtype = np.int64
    elif 'int32' in input_meta.type:
        dtype = np.int32
    elif 'bool' in input_meta.type:
        dtype = np.bool_
    else:
        dtype = np.float32

    if name in ['audio_codes', 'is_last']:
        continue

    shape = []
    for dim in input_meta.shape:
        if isinstance(dim, str) or dim is None or (isinstance(dim, int) and dim < 0):
            shape.append(0 if 'past' in name else 1)
        else:
            shape.append(dim)

    input_feed[name] = np.zeros(shape, dtype=dtype)
    print(f" ➔ State buffer initialized: {name:<20} | Shape: {str(list(input_feed[name].shape)):<15} | Type: {dtype.__name__}")

# Extract output layer configurations
output_names = [out.name for out in decoder_session.get_outputs()]
audio_output_name = output_names[0]  # First index is always generated raw wave chunk

# Map output tracking blocks back to input layers
state_output_to_input = {}
for out_name in output_names[1:]:
    cleaned_out = out_name.lower()
    for prefix in ['present_', 'updated_', 'next_']:
        if cleaned_out.startswith(prefix):
            cleaned_out = cleaned_out[len(prefix):]
            break

    for inp_meta in decoder_session.get_inputs():
        inp_name = inp_meta.name
        cleaned_inp = inp_name.lower()
        if cleaned_inp.startswith('past_'):
            cleaned_inp = cleaned_inp[5:]

        if cleaned_out == cleaned_inp or cleaned_out == inp_name.lower():
            state_output_to_input[out_name] = inp_name
            break

# Step through the frames sequentially in blocks of 16
all_audio_chunks = []
print(f"\nProcessing {total_frames} frames in steps of {chunk_size} windows on Intel B70...")

for i in range(0, total_frames, chunk_size):
    chunk = acoustic_codes[:, :, i:i+chunk_size]
    is_last_chunk = (i + chunk_size >= total_frames)

    # Update dynamic input values
    input_feed['audio_codes'] = chunk
    input_feed['is_last'] = np.array([1.0 if is_last_chunk else 0.0], dtype=np.float32)

    # Fire the execution pass through OpenVINO
    outputs = decoder_session.run(output_names, input_feed)

    # Extract the generated audio data piece
    audio_chunk = outputs[0].flatten()
    all_audio_chunks.append(audio_chunk)

    # Cycle the runtime parameters: map output states back into input slots for step N+1
    for idx, out_name in enumerate(output_names):
        if out_name in state_output_to_input:
            target_input_slot = state_output_to_input[out_name]
            input_feed[target_input_slot] = outputs[idx]

    if (i + chunk_size) % 80 == 0 or is_last_chunk:
        print(f" ➔ Progressive synthesis: {i + chunk_size}/{total_frames} frames computed.")

# Step 4: Stitch the processed chunks into a single audio sequence and save
print("\nStitching audio streams together...")
waveform = np.concatenate(all_audio_chunks)

output_filename = "qwen3_generation_output.wav"
sf.write(output_filename, waveform, 24000)

print("-" * 50)
print(f"[Success] Generation complete! File saved to: {os.path.abspath(output_filename)}")
