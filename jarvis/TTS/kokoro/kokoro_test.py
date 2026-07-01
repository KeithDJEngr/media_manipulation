#import torch
import soundfile as sf
from kokoro import KPipeline

# Initialize the pipeline directly onto the Intel B70 GPU
pipeline = KPipeline(lang_code='a', device='xpu')

text = "Hello! This is Kokoro eighty-two million running seamlessly on Intel Battlemage graphics."

# Generate generator object (uses default voice 'af_sarah')
generator = pipeline(text, voice='af_sarah', speed=1.0, split_pattern=r'\n+')

for i, (gs, ps, audio) in enumerate(generator):
    sf.write(f'kokoro_output_{i}.wav', audio, 24000)
    print(f"Saved chunk {i} successfully.")
