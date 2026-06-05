# Remaining tasks

- Setup git
- Switch audio generation to GPU
- Switch audio gen so it doesn't chunk and regenerate from all the chunks. It should chunk a piece, genrate audio, and continue to the next chunk. 
- LLM stream tokens not just the full text at the end. Make sure those chunks go to the audio gen so it's not waiting until the llm generation is complete
