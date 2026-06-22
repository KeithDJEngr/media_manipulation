# Remaining tasks

## Remaining TODOs

### Bugs
- [ ] VAD buffer issue: many "End of speech, buffer size: 0" messages indicate VAD fires before STT accumulates audio. May need to adjust silence threshold or add STT-side buffering.
- [ ] The LLM transcription prints to the right side top message always, not to the newest one.

### Improvements
- [ ] Add TTS voice selection UI control (currently hardcoded to "eric")
- [ ] Add conversation export (save chat history to file)
- [ ] Add audio playback volume control in UI
- [ ] The `llm_start` turn_id reset logic in browser can cause messages to appear out of order - investigate turn_id tracking

### Performance
- [ ] STT performance investigation shows real-time factors < 0.1x for some lengths - inconsistent. Investigate GPU memory contention between models.
- [ ] Consider batching multiple short utterances if VAD fires rapidly
- [ ] TTS model warmup on each generation adds overhead - consider keeping model in memory between turns

### Future features
- [ ] Consider adding a "listening history" panel separate from conversation
- [ ] Multi-language support (current STT is English-only)
- [ ] Wake word detection (always-listening mode)
- [ ] Custom system prompt via UI
- [ ] Conversation search
- [ ] Audio quality settings (sample rate, bit depth)
