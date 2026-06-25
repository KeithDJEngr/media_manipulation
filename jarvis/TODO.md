# Current state
The webpage works. It has a transcript with user messages and LLM responses. It has a strange voice responding slowly but it does respond.

# Remaining tasks

## Remaining TODOs

### Bugs
- [ ] Sometimes it takes gets stuck at starting. That's the state I quit it at last. It looks like the STT is having an error. Please review logs and resolve anything you find.
- [ ] Audio from later messages can be played before earlier messages. Make them go in order.
- [ ] Partial transcripts on the are not updating properly on the client. It starts as the first word heard and then doesn't update past that until the full message is done. 
- [ ] My phone browser refuses to work with the microphone. It says the server says: Microphone access denied or error: Permission denied. I don't want to have to download a certificate to each device I want to connect on.
- [ ] The first message it responds to is partial junk and it doesn't save it to the history (it's the first partial and it doesn't update).

### Performance
- [ ] Speech takes long to generate. Please speed it up.
- [ ] STT performance investigation shows real-time factors < 0.1x for some lengths - inconsistent. Investigate GPU memory contention between models.
- [ ] Consider batching multiple short utterances if VAD fires rapidly
- [ ] TTS model warmup on each generation adds overhead - consider keeping model in memory between turns

# Future tasks

## Future TODOs

### Improvements
- [ ] ?? The `llm_start` turn_id reset logic in browser can cause messages to appear out of order - investigate turn_id tracking

### Future features
- [ ] Add TTS voice selection UI control (currently hardcoded to "eric")
- [ ] Add conversation export (save chat history to file)
- [ ] Add audio playback volume control in UI
- [ ] Consider adding a "listening history" panel separate from conversation
- [ ] Multi-language support (current STT is English-only)
- [ ] Wake word detection (always-listening mode)
- [ ] Custom system prompt via UI
- [ ] Conversation search
- [ ] Audio quality settings (sample rate, bit depth)


