# Current state
The webpage works. It has a transcript with user messages and LLM responses.

# Completed tasks


# Remaining tasks

## General
- [ ] Conversation history is not working. It should contain the sytem instruction then (user msg 1) then (llm response 1) then (user message 2) then (llm response 2) and so on. Review the /tmp/ logs. It failed to keep the past and reordered the messages incorrectly.
- [ ] LLM response on the first user message is good but on new msgs the LLM service just responds with the first response. Fix the logic of how the receiving requrests.

## New features

## General

## Future TODOs

### Bugs
- [x] ???When the audio gets too long it appears to fail. Added conversation history truncation (max 20 messages) to prevent memory issues.
- [x] ???STT didn't appear to work with my mobile browser. Added sample rate detection and resampling to handle mobile browsers.

### Improvements
- [ ] Audio playback volume control in UI
- [ ] Listening history panel separate from conversation
- [ ] Multi-language support (current STT is English-only)
- [ ] Conversation search

### Performance
- [x] Reviewed and implemented performance improvements:
  - Added mobile browser sample rate detection and resampling
  - Added conversation history truncation (max 20 messages) to prevent memory issues
  - Added use_full_history toggle to reduce API payload size
  - Reduced AudioContext buffer size from 1024 to 512 samples for lower latency
  - Added wake word mute/unmute detection to skip audio processing

### New features
- [ ] Wake word detection (actual keyword-based, not just always-listening mode)
- [ ] Advanced audio processing (noise suppression, echo cancellation tuning)
- [ ] Dark/light theme toggle
- [ ] Keyboard shortcuts for common actions
