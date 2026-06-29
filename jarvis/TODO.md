# Current state
The webpage works. It has a transcript with user messages and LLM responses.

# Completed tasks


# Remaining tasks

## General
- [ ] Conversation history is not working.
- [ ] LLM response on new msgs is just the same as the first response. Fix the conversation history and counting and see if it's fixed.

## Review
- [ ] Review the repo. Generate a summary of all the elements involved, how they work together, and all the functionality included in detail.

## Bugs - spin up additional agents to handle these one at a time
- [ ] Not sure if settings are changing anything or not. Review the logs and see if you can spot anything missing with getting the settings to .

## Performance
- [ ] Are there any broad changes to make the system more efficient? Do another performance review like in PERFORMANCE.md and generate PERFORMANCE2.md.
- [ ] What could be done to speed up the TTS or STT, especially STT? Are there other models, ways of retaining loading params, or anything that would still provide good quality but be faster?

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
