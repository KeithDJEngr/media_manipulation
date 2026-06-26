# Current state
The webpage works. It has a transcript with user messages and LLM responses. It has a strange voice responding slowly but it does respond.

# Completed tasks

## Bugs
- [ ] I just connected 
- [x] STT stuck at starting - added timeout mechanism and fixed buffer clearing on speech end
- [x] After streaming STT text it went blank - fixed VAD result handling and waiting_for_audio_after_eos state
- [x] Audio from later messages played before earlier - fixed audio_seq per-turn tracking
- [x] First partial message junk - fixed conversation_history update when no user message exists
- [x] Partial transcripts not updating - STT now always sends partial_transcript (even empty)
- [x] Mobile browser mic denied - added better error messages and audio processing options
- [x] llm_start turn_id reset logic - fixed per-turn sequence offset tracking

## Performance
- [x] Speech takes long to generate - increased TTS chunk size, optimized generation parameters
- [x] VAD rapid-fire utterance batching - added 0.5s speech end debounce
- [x] TTS model warmup overhead - kept model loaded, reduced redundant generation, increased chunk thresholds

## New features
- [x] TTS voice selection UI control - dropdown with presets (eric, sarah, james, emma) and custom
- [x] Wake word detection toggle - always-listening mode in settings
- [x] Custom system prompt via UI - textarea in settings panel, sent to LLM service
- [x] Audio quality settings (sample rate) - 16kHz/22.05kHz/44.1kHz selection
- [x] Copy message to clipboard - per-message copy button with visual feedback
- [x] Conversation export - export button downloads conversation as text file
- [x] TTS voice instruct - configurable in settings panel

# Remaining tasks

## Future TODOs

### Improvements
- [ ] Audio playback volume control in UI
- [ ] Listening history panel separate from conversation
- [ ] Multi-language support (current STT is English-only)
- [ ] Conversation search

### New features
- [ ] Wake word detection (actual keyword-based, not just always-listening mode)
- [ ] Advanced audio processing (noise suppression, echo cancellation tuning)
- [ ] Dark/light theme toggle
- [ ] Keyboard shortcuts for common actions
