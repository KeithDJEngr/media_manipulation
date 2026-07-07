# Current state
The webpage works. It has a transcript with user messages and LLM responses.

# Current tasks
- [ ] Fix and LLM stop handling - on user pushing the stop button it should send a message to the LLM and the TTS to cancel their current requests and clear queues.
- [ ] Create a new test script that makes requests of the same LLM as you can see in the logs in /tmp. Test it many times and evalute its speed and ensure it's streaming not just sending the response at the end.
- [ ] Make TTS take LLM in batches (accepting whatever pieces it would generate itself [e.g. sentences])

# Completed tasks

## Need to verify complete

- [x] Add reset_history message handling in LLM service (LLM/llm_service.py)
- [x] Enable reset_history in server (Server/jarvis_server.py:423-424)
- [x] Desync mute button and "mute now" command - "mute now" only updates visual, doesn't mute mic (ProjectInterface/index.html:999-1004)
- [x] Clean up KokoroTTSProcessor dead code (removed unused text_buffer, current_chunk_text, processed_words)
- [x] Fix KokoroTTSProcessor tensor handling - .cpu().numpy() before .astype() (TTS/tts_service.py:344)
- [x] Fix KokoroTTSProcessor resampling from 24kHz to 16kHz (TTS/tts_service.py:347-350)
- [x] Fix KokoroTTSProcessor chunking yield placement inside inner loop (TTS/tts_service.py:361-366)

# Future tasks

## General
- [ ] LLM is sometimes not receiving streaming tokens. It works for some requests but others return 0 or 1 token. I see the LLM receiving the request and generating tokens but Is it sending the wrong request maybe? Or is there something about what it sends that would change it from streaming or how it's streaming? The latest run is an example of this and you can see the results in /tmp. See if you can find what's wrong and fix it.
- [ ] Make ordering TTS work for multiple message handling.

## New features

## General

## Future TODOs
- [ ] STT parials don't have the history so they're not really good for anything except keeping track that I'm listening.

# I believe resolved

## General
