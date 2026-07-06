# Current state
The webpage works. It has a transcript with user messages and LLM responses.

# Completed tasks


# Remaining tasks

## General
- [ ] LLM is sometimes not receiving streaming tokens? It is working for the first request but subsequent requests usually return 0 or 1 token. Is it sending the wrong request maybe? Or is there something about what it sends that would change it from streaming or how it's streaming? The latest run is an example of this. Review the log in /tmp to properly evaluate what's happening with it?
- [ ] Make TTS take LLM in batches (accepting whatever pieces it would generate itself [e.g. sentences])
- [ ] Make ordering TTS work for multiple message handling. 
- [ ] Stop TTS existing queue and audio if interrupted.
- [ ] Looks like the LLM still fails to stop when I interrupt
- [ ] Saying mute now mutes the microphone so I have to manually push the button because it stops even hearing me. Probably turn that off. And the mute button toggles mute/unmute so I have to say unmute after push the button. Yeah make them desynced.
- [ ] The reset and stop buttons don't clear the chat history in memory, just on the webpage. Maybe have 2 memories - keep export as the full history but the reset should erase the chat history.

## New features

## General

## Future TODOs
- [ ] STT parials don't have the history so they're not really good for anything except keeping track that I'm listening.





# I believe resolved

## General
- [ ] LLM server appears to stop at some points - not sure what's causing it. It's happening rarely with the last commit but with my present changes it's happening very fast and then the server just hangs and all the others fail to get the handshake. Have the logs saved: /tmp/ADDITIONAL_JARVIS_LOGS_LAST_COMMIT_STATE for the original commit.
