# Remaining tasks
## Review the repo and the contents of /tmp to see the results of the last run. Then take on the below tasks.

## Tasks

~Make the chat user/llm pane scrollable so the user can see all the chats.

~The llm response always updates the first in its list of messages and prints "Here" where the next message should go (at the end).
Example of failure. I replaced the LLM's response with "ACTUAL LATEST RESPONSE" for clarity.:
User
Explain the plot of Merlin.
LLM
ACTUAL LATEST RESPONSE
LLM
Here
User
Okay.
User
Mm-hmm.
User
Thank you. Could that
User
Hello.
LLM
Here
User
Please make that more concise.
User
Please make that more concise.
LLM
Here
User
Yeah.
LLM
Here
User
What did I just say?
User
Did I just say?
LLM
Here

~ When I reset the chat the STT still works fine but the LLM has an error displaying:
LLM:
2026-06-09 18:43:53,880 - INFO - An error occurred: sent 1011 (internal error) keepalive ping timeout; no close frame received
2026-06-09 18:43:53,880 - ERROR - LLM processing error: sent 1011 (internal error) keepalive ping timeout; no close frame received
2026-06-09 18:43:53,880 - INFO - LLM connection closed: sent 1011 (internal error) keepalive ping timeout; no close frame received

~ The TTS is having an error with generating. It was working at one point.
2026-06-09 18:43:56,923 - INFO - TTS synthesizing chunk: "Quick as you wanted!..."
2026-06-09 18:43:57,423 - ERROR - TTS generation error: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
2026-06-09 18:43:57,423 - INFO - Generating audio for sentence 46/46: "Let me know if you meant a different *Merlin* story or need ..."
2026-06-09 18:43:57,423 - INFO - TTS synthesizing chunk: "Let me know if you meant a different *Merlin* story or need anything else...."
2026-06-09 18:43:57,924 - ERROR - TTS generation error: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)


## Tasks for the future (please perform the above tasks then wait for the user confirmation to move onto these.
