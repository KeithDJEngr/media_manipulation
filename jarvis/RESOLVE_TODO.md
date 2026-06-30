# Resolve TODO.md Issues

## Issues from TODO.md

1. **Conversation history not working** -- The conversation_history should contain (system, user1, assistant1, user2, assistant2...) but it only contains (user, assistant) -- the latest exchange. The user message gets overwritten on every partial transcript.

2. **LLM responds with same response on new messages** -- Because the history only contains the last user message + last assistant response, the LLM receives nearly identical context each turn. E.g., turn 2 sends `[{user: "What were we talking about?"}, {assistant: "...Merlin..."}]` -- the LLM has no memory that the user previously asked "Can you tell me what the plot of Merlin is?" so it just replays the Merlin answer.

## Root Cause

The `handle_stt` function's `partial_transcript` and `final_transcript` handlers **update the last user entry in place** rather than **appending a new user entry per turn**:

```python
# partial_transcript: updates existing user entry (overwrites content)
for i in range(len(manager.conversation_history) - 1, -1, -1):
    if manager.conversation_history[i]["role"] == "user":
        manager.conversation_history[i]["content"] = partial_text  # OVERWRITES
        break
```

This means conversation_history never grows beyond 2 entries (last user message + last assistant response).

## Fix Applied

### partial_transcript handler (lines 544-569)

**Before:** Only updated the last user entry in place, never appended a new one.

**After:** Still only updates the last user entry in place (no change needed). The key change is in the final_transcript handler.

### final_transcript handler (lines 571-611)

**Before:** Updated the last user entry in place (same as partial).

**After:** Appends a new user entry after the last assistant entry, so history grows per turn:

```python
# Append a new user entry after the last assistant entry so history grows per turn
last_assistant_idx = -1
for i in range(len(manager.conversation_history) - 1, -1, -1):
    if manager.conversation_history[i]["role"] == "assistant":
        last_assistant_idx = i
        break
if last_assistant_idx >= 0:
    manager.conversation_history.insert(last_assistant_idx + 1, {"role": "user", "content": user_text})
```

## Expected Behavior After Fix

- Turn 1: partials update user1, final updates user1, llm_end appends assistant1 → history = [user1, assistant1]
- Turn 2: partials update user1, final inserts user2 after assistant1, llm_end appends assistant2 → history = [user1, assistant1, user2, assistant2]

The LLM service will receive the full history with all turns, allowing it to respond contextually to subsequent messages.
