# Context management

**What the model remembers.** Module: `regista/context/`.

The trigger is **provider-reported usage, never a local token estimate**: when a turn's
observed input tokens (input + cache read + cache write) cross `max_input_tokens`, the loop
compacts — it asks the session's own provider to summarize the older history and replaces
those messages with one summary message, keeping the most recent messages verbatim.

```python
from regista import Agent, ContextConfig

agent = Agent(
    provider=...,
    instructions=...,
    context=ContextConfig(
        max_input_tokens=150_000,   # None (default) disables compaction
        keep_recent_messages=4,     # preserved verbatim at the end of history
    ),
)
```

Rules the implementation keeps:

- A split never separates an assistant `tool_use` from its `tool_result` reply.
- Thinking blocks are excluded from the transcript sent to the summarizer.
- The summarization call is a **regular, hash-verified `llm.request`/`llm.response` pair**
  followed by a `context.compaction` event, and its usage/cost roll into the session
  totals.

## Why compaction goes through the provider seam

Because the summary flows through the same provider interface as everything else, a
replayed session re-runs the same compaction logic, is served the recorded summary, and
every post-compaction request hash still matches. `ContextConfig` is recorded in
`session.start` precisely so [replay](replay.md) can reconstruct it.
