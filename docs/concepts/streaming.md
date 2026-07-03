# Streaming

**See the session as it happens.** Module: `regista/streaming.py`.

```python
from regista import TextDelta, ToolCallStarted, ToolCallFinished, TurnCompleted, RunCompleted

async for event in agent.stream(task):
    match event:
        case TextDelta(text=t):            print(t, end="")
        case ToolCallStarted(name=n):      print(f"\n[{n} started]")
        case ToolCallFinished(content=c):  print(f"[-> {c}]")
        case TurnCompleted(turn=n, usage=u): ...
        case RunCompleted(result=r):       ...   # always last: the same RunResult run() returns
```

## The invariant

Streaming changes **when you see things, never what happened**. The trace records only
final requests and responses — never deltas — so a streamed session produces a trace with
byte-identical request hashes to a blocking run of the same script, and replays exactly the
same way. (This is tested.)

## How providers stream

Adapters yield `TextDelta`/`ThinkingDelta` items and finish with the complete
`ModelResponse`. Anthropic uses the SDK's native message stream; the OpenAI-compat adapter
parses SSE and assembles fragmented tool-call arguments; `FakeProvider` and
`ReplayProvider` synthesize deltas from completed messages with
`regista.streaming.synthetic_deltas` — which means **a $0 replay can drive a live UI**
exactly like the original run did.

A custom provider that can't stream natively needs three lines:

```python
async def stream(self, request):
    response = await self.complete(request)
    for delta in synthetic_deltas(response.message):
        yield delta
    yield response
```
