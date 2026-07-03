# Providers

**The LLM boundary.** Module: `regista/providers/`.

One protocol, four implementations. An adapter's whole job is normalization — wire format,
tool-call shape, usage extraction, retries:

```python
class Provider(Protocol):
    name: str    # recorded in session.start
    model: str   # chosen where the provider is constructed — never defaulted
    async def complete(self, request: ModelRequest) -> ModelResponse: ...
    def stream(self, request: ModelRequest) -> AsyncIterator[ProviderDelta | ModelResponse]: ...
```

regista's internal message model is a **superset of the Anthropic shape** (role + typed
content blocks, with tool_use/tool_result/thinking first-class). Rich→flat translation is
easy; flat→rich is lossy — so adapters translate *into* the rich shape, never out of it.

## `AnthropicProvider`

Official SDK. Blocks map 1:1; the system prompt carries a prompt-cache breakpoint; cache
read/write tokens land in `Usage` and flow into cost; thinking signatures round-trip. SDK
errors become `ProviderError` with an honest `retryable` flag.

```python
AnthropicProvider("claude-sonnet-4-6")           # reads ANTHROPIC_API_KEY
```

## `OpenAICompatProvider`

Raw httpx against any `/v1/chat/completions` endpoint — OpenAI, Ollama, vLLM, LM Studio —
with local exponential backoff. The lossy direction, made explicit: thinking blocks drop on
the way out; each tool result becomes its own `role="tool"` message; invalid JSON tool
arguments become `{"raw_arguments": ...}` so dispatch fails as error-data instead of
killing the session.

```python
OpenAICompatProvider("gpt-4o", api_key=...)
OpenAICompatProvider("llama3.1", base_url="http://localhost:11434/v1")   # no key needed
```

## `FakeProvider`

Scripted responses; records every request it receives. **Public API** — it's how you
unit-test your own agents and how contributors test loop changes. ~90% of regista's own
suite runs on it.

## `ReplayProvider`

Serves a recorded trace, hash-verifying each request — see
[deterministic replay](replay.md). Because it's just another provider, replay needed no
special support anywhere else in the harness.
