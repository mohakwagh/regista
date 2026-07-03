# regista

> The agent harness with a flight recorder.

**regista** is an observability-first agent harness for Python: the runtime layer that turns
a stateless LLM API into an agent that can act. Every session is recorded as a structured,
append-only trace — complete enough to **deterministically replay** the entire session with
zero API calls and zero cost.

> *regista* (Italian): the deep-lying playmaker who directs the game. Also: "director."

**Status: pre-release (v0.1 in active development).** The architecture is stable and
documented; the modules are landing in order. Watch/star to follow along.

## Why another harness?

Every serious agent harness treats observability as an add-on or a paid platform. regista
treats **the trace as the product**:

- 🔍 **Structured event log** — every LLM call, tool execution, permission decision, token
  count, and dollar of cost, as typed JSONL events.
- ⏪ **Deterministic replay** — re-run any recorded session with the LLM (and optionally the
  tools) stubbed from the recording. Time-travel debugging and zero-cost regression tests.
- 📊 **OpenTelemetry export** — the same trace as spans in Jaeger/Grafana/Langfuse.
- 🔌 **Provider-neutral** — native Anthropic adapter plus an OpenAI-compatible adapter
  (OpenAI, Ollama, vLLM, gateways).
- 🛡️ **Permission gate** — Allow/Deny/Ask policy callbacks on every tool call, honestly
  documented as a gate (not a sandbox).

## A taste of the API

```python
from regista import Agent, tool, replay
from regista.providers import AnthropicProvider

@tool
def get_ticket(ticket_id: str, include_comments: bool = False) -> str:
    """Fetch a ticket.

    Args:
        ticket_id: The ticket key, e.g. PROJ-123.
        include_comments: Whether to include the comment thread.
    """
    ...

agent = Agent(
    provider=AnthropicProvider(model="claude-sonnet-4-6"),
    instructions="You are a support-triage agent.",
    tools=[get_ticket],
    max_turns=20,
    max_cost_usd=1.00,
)

result = await agent.run("Summarize the open blockers on PROJ-123")
print(result.output, result.cost_usd, result.trace_path)

# ...later, with no API key and no cost:
replayed = await replay(result.trace_path)
assert replayed.output == result.output
```

## Architecture

regista is organized around **nine harness primitives**, each mapping 1:1 to a module —
read [ARCHITECTURE.md](ARCHITECTURE.md), which explains the whole system from first
principles (it's written for people who have never built an agent).

## License

[MIT](LICENSE)
