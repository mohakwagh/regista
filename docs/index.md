# regista

> The agent harness with a flight recorder.

**regista** is an observability-first agent harness for Python: the runtime layer that
turns a stateless LLM API into an agent that can act. Every session is recorded as a
structured, append-only trace — complete enough to **deterministically replay** the entire
session with zero API calls and zero cost.

## Where to go

- **[Quickstart](quickstart.md)** — an agent in five minutes, with or without an API key.
- **[The nine primitives](concepts/primitives.md)** — the map of the whole system; every
  concept page hangs off it.
- **[Deterministic replay](concepts/replay.md)** — the differentiator, explained.
- **[API reference](reference.md)** — generated from the docstrings.

The repository's [ARCHITECTURE.md](https://github.com/mohakwagh/regista/blob/main/ARCHITECTURE.md)
is the deep-dive companion: layer positioning, the dependency diagram, the life of a
request, and the trace schema contract.

## The pitch in one code block

```python
result = await agent.run("Fix the failing test")     # traced, budgeted, gated
replayed = await replay(result.trace_path)           # re-run it: $0, no keys, no effects
assert replayed.output == result.output              # hash-verified, call by call
```
