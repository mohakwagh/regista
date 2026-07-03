# Deterministic replay

**The differentiator.** Modules: `regista/replay.py`, `regista/providers/replay.py`.

Recording happens automatically — it's just [the trace](trace.md); there is no "record
mode." Replay re-runs the session through the **same loop** with two swaps: the provider
serves recorded responses, and a stub registry serves recorded tool results. Hermetic by
default: no network, no keys, no side effects, $0.

```python
from regista import replay

replayed = await replay(result.trace_path)
assert replayed.cost_usd == 0.0
```

## Replay is just another provider

`ReplayProvider` implements the same `Provider` protocol as the Anthropic adapter. On the
Nth call it computes the live **`request_hash`** — SHA-256 over a canonical JSON
serialization of the full request — and compares it to the recorded Nth request's hash.
Match → serve the recorded response. Mismatch → divergence:

| mode | on divergence | use case |
|---|---|---|
| `strict` (default) | raise `ReplayDivergence` with a structural diff | CI regression tests |
| `warn` | `ReplayDivergenceWarning`, keep serving positionally | time-travel debugging |
| `hybrid` | fall through to a real `fallback` provider from then on | resume-from-recording |

```python
await replay(path)                                      # strict
await replay(path, mode="warn")
await replay(path, mode="hybrid", fallback=AnthropicProvider("claude-sonnet-4-6"))
```

The diff points at exactly what changed:

```
request.system: recorded 'You are a greeter.' != live 'You are a greeter. Always answer in French.'
```

## What makes it deterministic

`replay()` reconstructs the entire configuration **from the trace alone**: task,
instructions, tool schemas, and context config from `session.start`; model, `max_tokens`,
and params from the first recorded request. Tool results are served by `tool_use_id` —
including permission denials, which were recorded as error tool results, so denied paths
replay byte-identically. Compaction re-runs and is served its recorded summary.

The test suite's core assertion: a strict replay's request-hash chain equals the
original's — the conversation rebuilt byte-for-byte from the log.

## What it buys you

- **Zero-cost regression tests** — commit one real trace; CI re-runs your agent's full loop
  forever without an API key.
- **Time-travel debugging** — re-run a failed production session locally with a debugger
  attached, without re-paying or re-triggering side effects.
- **Resume** — `hybrid` mode replayed past the end of a crashed recording *is* resumption;
  `Session.resume()` (v0.2) is a thin wrapper over it.

Replays write their own trace, tagged `replay_of: <original session_id>`, so a replay is
itself inspectable and diffable.
