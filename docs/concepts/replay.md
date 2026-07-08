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
  `agent.resume()` packages it (below).

Replays write their own trace, tagged `replay_of: <original session_id>`, so a replay is
itself inspectable and diffable.

## Resume

`Agent.resume(trace_path)` continues an interrupted session:

```python
result = await agent.resume(".regista/traces/01J....jsonl")
```

Three swaps on the agent's live config, all keyed on what the trace contains:

- the provider replays in **hybrid** mode — the recorded prefix is served for $0
  (hash-verified), and the first request the recording can't answer falls through to the
  agent's real provider;
- **recorded tool calls serve their recorded results** (their effects already happened
  once — a resume never re-runs them), while new calls execute for real. A call the crash
  cut short (no recorded result) is re-executed;
- recorded calls bypass the policy (they were gated in the original run); **new calls face
  the agent's real policy**.

Resume assumes the agent is configured like the recorded run — same instructions, tools,
params. Any difference makes the first request diverge, which simply means the whole task
re-runs live; the hash chain makes the fidelity self-verifying either way. The resumed
session writes its own trace, linked via `replay_of`.
