# Orchestration — the loop

**The turn engine.** Module: `regista/loop.py` — meant to be read top-to-bottom.

One call to `run_loop()` drives one session: assemble request → provider → parse blocks →
per tool_use: gate → dispatch → results → repeat until the model finishes or the harness
stops it. The loop owns no I/O of its own (providers, tools, and the trace writer do); it
is pure orchestration, and **every step emits a trace event**.

```
turn N:
  1. stop? max_turns reached, or total cost ≥ max_cost_usd  → stop
  2. assemble ModelRequest(system, history, tools, max_tokens, params)
     emit llm.request (+ request_hash)
  3. provider.complete(request)          ← the only network call
     emit llm.response (usage, cost, latency)
  4. stop_reason == "tool_use"?
       for each block: emit tool.call → policy gate (emit permission.decision)
       → execute (emit tool.result) → append results to history
     stop_reason final (end_turn, refusal, …)? → done
  5. over the context budget? → compact (see context management), then next turn
```

Details worth knowing:

- **Parallel tools**: a batch executes concurrently iff *every* permitted call in it is
  `parallel_safe`; results are returned to the model in its original call order regardless.
- **Denials** become error tool results in place — the reply preserves order and pairing.
- **Budget stops** use provider-reported usage and the pricing table (overridable per
  Agent; unknown models cost `None`, never a guess).
- **Provider errors** end the session with `stop_reason="error"` and an `error` trace
  event — except `ReplayDivergence`, which re-raises so strict replay fails loudly.

## Session and Agent

`session.py` brackets the loop: mints the ULID session id, opens the trace, emits
`session.start` / `session.end`, and folds the outcome into a `RunResult`. `agent.py` is
the composition root users touch — reusable config; each `run()` gets a fresh Session and a
fresh trace file, so one Agent can serve many tasks concurrently.
