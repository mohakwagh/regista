# The nine primitives

An agent harness decomposes into nine primitives. In regista each maps to exactly one
module with one narrow interface — this table is the contributor's map: pick a primitive
and you know which directory it lives in.

| # | Primitive | Module | What it owns |
|---|---|---|---|
| 1 | [Instructions](instructions.md) | `instructions.py` | What the agent is told |
| 2 | [Context management](context.md) | `context/` | What the model remembers |
| 3 | [Tools](tools.md) | `tools/` | What the model can ask for |
| 4 | [Execution environment](environment.md) | `environment/` | Where effects happen |
| 5 | [Durable state](trace.md) | `trace/` + `session.py` | What survives a crash |
| 6 | [Orchestration](loop.md) | `loop.py` | The turn engine |
| 7 | Subagents | — (v0.3) | Child agents with isolated context |
| 8 | Skills & procedures | — (v0.3) | Reusable instruction + tool bundles |
| 9 | [Verification & observability](trace.md) | `trace/` + [`policy/`](policy.md) | Whether it behaved |

## The boundary rules

Three boundaries are subtle enough to state as rules:

**Tools vs. environment.** A tool defines *what* capability the model sees (name, schema,
semantics); the environment defines *where* its effects happen (local filesystem today, a
container tomorrow). Swapping environments never changes a tool's schema — the model can't
tell the difference, and neither can a recorded trace.

**Observability vs. durable state.** One artifact — the trace — serves both. Replay,
resume, and OTel export are all *consumers* of the same log; there is no second
instrumentation path. **If a behavior isn't in the trace, it's a bug.**

**Policy vs. tools.** The permission gate sits in the loop, *before* dispatch. Tools never
self-police; a tool implementation can assume it was allowed to run.

## The dependency graph

Dependencies point strictly downward:

```
agent.py / session.py        (composition root)
        │
     loop.py                 (orchestration)
        │
providers/ tools/ policy/ context/ instructions.py
        │     └── environment/
     trace/                  (flight recorder + durable state)
        │
types.py + errors.py         (shared vocabulary — imports nothing)
```
