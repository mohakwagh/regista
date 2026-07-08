# regista — Architecture

> *regista* (Italian): the deep-lying playmaker who directs the game from the base of midfield.
> Also: "director."

regista is an **agent harness** — the runtime layer that turns a stateless LLM API into an
agent that can act. This document explains exactly what that means, what the harness is made
of, and how a request flows through it. It is written to be read top-to-bottom by someone who
has never built an agent before; if anything here is unclear, that is a documentation bug —
please open an issue.

---

## 1. What layer is this?

The agentic stack has three layers. Keeping them straight explains every design decision below.

| Layer | Examples | Responsibility |
|---|---|---|
| **Model API** | Anthropic Messages API, OpenAI Chat Completions | Stateless text/tool-call inference. The model never executes anything — it only *requests* tool calls. |
| **Harness / runtime** — **regista lives here** | Claude Agent SDK, OpenAI Agents SDK, pydantic-ai | The engine: the agent loop, tool registry and execution, permissions, context management, tracing. A library with no opinion about what the agent is *for*. |
| **Application** | Claude Code, Cursor, a support bot | A product built on a harness: picks the tools, prompts, and UI for a domain. |

regista is not a coding agent, a chatbot, or a workflow framework. It is the layer those
things are built on. (A mini coding agent takes ~50 lines on top of regista — see the README —
but so does a ticket-triage agent; the harness doesn't care.)

**The one-sentence pitch:** every serious harness treats observability as an add-on or a paid
platform; regista treats the **trace as the product** — an append-only event log complete
enough to *deterministically replay* any session, for free, forever.

## 2. The nine primitives

An agent harness decomposes into nine primitives. Each maps to exactly one module in this
repo, and each module has exactly one responsibility and one narrow interface. This table is
the contributor's map: pick a primitive, and you know which directory it lives in.

| # | Primitive | Module | What it owns | Status |
|---|---|---|---|---|
| 1 | **Instructions** | `regista/instructions.py` | What the agent is told: a layered system prompt (base + sections + task), rendered once per session and recorded in the trace. | v0.1 |
| 2 | **Context management** | `regista/context/` | What the model remembers: token/cost budgets from provider-reported usage; compaction (summarizing old turns) when the window fills. | v0.1 |
| 3 | **Tool interface** | `regista/tools/` | What the model can ask for: `@tool` turns a typed Python function into a JSON-Schema tool; the registry dispatches calls. Defines **what** a capability is, never **where** it runs. | v0.1 (MCP client: v0.2) |
| 4 | **Execution environment** | `regista/environment/` | Where effects happen: file ops and process execution behind one protocol, pinned to a workspace. `LocalEnvironment` today; a container backend is a drop-in, not a rewrite. | v0.1 |
| 5 | **Durable state** | `regista/trace/` + `session.py` | What survives a crash: **the trace IS the durable state.** It holds the full history, so resuming a session is just replaying its trace and continuing — `Agent.resume(trace_path)`. | v0.1 (resume: v0.2, shipped) |
| 6 | **Orchestration** | `regista/loop.py` | The turn engine: request → response → tool dispatch → repeat. ~250 lines, readable top-to-bottom, owns no I/O of its own. | v0.1 |
| 7 | **Subagents** | — | Child agents with isolated context, restricted policies, and budget carve-outs, linked in the parent's trace. | v0.3 |
| 8 | **Skills & procedures** | — | Reusable bundles of instruction fragments + tools, loadable into an agent. | v0.3 |
| 9 | **Verification & observability** | `regista/trace/` + `regista/policy/` | Whether it behaved: the event log, deterministic replay, OTel export; and the permission gate (Allow/Deny/Ask) consulted before every tool execution. | v0.1 (eval runner: v0.2) |

### Boundary rules

Three boundaries are subtle enough to state as rules:

- **Tools vs. environment** — a tool defines *what* capability the model sees (name, schema,
  semantics); the environment defines *where* its effects happen (local filesystem today, a
  container tomorrow). Swapping environments never changes a tool's schema, so the model
  can't tell the difference — and neither can a recorded trace.
- **Observability vs. durable state** — one artifact, the trace, serves both. Replay, resume,
  and OTel export are all *consumers* of the same log; there is no second instrumentation
  path. **If a behavior isn't in the trace, it's a bug.**
- **Policy vs. tools** — the permission gate sits in the loop, *before* dispatch. Tools never
  self-police; a tool implementation can assume it was allowed to run.

## 3. The dependency graph

Dependencies point strictly downward. Nothing imports `agent.py` except users; nothing in the
core imports `cli/` (a reserved, empty namespace for a future CLI).

```
            ┌────────────────────────────────────────────────┐
            │ agent.py / session.py  (composition root, API) │
            └───────────────────────┬────────────────────────┘
            ┌───────────────────────▼────────────────────────┐
            │ loop.py         (orchestration — turn engine)  │
            └──┬────────┬─────────┬─────────┬─────────┬──────┘
               │        │         │         │         │
        ┌──────▼───┐ ┌──▼─────┐ ┌─▼──────┐ ┌▼───────┐ ┌▼─────────────┐
        │providers/│ │ tools/ │ │policy/ │ │context/│ │instructions  │
        │ (LLM IO) │ │ (what) │ │ (gate) │ │(budget)│ │ (system/task)│
        └──────┬───┘ └──┬─────┘ └─┬──────┘ └┬───────┘ └┬─────────────┘
               │     ┌──▼───────────┐       │          │
               │     │ environment/ │       │          │
               │     │   (where)    │       │          │
               │     └──┬───────────┘       │          │
            ┌──▼────────▼────────────────────▼──────────▼──┐
            │ trace/   (flight recorder + durable state;   │
            │ replay, resume, OTel are consumers of it)    │
            └──────────────────────┬───────────────────────┘
            ┌──────────────────────▼───────────────────────┐
            │ types.py + errors.py  (shared vocabulary)    │
            └──────────────────────────────────────────────┘
```

### Component responsibilities, one line each

- **`types.py`** — the shared vocabulary: provider-neutral `Message` (typed content blocks),
  `Usage`, `StopReason`. Depends on nothing.
- **`trace/`** — the flight recorder: versioned event schema (`events.py`), crash-safe JSONL
  writer, `Trace` reader, OTel exporter.
- **`providers/`** — the LLM boundary: one protocol (`complete()` / `stream()`), four
  implementations — `anthropic`, `openai_compat`, `fake` (for tests), `replay` (see §6).
- **`tools/`** — capability registry: schema generation from Python signatures, dispatch,
  built-in tools (files/shell/search/fetch).
- **`environment/`** — effect boundary: file ops + process exec behind a protocol, workspace-scoped.
- **`policy/`** — the gate: a pure function `(PermissionRequest) → Allow | Deny | Ask`.
- **`context/`** — the memory manager: budgets and compaction.
- **`instructions.py`** — layered system-prompt construction.
- **`loop.py`** — pure orchestration over all of the above; every step emits a trace event.
- **`agent.py` / `session.py`** — the composition root users touch: `Agent` is reusable
  config; each `run()` creates a fresh `Session` with a fresh trace file.

## 4. Life of a request: `await agent.run(task)`

This is the whole system in ten steps. Everything else in the repo is detail.

```
 user ──► Agent.run(task)
            │
            ▼
      ┌─ Session ────────────────────────────────────────────────┐
      │  1. open trace file, emit session.start                  │
      │     (rendered instructions, model, tool schemas, policy) │
      │                                                          │
      │  ┌─ loop: one "turn" ─────────────────────────────────┐  │
      │  │ 2. context: check token/cost budget                │  │
      │  │ 3. assemble ModelRequest (system+history+tools)    │  │
      │  │    emit llm.request  (+ request_hash — see §6)     │  │
      │  │ 4. provider.complete(request)  ◄── the ONLY        │  │
      │  │    emit llm.response (blocks,      network call    │  │
      │  │    usage, cost, latency)                           │  │
      │  │ 5. stop_reason == "tool_use"?                      │  │
      │  │    no  ──► exit loop                               │  │
      │  │    yes ──► for each tool_use block:                │  │
      │  │      6. emit tool.call                             │  │
      │  │      7. policy(request) → Allow / Deny / Ask       │  │
      │  │         emit permission.decision                   │  │
      │  │         (Deny → tool_result(is_error=True) — the   │  │
      │  │          model sees the refusal and adapts)        │  │
      │  │      8. registry dispatches; effects go through    │  │
      │  │         the environment; emit tool.result          │  │
      │  │      9. append results to history; next turn ──►2  │  │
      │  └────────────────────────────────────────────────────┘  │
      │ 10. emit session.end (aggregates)                        │
      │     return RunResult(output, stop_reason, usage,         │
      │                      cost_usd, trace_path)               │
      └──────────────────────────────────────────────────────────┘
```

Stop conditions: the model finishes (`end_turn`), or the harness stops it (`max_turns`,
`max_cost_usd` budget, unrecoverable error). Whichever fires is recorded in `session.end`
and surfaced as `RunResult.stop_reason`.

One extra move can happen between turns: if the last response's observed input tokens
crossed the configured budget, the loop **compacts** — it asks the same provider to
summarize the older history (a regular, hash-verified `llm.request`/`llm.response` pair)
and replaces those messages with the summary, then emits `context.compaction`. Because the
summarization flows through the same provider seam, compacted sessions replay exactly.

## 5. The trace: event schema

One JSON object per line, append-only, flushed per event (a crash loses at most the event
being written). Every event shares an envelope:

```json
{"schema_version": 1, "session_id": "01J...", "seq": 17, "ts": "2026-07-02T18:04:11Z", "type": "tool.call", ...}
```

`seq` is the replay ordering key; `ts` is informational only (replay never depends on wall
clocks). The event types:

| type | carries | replay-critical? |
|---|---|---|
| `session.start` | rendered instructions, model, provider, full tool JSON schemas, policy name, context config, regista version | yes — replay reconstructs its config from this |
| `llm.request` | the full normalized `ModelRequest` + `request_hash` | yes |
| `llm.response` | full normalized response (all content blocks), usage, cost, latency, replayed flag | **yes — the replay payload** |
| `tool.call` | tool_use_id, name, input | yes |
| `tool.result` | tool_use_id, content, is_error, duration | yes (stubbed-tool replay + divergence checks) |
| `permission.decision` | decision, policy name, reason | yes (deny paths must replay identically) |
| `context.compaction` | tokens before/after, summary, dropped range | yes — changes subsequent requests |
| `error` | phase, exception type, whether retried | yes |
| `session.end` | stop reason, aggregate usage/cost, wall time | no (summary) |

The schema is versioned; any change requires a `schema_version` bump and a migration note.
Because compaction summaries, permission denials, and retries all flow through the trace,
the trace is a *complete* record of every input that shaped every request — which is what
makes §6 possible.

## 6. Deterministic replay (the differentiator)

**Replay is just another provider.** `ReplayProvider(trace)` implements the same protocol as
the Anthropic adapter; the loop, tools, policy, and tracing all run for real — only the LLM
is stubbed from the recording. This one seam is why replay costs regista almost no extra
machinery.

Mechanics:

1. Recording happens automatically — it's just the trace (§5). There is no "record mode."
2. On replay, the loop runs normally. On the Nth LLM call, `ReplayProvider` computes the
   live **`request_hash`** — SHA-256 over a canonical JSON serialization of the request
   (sorted keys; messages, tool schemas, sampling params; nothing nondeterministic) — and
   compares it to the recorded Nth request's hash.
3. **Match** → return the recorded response. Cost counts as $0. **Mismatch** → divergence:

| mode | behavior on divergence | use case |
|---|---|---|
| `strict` (default) | raise `ReplayDivergence` with a structural diff pointing at the exact field that changed | CI regression tests |
| `warn` | emit a `ReplayDivergenceWarning`, continue serving positionally | time-travel debugging |
| `hybrid` | fall through to a real `fallback` provider from that point on | resume-from-recording |

Tool execution during replay is **stubbed**: results are served from recorded
`tool.result` events by `tool_use_id` — fully hermetic, zero side effects, and it covers
permission denials too (a denial was recorded as an error tool_result, so the conversation
replays byte-identically). Live tool re-execution during replay is on the roadmap; when a
re-executed tool's output drifts, the *next* request's hash catches it.

What this buys you concretely:

- **Zero-cost regression tests**: record one real session, commit the trace, and CI re-runs
  your agent's full loop forever without an API key. regista's own CI does exactly this.
- **Time-travel debugging**: re-run a failed production session locally, with a debugger
  attached, without re-paying for tokens or re-triggering side effects.
- **Resume**: `Agent.resume(trace_path)` continues an interrupted session — `hybrid` mode
  replayed to the end of the recording, with one twist on hermetic replay: recorded tool
  calls serve their recorded results (effects are never re-run), while calls the recording
  doesn't answer — including one a crash cut short — execute live, gated by the agent's
  real policy.

Replays write their own trace (tagged `replay_of: <original session_id>`), so a replay is
itself inspectable and diffable against the original.

## 7. Providers

The protocol every adapter implements:

```python
class Provider(Protocol):
    name: str    # recorded in session.start
    model: str   # chosen where the provider is constructed — never defaulted
    async def complete(self, request: ModelRequest) -> ModelResponse: ...
    def stream(
        self, request: ModelRequest
    ) -> AsyncIterator[ProviderDelta | ModelResponse]: ...  # deltas, then the final response
```

The internal message model is a **superset of the Anthropic shape** (role + a list of typed
content blocks, including tool_use/tool_result/thinking as first-class blocks). Rich→flat
translation (for OpenAI-style APIs) is easy; flat→rich is lossy — so adapters translate
*into* the rich shape, never out of the internal one.

- **`anthropic.py`** — official SDK (which handles transient retries); maps tool_use blocks
  1:1; extracts cache read/write tokens into `Usage`; places a prompt-cache breakpoint on
  the system prompt by default; preserves thinking-block signatures.
- **`openai_compat.py`** — raw httpx against any `/v1/chat/completions` endpoint (OpenAI,
  Ollama, vLLM, gateways); translates `tool_calls` ↔ tool_use blocks; local backoff retries.
- **`fake.py`** — scripted responses, records requests. **Public API**, not a test-only
  hack: it's how users unit-test their own agents, and how contributors test loop changes
  with zero API keys. ~90% of regista's own test suite runs on it.
- **`replay.py`** — §6.

Streaming: adapters yield normalized deltas and guarantee a final accumulated
`ModelResponse`. The trace records **only the final response**, never deltas — replay
re-chunks recorded text into synthetic deltas, which keeps the trace schema simple.

## 8. Safety model (read this honestly)

regista's permission layer is a **policy gate, not a sandbox**. The `Environment` scopes file
operations and subprocess working directories to a workspace, applies timeouts, and caps
output sizes — but a shell command that is *allowed* can still do anything your user account
can do. The layered posture:

1. **Policy callback** sees every tool call (including the full shell command) before
   execution: `Allow`, `Deny` (model sees an error result and adapts), or `Ask` (escalates
   to a user-supplied handler; if none is configured, Ask auto-denies — the harness never
   hangs waiting for input that can't arrive).
2. **Environment scoping**: workspace-pinned paths and cwd (symlink escapes included),
   minimal env passthrough (no inherited API keys), hard timeouts that kill the whole
   process group, output truncation.
3. **Trace**: every decision and every effect is recorded, so there is always an audit log.

For untrusted tasks, run the whole process in a container. A `ContainerEnvironment` that
makes this a one-line change is on the roadmap — the `environment/` boundary exists
precisely so that it's a drop-in, not a rewrite. See SECURITY.md for the threat model.

## 9. Design principles (the shortlist)

1. **The trace is the product.** Every subsystem writes to it; replay/resume/OTel consume it.
2. **Replay is just another provider.** Keep the seam clean and replay stays nearly free.
3. **One primitive, one module, one interface.** The §2 table is enforceable structure, not
   documentation garnish.
4. **Small, typed, boring core.** Pydantic models, `Protocol` extension points, no metaclass
   magic, no hidden defaults (the model name is always explicit).
5. **Fail honestly.** Deny is data, not an exception; unknown model pricing is `null`, never
   a guess; the safety section above says "gate," not "sandbox."

## 10. Roadmap

- **v0.1** — everything marked v0.1 above: the loop, both providers, tools + environment +
  policy, budgets + compaction, trace + replay + OTel export.
- **v0.2** — `Agent.resume()` (shipped) · eval/regression runner (task suites with outcome
  and trace-shape assertions, replay-powered $0 CI mode) · MCP client (any MCP server's
  tools join the registry).
- **v0.3** — subagents · Skills · `ContainerEnvironment`.
- **Later** — file checkpoints/rollback, cross-session memory.
