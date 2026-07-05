# regista

> The agent harness with a flight recorder.

**regista** is an observability-first agent harness for Python: the runtime layer that turns
a stateless LLM API into an agent that can act. Every session is recorded as a structured,
append-only trace — complete enough to **deterministically replay** the entire session with
zero API calls and zero cost.

> *regista* (Italian): the deep-lying playmaker who directs the game. Also: "director."

**Status: pre-release.** The v0.1 core is feature-complete — loop, tools, environment,
policy, both providers, streaming, compaction, replay, OTel export — with 140+ tests, strict
mypy, and every subsystem traced. Docs site and the PyPI release are landing next.

## Why another harness?

Every serious agent harness treats observability as an add-on or a paid platform. regista
treats **the trace as the product**:

- 🔍 **Structured event log** — every LLM call, tool execution, permission decision, token
  count, and dollar of cost, as typed JSONL events. If a behavior isn't in the trace, it's a bug.
- ⏪ **Deterministic replay** — re-run any recorded session with the LLM and tools served
  from the recording: hermetic, keyless, $0. Strict mode hash-verifies every request and
  fails with a structural diff — zero-cost regression tests for CI.
- 📊 **OpenTelemetry export** — the same trace as session/turn/llm/tool spans in
  Jaeger/Grafana, post-hoc, with the recorded timestamps.
- 🔌 **Provider-neutral** — native Anthropic adapter plus an OpenAI-compatible adapter
  (OpenAI, Ollama, vLLM, gateways). Streaming on both.
- 🛡️ **Permission gate** — Allow/Deny/Ask policies on every tool call, workspace-scoped
  execution, honestly documented as a gate (not a sandbox).

## A coding agent in ~30 lines

```python
import asyncio

from regista import Agent
from regista.environment import LocalEnvironment
from regista.policy import PermissionRequest, workspace
from regista.providers import AnthropicProvider
from regista.tools.builtin import builtin_tools

env = LocalEnvironment("./sandbox")          # every file op is pinned here

async def approve(request: PermissionRequest) -> bool:
    return input(f"allow {request.tool_input}? [y/N] ") == "y"

agent = Agent(
    provider=AnthropicProvider("claude-sonnet-4-6"),   # the model is always explicit
    instructions="You are a careful coding agent. Run the tests after every change.",
    tools=builtin_tools(env),                # read/write/list/glob/search/shell/fetch
    policy=workspace(),                      # file tools allowed; shell escalates to Ask
    ask_handler=approve,
    max_turns=30,
    max_cost_usd=1.00,                       # hard budget, from provider-reported usage
)

result = asyncio.run(agent.run("pytest is failing in this repo — find the bug and fix it"))
print(result.output, f"${result.cost_usd:.2f}", result.trace_path)
```

The same `Agent` streams: `async for event in agent.stream(task)` yields text deltas, tool
start/finish, and per-turn usage as they happen — and writes the identical trace.

## The flight recorder

Every run writes a JSONL trace: `session.start`, `llm.request` (with a SHA-256
`request_hash`), `llm.response`, `tool.call`, `permission.decision`, `tool.result`,
`context.compaction`, `session.end`. That log is complete enough to re-run the session:

```python
from regista import replay

replayed = await replay(result.trace_path)   # no API key, no network, no side effects
assert replayed.cost_usd == 0.0
assert replayed.output == result.output
```

Replay is hash-verified call by call. Change anything that shaped a request — the prompt, a
tool schema, a sampling param — and strict mode fails with a diff pointing at exactly what
drifted:

```
ReplayDivergence: request 1 diverged from the recording (trace seq 1)
request.system: recorded 'You are a greeter.' != live 'You are a greeter. Always answer in French.'
```

`mode="warn"` keeps serving positionally for time-travel debugging; `mode="hybrid"` falls
through to a live provider mid-session — which is how a crashed trace resumes.

## Zero-cost testing

`FakeProvider` is public API. Script the model, run the real loop, assert on the real trace:

```python
from regista.providers import FakeProvider, text_response, tool_use_response

provider = FakeProvider([
    tool_use_response(("tu_1", "greet", {"name": "world"})),
    text_response("done"),
])
```

~90% of regista's own test suite runs on it — and your agent's tests can too. Committed
traces replayed strictly in CI are your end-to-end tests, forever, for free.

## Architecture

regista is organized around **nine harness primitives** — instructions, context management,
tool interface, execution environment, durable state, orchestration, subagents, skills,
verification & observability — each mapping 1:1 to a module with one interface.
[ARCHITECTURE.md](ARCHITECTURE.md) explains the whole system from first principles,
including the dependency diagram and the life of a request; it's written for people who
have never built an agent.

## Try it

On PyPI the distribution is named **`regista-harness`** (the bare name was squatted);
the import name is `regista` everywhere. From source:

```bash
git clone https://github.com/mohakwagh/regista && cd regista
uv sync
uv run python examples/01_hello_agent.py      # $0: scripted model, real loop, real trace
uv run python examples/05_replay.py           # $0: replay + a divergence diff
ANTHROPIC_API_KEY=... uv run python examples/04_real_provider.py   # ~1 cent
```

## Roadmap

- **v0.2** — eval/regression runner (replay-powered $0 CI) · MCP client · `Session.resume()`
- **v0.3** — subagents · Skills · `ContainerEnvironment`

## Safety, honestly

The permission layer is a **policy gate, not a sandbox**: an allowed shell command can do
anything your user account can. The environment scopes paths (symlinks included), strips
secrets from subprocess env, and enforces timeouts — and every decision is in the trace.
For untrusted tasks, run the process in a container. [SECURITY.md](SECURITY.md) has the
full threat model.

## License

[MIT](LICENSE)
