# Quickstart

## Install

Until the PyPI release lands:

```bash
git clone https://github.com/mohakwagh/regista && cd regista
uv sync
```

Once released, the distribution is **`regista-harness`** (`pip install regista-harness`);
the import name stays `regista`.

Runtime dependencies are just pydantic v2, httpx, and the anthropic SDK. Python 3.10+.

## An agent with no API key

`FakeProvider` is public API: you script the model's responses, the **real** loop runs —
tools execute, policies gate, the trace is written.

```python
import asyncio

from regista import Agent, tool
from regista.providers import FakeProvider, text_response, tool_use_response

@tool
def greet(name: str) -> str:
    """Greet someone by name.

    Args:
        name: Who to greet.
    """
    return f"Hello, {name}!"

provider = FakeProvider([
    tool_use_response(("tu_1", "greet", {"name": "world"})),
    text_response("Greeted the world."),
])

agent = Agent(provider=provider, instructions="You are a greeter.", tools=[greet])
result = asyncio.run(agent.run("Greet the world"))
print(result.output, result.trace_path)
```

Everything the harness did is in `result.trace_path` — one JSON event per line.

## A real agent

Swap one line:

```python
from regista.providers import AnthropicProvider

agent = Agent(
    provider=AnthropicProvider("claude-sonnet-4-6"),   # reads ANTHROPIC_API_KEY
    instructions="You are a helpful research assistant.",
    tools=[greet],
    max_turns=20,
    max_cost_usd=1.00,
)
```

Or point the OpenAI-compatible adapter at any Chat Completions endpoint:

```python
from regista.providers import OpenAICompatProvider

provider = OpenAICompatProvider("llama3.1", base_url="http://localhost:11434/v1")  # Ollama
```

## Give it hands: built-in tools + an environment

```python
from regista.environment import LocalEnvironment
from regista.policy import workspace
from regista.tools.builtin import builtin_tools

env = LocalEnvironment("./sandbox")     # file effects are pinned to this directory

agent = Agent(
    provider=AnthropicProvider("claude-sonnet-4-6"),
    instructions="You are a coding agent.",
    tools=builtin_tools(env),           # read/write/list/glob/search/shell/fetch
    policy=workspace(),                 # file tools allowed; shell escalates to Ask
)
```

## Replay it

```python
from regista import replay

replayed = asyncio.run(replay(result.trace_path))
assert replayed.cost_usd == 0.0
```

See [Deterministic replay](concepts/replay.md) for divergence modes and what makes this
work. The `examples/` directory in the repo has five runnable scripts covering all of the
above.
