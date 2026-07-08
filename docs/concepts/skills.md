# Skills

**Reusable expertise.** Module: `regista/skills.py`.

A `Skill` is declarative data: a named instruction fragment plus the tools that make it
actionable. Loading skills into an `Agent` appends each fragment to the system prompt as
an [Instructions](instructions.md) section titled `Skill: {name}` and registers the tools
alongside the agent's own.

```python
from regista import Agent, Skill

reviewer = Skill(
    name="code-review",
    instructions="Review diffs for correctness first, style second. Cite line numbers.",
    tools=[fetch_diff, post_comment],
)

agent = Agent(provider=..., instructions="You are a maintainer.", skills=[reviewer])
```

Skills compose: pass several and each becomes its own section, each tool joins the one
registry (a name collision with an existing tool fails fast at construction, like any
duplicate).

## Nothing new to trace

A skill is sugar over two primitives that are already recorded: the rendered instructions
(fragments included) and the full tool schemas land in `session.start` exactly as always,
plus the skill *names* for provenance. A skilled session therefore replays like any other —
there is no separate "skill state" to persist or reconstruct.
