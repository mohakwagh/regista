# Instructions

**What the agent is told.** Module: `regista/instructions.py`.

An `Instructions` object is a layered system prompt: a base identity plus named sections,
rendered once per session and recorded verbatim in the `session.start` trace event.

```python
from regista import Agent, Instructions

instructions = Instructions(
    base="You are a code-maintenance agent.",
    sections={"Style": "Prefer small diffs. Never reformat unrelated code."},
)
print(instructions.render())
# You are a code-maintenance agent.
#
# ## Style
#
# Prefer small diffs. Never reformat unrelated code.

agent = Agent(provider=..., instructions=instructions)
```

A plain string works too — `Agent` coerces it to `Instructions(base=...)`:

```python
agent = Agent(provider=..., instructions="You are a greeter.")
```

`with_section(title, body)` returns a new frozen copy, which is how Skills (v0.3) will
contribute instruction fragments without mutating anything.

## Why it matters for replay

The rendered string is part of every `ModelRequest`, so it is part of every
`request_hash`. Change one word of the prompt and a strict replay fails with a diff
pointing at `request.system` — prompt drift is a first-class, detectable change.
