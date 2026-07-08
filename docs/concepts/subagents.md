# Subagents

**Delegation with boundaries.** API: `Agent.as_tool()`.

A subagent is an `Agent` exposed as a tool of another agent:

```python
researcher = Agent(
    provider=...,
    instructions="You are a research specialist.",
    tools=[search, fetch],
    policy=read_only(),          # the child can be *more* restricted
    max_cost_usd=0.50,           # the budget carve-out: the child polices itself
)

parent = Agent(
    provider=...,
    instructions="You are an orchestrator. Delegate research.",
    tools=[researcher.as_tool(
        name="researcher",
        description="Delegate a research question to the research subagent.",
    )],
)
```

The parent model sees an ordinary tool — `researcher(task="...")`. When it calls, the
child runs the task in a **fresh session**: isolated context (it sees only the delegated
task, none of the parent's conversation), its own policy, its own `max_turns` and
`max_cost_usd`. The child's final output becomes the tool result.

## What lands in the traces

- The **child writes its own trace**, tagged `parent_session_id`, so the whole delegation
  tree is reconstructable from trace files alone — nesting included (children can have
  children).
- The **parent's trace records the delegation as a tool call** with the child's output as
  the result. That means a parent replay is hermetic: the child is *never re-run*, its
  recorded output is served like any tool result. The child's trace replays independently.
- A child run that ends in an error outcome raises `SubagentError` inside the tool, which
  the registry converts to `tool_result(is_error=True)` — the parent model sees the
  failure as data and can adapt.

## Honest limits (v0.3)

The parent's `max_cost_usd` does not see the child's spend — budgets are per-session, and
the carve-out is the child's own limit. Child streaming events don't propagate to the
parent's `stream()`. Both are candidates for later versions.
