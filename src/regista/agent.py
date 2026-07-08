"""The composition root: Agent wires every primitive together.

An Agent is reusable configuration — provider, instructions, tools, policy,
budgets. Each ``run()`` creates a fresh Session (new ULID, new trace file), so
one Agent can serve many tasks concurrently without shared mutable state.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from regista.context import ContextConfig
from regista.errors import ConfigurationError, SubagentError
from regista.instructions import Instructions
from regista.loop import LoopConfig
from regista.policy import allow_all
from regista.replay import resume_from_trace
from regista.session import Session
from regista.streaming import RunCompleted
from regista.tools import Tool, ToolRegistry
from regista.types import ToolSpec

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from regista.policy import AskHandler, PermissionPolicy
    from regista.pricing import ModelPrice
    from regista.providers.base import Provider
    from regista.session import RunResult
    from regista.skills import Skill
    from regista.streaming import StreamEvent


class Agent:
    """Reusable agent configuration; ``run()`` executes one traced session.

    >>> agent = Agent(provider=..., instructions="You are a helpful agent.")
    >>> result = await agent.run("Summarize README.md")
    """

    def __init__(
        self,
        *,
        provider: Provider,
        instructions: str | Instructions,
        tools: list[Tool] | None = None,
        skills: list[Skill] | None = None,
        policy: PermissionPolicy | None = None,
        ask_handler: AskHandler | None = None,
        trace_dir: Path | str = "./.regista/traces",
        max_turns: int = 50,
        max_cost_usd: float | None = None,
        max_tokens: int = 8192,
        params: dict[str, Any] | None = None,
        price_overrides: dict[str, ModelPrice] | None = None,
        context: ContextConfig | None = None,
    ) -> None:
        if max_turns < 1:
            raise ConfigurationError(f"max_turns must be >= 1, got {max_turns}")
        self.instructions = Instructions.coerce(instructions)
        self.skill_names = tuple(s.name for s in skills or [])
        for skill in skills or []:
            self.instructions = self.instructions.with_section(
                f"Skill: {skill.name}", skill.instructions
            )
        all_tools = [*(tools or []), *(t for s in skills or [] for t in s.tools)]
        self.trace_dir = trace_dir
        self._config = LoopConfig(
            provider=provider,
            registry=ToolRegistry(all_tools),
            policy=policy if policy is not None else allow_all(),
            ask_handler=ask_handler,
            system=self.instructions.render(),
            max_turns=max_turns,
            max_cost_usd=max_cost_usd,
            max_tokens=max_tokens,
            params=params or {},
            price_overrides=price_overrides,
            context=context or ContextConfig(),
        )

    async def run(self, task: str) -> RunResult:
        """Run one task to completion in a fresh traced session."""
        return await Session(task, self._config, self.trace_dir, skills=self.skill_names).run()

    async def resume(self, trace_path: Path | str) -> RunResult:
        """Continue an interrupted session from its trace.

        The recorded prefix replays for $0 — every request hash-verified,
        recorded tool results served without re-running their effects — and
        the first request the recording can't answer falls through to this
        agent's live provider. From there new tool calls execute for real,
        gated by this agent's policy. A tool call the crash cut short (no
        recorded result) is re-executed.

        Assumes this Agent is configured like the recorded run; any
        difference makes the first request diverge, which just means the
        whole task re-runs live. Writes a new trace linked to the original
        via ``replay_of``.
        """
        return await resume_from_trace(trace_path, self._config, self.trace_dir)

    async def stream(self, task: str) -> AsyncIterator[StreamEvent]:
        """Run one task, yielding events as they happen.

        Yields TextDelta/ThinkingDelta/ToolCallStarted/ToolCallFinished/
        TurnCompleted as the session progresses, then RunCompleted with the
        same RunResult ``run()`` would return. The trace is identical to a
        blocking run — streaming changes when you see things, not what
        happened.
        """
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        session = Session(task, self._config, self.trace_dir, skills=self.skill_names)

        async def pump() -> RunResult:
            try:
                return await session.run(on_event=queue.put_nowait)
            finally:
                queue.put_nowait(None)  # sentinel: no more events

        runner = asyncio.ensure_future(pump())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            yield RunCompleted(await runner)  # re-raises if the session raised
        finally:
            if not runner.done():
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)

    def as_tool(self, *, name: str, description: str, parallel_safe: bool = False) -> Tool:
        """Expose this agent as a tool of another agent — a subagent.

        The parent model calls ``{name}(task=...)``; this agent runs the task
        in its own fresh session — isolated context, its own policy, its own
        ``max_turns``/``max_cost_usd`` (the budget carve-out) — and the final
        output becomes the tool result. The child writes its own trace, tagged
        ``parent_session_id``, while the parent's trace records the delegation
        like any tool call — so a parent replay is hermetic (the child is
        never re-run) and the child's trace replays independently.

        A child run that ends in an error outcome raises `SubagentError`,
        which the registry hands to the parent model as error-data.
        """

        async def run_subagent(task: str) -> str:
            result = await self.run(task)
            if result.error is not None:
                raise SubagentError(f"subagent '{name}' failed: {result.error}")
            return result.output

        spec = ToolSpec(
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "properties": {"task": {"type": "string", "description": "The task to delegate."}},
                "required": ["task"],
                "additionalProperties": False,
            },
            parallel_safe=parallel_safe,
        )
        return Tool(run_subagent, spec)

    def run_sync(self, task: str) -> RunResult:
        """``run()`` for synchronous callers (scripts, notebooks without top-level await)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(task))
        raise ConfigurationError(
            "run_sync() was called inside a running event loop; use `await agent.run(task)`"
        )
