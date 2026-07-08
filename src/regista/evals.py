"""Eval/regression runner: assert on outcomes and trace shape, live or for $0.

An :class:`EvalSuite` is a list of tasks, each with checks — small functions
that inspect the run's :class:`~regista.session.RunResult` and its
:class:`~regista.trace.reader.Trace` and return ``None`` (pass) or a failure
message. Three ways to run the same suite:

- ``await suite.run(agent)`` — live sessions, real cost. Development.
- ``await suite.record(agent)`` — live sessions whose traces are saved as
  fixtures (only when the task passes — a failing fixture is never committed).
- ``await suite.replay()`` — no agent, no keys, $0: each fixture is strictly
  replayed (any divergence fails the task — the regression signal) and the
  checks are judged against the *recorded* run, so cost/turn assertions mean
  what they meant when the fixture was recorded.

The intended CI shape is one pytest test::

    report = await suite.replay()
    assert report.passed, f"\\n{report}"
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from regista.errors import ReplayDivergence
from regista.replay import replay as _replay
from regista.session import RunResult
from regista.trace.events import ErrorEvent, ToolCall, ToolResult
from regista.trace.reader import Trace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from regista.agent import Agent
    from regista.loop import StopReason

Check = Callable[[RunResult, Trace], "str | None"]
"""A check inspects one run and returns None (pass) or a failure message.

Any function with this shape works; its ``__name__`` labels it in reports.
"""


def _short(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _named(name: str, fn: Callable[[RunResult, Trace], str | None]) -> Check:
    fn.__name__ = name
    return fn


def output_contains(text: str) -> Check:
    """The final output includes ``text``."""

    def check(result: RunResult, trace: Trace) -> str | None:
        if text in result.output:
            return None
        return f"output does not contain {text!r}: {_short(result.output)!r}"

    return _named(f"output_contains({text!r})", check)


def stop_reason_is(expected: str) -> Check:
    """The run stopped for ``expected`` ("completed", "max_turns", …)."""

    def check(result: RunResult, trace: Trace) -> str | None:
        if result.stop_reason == expected:
            return None
        return f"stop_reason is {result.stop_reason!r}, expected {expected!r}"

    return _named(f"stop_reason_is({expected!r})", check)


def max_turns_used(limit: int) -> Check:
    """The run took at most ``limit`` turns."""

    def check(result: RunResult, trace: Trace) -> str | None:
        if result.turns <= limit:
            return None
        return f"used {result.turns} turns, limit {limit}"

    return _named(f"max_turns_used({limit})", check)


def max_cost_usd(limit: float) -> Check:
    """The run cost at most ``limit`` dollars (recorded cost, under replay)."""

    def check(result: RunResult, trace: Trace) -> str | None:
        if result.cost_usd <= limit:
            return None
        return f"cost ${result.cost_usd:.4f}, limit ${limit:.4f}"

    return _named(f"max_cost_usd({limit})", check)


def tool_was_called(name: str) -> Check:
    """At least one ``tool.call`` for ``name`` appears in the trace."""

    def check(result: RunResult, trace: Trace) -> str | None:
        if any(isinstance(e, ToolCall) and e.name == name for e in trace):
            return None
        return f"tool {name!r} was never called"

    return _named(f"tool_was_called({name!r})", check)


def tool_never_called(name: str) -> Check:
    """No ``tool.call`` for ``name`` appears in the trace."""

    def check(result: RunResult, trace: Trace) -> str | None:
        count = sum(1 for e in trace if isinstance(e, ToolCall) and e.name == name)
        if count == 0:
            return None
        return f"tool {name!r} was called {count}x"

    return _named(f"tool_never_called({name!r})", check)


def no_errors() -> Check:
    """The trace has no error events and no is_error tool results."""

    def check(result: RunResult, trace: Trace) -> str | None:
        errors = [e for e in trace if isinstance(e, ErrorEvent)]
        failed_tools = [e for e in trace if isinstance(e, ToolResult) and e.is_error]
        if not errors and not failed_tools:
            return None
        return f"{len(errors)} error event(s), {len(failed_tools)} failed tool result(s)" + (
            f"; first: {_short(errors[0].message)}" if errors else ""
        )

    return _named("no_errors", check)


@dataclass(frozen=True)
class EvalTask:
    """One task in a suite: a prompt, its checks, and (optionally) where its
    recorded fixture lives for ``record()``/``replay()``."""

    name: str
    task: str
    checks: Sequence[Check] = ()
    trace: Path | str | None = None


@dataclass(frozen=True)
class CheckOutcome:
    name: str
    passed: bool
    message: str | None = None


@dataclass(frozen=True)
class TaskReport:
    name: str
    passed: bool
    checks: list[CheckOutcome]
    result: RunResult | None = None
    error: str | None = None  # the run itself failed (nothing to judge)


@dataclass(frozen=True)
class EvalReport:
    """The suite verdict. ``assert report.passed, f"\\n{report}"`` reads well."""

    tasks: list[TaskReport]

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.tasks)

    @property
    def total_cost_usd(self) -> float:
        return sum(t.result.cost_usd for t in self.tasks if t.result is not None)

    def __str__(self) -> str:
        done = sum(1 for t in self.tasks if t.passed)
        lines = [f"{done}/{len(self.tasks)} tasks passed, cost ${self.total_cost_usd:.4f}"]
        for t in self.tasks:
            lines.append(f"  {'PASS' if t.passed else 'FAIL'}  {t.name}")
            if t.error is not None:
                lines.append(f"        run failed: {_short(t.error, 200)}")
            for c in t.checks:
                if not c.passed:
                    lines.append(f"        {c.name}: {c.message}")
        return "\n".join(lines)


class EvalSuite:
    """A named set of tasks judged by the same machinery live or replayed."""

    def __init__(self, tasks: Sequence[EvalTask]) -> None:
        names = [t.name for t in tasks]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate task names: {duplicates}")
        self._tasks = list(tasks)

    async def run(self, agent: Agent) -> EvalReport:
        """Run every task live against ``agent`` and judge the checks."""
        return EvalReport([await self._run_one(task, agent) for task in self._tasks])

    async def record(self, agent: Agent) -> EvalReport:
        """Run live and save each *passing* task's trace to its fixture path.

        A failing task's fixture is never written — you can't accidentally
        commit a regression as the new baseline. Tasks without a ``trace``
        path just run.
        """
        reports = []
        for task in self._tasks:
            report = await self._run_one(task, agent)
            if report.passed and task.trace is not None and report.result is not None:
                destination = Path(task.trace)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(report.result.trace_path, destination)
            reports.append(report)
        return EvalReport(reports)

    async def replay(self, *, trace_dir: Path | str | None = None) -> EvalReport:
        """Judge every task against its recorded fixture, for $0.

        Each fixture is strictly replayed first — a `ReplayDivergence` fails
        the task, because it means the harness/tool/prompt behavior no longer
        reproduces the recording. The checks are then judged against the
        *recorded* run, so cost and turn numbers are the original ones.
        """
        reports = []
        for task in self._tasks:
            reports.append(await self._replay_one(task, trace_dir))
        return EvalReport(reports)

    async def _run_one(self, task: EvalTask, agent: Agent) -> TaskReport:
        try:
            result = await agent.run(task.task)
        except Exception as exc:  # one broken task must not sink the suite
            return TaskReport(task.name, False, [], error=f"{type(exc).__name__}: {exc}")
        # a run that ended in an error outcome fails its task regardless of checks
        return _judge(task, result, Trace.load(result.trace_path), run_error=result.error)

    async def _replay_one(self, task: EvalTask, trace_dir: Path | str | None) -> TaskReport:
        if task.trace is None:
            return TaskReport(task.name, False, [], error="task has no trace fixture path")
        fixture = Path(task.trace)
        if not fixture.exists():
            return TaskReport(
                task.name, False, [], error=f"fixture {fixture} not found — run suite.record()"
            )
        recorded = Trace.load(fixture)
        result = _result_from_trace(recorded)
        if result is None:
            return TaskReport(
                task.name, False, [], error=f"fixture {fixture} has no session.end (crashed run?)"
            )
        try:
            await _replay(fixture, trace_dir=trace_dir)
            faithful = CheckOutcome("strict_replay", True)
        except ReplayDivergence as exc:
            faithful = CheckOutcome("strict_replay", False, _short(str(exc), 400))
        return _judge(task, result, recorded, extra=[faithful])


def _judge(
    task: EvalTask,
    result: RunResult,
    trace: Trace,
    extra: list[CheckOutcome] | None = None,
    run_error: str | None = None,
) -> TaskReport:
    outcomes = list(extra or [])
    for check in task.checks:
        try:
            message = check(result, trace)
        except Exception as exc:  # a buggy check is a failed check, not a crash
            message = f"check raised {type(exc).__name__}: {exc}"
        outcomes.append(CheckOutcome(getattr(check, "__name__", "check"), message is None, message))
    passed = run_error is None and all(o.passed for o in outcomes)
    return TaskReport(task.name, passed, outcomes, result=result, error=run_error)


def _result_from_trace(trace: Trace) -> RunResult | None:
    """Rebuild the RunResult a finished recording summarized in session.end."""
    end = trace.end
    if end is None:
        return None
    return RunResult(
        session_id=trace.start.session_id,
        output=end.final_output,
        stop_reason=cast("StopReason", end.stop_reason),
        usage=end.usage,
        cost_usd=end.cost_usd,
        turns=end.turns,
        trace_path=trace.path if trace.path is not None else Path(),
    )
