"""EvalSuite: the same checks judge live runs, recordings, and $0 replays."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from regista import Agent, tool
from regista.evals import (
    EvalReport,
    EvalSuite,
    EvalTask,
    TaskReport,
    max_cost_usd,
    max_turns_used,
    no_errors,
    output_contains,
    stop_reason_is,
    tool_never_called,
    tool_was_called,
)
from regista.providers.fake import FakeProvider, text_response, tool_use_response

if TYPE_CHECKING:
    from pathlib import Path

    from regista.session import RunResult
    from regista.trace.reader import Trace


@tool
def echo(text: str) -> str:
    """Echo the text back.

    Args:
        text: What to echo.
    """
    return f"echo: {text}"


def scripted_provider() -> FakeProvider:
    return FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"})),
            text_response("done"),
        ]
    )


def make_agent(tmp_path: Path, provider: FakeProvider | None = None) -> Agent:
    return Agent(
        provider=provider or scripted_provider(),
        instructions="You are a test agent.",
        tools=[echo],
        trace_dir=tmp_path / "traces",
    )


def standard_checks() -> list:  # type: ignore[type-arg]
    return [
        output_contains("done"),
        stop_reason_is("completed"),
        max_turns_used(3),
        max_cost_usd(0.01),
        tool_was_called("echo"),
        tool_never_called("shell"),
        no_errors(),
    ]


async def test_live_run_judges_all_checks(tmp_path: Path) -> None:
    suite = EvalSuite([EvalTask(name="echoes", task="Say hi", checks=standard_checks())])

    report = await suite.run(make_agent(tmp_path))

    assert report.passed
    assert len(report.tasks[0].checks) == 7
    assert "1/1 tasks passed" in str(report)


async def test_failing_checks_name_themselves(tmp_path: Path) -> None:
    suite = EvalSuite(
        [
            EvalTask(
                name="impossible",
                task="Say hi",
                checks=[output_contains("MISSING"), tool_was_called("shell")],
            )
        ]
    )

    report = await suite.run(make_agent(tmp_path))

    assert not report.passed
    rendered = str(report)
    assert "FAIL  impossible" in rendered
    assert "output_contains('MISSING')" in rendered
    assert "tool 'shell' was never called" in rendered


async def test_a_crashing_run_fails_only_its_task(tmp_path: Path) -> None:
    good = EvalTask(name="good", task="Say hi", checks=[output_contains("done")])
    bad = EvalTask(name="bad", task="Say hi", checks=[])
    # one agent, two tasks, but the script only covers the first run
    agent = make_agent(tmp_path)
    suite = EvalSuite([good, bad])

    report = await suite.run(agent)

    assert [t.passed for t in report.tasks] == [True, False]
    assert report.tasks[1].error is not None
    assert "script exhausted" in report.tasks[1].error


async def test_record_then_replay_for_zero_dollars(tmp_path: Path) -> None:
    fixture = tmp_path / "fixtures" / "echoes.jsonl"
    suite = EvalSuite(
        [EvalTask(name="echoes", task="Say hi", checks=standard_checks(), trace=fixture)]
    )

    recorded = await suite.record(make_agent(tmp_path))
    assert recorded.passed
    assert fixture.exists()

    # no agent, no provider, no keys — judged against the recording
    report = await suite.replay(trace_dir=tmp_path / "replays")
    assert report.passed
    assert report.tasks[0].checks[0].name == "strict_replay"
    assert report.tasks[0].checks[0].passed


async def test_record_never_saves_a_failing_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "fixtures" / "never.jsonl"
    suite = EvalSuite(
        [EvalTask(name="never", task="Say hi", checks=[output_contains("MISSING")], trace=fixture)]
    )

    report = await suite.record(make_agent(tmp_path))

    assert not report.passed
    assert not fixture.exists()


async def test_replay_flags_a_tampered_fixture_as_divergence(tmp_path: Path) -> None:
    fixture = tmp_path / "fixtures" / "echoes.jsonl"
    suite = EvalSuite(
        [EvalTask(name="echoes", task="Say hi", checks=[output_contains("done")], trace=fixture)]
    )
    await suite.record(make_agent(tmp_path))

    lines = fixture.read_text().splitlines()
    start = json.loads(lines[0])
    start["instructions"] = "You are a DIFFERENT agent."
    fixture.write_text("\n".join([json.dumps(start), *lines[1:]]) + "\n")

    report = await suite.replay(trace_dir=tmp_path / "replays")

    assert not report.passed
    strict = report.tasks[0].checks[0]
    assert strict.name == "strict_replay" and not strict.passed
    assert strict.message is not None and "diverged" in strict.message
    # the ordinary checks still judged the recording itself
    assert report.tasks[0].checks[1].passed


async def test_replay_without_a_fixture_fails_helpfully(tmp_path: Path) -> None:
    missing = EvalTask(name="missing", task="x", trace=tmp_path / "nope.jsonl")
    pathless = EvalTask(name="pathless", task="x")

    report = await EvalSuite([missing, pathless]).replay()

    assert not report.passed
    assert "run suite.record()" in (report.tasks[0].error or "")
    assert "no trace fixture path" in (report.tasks[1].error or "")


async def test_custom_checks_are_plain_functions(tmp_path: Path) -> None:
    def echo_ran_before_answering(result: RunResult, trace: Trace) -> str | None:
        return None if trace.tool_results() else "no tool ran"

    suite = EvalSuite([EvalTask(name="custom", task="Say hi", checks=[echo_ran_before_answering])])
    report = await suite.run(make_agent(tmp_path))

    assert report.passed
    assert report.tasks[0].checks[0].name == "echo_ran_before_answering"


async def test_a_raising_check_is_a_failed_check(tmp_path: Path) -> None:
    def buggy(result: RunResult, trace: Trace) -> str | None:
        raise RuntimeError("oops")

    report = await EvalSuite([EvalTask(name="t", task="Say hi", checks=[buggy])]).run(
        make_agent(tmp_path)
    )

    assert not report.passed
    assert "check raised RuntimeError: oops" in (report.tasks[0].checks[0].message or "")


def test_duplicate_task_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate task names"):
        EvalSuite([EvalTask(name="x", task="a"), EvalTask(name="x", task="b")])


def test_report_totals_recorded_cost() -> None:
    report = EvalReport(
        tasks=[
            TaskReport("a", True, [], result=None),
            TaskReport("b", False, [], error="boom"),
        ]
    )
    assert report.total_cost_usd == 0.0
    assert not report.passed
