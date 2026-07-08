# Evals (the regression runner)

**Whether it behaved — checked continuously.** Module: `regista/evals.py`.

An `EvalSuite` is a list of tasks, each judged by **checks**: small functions that inspect
one run's `RunResult` and its `Trace` and return `None` (pass) or a failure message.

```python
from regista.evals import (
    EvalSuite, EvalTask,
    output_contains, stop_reason_is, max_turns_used,
    max_cost_usd, tool_was_called, tool_never_called, no_errors,
)

suite = EvalSuite([
    EvalTask(
        name="fixes the failing test",
        task="Run the tests and fix the failure",
        checks=[
            output_contains("all tests pass"),
            tool_was_called("shell"),
            tool_never_called("fetch"),
            max_turns_used(10),
            max_cost_usd(0.50),
            no_errors(),
        ],
        trace="tests/fixtures/fixes_failing_test.jsonl",
    ),
])
```

Because checks see the whole trace, they assert on *shape*, not just outcome: which tools
ran, how many turns, what it cost. A custom check is any `(RunResult, Trace) -> str | None`
function — its `__name__` labels it in the report.

## Three ways to run one suite

```python
report = await suite.run(agent)      # live sessions, real cost — development
report = await suite.record(agent)   # live + save passing traces as fixtures
report = await suite.replay()        # $0: no agent, no keys — CI
assert report.passed, f"\n{report}"
```

- **`run`** executes every task in a fresh session. A run that ends in an error outcome
  fails its task regardless of checks.
- **`record`** is `run` plus fixture-saving — and it only writes a task's fixture when the
  task *passed*, so a regression can never quietly become the new baseline.
- **`replay`** is the payoff: each fixture is strictly [replayed](replay.md) (a
  `ReplayDivergence` fails the task — the harness/prompt/tool behavior no longer reproduces
  the recording), then the checks are judged against the **recorded** run, so cost and turn
  numbers mean what they meant on recording day. This is the "$0 agent regression tests in
  CI" story: record once, assert forever.

The intended CI shape is one pytest test:

```python
async def test_agent_regressions():
    report = await suite.replay()
    assert report.passed, f"\n{report}"
```

A failing report prints one line per task and one per failed check:

```
1/2 tasks passed, cost $0.4210
  PASS  fixes the failing test
  FAIL  refuses out-of-scope work
        output_contains('cannot'): output does not contain 'cannot': 'Sure, deleting...'
        strict_replay: request 3 diverged from the recording (trace seq 14) ...
```
