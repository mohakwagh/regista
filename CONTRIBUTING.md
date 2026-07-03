# Contributing to regista

Thanks for your interest! regista is deliberately structured to be easy to contribute to.

## Getting oriented

Read [ARCHITECTURE.md](ARCHITECTURE.md) first — it maps the nine harness primitives to the
nine modules, so you can pick the area you care about and know exactly which directory it
lives in.

## Dev setup (three commands)

```bash
uv sync --group dev
uv run pytest
uv run ruff check . && uv run mypy
```

No API keys required: the test suite runs on `FakeProvider` and recorded trace fixtures.
Live tests are opt-in (`REGISTA_LIVE_TESTS=1`) and never run in PR CI.

## Ground rules

- **Every behavior must emit a trace event.** If your change alters what the loop does and
  the trace can't show it, that's a bug (and replay will break).
- Trace schema changes require a `schema_version` bump and a migration note in the PR.
- `mypy --strict` and `ruff` must pass; new code needs tests (FakeProvider makes this cheap).
- PR titles follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`,
  `fix:`, `docs:`, ...). Individual commit messages are up to you.

## Not sure where to start?

Issues labeled `good first issue` are scoped and mentored. Opening an issue to discuss an
idea before writing code is always welcome.
