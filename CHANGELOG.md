# Changelog

All notable changes to regista are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **MCP client** (`regista.tools.mcp`, `[mcp]` extra): `MCPServer.stdio(...)` /
  `MCPServer.http(...)` connect to any Model Context Protocol server and wrap its tools
  as ordinary regista tools — same trace events, same policy gate, `prefix=` for
  namespacing, server failures surfaced as error-data. Sessions that used MCP tools
  replay hermetically without the server.
- **`Agent.resume(trace_path)`**: continue an interrupted session from its trace. The
  recorded prefix replays for $0 (hash-verified, recorded tool results served — effects
  never re-run); the first request the recording can't answer falls through to the
  agent's live provider, and new tool calls execute for real, gated by the agent's
  policy. A tool call the crash cut short is re-executed. The resumed session's trace
  links to the original via `replay_of`.

## [0.1.0] — 2026-07-05

The PyPI distribution is `regista-harness` (the bare name `regista` was squatted by a
placeholder upload; a PEP 541 claim is pending). The import name is `regista`.

### Added
- **Core loop**: `Agent` / `Session` / `run_loop` — traced turn engine with
  `max_turns` and `max_cost_usd` stops, parallel-safe tool batching, and a
  `run_sync()` facade.
- **Types**: provider-neutral message vocabulary (text/thinking/tool_use/tool_result
  blocks), `Usage`, `ToolSpec`.
- **Trace**: versioned JSONL event schema (v1), crash-safe writer, `Trace` reader,
  and post-hoc OpenTelemetry export (`[otel]` extra) with `gen_ai.*` span attributes.
- **Deterministic replay**: `replay(trace_path)` with strict/warn/hybrid divergence
  modes, per-call `request_hash` verification, structural diffs, stubbed tool results,
  `replay_of` trace linkage, and $0 accounting.
- **Tools**: `@tool` decorator (signature + Google-docstring → JSON Schema, fail-fast),
  registry with error-as-data execution, and seven environment-backed built-ins
  (read_file, write_file, list_dir, glob, search_files, shell, fetch) with output caps.
- **Execution environment**: `Environment` protocol and `LocalEnvironment` —
  workspace-pinned paths (symlink escapes rejected via `WorkspaceViolation`),
  allowlisted subprocess env, hard timeouts.
- **Policy**: `Allow`/`Deny`/`Ask` permission gate in the loop with traced decisions;
  presets `allow_all`, `read_only`, `workspace`, `compose`; Ask auto-denies without a
  handler.
- **Providers**: `AnthropicProvider` (official SDK, prompt-cache breakpoint, cache
  tokens in `Usage`, thinking signatures preserved), `OpenAICompatProvider` (raw httpx
  for OpenAI/Ollama/vLLM with local backoff), `FakeProvider` (public, scripted), and
  `ReplayProvider`.
- **Streaming**: `agent.stream()` and provider-native `stream()` on all four adapters;
  the trace records final responses only, so streamed sessions replay identically.
- **Context management**: `ContextConfig` budget-triggered compaction — summarization
  through the session's own provider, traced and hash-replayable.
- **Pricing**: best-effort cost from provider-reported usage with per-Agent overrides;
  unknown models cost `None`, never a guess.
- **Instructions**: layered system prompt (`base` + sections), recorded in
  `session.start`.
- Docs site (mkdocs-material, page per primitive), runnable `examples/01–05`,
  opt-in live smoke test, and a 90% coverage gate in CI.
