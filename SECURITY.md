# Security Policy

## Threat model (read this before running agents)

regista's permission layer is a **policy gate, not a sandbox**:

- The policy callback (Allow/Deny/Ask) intercepts every tool call before execution, and the
  `Environment` scopes file paths and subprocess working directories to a workspace with
  hard timeouts — but an *allowed* shell command can do anything your user account can do.
- Model outputs are untrusted input. A prompt-injected agent will attempt tool calls you did
  not intend; the policy gate is your control point, so prefer `Ask`-by-default policies for
  anything with side effects.
- For untrusted or high-risk tasks, use `ContainerEnvironment`: commands run inside a
  Docker container with none of your host environment variables, and only the bind-mounted
  workspace is shared. Note the file boundary honestly — bind-mounted files are shared by
  design, and container escape is out of scope for this threat model (Docker is the
  isolation layer, not regista).

Every tool call and permission decision is recorded in the session trace, so there is always
an audit log. Traces can contain sensitive data (prompts, file contents, command output) —
treat trace files with the same care as logs.

## Reporting a vulnerability

Please report vulnerabilities privately via GitHub Security Advisories
("Report a vulnerability" on the repo) rather than public issues. You should receive a
response within a week.
