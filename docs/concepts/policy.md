# Policy — the permission gate

**Whether a tool call may run.** Module: `regista/policy/`.

A policy is a pure decision function, consulted by the loop before every dispatch:

```python
def my_policy(request: PermissionRequest) -> Allow | Deny | Ask: ...
```

`PermissionRequest` carries the tool name, the full input (including e.g. the exact shell
command), the `tool_use_id`, and the turn number. Policies may be sync or async.

## The three decisions

- **`Allow()`** — dispatch proceeds.
- **`Deny(reason)`** — the model receives `tool_result(is_error=True, content="Permission
  denied: <reason>")`. **Deny is data, not an exception**: the model sees the refusal and
  adapts; the session continues.
- **`Ask(prompt)`** — escalates to the Agent's `ask_handler` (an async
  `(PermissionRequest) -> bool`). With no handler configured, Ask resolves to deny — the
  harness never hangs waiting for input that can't arrive.

Every decision — including how an Ask was resolved — is a `permission.decision` trace
event.

## Presets

```python
from regista.policy import allow_all, read_only, workspace, compose

read_only()            # only read_file/list_dir/glob/search_files; deny everything else
workspace()            # file tools allowed; shell, fetch, and custom tools escalate to Ask
compose(deny_shell, workspace())   # first Deny or Ask wins — composing only tightens
allow_all()            # fine for FakeProvider tests and trusted tools
```

`read_only` deliberately denies `fetch`: a GET writes nothing locally but still reaches the
network, and read-only promises no effects beyond reading.

## Boundary

The gate sits **in the loop, before dispatch**. Tools never self-police — a tool
implementation can assume it was allowed to run — and the environment doesn't decide
permissions either; it enforces *scope* (see [environment](environment.md)).
