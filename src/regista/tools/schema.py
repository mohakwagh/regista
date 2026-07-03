"""Turn a typed Python function into a JSON Schema tool definition.

Deliberately strict: an unsupported parameter type raises ConfigurationError at
decoration time, not at call time — a tool that can't be described to the model
accurately should fail before any agent runs.

Supported parameter types (v0.1): str, int, float, bool, Literal[...],
list[...], and X | None. Docstrings follow Google style; the summary becomes
the tool description and ``Args:`` lines become parameter descriptions.
"""

from __future__ import annotations

import inspect
import re
import types
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from collections.abc import Callable

from regista.errors import ConfigurationError

_PRIMITIVES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _annotation_to_schema(annotation: Any, *, context: str) -> dict[str, Any]:
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    origin = get_origin(annotation)

    if origin is Literal:
        values = list(get_args(annotation))
        schema: dict[str, Any] = {"enum": values}
        value_types = {type(v) for v in values}
        if len(value_types) == 1 and (t := value_types.pop()) in _PRIMITIVES:
            schema["type"] = _PRIMITIVES[t]
        return schema

    if origin is list:
        (item,) = get_args(annotation) or (None,)
        if item is None:
            raise ConfigurationError(f"{context}: bare `list` needs an item type, e.g. list[str]")
        return {"type": "array", "items": _annotation_to_schema(item, context=context)}

    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0], context=context)
        raise ConfigurationError(f"{context}: unions other than `X | None` are not supported")

    raise ConfigurationError(
        f"{context}: unsupported parameter type {annotation!r} "
        "(supported: str, int, float, bool, Literal, list[...], X | None)"
    )


def _parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Returns (summary, {param_name: description}) from a Google-style docstring."""
    if not docstring:
        return "", {}
    text = inspect.cleandoc(docstring)
    parts = re.split(r"\n\s*Args:\s*\n", text, maxsplit=1)
    summary = parts[0].strip()
    param_docs: dict[str, str] = {}
    if len(parts) == 2:
        args_block = re.split(r"\n\s*(?:Returns|Raises|Yields|Examples?):\s*\n", parts[1])[0]
        current: str | None = None
        for line in args_block.splitlines():
            match = re.match(r"^\s+(\w+)\s*(?:\([^)]*\))?:\s*(.*)$", line)
            if match and not line.startswith(" " * 12):
                current = match.group(1)
                param_docs[current] = match.group(2).strip()
            elif current and line.strip():
                param_docs[current] += " " + line.strip()
    return summary, param_docs


def build_input_schema(fn: Callable[..., Any]) -> tuple[str, dict[str, Any]]:
    """Returns (description, JSON Schema) for a tool function's parameters."""
    context = f"tool '{fn.__name__}'"
    signature = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception as exc:
        raise ConfigurationError(f"{context}: could not resolve type hints: {exc}") from exc

    summary, param_docs = _parse_docstring(fn.__doc__)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise ConfigurationError(f"{context}: *args/**kwargs parameters are not supported")
        if name not in hints:
            raise ConfigurationError(f"{context}: parameter '{name}' is missing a type annotation")
        prop = _annotation_to_schema(hints[name], context=f"{context}, parameter '{name}'")
        if name in param_docs:
            prop["description"] = param_docs[name]
        if parameter.default is parameter.empty:
            required.append(name)
        elif parameter.default is not None:
            prop["default"] = parameter.default
        properties[name] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return summary, schema
