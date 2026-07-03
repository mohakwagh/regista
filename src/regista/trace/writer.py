"""Append-only JSONL trace writer.

One event per line, flushed to the OS per event: a crash loses at most the
event being written, never earlier history. The writer stamps the envelope
(session_id, seq, ts) so callers construct payload fields only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, TypeVar

if TYPE_CHECKING:
    from types import TracebackType

from regista.trace.events import _Event

E = TypeVar("E", bound=_Event)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TraceWriter:
    """Owns one session's trace file for the duration of the session."""

    def __init__(self, path: Path | str, session_id: str) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self._seq = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = self.path.open("a", encoding="utf-8")

    def emit(self, event: E) -> E:
        """Stamp the envelope, append one JSON line, flush, return the stamped event."""
        if self._file is None:
            raise ValueError(f"trace writer for {self.path} is closed")
        stamped: E = event.model_copy(
            update={"session_id": self.session_id, "seq": self._seq, "ts": _now_iso()}
        )
        self._seq += 1
        self._file.write(stamped.model_dump_json() + "\n")
        self._file.flush()
        return stamped

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
