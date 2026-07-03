"""Session and event identifiers.

ULIDs rather than UUIDs: lexicographic order equals creation order, so trace
filenames and session ids sort chronologically in a directory listing.
"""

from __future__ import annotations

import os
import time

_CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    """A 26-character ULID: 48-bit millisecond timestamp + 80 random bits."""
    timestamp = int(time.time() * 1000) & ((1 << 48) - 1)
    randomness = int.from_bytes(os.urandom(10), "big")
    value = (timestamp << 80) | randomness
    return "".join(_CROCKFORD32[(value >> (5 * i)) & 31] for i in range(25, -1, -1))
