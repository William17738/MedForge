"""
MedForge Filesystem Utilities

Provides:
- Cross-platform advisory file locks (process-safe)
- Atomic text/JSON writes via temp file + os.replace
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


if os.name == "nt":
    import msvcrt

    def _lock_handle(handle) -> None:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write("0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock_handle(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_handle(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_handle(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def file_lock(
    lock_path: Path,
    *,
    timeout_s: float | None = 60.0,
    poll_interval_s: float = 0.1,
) -> Iterator[None]:
    """
    Acquire an exclusive advisory lock using a dedicated lock file.

    This is safe to use together with atomic writes that replace the target file,
    because the lock is held on a separate path.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    with open(lock_path, "a+", encoding="utf-8") as handle:
        while True:
            try:
                _lock_handle(handle)
                break
            except OSError:
                if timeout_s is not None and (time.monotonic() - start) >= timeout_s:
                    raise TimeoutError(f"Timed out acquiring lock: {lock_path}")
                time.sleep(poll_interval_s)

        try:
            yield
        finally:
            try:
                _unlock_handle(handle)
            except Exception:
                pass


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    payload = json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
    _atomic_write_bytes(path, payload.encode(encoding))

