"""Every syscall the live pane makes. Nothing here is pure; nothing elsewhere
in `surfaces/` touches the terminal.

The split exists because `render_frame`'s purity was already broken once, by a
metric that read the filesystem. Interactivity is the obvious second way to
break it, so the input path is quarantined here.
"""
from __future__ import annotations

import select
import sys
import termios
import tty
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TextIO


def _ready(stream: TextIO, timeout: float) -> bool:
    """True when `stream` has at least one byte pending within `timeout`
    seconds. Isolated behind a name so `read_key` can take an injected fake
    in tests instead of a real `select` call."""
    try:
        fd = stream.fileno()
    except (OSError, ValueError, AttributeError):
        return False
    try:
        ready, _, _ = select.select([fd], [], [], timeout)
    except (OSError, ValueError):
        return False
    return bool(ready)


def _restore(fd: int, old: Any) -> None:
    """Best-effort restore of the terminal's prior settings. A failure here
    must not mask whatever exception is already propagating out of the
    `with` body -- this runs from a `finally`."""
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except OSError:
        pass


@contextmanager
def raw_mode(stream: TextIO | None = None,
             _setraw: Callable[[int], Any] = tty.setcbreak) -> Iterator[bool]:
    """Enter raw mode for `stream` for the lifetime of the `with` body,
    restoring the previous terminal settings in a `finally` on every exit
    path -- a normal return, an exception, or a `KeyboardInterrupt`.

    Yields `False` -- without touching the terminal at all -- when `stream`
    is not a TTY, or when `_setraw` itself raises `OSError` (no controlling
    terminal, permission denied). A terminal quirk degrades this read-only
    diagnostic pane to the non-interactive path; it must never crash the
    run over it.
    """
    stream = sys.stdin if stream is None else stream
    if not stream.isatty():
        yield False
        return

    fd = stream.fileno()
    try:
        old = _setraw(fd)
    except OSError:
        yield False
        return

    try:
        yield True
    finally:
        _restore(fd, old)


def read_key(stream: TextIO | None = None, timeout: float = 0.2,
             _ready: Callable[[TextIO, float], bool] = _ready) -> str | None:
    """Return one key, or `None` if nothing arrives within `timeout` seconds
    or `stream` is not a TTY -- so the redraw loop never blocks on input.

    An arrow key is assembled from `ESC` only when the bytes after it are
    already pending (checked with a zero timeout -- never a blocking read);
    a bare `ESC` with nothing behind it is returned as itself instead of
    stalling on a read that may never come.
    """
    stream = sys.stdin if stream is None else stream
    if not stream.isatty():
        return None
    if not _ready(stream, timeout):
        return None
    ch = stream.read(1)
    if ch == "":
        return None
    if ch != "\x1b" or not _ready(stream, 0):
        return ch
    nxt = stream.read(1)
    if nxt != "[" or not _ready(stream, 0):
        return ch + nxt
    return ch + nxt + stream.read(1)
