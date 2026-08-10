import io
import sys

import pytest

from agentview.surfaces import term


class _FakeStdin(io.StringIO):
    """A stdin stand-in whose TTY-ness is chosen by the test, not by the
    real file descriptor the process happens to be running under."""

    def __init__(self, initial_value: str = "", tty: bool = True):
        super().__init__(initial_value)
        self._tty = tty

    def isatty(self):
        return self._tty

    def fileno(self):
        return 0


def test_raw_mode_is_skipped_when_stdin_is_not_a_tty():
    calls = []
    with term.raw_mode(_FakeStdin(tty=False), _setraw=calls.append):
        pass
    assert calls == []


def test_raw_mode_is_restored_after_an_exception(monkeypatch):
    restored = []
    monkeypatch.setattr(term, "_restore", lambda *a: restored.append(a))
    with pytest.raises(RuntimeError):
        with term.raw_mode(_FakeStdin(tty=True), _setraw=lambda fd: None):
            raise RuntimeError("boom")
    assert restored, "terminal left in raw mode after an exception"


def test_read_key_returns_none_when_nothing_is_pending():
    # A real character sits in the buffer even though nothing is "pending"
    # per the (faked) readiness check -- a `read_key` that skipped the
    # readiness gate and read anyway would return "x", not None, so this
    # fixture actually discriminates the check rather than merely matching
    # an empty stream's own end-of-input behaviour.
    assert term.read_key(_FakeStdin("x", tty=True), timeout=0.0,
                         _ready=lambda *a: False) is None


def test_read_key_reads_one_character():
    assert term.read_key(_FakeStdin("j", tty=True), timeout=0.0,
                         _ready=lambda *a: True) == "j"


def test_read_key_is_none_off_a_tty():
    assert term.read_key(_FakeStdin("j", tty=False), timeout=0.0) is None


def test_raw_mode_leaves_isig_set_so_ctrl_c_still_interrupts():
    # `tty.setraw` (the old default) clears `ISIG`, which is what makes
    # Ctrl-C generate `SIGINT`/`KeyboardInterrupt` in the first place -- with
    # it cleared, the pane's advertised "ctrl-c to exit" footer would be a
    # lie. `tty.setcbreak` (the current default) disables canonical mode and
    # echo, exactly what `read_key`'s single-character reads need, without
    # touching `ISIG`. A real pty (no subprocess) proves this against the
    # real stdlib default rather than an injected fake `_setraw`.
    import os
    import pty
    import termios

    leader, follower = pty.openpty()
    try:
        stream = os.fdopen(follower, "r", closefd=False)
        with term.raw_mode(stream):
            attrs = termios.tcgetattr(follower)
            assert attrs[3] & termios.ISIG, (
                "raw_mode cleared ISIG -- Ctrl-C would no longer interrupt")
    finally:
        os.close(follower)
        os.close(leader)


def test_q_exits_the_real_cli_with_status_0_under_a_pty(tmp_path):
    import fcntl
    import json
    import os
    import pty
    import select
    import subprocess
    import termios
    import time

    # A minimal on-disk run, built under pytest's own tmp_path rather than a
    # personal directory that only happens to exist on one machine -- the
    # same "state dir holding a ledger.json" shape `find_runs`
    # (`agentview/discovery.py`) requires, as already used by
    # `tests/test_discovery.py`'s own `_project` fixture. Its content does
    # not matter here: the pty exercises `q`-handling in the watch loop, not
    # what the run reports.
    run_root = tmp_path / "proj"
    state_dir = run_root / "_planrunner" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "ledger.json").write_text(json.dumps({"steps": []}))

    leader, follower = pty.openpty()

    def _make_the_pty_our_controlling_terminal():
        # A test process spawned by a step-runner/CI harness commonly has no
        # controlling terminal anywhere in its own ancestry. Plain `Popen`
        # only dup2's `follower` onto the child's stdin -- it never makes
        # the pty that session's controlling terminal. Without one, this
        # process's `tcsetattr` calls (via `tty.setraw`/`term.raw_mode`)
        # block inside the kernel indefinitely instead of erroring or
        # succeeding -- confirmed with `sample` during development, which
        # showed the child parked in the `ioctl` syscall underneath
        # `tcsetattr` with no progress at all. `setsid` + `TIOCSCTTY` is the
        # standard fix for exactly this (the same one `pexpect`/`ptyprocess`
        # apply internally); it is test-only subprocess-spawning plumbing,
        # not a change to shipped `term.py` or `live.py`.
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    proc = subprocess.Popen(
        [sys.executable, "-m", "agentview", "watch", str(run_root)],
        stdin=follower, stdout=follower, stderr=follower, close_fds=True,
        preexec_fn=_make_the_pty_our_controlling_terminal)
    os.close(follower)
    try:
        time.sleep(0.3)      # let the first frame render before sending input
        os.write(leader, b"q")

        # Drain the pty while waiting rather than a single blocking `wait()`:
        # if nobody reads the master side, the child's own frame writes can
        # block on a full kernel pty buffer, which would masquerade as `q`
        # never being noticed. A generous overall bound (this environment's
        # `tty.setraw`/`tcsetattr` calls were observed, during development,
        # taking anywhere from under a second to several seconds) still
        # catches a real hang.
        deadline = time.time() + 15
        while time.time() < deadline and proc.poll() is None:
            ready, _, _ = select.select([leader], [], [], 0.2)
            if ready:
                try:
                    os.read(leader, 65536)
                except OSError:
                    break
        assert proc.wait(timeout=5) == 0
    finally:
        # If the assertion above failed or the wait timed out, the child
        # (and the pty fds it holds open) must not be left running -- an
        # orphaned `watch` process is exactly the leak R5 flagged.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        os.close(leader)
