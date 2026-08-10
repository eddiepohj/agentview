from pathlib import Path
from agentview.paths import encode_cwd, session_dir


def test_encode_plain_path():
    assert encode_cwd("/Users/example") == "-Users-example"


def test_encode_replaces_dots_producing_double_dash():
    assert encode_cwd("/Users/example/.claude/primary-project") == \
        "-Users-example--claude-primary-project"


def test_encode_replaces_spaces():
    assert encode_cwd("/Users/example/Claude code testing") == \
        "-Users-example-Claude-code-testing"


def test_encode_strips_trailing_slash():
    assert encode_cwd("/Users/example/") == encode_cwd("/Users/example")


def test_session_dir_joins_under_root():
    d = session_dir("/Users/example", root=Path("/tmp/projects"))
    assert d == Path("/tmp/projects/-Users-example")
