from agentview.sources.buildlog import read_buildlog, buildlog_events, VOCAB


def _log(tmp_path, body):
    p = tmp_path / "BUILD-LOG.md"
    p.write_text(body)
    return p


def test_parses_a_well_formed_line(tmp_path):
    p = _log(tmp_path, "25/06/2026 14:26 | F.2 | complete | Authoring report.\n")
    row = read_buildlog(p)[0]
    assert row["step"] == "F.2"
    assert row["status"] == "complete"
    assert row["date"] == "25/06/2026"
    assert row["off_vocab"] is False


def test_ignores_prose_and_headers(tmp_path):
    p = _log(tmp_path, "# BUILD-LOG\n\nFormat: blah\n"
                       "25/06/2026 14:26 | F.2 | complete | ok\n")
    assert len(read_buildlog(p)) == 1


def test_off_vocabulary_status_is_flagged_not_dropped(tmp_path):
    p = _log(tmp_path, "10/06/2026 09:00 | X.1 | FAIL (predicted) | nope\n")
    rows = read_buildlog(p)
    assert len(rows) == 1
    assert rows[0]["off_vocab"] is True
    assert rows[0]["status"] == "FAIL (predicted)"


def test_decorated_complete_is_off_vocabulary(tmp_path):
    p = _log(tmp_path, "24/06/2026 09:00 | X.1 | complete (5/5 ACs MET) | ok\n")
    assert read_buildlog(p)[0]["off_vocab"] is True


def test_vocab_contains_the_documented_twelve():
    assert len(VOCAB) == 12
    assert "red-verified" in VOCAB and "gate-cleared" in VOCAB


def test_events_carry_no_utc_timestamp(tmp_path):
    p = _log(tmp_path, "25/06/2026 14:26 | F.2 | complete | ok\n")
    e = buildlog_events(p)[0]
    assert e.ts is None
    assert e.payload["local_time"] == "25/06/2026 14:26"


def test_prose_line_containing_a_date_is_not_parsed_as_a_row(tmp_path):
    # A narrative sentence that merely mentions a date must not be mistaken
    # for a log row just because a DD/MM/YYYY HH:MM substring appears in it.
    p = _log(tmp_path, "Note: as of 25/06/2026 14:26 the schema changed, "
                       "see step F.2 for details.\n"
                       "25/06/2026 14:27 | F.2 | complete | ok\n")
    rows = read_buildlog(p)
    assert len(rows) == 1
    assert rows[0]["line_no"] == 2


def test_line_with_fewer_than_three_fields_is_rejected(tmp_path):
    # Only two pipe-separated fields after the timestamp (step | status,
    # no summary) must be dropped rather than silently accepted with a
    # missing/garbled summary.
    p = _log(tmp_path, "25/06/2026 14:26 | F.2 | complete\n"
                       "25/06/2026 14:27 | F.2 | complete | ok\n")
    rows = read_buildlog(p)
    assert len(rows) == 1
    assert rows[0]["line_no"] == 2


def test_summary_containing_pipes_is_preserved_in_full(tmp_path):
    # A literal "|" inside the summary must not truncate the line: splitting
    # on every "|" instead of maxsplit=2 would silently drop everything past
    # the third field, with no anomaly and no signal. Assert the exact
    # expected string (not just that a pipe survived) so a partially-
    # truncated result still fails.
    p = _log(tmp_path, "01/07/2026 09:00 | F.2 | complete | rate: 4 | 5 stars\n")
    row = read_buildlog(p)[0]
    assert row["summary"] == "rate: 4 | 5 stars"
    assert row["off_vocab"] is False


def test_empty_summary_still_parses(tmp_path):
    # Exactly three fields where the summary is the empty string must still
    # be accepted, not rejected by the len(fields) < 3 guard.
    p = _log(tmp_path, "01/07/2026 09:00 | F.2 | complete |\n")
    rows = read_buildlog(p)
    assert len(rows) == 1
    assert rows[0]["summary"] == ""
