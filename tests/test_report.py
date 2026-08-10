import json
from pathlib import Path

from agentview.discovery import find_runs
from agentview.model import build_run
from agentview.surfaces.report import render, write_report


def _run(tmp_path):
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "A", "status": "complete", "owner": "b", "tier": "sonnet",
         "attempts": 1, "deliverable": "a.md", "depends_on": [],
         "updated": "2026-06-25T11:00:00Z"}]}))
    (tmp_path / "proj" / "BUILD-LOG.md").write_text(
        "25/06/2026 11:00 | A | milestone | drifted\n")
    return build_run(find_runs(tmp_path)[0], sessions=[])


def _empty_run(tmp_path):
    """A run with no steps, no docs, no build-log at all."""
    sd = tmp_path / "proj" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": []}))
    return build_run(find_runs(tmp_path)[0], sessions=[])


def test_render_produces_a_complete_html_document(tmp_path):
    html = render([_run(tmp_path)])
    assert html.startswith("<!doctype html>")
    assert "</html>" in html


def test_render_includes_step_ids_and_status(tmp_path):
    html = render([_run(tmp_path)])
    assert "A" in html and "complete" in html


def test_render_surfaces_vocabulary_drift(tmp_path):
    assert "milestone" in render([_run(tmp_path)])


def test_render_escapes_html_in_free_text(tmp_path):
    run = _run(tmp_path)
    run.steps[0].owner = "<script>x</script>"
    assert "<script>x</script>" not in render([run])
    assert "&lt;script&gt;" in render([run])


def test_write_report_creates_only_the_requested_file(tmp_path):
    run = _run(tmp_path)
    out = tmp_path / "out" / "report.html"
    before = {p for p in tmp_path.rglob("*")}
    write_report([run], out)
    created = {p for p in tmp_path.rglob("*")} - before
    assert created == {out.parent, out}


# --- Strengthening beyond the brief ---

def test_render_escapes_html_in_every_free_text_field(tmp_path):
    """Owner-only escaping proof would miss a leak from status, a deliverable
    that resolves to a path, a deliverable that is prose, or a waiver path --
    each is rendered through a different code path (_table row cell vs.
    <li><code> vs. plain <li> vs. waiver table).

    Deliverables are now split into `doc_paths` and `doc_notes` and rendered
    by two different branches, so both must be proven to escape. Setting only
    `docs` -- which no longer feeds either branch -- would leave this test
    passing while asserting nothing about the markup that is actually
    emitted."""
    run = _run(tmp_path)
    run.steps[0].status = "<b>weird</b>"
    run.doc_paths = ["<img src=x onerror=alert(1)>.md"]
    run.doc_notes = ["<svg onload=alert(2)> prose deliverable"]
    html = render([run])
    assert "<b>weird</b>" not in html
    assert "&lt;b&gt;weird&lt;/b&gt;" in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<svg onload=alert(2)>" not in html
    assert "&lt;svg onload=alert(2)&gt;" in html


def test_render_multiple_runs_in_one_document(tmp_path):
    """render() takes a list -- prove it actually renders every run, not
    just runs[0], and keeps them clearly separated by project/slug heading."""
    run_a = _run(tmp_path)
    other_root = tmp_path / "other"
    sd = other_root / "proj2" / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "Z", "status": "complete", "owner": "c", "tier": "haiku",
         "attempts": 1, "deliverable": "z.md", "depends_on": []}]}))
    run_b = build_run(find_runs(other_root)[0], sessions=[])

    html = render([run_a, run_b])
    assert html.count("<h2>") == 2
    assert "A" in html and "Z" in html
    assert "proj2" in html
    assert html.index("A") < html.index("Z")


def test_render_empty_run_has_no_step_rows_and_no_docs_section(tmp_path):
    """A run with zero steps, zero docs, and no vocabulary drift must not
    raise, and must not emit an empty <ul></ul> docs shell or a phantom
    drift/waiver warning section."""
    html = render([_empty_run(tmp_path)])
    assert "<!doctype html>" in html
    assert "<tbody></tbody>" in html  # the step table itself may be empty
    assert "documents" not in html
    assert "vocabulary drift" not in html
    assert "waivers without a named authoriser" not in html


def test_render_has_no_script_tags_or_network_references(tmp_path):
    """Pin the 'static, self-contained, no JS' property with an assertion
    instead of relying on nobody adding a <script> or a CDN link later."""
    html = render([_run(tmp_path)])
    assert "<script" not in html.lower()
    assert "http://" not in html
    assert "https://" not in html


def test_render_escapes_html_in_run_heading(tmp_path):
    """The run heading interpolates ref.project and ref.slug directly
    (report.py:49). Both come from on-disk paths this tool does not
    control -- a project directory or slug containing markup must not
    inject unescaped HTML into the document."""
    run = _run(tmp_path)
    run.ref.project = Path(str(run.ref.project)) / "<script>evil</script>"
    run.ref.slug = "\"><img>"
    html = render([run])
    assert "<script>evil</script>" not in html
    assert "&lt;script&gt;evil&lt;/script&gt;" in html
    assert "\"><img>" not in html
    assert "&quot;&gt;&lt;img&gt;" in html


def test_redacted_report_hides_absolute_project_and_document_paths(tmp_path):
    run = _run(tmp_path)
    private_doc = tmp_path / "outside" / "private.md"
    private_doc.parent.mkdir()
    private_doc.write_text("x")
    run.doc_paths = [str(private_doc)]
    html = render([run], redact_paths=True)
    assert str(tmp_path) not in html
    assert "external/private.md" in html


def test_render_escapes_html_in_table_first_column(tmp_path):
    """_table's first column (a step id, or a waiver path) had no markup
    coverage -- every existing escaping test puts markup in owner or
    status, both later columns of the same table. A mutation that skips
    escaping only on column 0 would still pass all of them."""
    run = _run(tmp_path)
    run.steps[0].id = "<i>A</i>"

    waiver_dir = run.ref.project / "<b>sub</b>" / "changes"
    waiver_dir.mkdir(parents=True)
    (waiver_dir / "w.md").write_text("Kind: waiver\n")

    html = render([run])
    assert "<i>A</i>" not in html
    assert "&lt;i&gt;A&lt;/i&gt;" in html
    assert "<b>sub</b>" not in html
    assert "&lt;b&gt;sub&lt;/b&gt;" in html


# --- Fix wave, item 4 (C1): the report must state what is not clean ----------


def _overlapping_pair(tmp_path, write_jsonl, assistant):
    base = tmp_path / "proj" / "_planrunner"
    refs = []
    for name, created, updated in (("state-a", "2026-06-25T10:00:00Z",
                                    "2026-06-25T13:00:00Z"),
                                   ("state-b", "2026-06-25T12:00:00Z",
                                    "2026-06-25T15:00:00Z")):
        sd = base / name
        sd.mkdir(parents=True)
        (sd / "ledger.json").write_text(json.dumps({
            "created": created,
            "steps": [{"id": f"{name}-S1", "status": "complete",
                       "updated": updated}]}))
        refs.append(sd)
    session = write_jsonl("shared.jsonl", [
        assistant("2026-06-25T12:30:00Z", out=200),
        assistant("2026-06-25T12:45:00Z", out=300)])
    found = find_runs(tmp_path)
    return [build_run(r, [session]) for r in found]


def test_report_states_the_overlap_prominently_not_buried_in_a_table(
        tmp_path, write_jsonl, assistant):
    """The headline of the whole fix wave: two runs counting the same turns
    must not both present clean totals. The warning belongs above the step
    table, in the alert box, with the shared figures spelled out."""
    html = render(_overlapping_pair(tmp_path, write_jsonl, assistant))

    assert "these figures are not clean" in html
    assert "500 output tokens are counted in both runs" in html
    assert "2 turns" in html
    # Above the first run's step table, not appended after everything.
    assert html.index('class="alert"') < html.index("<table>")


def test_each_run_section_carries_its_own_overlap_warning(tmp_path,
                                                          write_jsonl,
                                                          assistant):
    """A single warning at the top of a multi-run document leaves a reader
    scrolled to one run's table with no sign its numbers are contaminated.
    Both runs are implicated, so both sections must say so."""
    html = render(_overlapping_pair(tmp_path, write_jsonl, assistant))
    assert html.count("overlaps sibling run") >= 4  # 2 top-level + 2 sections
    for slug in ("a", "b"):  # `_slug` strips the "state-" prefix
        after = html[html.index(f"— {slug}</h2>"):]
        assert 'class="alert"' in after[:after.index("<table>")]


def test_a_clean_run_gets_no_alert_box(tmp_path):
    """The box must mean something. A run with nothing wrong must not grow a
    permanent red banner."""
    html = render([_run(tmp_path)])
    assert 'class="alert"' not in html
    assert "these figures are not clean" not in html


# --- Item B: the report must not dress prose up as a file path --------------

def _mixed_run(tmp_path):
    proj = tmp_path / "proj"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (proj / "notes").mkdir()
    (proj / "notes" / "design.md").write_text("real\n")
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "S1", "status": "complete", "deliverable": "notes/design.md",
         "updated": "2026-06-25T11:00:00Z"},
        {"id": "S2", "status": "complete",
         "deliverable": "E.4 skeptic verdict (BUILD-LOG) + E.3 reply evidence",
         "updated": "2026-06-25T12:00:00Z"}]}))
    return build_run(find_runs(tmp_path)[0], sessions=[])


def test_report_marks_up_real_paths_as_code_and_prose_as_plain_text(tmp_path):
    """`<code>` under a heading called "documents" asserts "this is a file".
    Only the entry that is one may carry it."""
    html = render([_mixed_run(tmp_path)])

    assert "<code>notes/design.md</code>" in html
    assert "E.4 skeptic verdict (BUILD-LOG) + E.3 reply evidence" in html
    assert ("<code>E.4 skeptic verdict (BUILD-LOG) + E.3 reply evidence"
            "</code>") not in html


def test_report_keeps_prose_deliverables_rather_than_dropping_them(tmp_path):
    """They are real ledger content and a reader may want them; they are
    simply not paths. Filtering them out would trade one false statement for
    a silent omission."""
    run = _mixed_run(tmp_path)
    html = render([run])

    assert len(run.doc_notes) == 1
    assert "ledger deliverables that are not paths" in html
    for d in run.docs:
        assert d in html


def test_report_omits_the_documents_heading_when_nothing_resolves(tmp_path):
    """A run whose every deliverable is prose must not show a "documents"
    section at all -- that heading is the claim being corrected."""
    proj = tmp_path / "proj"
    sd = proj / "_planrunner" / "state"
    sd.mkdir(parents=True)
    (sd / "ledger.json").write_text(json.dumps({"steps": [
        {"id": "S1", "status": "complete",
         "deliverable": "launchd job installed + loaded",
         "updated": "2026-06-25T11:00:00Z"}]}))
    html = render([build_run(find_runs(tmp_path)[0], sessions=[])])

    assert "<h3>documents</h3>" not in html
    assert "ledger deliverables that are not paths" in html
    assert "launchd job installed + loaded" in html
