from src.render.checks import check_all
from src.render.markdown import render_markdown
from src.schema.models import GMPRecord

BASE = {
    "document_id": "CR-2026-00601", "document_type": "cleaning_record",
    "equipment": {"id": "MX-101", "name": "Ribbon Blender", "location": "Room B-12"},
    "date": "2026-07-02", "performed_by": "A. Patel", "reviewed_by": None,
    "steps": [
        {"no": 1, "description": "Rinse | until clear", "result": "pass", "initials": "AP"},
        {"no": 2, "description": "Wipe with 70% IPA", "result": "fail", "initials": "KS"},
    ],
    "measurements": [
        {"parameter": "TOC", "value": 612.5, "unit": "ppb", "limit_min": 0.0, "limit_max": 500.0},
        {"parameter": "pH of final rinse", "value": 7.3, "unit": "pH", "limit_min": None, "limit_max": None},
    ],
    "deviations": ["Residue found on gasket"],
}


def rec(**over):
    return GMPRecord(**{**BASE, **over})


def test_render_passes_all_checks():
    r = rec()
    assert check_all(render_markdown(r), r) == []


def test_flags_and_computed_columns():
    md = render_markdown(rec())
    assert "OUT OF LIMITS" in md and "no limits" in md
    assert "Failed step 2" in md and "Not yet reviewed" in md and "Deviation recorded" in md
    assert "Reviewed by | Not recorded" in md


def test_pipe_in_text_is_escaped():
    r = rec()
    md = render_markdown(r)
    assert "Rinse \\| until clear" in md and check_all(md, r) == []


def test_empty_lists_render_and_pass():
    r = rec(steps=[], measurements=[], deviations=[], reviewed_by="S. Das")
    md = render_markdown(r)
    assert "No measurements recorded." in md and "None recorded." in md
    assert check_all(md, r) == []


def test_tampered_value_is_caught():
    r = rec()
    md = render_markdown(r).replace("612.5", "512.5")
    errs = check_all(md, r)
    assert any("differ from record" in e for e in errs) and any("512.5" in e for e in errs)


def test_invented_number_is_caught():
    r = rec()
    assert any("99" in e for e in check_all(render_markdown(r) + "\nTotal: 99 items", r))


def test_missing_section_is_caught():
    r = rec()
    md = render_markdown(r).replace("## Deviations", "## Notes")
    assert any("Deviations" in e for e in check_all(md, r))