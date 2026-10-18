import random

import pytest

from src.generator import layouts
from src.generator.values import sample_record
from src.schema.models import GMPRecord

RENDER = {"A": layouts.render_a, "B": layouts.render_b, "C": layouts.render_c, "D": layouts.render_d}


def test_records_include_null_limits_and_validate():
    rng = random.Random(42)
    recs = [sample_record(rng, i) for i in range(1, 200)]
    for r in recs:
        GMPRecord(**r)
    ms = [m for r in recs for m in r["measurements"]]
    n_min = sum(m["limit_min"] is None for m in ms)
    n_max = sum(m["limit_max"] is None for m in ms)
    assert n_min > 0 and n_max > 0                       # nulls now exist
    assert 0.1 < n_min / len(ms) < 0.45                  # roughly the intended 25%


@pytest.mark.parametrize("key", list("ABCD"))
def test_layout_renders_and_never_prints_None(key, tmp_path):
    rng = random.Random(7)
    for i in range(1, 25):
        rec = sample_record(rng, i)
        out = tmp_path / f"{key}_{i}.pdf"
        gt = RENDER[key](rec, str(out))
        assert out.stat().st_size > 500
        # "Deviations: None" is legitimate printed text for "no deviations"; a null LIMIT must never print as None.
        lines = [l for l in gt["gt_lines"] if not l.startswith("Deviations")]
        assert not any("None" in l for l in lines), f"layout {key} printed a null as the string 'None'"
