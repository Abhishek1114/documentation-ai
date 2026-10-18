import copy
import json

from src.eval.extraction_metrics import (TYPES, evaluate, field_type, flatten, normalize, prf,
                                          score, salvage_parse, try_parse)

GOLD = {
    "document_id": "CR-2026-00001", "document_type": "cleaning_record",
    "equipment": {"id": "MX-101", "name": "Ribbon Blender", "location": "Room B-12"},
    "date": "2026-10-16", "performed_by": "A. Patel", "reviewed_by": None,
    "steps": [
        {"no": 1, "description": "Remove product residue", "result": "pass", "initials": "RM"},
        {"no": 2, "description": "Rinse until clear", "result": "fail", "initials": "KS"},
        {"no": 3, "description": "Visual inspection", "result": "n/a", "initials": "NJ"},
    ],
    "measurements": [
        {"parameter": "pH of final rinse", "value": 6.1, "unit": "pH", "limit_min": 5.0, "limit_max": 7.5},
        {"parameter": "TOC", "value": 120.5, "unit": "ppb", "limit_min": None, "limit_max": 500.0},
    ],
    "deviations": [],
}


def totals(counts):
    return tuple(sum(counts[t][k] for t in TYPES) for k in ("tp", "n_pred", "n_gold"))


def gold():
    return copy.deepcopy(GOLD)


def test_perfect_prediction():
    pred, err = try_parse(json.dumps(GOLD))
    assert err is None
    tp, n_pred, n_gold = totals(score(pred, gold()))
    assert tp == n_pred == n_gold > 0
    assert prf(tp, n_pred, n_gold)[2] == 1.0


def test_wrong_date_hits_only_date_bucket():
    bad = gold(); bad["date"] = "2026-10-17"
    pred, _ = try_parse(json.dumps(bad))
    c = score(pred, gold())
    assert c["date"]["tp"] == 0 and c["date"]["n_gold"] == 1
    tp, n_pred, n_gold = totals(c)
    assert tp == n_pred - 1 == n_gold - 1


def test_invalid_json_counts_zero_pred_all_gold():
    pred, err = try_parse('{"document_id": "CR-1", ')
    assert pred is None and err.startswith("json_decode")
    c = score(pred, gold())
    tp, n_pred, n_gold = totals(c)
    assert tp == 0 and n_pred == 0 and n_gold > 0


def test_schema_error_reports_field():
    bad = gold(); bad["steps"][0]["result"] = "PASS"          # layout D prints upper case; schema wants lower
    pred, err = try_parse(json.dumps(bad))
    assert pred is None and err.startswith("schema") and "steps.0.result" in err


def test_no_json_at_all():
    assert try_parse("Sorry, I cannot do that.")[1].startswith("no_json")


def test_fences_and_prose_are_ignored():
    raw = "Here you go {not json}:\n```json\n" + json.dumps(GOLD) + "\n```\nHope that helps."
    pred, err = try_parse(raw)
    assert err is None and pred["document_id"] == "CR-2026-00001"


def test_skipped_step_does_not_shift_later_steps():
    bad = gold(); del bad["steps"][0]                          # model drops step 1
    pred, _ = try_parse(json.dumps(bad))
    c = score(pred, gold())
    # steps 2 and 3 still match exactly; only step 1's items are missed, plus the length
    assert c["enum"]["tp"] == c["enum"]["n_gold"] - 1                 # step 1 'result' missed; document_type + steps 2,3 ok
    assert c["count"]["tp"] == c["count"]["n_gold"] - 1               # steps.len wrong
    miss = sum(c[t]["n_gold"] - c[t]["tp"] for t in TYPES)
    # step 1 has 4 leaves (no, description, result, initials) + steps.len = 5 misses, nothing else
    assert miss == 5
    assert totals(c)[1] == totals(c)[2] - 4                           # no invented items; 4 leaves simply absent


def test_hallucinated_step_is_penalised_in_precision_and_count():
    bad = gold()
    bad["steps"].append({"no": 4, "description": "Invented step", "result": "pass", "initials": "XX"})
    pred, _ = try_parse(json.dumps(bad))
    c = score(pred, gold())
    tp, n_pred, n_gold = totals(c)
    assert n_pred == n_gold + 4 and tp == n_gold - 1           # 4 invented leaves; steps.len now wrong
    assert prf(tp, n_pred, n_gold)[0] < 1.0


def test_duplicate_step_number_counts_as_extra():
    bad = gold(); bad["steps"].append(copy.deepcopy(bad["steps"][0]))
    flat = flatten(try_parse(json.dumps(bad))[0])
    assert "steps[1#2].description" in flat


def test_null_bucket_and_missing_optional_key():
    bad = gold(); del bad["reviewed_by"]                       # missing optional key == null
    for m in bad["measurements"]:
        m.pop("limit_min", None)                               # second measurement gold null: fine; first gold 5.0: miss
    pred, _ = try_parse(json.dumps(bad))
    c = score(pred, gold())
    assert c["null"]["n_gold"] == 2 and c["null"]["tp"] == 2   # reviewed_by + TOC limit_min both gold null
    assert c["number"]["tp"] == c["number"]["n_gold"] - 1      # pH limit_min 5.0 was dropped


def test_model_says_null_but_gold_has_value_is_a_miss():
    bad = gold(); bad["measurements"][0]["limit_max"] = None
    pred, _ = try_parse(json.dumps(bad))
    c = score(pred, gold())
    assert c["number"]["tp"] == c["number"]["n_gold"] - 1


def test_normalisation_policy():
    assert normalize("  K.  Shah ", "text") == normalize("k. shah", "text")
    assert normalize("µS/cm", "unit") == normalize("uS/cm", "unit")
    assert normalize("5", "number") == normalize(5.0, "number") == normalize("5.0", "number")
    assert normalize("PASS", "enum") == "pass"
    assert normalize(None, "text") is None and normalize(None, "text") != normalize("", "text")


def test_field_types():
    assert field_type("steps[3].result") == "enum"
    assert field_type("measurements[toc].unit") == "unit"
    assert field_type("steps.len") == "count"
    assert field_type("deviations[0]") == "text"


def test_evaluate_groups_and_validity():
    rows = [
        {"id": "a", "mode": "zero", "split": "val", "layout": "C", "kind": "digital",
         "raw": json.dumps(GOLD), "target": json.dumps(GOLD)},
        {"id": "b", "mode": "zero", "split": "val", "layout": "C", "kind": "digital",
         "raw": "not json {", "target": json.dumps(GOLD), "truncated": True},
    ]
    (r,) = evaluate(rows)
    assert r["n_docs"] == 2 and r["validity"] == 0.5 and r["doc_exact"] == 0.5
    assert r["recall"] == 0.5 and r["precision"] == 1.0 and r["truncated"] == 1
    assert r["fail_kinds"] == {"json_decode": 1}


# ---------------------------------------------------------------- salvage metric
def _f1(pred):
    return prf(*totals(score(pred, gold())))[2]


def _misses(c):
    return sum(c[t]["n_gold"] - c[t]["tp"] for t in TYPES)


def test_salvage_credits_uppercase_result():
    bad = gold()
    for s in bad["steps"]:
        s["result"] = s["result"].upper()
    raw = json.dumps(bad)
    assert try_parse(raw)[0] is None
    assert _f1(salvage_parse(raw)) == 1.0


def test_salvage_wrong_document_type_costs_exactly_one_item():
    bad = gold(); bad["document_type"] = "cleaning_log"
    c = score(salvage_parse(json.dumps(bad)), gold())
    assert _misses(c) == 1 and c["enum"]["tp"] == c["enum"]["n_gold"] - 1


def test_salvage_string_na_limit_is_a_miss():
    bad = gold(); bad["measurements"][1]["limit_min"] = "N/A"
    c = score(salvage_parse(json.dumps(bad)), gold())
    assert c["null"]["tp"] == c["null"]["n_gold"] - 1


def test_salvage_equals_strict_on_valid_output_with_extra_keys():
    ok = gold(); ok["verified_by"] = None; ok["steps"][0]["note"] = "x"
    raw = json.dumps(ok)
    assert _f1(salvage_parse(raw)) == _f1(try_parse(raw)[0]) == 1.0


def test_broken_json_never_falls_back_to_inner_object():
    raw = json.dumps(gold())[:-30]
    assert salvage_parse(raw) is None
    assert try_parse(raw)[1].startswith("json_decode")


def test_salvage_does_not_crash_on_wrong_shapes():
    bad = gold(); bad["steps"] = ["oops", bad["steps"][0]]; bad["measurements"] = "none"
    assert salvage_parse(json.dumps(bad)) is not None
    assert salvage_parse("[1, 2]") is None