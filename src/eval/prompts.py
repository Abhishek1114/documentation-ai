"""Prompt construction for baselines (Phase 2) and, unchanged, for fine-tuning and serving (Phases 3-4).

IMPORTANT: this returns chat MESSAGES (role/content dicts). The special tokens (<|im_start|> etc.) are
added by tokenizer.apply_chat_template in the notebook. Never hand-write them here.

Use the SAME build_messages for zero-shot, few-shot, training and serving. If the prompt changes
between baseline and fine-tune, the before/after comparison is no longer apples to apples.
"""
import json

SYSTEM_PROMPT = """You extract structured data from one page of an equipment cleaning record.
Return ONE JSON object and nothing else: no explanation, no code fences.

Schema (all keys required; use null where allowed):
{
 "document_id": string,
 "document_type": "cleaning_record" | "maintenance_log" | "sop_excerpt",
 "equipment": {"id": string, "name": string, "location": string},
 "date": "YYYY-MM-DD",
 "performed_by": string,
 "reviewed_by": string | null,
 "steps": [{"no": integer, "description": string, "result": "pass" | "fail" | "n/a", "initials": string}],
 "measurements": [{"parameter": string, "value": number, "unit": string, "limit_min": number | null, "limit_max": number | null}],
 "deviations": [string]
}

Rules:
- date: convert any date format to YYYY-MM-DD.
- steps[].result: lower case. Keep "n/a" as the string "n/a".
- Numbers are JSON numbers, not strings.
- A value that is blank or marked as absent / not applicable becomes null (reviewed_by, limit_min, limit_max).
- A limit written in words ("up to X", "at least X") fills limit_max / limit_min respectively.
- deviations is [] when the page says there were none.
- Copy names, ids and descriptions exactly as printed. Never invent values."""


def user_message(pair: dict, tables: bool = True) -> str:
    msg = f"Extract the record from this page.\n\n<page>\n{pair['input_text']}\n</page>"
    if tables and pair.get("input_tables"):
        msg += f"\n\n<tables>\n{pair['input_tables']}\n</tables>"
    return msg


def build_messages(pair: dict, shots=(), tables: bool = True, with_answer: bool = False) -> list:
    """system + (user, assistant) per shot + user [+ assistant answer when building TRAINING text]."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s in shots:
        msgs.append({"role": "user", "content": user_message(s, tables)})
        msgs.append({"role": "assistant", "content": s["target"]})
    msgs.append({"role": "user", "content": user_message(pair, tables)})
    if with_answer:
        msgs.append({"role": "assistant", "content": pair["target"]})
    return msgs


def _good_shot(pair: dict) -> bool:
    """A teaching example should show a null limit, so the model sees how absence is written."""
    rec = json.loads(pair["target"])
    ms = rec["measurements"]
    return bool(ms) and any(m["limit_min"] is None or m["limit_max"] is None for m in ms)


def pick_shots(train_pairs, kind: str, n: int = 2, layouts=("A", "B")):
    """Deterministic few-shot choice: lowest id per layout (round robin A, B, A, ...), same input kind
    as the document being evaluated, TRAIN layouts only. Never draws from val/test layouts."""
    pools = []
    for L in layouts:
        c = sorted((p for p in train_pairs if p["layout"] == L and p["kind"] == kind), key=lambda p: p["id"])
        good = [p for p in c if _good_shot(p)]
        pools.append(good or c)
    shots = []
    for i in range(n):
        pool = pools[i % len(pools)]
        shots.append(pool[i // len(pools)])
    return shots
