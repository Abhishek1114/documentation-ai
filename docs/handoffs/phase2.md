# Phase 2 handoff: DRAFT (fill the TO FILL fields from `results/`, do not type numbers from memory)

**For the next Claude session:** read `documentation_ai_handoff.md`, then `phase1.md`, then this file. This file overrides earlier ones where they differ.

## 1. Status

- Phase 2 code written and tested locally (19 unit tests pass, pipeline smoke-tested on 16 docs).
- NOT yet done: full dataset regeneration, full OCR cache, Colab baseline run, scoring. Those produce the numbers below.
- Next: Phase 3 (QLoRA), after the baseline table is filled in.

## 2. Decisions made (with reasons)

| Decision | Reason |
|---|---|
| Model input = parsed page text; extracted tables appended as ` \| ` rows (`--tables` ablation supported) | Table accuracy is poor on B, C and scans (Phase 1). The text already contains table rows, so appending duplicates them; whether it helps is an ablation (`USE_TABLES` in the notebook). |
| Mix digital and scan inputs; report separately | Model sees OCR noise; gap is measurable. |
| `--per-layout 300` -> 600 train (A,B), 300 val (C), 300 test (D) | 100 train docs too few. "Held-out layouts" is exactly ONE val layout and ONE test layout; say so in the README. |
| Null `limit_min`/`limit_max` added (25% each) | Makes null handling testable. Each layout prints null differently: A `-`, B `n.a.`, C prose ("up to X" / "at least X" / "no limit"), D `N/A`. |
| `document_type` still always `cleaning_record` | Limitation: F1 on this field is trivial. |
| Baselines on Colab T4, fp16, greedy, `repetition_penalty=1.0` | T4 has no fast bf16. A repetition penalty hurts exact copying. |
| Same `build_messages` for zero-shot, few-shot, training, serving | Keeps before/after comparable. |
| Few-shot: 2 fixed examples, layouts A and B only, same input kind as the test doc, chosen to show a null limit | No val/test leakage. |
| Metric policies | See the docstring at the top of `src/eval/extraction_metrics.py`. Key ones: list items aligned by key (`steps` by `no`, `measurements` by `parameter`), `.len` penalises invented/dropped items, invalid output = zero predicted items and all gold items missed, missing Optional key == null. |

## 3. Known limitations (README material)

- Extra keys the model invents are dropped by pydantic and not penalised.
- A misspelled `parameter` makes the whole measurement count as missed + invented (key alignment is exact after normalisation).
- Per-batch time is amortised over a batch; it is not single-document latency (Phase 4 measures that).
- The system prompt tells the model that blank/absent/not-applicable values become null, and how limits written in words map. It does not name any layout's marker, but it is still a hint that a baseline gets and a fine-tune also sees.
- Only 1 val layout and 1 test layout: results say "generalises to this unseen layout", not "to unseen layouts in general".

## 4. How to reproduce (from repo root, venv active)

```
python -m src.generator.build --per-layout 300 --clean
python -m src.parsing.cache --workers 3          # resumable; first check s/file on a few docs
python -m src.data.build_pairs
python -m src.data.make_colab_bundle             # -> colab_bundle.zip, upload to notebooks/baselines.ipynb
python -m pytest tests -q
```

## 5. Results (TO FILL: copy from `results/baselines_*.csv`)

Dataset tag: TO FILL (from `data/synthetic/dataset_meta.json`). Model: Qwen2.5-1.5B-Instruct. Docs per layout scored: TO FILL.

| Mode | Split (layout) | Input | Validity | Field F1 | Doc exact | F1 date | F1 enum | F1 number | F1 unit | F1 text | F1 count | F1 null |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zero-shot | val (C) | digital | TO FILL | | | | | | | | | |
| zero-shot | val (C) | scan | | | | | | | | | | |
| zero-shot | test (D) | digital | | | | | | | | | | |
| zero-shot | test (D) | scan | | | | | | | | | | |
| few-shot | val (C) | digital | | | | | | | | | | |
| few-shot | val (C) | scan | | | | | | | | | | |
| few-shot | test (D) | digital | | | | | | | | | | |
| few-shot | test (D) | scan | | | | | | | | | | |

Ablation (tables appended vs text only): TO FILL.
Speed: s/doc (batch 8), peak VRAM, mean prompt tokens: TO FILL.
Invalid outputs by kind (no_json / json_decode / schema) and truncation count: TO FILL.
What the failures look like (3 examples read by eye): TO FILL.

## 6. Open issues carried forward

1. Phase 1 issues 1 to 3 (digital tables on B/C, layout A scan CER ~0.47, heuristic scan tables) are still open. Layout A scans are ~1/3 of the scan training data; diagnose before Phase 3 (print REF vs HYP for one A scan).
2. Public test set (10-20 DailyMed / GMP PDFs) not started.
3. Phase 1 quiz answers not recorded.

## 7. Handoff log entry

```
### Phase 2 handoff: TO FILL (date)
Done: metrics + tests, null-limit generator, parse cache, pair builder, prompts, Colab baseline notebook
Numbers: section 5
Decisions + reasons: section 2
Problems / open issues: sections 3 and 6
Next step: Phase 3, QLoRA on train pairs (A+B, digital+scan), evaluate on val/test with the same prompt
```
