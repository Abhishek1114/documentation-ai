# Documentation AI: PDF to validated JSON to GMP-style document

A pipeline that parses digital and scanned PDFs (text + tables, OCR fallback), converts them to a fixed
Pydantic schema with a QLoRA-fine-tuned Qwen2.5-1.5B-Instruct, and renders a GMP-*style* document.
A LangGraph agent validates every output and repairs failures, writing one JSONL trace per run.

**Portfolio project on synthetic data. "GMP-style" only: not GMP-compliant, not validated, not regulatory-ready.**

## Data

Synthetic only: a generator fills random values into 4 layout templates (A, B, C, D) and produces a digital PDF and
a scanned variant (rotation, blur, noise, JPEG) for each, so the labels are exact.
Split **by layout**: A, B train; C validation; D test. Training pairs: <<N pairs, N documents: check data/synthetic/manifest.csv>>.

## Results

All numbers are strict (JSON must decode and pass Pydantic), n = 100 documents per layout and kind unless stated.
"Salvage F1" is reporting only: it scores the raw JSON of invalid outputs without repairing values.



### Extraction, unseen layout C (validation)
| model | kind | valid | strict F1 | salvage F1 | exact docs |
|---|---|---|---|---|---|
| zero-shot | digital | 0.960 | 0.890 | 0.907 | 0.05 |
| zero-shot | scan | 0.910 | 0.843 | 0.878 | 0.03 |
| fine-tuned (r16, merged fp16) | digital | 0.980 | 0.984 | 0.991 | 0.78 |
| fine-tuned (r16, merged fp16) | scan | 0.990 | 0.984 | 0.988 | 0.69 |

### Extraction, unseen layout D (test, scored once)
<<Paste the table after the Colab run: zero-shot D, few-shot D, fine-tuned D, strict and salvage side by side.
Note: fine-tune vs few-shot compares a trained model with a prompt-only method that had 2 in-context examples.>>

### Training
QLoRA: 4-bit NF4 + double quant, fp16 compute; LoRA r=16, alpha=32, dropout 0.05, all 7 linear projections;
lr 2e-4 cosine, batch 2 x 8 accumulation, 3 epochs = 225 steps. 18.46M trainable parameters (about 1.2% of the model).
Colab T4: 79.4 min (includes about 12.5 min of evaluation), peak VRAM 4.10 GB. Adapter weights: <<size of adapter_model.safetensors>>.

![loss curves]

![alt text](results/loss_curves.png)

Train loss 0.093 to 0.0028; validation loss floors at about 0.0033 from step 75-100 and tracks train. No sign of overfitting.
Loss is per answer token and most of the JSON is copyable structure, so low loss does not by itself show field-level accuracy.

### Serving (CPU, i5-12500H, Ollama, greedy, layout C)
| model | n per kind | strict F1 digital / scan | median s/doc digital / scan | tokens/s | file size |
|---|---|---|---|---|---|
| Q4_K_M | 100 | 0.991 / 0.983 | 13.6 / 12.6 | about 26 | <<GB>> |
| Q8_0 | 50 | 0.987 / 0.984 | 16.2 / 16.0 | about 19 | <<GB>> |

No measurable quality loss from quantization on layout C (differences are 1-2 documents). Q4 decodes about 1.35x faster than Q8.
The Q8 row covers 50 documents and its latency is probably understated (inferred, not checked).

### Agent (Q4_K_M, layout C, 30 documents per kind)
- Normal input: 100% valid on the first try, 0 repairs. **No agent improvement is claimed on clean input.**
- Stress test (30 scans, first OCR deliberately degraded): valid 0.900 to 1.000, strict F1 0.836 to 0.888, exact documents 0.133 to 0.200.
  3 of 30 documents were repaired; repaired documents took about 2.7x longer.
- "Success" means Pydantic-valid, not correct. After repair only 20% of stress documents are exactly right.

### Rendering
Deterministic template code, no model call. 30/30 digital and 30/30 scan agent outputs pass all Markdown and DOCX checks
(required sections, tables match the record, every number in the output exists in the record). Samples are in `results/rendered/`.

## Failure analysis (layout C, fine-tuned model)
| | digital | scan |
|---|---|---|
| documents with at least one wrong item | 22 / 100 | 31 / 100 |
| most common error | `limit_min` written where gold is null (15 documents) | `limit_min` null to value (9), `limit_max` value to null (7) |
| measurement rows copied into `steps` | 5 documents | 1 document |

- The model never replaced a real value with null in the digital set; it only invented values for null fields.
  <<Cause of the `limit_min` errors: fill in after checking the input text.>>
- 12 documents are perfect as digital and wrong as scans. Mostly `limit_max` dropped, plus 2 misread `document_id`s.
- <<3-5 public PDFs: what the parser and model do on pages that are not cleaning records. Add after the sample.>>

## Limitations
- One validation layout (C) and one test layout (D); results are specific to these layouts.
- Digital and scan variants share underlying documents, so they are not independent samples.
- Synthetic data only. Public documents are a small qualitative stress test, not a benchmark.
- Layout D failure categories were inspected before salvage F1 was added (a reporting change; the prompt was untouched).
- The "one change" training run was skipped: C is saturated (a 1-2 document difference is noise) and D must not be used to choose between runs. Run 1 is the final model.
- The adapter was trained against the 4-bit base, evaluated merged into fp16, then converted to GGUF and quantized. Each step was measured end to end.
- Groundedness checks compare the document with the extracted JSON, not with the original page. An extraction error passes through.
- The parser reads page 0 only for scans.


