"""Phase 3: QLoRA fine-tuning of Qwen2.5-1.5B-Instruct on the extraction pairs.

Run on a Colab T4 (needs CUDA). Example:
    python -m src.training.train_qlora --run-name r16_lr2e-4 --out-dir runs

Writes <out-dir>/<run-name>/adapter/   (LoRA weights + tokenizer)
       <out-dir>/<run-name>/metrics.json (config, loss history, time, peak VRAM, adapter size)
"""
import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.training.data import IGNORE_INDEX, load_jsonl, tokenize_split

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class ListDataset(Dataset):
    """Wraps a list of {"input_ids", "attention_mask", "labels"} dicts."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class PadCollator:
    """Right-pad a batch to its longest example.

    input_ids pad with the pad token, attention_mask pads with 0, labels pad with -100
    (so padding never contributes to the loss).
    """

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        longest = max(len(b["input_ids"]) for b in batch)
        ids, mask, labels = [], [], []
        for b in batch:
            pad = longest - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            mask.append(b["attention_mask"] + [0] * pad)
            labels.append(b["labels"] + [IGNORE_INDEX] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--train", default="data/pairs/train.jsonl")
    ap.add_argument("--val", default="data/pairs/val.jsonl")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--eval-n", type=int, default=300, help="val docs used for eval loss")
    ap.add_argument("--eval-steps", type=int, default=25)
    ap.add_argument("--grad-ckpt", action="store_true", help="enable if you hit out-of-memory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "No GPU. In Colab: Runtime > Change runtime type > T4 GPU."
    set_seed(args.seed)
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # ---- data -------------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(MODEL)
    train_pairs = load_jsonl(args.train)
    val_pairs = load_jsonl(args.val)
    if args.eval_n < len(val_pairs):  # fixed subsample so every run evaluates the same docs
        idx = sorted(random.Random(0).sample(range(len(val_pairs)), args.eval_n))
        val_pairs = [val_pairs[i] for i in idx]
    train_rows = tokenize_split(train_pairs, tok, args.max_len)  # raises if anything is too long
    val_rows = tokenize_split(val_pairs, tok, args.max_len)
    print(f"train={len(train_rows)} val(eval subset)={len(val_rows)}")

    # ---- 4-bit base model -------------------------------------------------------------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map={"": 0}
    )
    model.config.use_cache = False  # KV cache is for generation; useless (and conflicting) in training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_ckpt)

    # ---- LoRA adapters ----------------------------------------------------------------
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    # ---- training ---------------------------------------------------------------------
    targs = TrainingArguments(
        output_dir=os.path.join(run_dir, "trainer_tmp"),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        max_grad_norm=1.0,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=5,
        eval_strategy="steps",  # older transformers call this evaluation_strategy
        eval_steps=args.eval_steps,
        save_strategy="no",  # we save the adapter ourselves at the end
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_ckpt else None,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ListDataset(train_rows),
        eval_dataset=ListDataset(val_rows),
        data_collator=PadCollator(tok.pad_token_id),
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 2**30  # allocated, not reserved

    # ---- save -------------------------------------------------------------------------
    adapter_dir = os.path.join(run_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    metrics = {
        "args": vars(args),
        "trainable_params": trainable,
        "total_params": total,
        "train_seconds": round(train_seconds, 1),
        "peak_vram_gb_allocated": round(peak_vram_gb, 2),
        "adapter_size_mb": round(dir_size_mb(adapter_dir), 1),
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"done: {train_seconds / 60:.1f} min | peak VRAM {peak_vram_gb:.2f} GB | "
        f"adapter {metrics['adapter_size_mb']} MB -> {run_dir}"
    )


if __name__ == "__main__":
    main()