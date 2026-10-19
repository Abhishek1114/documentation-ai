"""Read <run-dir>/metrics.json, plot train vs val loss, print the numbers for the overfitting read.

Usage: python -m src.training.plot_curves /content/drive/MyDrive/documentation-ai/runs/r16_lr2e-4
(needs matplotlib: pip install matplotlib)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # write a PNG, no window
import matplotlib.pyplot as plt


def main(run_dir):
    with open(os.path.join(run_dir, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    hist = m["log_history"]
    train = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    if not train or not evals:
        raise SystemExit("metrics.json has no train/eval loss entries; did training finish?")

    plt.figure(figsize=(7, 4))
    plt.plot(*zip(*train), label="train loss", alpha=0.8)
    plt.plot(*zip(*evals), label="val loss (layout C)", marker="o")
    plt.xlabel("optimizer step")
    plt.ylabel("loss (answer tokens only)")
    plt.title(os.path.basename(run_dir.rstrip("/\\")))
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(run_dir, "loss_curves.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)

    best_step, best_val = min(evals, key=lambda x: x[1])
    print(f"saved {out}")
    print(f"first train loss logged: {train[0][1]:.4f} at step {train[0][0]}")
    print(f"last train loss (log window): {train[-1][1]:.4f} at step {train[-1][0]}")
    print(f"first val loss: {evals[0][1]:.4f} at step {evals[0][0]}")
    print(f"last val loss: {evals[-1][1]:.4f} at step {evals[-1][0]}")
    print(f"best val loss: {best_val:.4f} at step {best_step}")
    print(
        f"train time {m['train_seconds']} s | peak VRAM {m['peak_vram_gb_allocated']} GB | "
        f"adapter {m['adapter_size_mb']} MB | trainable {m['trainable_params']:,}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
