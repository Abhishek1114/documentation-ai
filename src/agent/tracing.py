"""One JSONL trace file per document run: one line per node execution, so a failure can be replayed."""
import json
import time
from pathlib import Path


class Tracer:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)  # a rerun overwrites, never appends to a stale trace
        self.t0 = time.perf_counter()
        self.n = 0

    def log(self, node, **fields):
        self.n += 1
        row = {"step": self.n, "t": round(time.perf_counter() - self.t0, 3), "node": node, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")