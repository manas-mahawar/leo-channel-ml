"""Train several model variants one after another (one run at a time).

Reads lines "<variant>" or "<variant> <seed>" from a queue file (default
training/queue.txt; lines starting with # are ignored). A run is skipped
when its run_config.json exists (finished). An interrupted run resumes from
its last finished epoch because train_paper_model.py uses BackupAndRestore.
The file is re-read after every run, so lines can be appended while the
queue is running.

Usage:
    python training/run_queue.py                              # training/queue.txt
    python training/run_queue.py training/queue_seeds.txt     # repeated seeds
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "training" / "queue.txt"
LOG_DIR = ROOT / "models" / "paper" / "logs"
DEFAULT_SEED = 42   # config.SEED; runs with this seed live in models/paper/<variant>/


def run_dir(variant, seed):
    if seed == DEFAULT_SEED:
        return ROOT / "models" / "paper" / variant
    return ROOT / "models" / "paper" / "seeds" / f"{variant}_seed{seed}"


def next_run():
    for line in QUEUE.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            variant, seed = parts[0], int(parts[1]) if len(parts) > 1 else DEFAULT_SEED
            if not (run_dir(variant, seed) / "run_config.json").exists():
                return variant, seed
    return None


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    while (run := next_run()) is not None:
        variant, seed = run
        name = variant if seed == DEFAULT_SEED else f"{variant}_seed{seed}"
        print(f"[{time.strftime('%H:%M:%S')}] training {name}", flush=True)
        with open(LOG_DIR / f"{name}.log", "a", encoding="utf-8") as log:
            code = subprocess.call([sys.executable, "-u", "training/train_paper_model.py", variant, str(seed)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"[{time.strftime('%H:%M:%S')}] {name} exited with code {code}", flush=True)
        if code != 0:
            print("Stopping the queue; see the log above.", flush=True)
            raise SystemExit(code)
    print("Queue finished.", flush=True)


if __name__ == "__main__":
    main()
