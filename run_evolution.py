"""Run train.py across N 'generations', varying DEPTH. Record val_bpb and plot."""
import re
import subprocess
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
TRAIN = ROOT / "train.py"
RESULTS = ROOT / "evolution_results.json"

# Each generation: (label, DEPTH value)
GENERATIONS = [
    ("gen1 d=4", 4),
    ("gen2 d=6", 6),
    ("gen3 d=8", 8),
    ("gen4 d=10", 10),
]

def patch_depth(depth: int):
    text = TRAIN.read_text()
    new = re.sub(r"^DEPTH\s*=\s*\d+", f"DEPTH = {depth}", text, count=1, flags=re.M)
    TRAIN.write_text(new)

def run_one() -> float:
    proc = subprocess.run(
        ["uv", "run", "train.py"],
        cwd=ROOT, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    m = re.search(r"val_bpb:\s*([0-9.]+)", out)
    if not m:
        raise RuntimeError(f"val_bpb not found. Tail:\n{out[-2000:]}")
    return float(m.group(1))

def main():
    original = TRAIN.read_text()
    results = []
    try:
        for label, depth in GENERATIONS:
            print(f"=== {label} (DEPTH={depth}) ===", flush=True)
            patch_depth(depth)
            bpb = run_one()
            print(f"  val_bpb = {bpb:.4f}", flush=True)
            results.append({"label": label, "depth": depth, "val_bpb": bpb})
            RESULTS.write_text(json.dumps(results, indent=2))
    finally:
        TRAIN.write_text(original)

    labels = [r["label"] for r in results]
    bpbs = [r["val_bpb"] for r in results]
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(bpbs) + 1), bpbs, marker="o")
    for i, (l, b) in enumerate(zip(labels, bpbs), 1):
        plt.annotate(f"{b:.3f}", (i, b), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xticks(range(1, len(bpbs) + 1), labels, rotation=20)
    plt.ylabel("val_bpb (lower = better)")
    plt.xlabel("generation")
    plt.title("Loss across generations of evolution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = ROOT / "evolution_loss.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
