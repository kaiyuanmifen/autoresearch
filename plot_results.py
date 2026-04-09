import csv
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results.tsv"), delimiter="\t"))
xs = list(range(len(rows)))
ys = [float(r["val_bpb"]) for r in rows]
status = [r["status"] for r in rows]
desc = [r["description"] for r in rows]

color = {"keep": "#2ca02c", "discard": "#d62728", "crash": "#7f7f7f"}
colors = [color[s] for s in status]

# best-so-far among kept
best = float("inf")
best_curve = []
for s, y in zip(status, ys):
    if s == "keep" and y < best:
        best = y
    best_curve.append(best if best != float("inf") else None)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(xs, ys, c=colors, s=110, zorder=3, edgecolor="black")
ax.plot(xs, best_curve, color="#1f77b4", lw=2, label="best-so-far (kept)")
for x, y, d in zip(xs, ys, desc):
    ax.annotate(d, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel("generation")
ax.set_ylabel("val_bpb (lower = better)")
ax.set_title("autoresearch agent: val_bpb across generations")
ax.set_xticks(xs)
from matplotlib.lines import Line2D
legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=color["keep"], markersize=10, label="keep"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=color["discard"], markersize=10, label="discard"),
    Line2D([0], [0], color="#1f77b4", lw=2, label="best-so-far"),
]
ax.legend(handles=legend, loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("agent_evolution.png", dpi=120)
print("saved agent_evolution.png")
