"""Aggregate sweep results and print the best configuration.

Works with any sweep — auto-detects saved hyperparameters from the .npz files.

Usage:
    python sweep_aggregate.py <sweep_name>
"""

import argparse
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("sweep_name", type=str, help="Name of the sweep to aggregate")
args = parser.parse_args()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", args.sweep_name, "results")

if not os.path.isdir(out_dir):
    print(f"Sweep directory not found: {out_dir}")
    raise SystemExit(1)

# ── Collect results ──────────────────────────────────────────────────
SKIP_KEYS = {"val_bits_per_spike", "best_iteration", "task_id", "dataset"}
results = []
param_keys = None

for fname in sorted(os.listdir(out_dir)):
    if not fname.endswith(".npz"):
        continue
    d = dict(np.load(os.path.join(out_dir, fname), allow_pickle=True))
    d["task_id"] = fname.replace(".npz", "")
    results.append(d)

    # Discover hyperparameter keys from the first file
    if param_keys is None:
        param_keys = [k for k in d if k not in SKIP_KEYS]

if not results:
    print(f"No .npz files found in {out_dir}. Are the jobs done?")
    raise SystemExit(1)

print(f"Sweep '{args.sweep_name}': collected {len(results)} results")
if "dataset" in results[0]:
    print(f"Dataset: {str(results[0]['dataset'])}")
print()

# ── Sort and display ─────────────────────────────────────────────────
results.sort(key=lambda r: float(r["val_bits_per_spike"]), reverse=True)

def fmt_params(r):
    return ", ".join(f"{k}={float(r[k]):.4g}" for k in param_keys)

print("Top 10 configurations:")
for i, r in enumerate(results[:10]):
    print(f"  {i+1}. bps_val={float(r['val_bits_per_spike']):.4f}  "
          f"{fmt_params(r)}  (iter {int(r['best_iteration'])})")

best = results[0]
print(f"\nBest configuration:")
for k in param_keys:
    print(f"  {k:20s} = {float(best[k]):.4g}")
print(f"  {'val bits/spike':20s} = {float(best['val_bits_per_spike']):.4f}")
