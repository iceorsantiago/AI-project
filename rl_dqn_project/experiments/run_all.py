"""Reproduce EVERYTHING with one command:   python experiments/run_all.py

Runs (or re-uses cached) Tier 1 CartPole, Tier 1 MountainCar, Tier 2 LunarLander, the observation-range
study, then rebuilds every figure in results/figures/ and every table in results/tables/.
Extra arguments are forwarded to each experiment, e.g.  --quick, --plot-only, --workers 4, --seeds 3.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["obs_ranges.py", "tier1_cartpole.py", "tier1_mountaincar.py", "tier2_ablation.py"]

if __name__ == "__main__":
    t0 = time.perf_counter()
    for script in SCRIPTS:
        args = [a for a in sys.argv[1:] if script != "obs_ranges.py" or a == "--quick"]
        print(f"\n=== {script} {' '.join(args)} ===", flush=True)
        subprocess.run([sys.executable, str(HERE / script), *args], check=True)
    print(f"\nAll done in {(time.perf_counter() - t0) / 60:.1f} min. Figures: results/figures/, tables: results/tables/")
