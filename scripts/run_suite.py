"""Run a sequence of Hydra experiment configs back‑to‑back.

Why?  Our ablation study has five configs (baseline → full).  Rather than
starting each manually, run:

    python scripts/run_suite.py --suite default --seeds 0 1 2

and the script will launch `scripts/train.py` for every (config × seed)
combination in a *fresh* subprocess.  This avoids CUDA‑memory creep and keeps
Hydra runtimes separate.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Define experiment suites here
# ---------------------------------------------------------------------------
SUITES = {
    "default": [
        "exp_sac",
        "exp_sac_her",
        "exp_sac_her_norm",
        "exp_sac_her_norm_aux",
        "exp_full",
    ],
    # Add more suites if needed (e.g. td3, tqc)
}


# ---------------------------------------------------------------------------
# Helper to launch a single training run as a subprocess
# ---------------------------------------------------------------------------
def launch(config_name: str, seed: int, log_root: str, extra_args: List[str]) -> None:
    run_dir = f"{log_root}/{config_name}/seed_{seed}"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("train.py")),
        f"hydra.run.dir={run_dir}",
        f"seed={seed}",
        f"experiment={config_name}",
    ] + extra_args

    print("\n🚀 Launching:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)

    if completed.returncode != 0:
        print(f"❌ Run failed (return code {completed.returncode}) → aborting.")
        sys.exit(completed.returncode)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a suite of ablation configs.")
    p.add_argument("--suite", default="default", help=f"Suite key in {list(SUITES)}")
    p.add_argument("--seeds", type=int, nargs="*", default=[0], help="List of RNG seeds")
    p.add_argument("--log_root", default="logs/multirun", help="Base directory for logs")
    p.add_argument("--extra", nargs=argparse.REMAINDER, help="Overrides after '--'")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.suite not in SUITES:
        print(f"Unknown suite '{args.suite}'. Available: {list(SUITES)}")
        sys.exit(1)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = f"{args.log_root}_{timestamp}"

    configs = SUITES[args.suite]
    extra_args = args.extra or []

    for seed in args.seeds:
        for cfg_name in configs:
            launch(cfg_name, seed, log_root, extra_args)

    print(f"\n✅ All runs completed → saved to {log_root}")



if __name__ == "__main__":
    main()
