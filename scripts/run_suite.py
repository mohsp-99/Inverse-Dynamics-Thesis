# scripts/run_suite.py

import os
import subprocess
from datetime import datetime
from pathlib import Path

# Define the list of experiments and seeds
EXPERIMENTS = [
    "exp_sac",
    "exp_sac_her",
    "exp_sac_her_norm",
    "exp_sac_her_norm_aux",
]
SEEDS = [0, 1, 2]

# Base directory for this run
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
multirun_root = Path("logs") / f"multirun_{timestamp}"

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = BASE_DIR / "scripts" / "train.py"

# Setup PYTHONPATH so imports work correctly
env = {
    **os.environ,
    "PYTHONPATH": str(BASE_DIR),
}

# Run each experiment and seed
for exp_name in EXPERIMENTS:
    for seed in SEEDS:
        run_dir = multirun_root / exp_name / f"seed_{seed}"
        cmd = [
            "python", str(TRAIN_SCRIPT),
            f"experiment={exp_name}",
            f"train.seed={seed}",
            f"hydra.run.dir={run_dir}",
        ]
        print(f"🚀 Launching: {' '.join(cmd)}")
        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            print(f"❌ Run failed (return code {ret.returncode}) → aborting.")
            exit(1)
