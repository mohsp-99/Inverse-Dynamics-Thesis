# 🧠 Training Guide

This guide explains how to run training experiments using the project’s modular Hydra + SB3 pipeline.

---

## 📦 Prerequisites

Make sure the following are installed and your virtual environment is active:

```bash
pip install -r requirements.txt
```

To make the project modules like `envs/`, `agents/`, and `utils/` importable, run everything from the `scripts/` folder using:

```bash
export PYTHONPATH=..
```

---

## 🚀 1. Run a Single Training Experiment

Use the `train.py` script with a base or composed config.

### Example:

```bash
cd scripts
export PYTHONPATH=..

python train.py experiment=exp_sac
```

### With CLI Overrides:

You can override any config parameter inline:

```bash
python train.py experiment=exp_sac agent.learning_rate=1e-4 env.dense_reward=false
```

### With Custom Seed:

```bash
python train.py experiment=exp_sac train.seed=123
```

### Custom Output Folder (optional):

```bash
python train.py experiment=exp_sac hydra.run.dir=logs/my_custom_run
```

---

## 🧪 2. Run Sweeps with `run_suite.py`

Use this when you want to train multiple experiments with multiple seeds.

### Example:

```bash
cd scripts
export PYTHONPATH=..

python run_suite.py
```

This will:

* Read a list of experiments and seeds.
* Launch each with `subprocess`.
* Automatically log outputs under `logs/multirun_{TIMESTAMP}/exp_NAME/seed_N`.

---

## 🧬 3. Available Experiments

| Name                   | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| `exp_sac`              | SAC on sparse reward push task                            |
| `exp_sac_her`          | Adds Hindsight Experience Replay (HER)                    |
| `exp_sac_her_norm`     | Adds action normalization                                 |
| `exp_sac_her_norm_aux` | Adds inverse dynamics auxiliary loss                      |
| `exp_full`             | Full pipeline: HER + normalization + aux + reward shaping |

You can define your own experiment files in `configs/experiment/`.

---

## 🛠 4. Changing Parameters

Use dot notation to override **any field** in any config:

```bash
python train.py experiment=exp_full env.reward.w_obj_goal=5.0
```

To change network architecture:

```bash
python train.py experiment=exp_full agent.net_arch=[128,128]
```

---

## 🧾 5. Outputs & Logs

* TensorBoard: `logs/.../tb/`
* Checkpoints: `logs/.../checkpoints/`
* Videos (if enabled): `logs/.../videos/`
* Best eval results: `logs/.../eval/`
* Final model: `logs/.../model_final.zip`

---

## 🧼 Tips

* If configs fail to override: check for nesting issues (`agent.agent.name` vs `agent.name`).
* Always clean `logs/` between incompatible changes.
* You can use `python -m scripts.train` if directly invoking fails.
* To check current config: `python train.py experiment=exp_full --cfg job`
