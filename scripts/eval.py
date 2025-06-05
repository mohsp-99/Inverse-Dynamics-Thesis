from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC, TD3
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Extend PYTHONPATH to import project modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.push_wrappers import make_push_env

# Agent registry
AGENT_REGISTRY: Dict[str, Any] = {
    "sac": SAC,
    "td3": TD3,
    "tqc": TQC,
}
try:
    from agents.sac_id_aux import SACIDAux
    AGENT_REGISTRY["sac_id_aux"] = SACIDAux
except Exception:
    pass


def build_env(seed: int, cfg_path: Path | None, video_dir: Path | None = None):
    """Recreate a deterministic environment stack for evaluation."""
    env_cfg = {}
    if cfg_path and cfg_path.exists():
        env_cfg = json.loads(cfg_path.read_text())

    def _factory():
        env = make_push_env(env_cfg, seed=seed, deterministic=True)
        if video_dir:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_dir),
                name_prefix="eval",
                episode_trigger=lambda i: i == 0,
                disable_logger=True,
            )
        return env

    env = DummyVecEnv([_factory])

    if env_cfg.get("normalize_obs", False):
        env = VecNormalize.load(env_cfg["normalize_path"], env)
        env.training = False
        env.norm_reward = False

    return env


def run_eval(
    checkpoint: Path,
    agent_name: str | None,
    seed: int,
    n_episodes: int,
    record_video: bool = False,
):
    assert checkpoint.exists(), f"Checkpoint not found: {checkpoint}"
    ckpt_dir = checkpoint.parent

    # Try to auto-detect agent type if not provided
    if agent_name is None:
        # Try agent.txt first
        agent_file = ckpt_dir / "agent.txt"
        if agent_file.exists():
            agent_name = agent_file.read_text().strip()
        else:
            # Try .hydra/config.yaml fallback
            hydra_cfg = ckpt_dir / ".hydra" / "config.yaml"
            if hydra_cfg.exists():
                import yaml
                with open(hydra_cfg) as f:
                    config = yaml.safe_load(f)
                    agent_name = config.get("agent", {}).get("name")
            if not agent_name:
                raise ValueError("Could not infer agent type. Please provide --agent or create agent.txt.")


    assert agent_name in AGENT_REGISTRY, f"Unknown agent type: {agent_name}"

    # Set up evaluation output folders
    eval_dir = ckpt_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    video_dir = eval_dir / "videos" if record_video else None
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild environment
    env_cfg_path = ckpt_dir / "env_cfg.json"
    env = build_env(seed, cfg_path=env_cfg_path, video_dir=video_dir)

    # Load agent safely, bypassing dummy env crash
    AgentClass = AGENT_REGISTRY[agent_name]
    model = AgentClass.load(checkpoint, print_system_info=False, custom_objects={"env": env})
    model.set_env(env)

    # Evaluation loop
    success, ep_lens = [], []

    for _ in range(n_episodes):
        obs = env.reset()
        done, steps = False, 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
            if "is_success" in info[0]:
                success.append(info[0]["is_success"])
        ep_lens.append(steps)

    # Save metrics
    results = {
        "success_rate": float(np.mean(success)),
        "avg_episode_length": float(np.mean(ep_lens)),
        "n_episodes": n_episodes,
        "seed": seed,
        "agent": agent_name,
    }

    print(json.dumps(results, indent=2))

    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained RL agent.")
    p.add_argument("checkpoint", type=str, help="Path to model_*.zip")
    p.add_argument("--agent", type=str, default=None, help="Agent name (e.g. sac, sac_id_aux). Auto-detects if not set.")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record_video", action="store_true", help="Save rollout videos")
    return p.parse_args()


def main():
    args = parse_args()
    run_eval(
        checkpoint=Path(args.checkpoint).expanduser(),
        agent_name=args.agent,
        seed=args.seed,
        n_episodes=args.n_episodes,
        record_video=args.record_video,
    )


if __name__ == "__main__":
    main()
