"""Utility callbacks for training & evaluation.

This module wires together Stable‑Baselines3 callbacks so that *every run* gets

* **periodic checkpoints** (`model_step_x.zip`, `vecnormalize.pkl`).
* **evaluation** on a hold‑out environment with success‑rate metrics.
* **optional MP4 videos** of roll‑outs (train & eval).
* **TensorBoard + optional Weights‑and‑Biases logging** with identical tags.

All parameters are driven by the Hydra config dictionary *cfg.callbacks* – see
`configs/base.yaml` for the canonical schema.
"""
from __future__ import annotations

import imageio
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch as th
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

# `VideoRecorderCallback` exists in SB3 ≥2.5.
try:
    from stable_baselines3.common.callbacks import VideoRecorderCallback
except ImportError:  # pragma: no cover – older SB3 fallback
    VideoRecorderCallback = None  # type: ignore


class SimpleGymVideoRecorder(BaseCallback):
    """
    Records one episode every `record_freq` steps using Gym's native RecordVideo.
    Avoids SB3's tensorboard logging. Pure .mp4 saving.

    Args:
        eval_env_factory: a callable that returns a fresh unwrapped Gym env
        run_dir: base log directory (Hydra run dir)
        record_freq: every N steps, trigger a recording
        deterministic: use deterministic actions
    """

    def __init__(self, eval_env_factory, run_dir, record_freq=5000, deterministic=True):
        super().__init__()
        self.eval_env_factory = eval_env_factory
        self.run_dir = Path(run_dir)
        self.record_freq = record_freq
        self.deterministic = deterministic
        self.counter = 0

    def _on_step(self) -> bool:
        self.counter += 1
        if self.counter % self.record_freq != 0:
            return True

        print(f"[SimpleGymVideoRecorder] Recording at step {self.num_timesteps}")

        # Create fresh environment wrapped with video recorder
        video_dir = self.run_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        env = self.eval_env_factory()
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            name_prefix=f"step_{self.num_timesteps}",
            episode_trigger=lambda x: True,  # Record first episode
            disable_logger=True,
        )

        obs, _ = env.reset()
        done, step= False, 0
        max_steps = 50
        while not done and step < max_steps:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _, done, _, _ = env.step(action)
            step+=1

        env.close()
        return True
    

# Optional Weights & Biases integration (lazy‑import)
try:
    import wandb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    wandb = None  # type: ignore

__all__ = [
    "build_default_callbacks",
]


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------


def _make_path(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class WandbCallback(BaseCallback):
    """Light wrapper that logs SB3 scalars to a WandB run.

    If *wandb* is not installed or `wandb_run` is *None*, the callback becomes a
    no‑op so downstream code does not need to handle that edge case.
    """

    def __init__(self, wandb_run: "wandb.sdk.wandb_run.Run | None" = None):
        super().__init__()
        self._wandb_run = wandb_run

    def _on_step(self) -> bool:  # noqa: D401
        if self._wandb_run is None:
            return True  # no‑op

        # Current logger dict (already contains ep_* & loss keys)
        log_dict = {k: v[0] for k, v in self.logger.get_log_dict().items()}
        if log_dict:
            self._wandb_run.log(log_dict, step=self.num_timesteps)
        return True


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------


def build_default_callbacks(cfg, eval_env, run_dir):
    """
    Build CallbackList from cfg.callbacks.{checkpoint, eval, video}.
    All three blocks are optional.
    """
    callbacks = []

    cb_cfg = cfg.get("callbacks", {})  # ← new root

    # --- checkpoint -------------------------------------------------------
    ckpt_cfg = cb_cfg.get("checkpoint", None)
    if ckpt_cfg and ckpt_cfg.save_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=ckpt_cfg.save_freq,
                save_path=os.path.join(run_dir, "checkpoints"),
                name_prefix="model",
                save_replay_buffer=ckpt_cfg.save_replay_buffer,
            )
        )

    # --- evaluation -------------------------------------------------------
    eval_cfg = cb_cfg.get("eval", None)
    if eval_cfg and eval_cfg.eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(run_dir, "best"),
                log_path=os.path.join(run_dir, "eval"),
                eval_freq=eval_cfg.eval_freq,
                n_eval_episodes=eval_cfg.n_eval_episodes,
                deterministic=eval_cfg.deterministic,
            )
        )

    # --- video recording --------------------------------------------------
    vid_cfg = cb_cfg.get("video", None)
    if vid_cfg and vid_cfg.record:
        # We create a fresh eval_env for each video
        def eval_env_factory():
            from envs.push_wrappers import make_push_env
            return make_push_env(
                cfg.env,
                seed=cfg.train.seed + 999,  # avoid training collision
                deterministic=True
            )

        callbacks.append(
            SimpleGymVideoRecorder(
                eval_env_factory=eval_env_factory,
                run_dir=run_dir,
                record_freq=vid_cfg.record_freq,
                deterministic=True,
            )
        )

    # TensorBoard logger (always on)
    return CallbackList(callbacks)
