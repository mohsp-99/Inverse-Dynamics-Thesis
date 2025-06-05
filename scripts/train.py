# scripts/train.py

import hydra
from omegaconf import DictConfig
import os
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils.callbacks import build_default_callbacks
from envs.push_wrappers import make_push_env
from agents.sac_id_aux import SACIDAux

AGENT_CLASSES = {
    "SAC": SAC,
    "SAC_ID_AUX": SACIDAux,
}


def build_env(cfg: DictConfig, seed: int, eval: bool = False):
    def make_env_fn():
        return make_push_env(cfg.env, seed=seed, deterministic=eval)

    env_fns = [lambda: make_env_fn() for _ in range(cfg.train.n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    return vec_env


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
    seed = cfg.train.seed

    # Train environment
    train_env = build_env(cfg, seed=seed, eval=False)

    # Agent selection
    agent_name = cfg.agent.name
    AgentClass = AGENT_CLASSES.get(agent_name)
    if AgentClass is None:
        raise ValueError(f"Unknown agent type: {agent_name}")

    # Agent instantiation
    agent_kwargs = {k: v for k, v in cfg.agent.items() if k not in ["name", "policy", "auxiliary", "net_arch"]}
    policy_kwargs = dict(net_arch=list(cfg.agent.net_arch))

    model = AgentClass(
        policy=cfg.agent.policy,
        env=train_env,
        seed=seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        policy_kwargs=policy_kwargs,
        **agent_kwargs,
    )


    # Callbacks
    callbacks = build_default_callbacks(cfg, eval_env=build_env(cfg, seed, eval=True), run_dir=run_dir)

    # Training
    print(f"[train] Starting learning for {cfg.train.total_timesteps} timesteps …")
    model.learn(
        total_timesteps=cfg.train.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save
    print("[train] Saving final model and VecNormalize stats…")
    model.save(run_dir / "model_final")
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))


if __name__ == "__main__":
    main()
