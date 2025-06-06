# push_wrappers.py – PandaPush extensions for stability and reproducibility
# Includes:
# - Object-goal minimum distance constraint
# - Safe obstacle placement
# - Termination on object falling off table
# - Reward shaping for distance and obstacle collisions
# - Full deterministic seeding
# - Logged reward shaping components for validation

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics
from panda_gym.envs.tasks.push import Push


def _inside_table(x: float, y: float, half_size: Tuple[float, float]) -> bool:
    return -half_size[0] <= x <= half_size[0] and -half_size[1] <= y <= half_size[1]

def _dist(a, b):
    return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))


def _valid_position(p, object_pos, goal_pos, margin=0.05):
    return (
        _dist(p, object_pos) > margin
        and _dist(p, goal_pos) > margin
        and _inside_table(p[0], p[1], (0.5, 0.5))
    )


class TerminateIfOffTableWrapper(gym.Wrapper):
    def __init__(self, env, table_half_size, obj_z_threshold=-0.01):
        super().__init__(env)
        self.half_size = table_half_size
        self.obj_z_threshold = obj_z_threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not terminated:
            x, y, z = self.unwrapped.sim.get_base_position("object")
            if not _inside_table(x, y, self.half_size) or z < self.obj_z_threshold:
                terminated = True
                info["off_table"] = True
        return obs, reward, terminated, truncated, info


class MinObjectGoalDistanceWrapper(gym.Wrapper):
    def __init__(self, env, min_distance=0.05, max_attempts=100):
        super().__init__(env)
        self.min_distance = min_distance
        self.max_attempts = max_attempts

    def reset(self, **kwargs):
        for _ in range(self.max_attempts):
            obs, info = self.env.reset(**kwargs)
            obj = self.unwrapped.sim.get_base_position("object")
            goal = self.unwrapped.task.goal
            if _dist(obj, goal) >= self.min_distance:
                return obs, info
        return obs, info  # fallback after max attempts


class ObstacleWrapper(gym.Wrapper):
    def __init__(self, env, n_obstacles, radius_range, height_range, table_half_size, seed=None):
        super().__init__(env)
        self.n_obs = n_obstacles
        self.r_range = radius_range
        self.h_range = height_range
        self.half_size = table_half_size
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

def reset(self, **kwargs):
    if hasattr(self.unwrapped, "_obstacle_ids"):
        for bid in self.unwrapped._obstacle_ids:
            self.unwrapped.sim._p.removeBody(bid)
    self.unwrapped._obstacle_ids: List[int] = []

    if self.n_obs > 0:
        object_pos = self.unwrapped.sim.get_base_position("object")
        goal_pos = self.unwrapped.task.goal

        attempts = 0
        placed_count = 0

        while placed_count < self.n_obs and attempts < 100 * self.n_obs:
            x = self._rng.uniform(-self.half_size[0] * 0.9, self.half_size[0] * 0.9)
            y = self._rng.uniform(-self.half_size[1] * 0.9, self.half_size[1] * 0.9)
            r = self._rng.uniform(*self.r_range)
            h = self._rng.uniform(*self.h_range)
            pos = np.array([x, y, h / 2])

            if _valid_position((x, y), object_pos, goal_pos):
                body_name = f"obstacle_{placed_count}"
                body_id = self.unwrapped.sim.create_cylinder(
                    body_name=body_name,
                    radius=r,
                    height=h,
                    mass=0.0,
                    position=pos,
                    rgba_color=np.array([0.8, 0.2, 0.2, 1.0]),
                    lateral_friction=1.0,
                )
                self.unwrapped._obstacle_ids.append(body_id)
                placed_count += 1

            attempts += 1

    return super().reset(**kwargs)


class RewardShapingWrapper(gym.RewardWrapper):
    def __init__(self, env, w_ee_obj=1.0, w_obj_goal=1.0, success_bonus=5.0, obstacle_penalty=2.0):
        super().__init__(env)
        self.w_ee_obj = w_ee_obj
        self.w_obj_goal = w_obj_goal
        self.success_bonus = success_bonus
        self.obstacle_penalty = obstacle_penalty
        self._last_log: Dict[str, float] = {}

    def reward(self, reward):
        sim = self.unwrapped.sim
        ee = self.unwrapped.robot.get_ee_position()
        obj = sim.get_base_position("object")
        goal = self.unwrapped.task.goal

        d1 = _dist(ee, obj)
        d2 = _dist(obj, goal)
        success = reward > 0

        shaped = reward - self.w_ee_obj * d1 - self.w_obj_goal * d2
        if success:
            shaped += self.success_bonus

        obstacle_hit = False
        if hasattr(self.unwrapped, "_obstacle_ids"):
            for oid in self.unwrapped._obstacle_ids:
                pos, _ = sim._p.getBasePositionAndOrientation(oid)
                if _dist(pos, ee) < 0.05 or _dist(pos, obj) < 0.05:
                    shaped -= self.obstacle_penalty
                    obstacle_hit = True
                    break

        self._last_log = {
            "base_reward": reward,
            "ee_obj_dist": d1,
            "obj_goal_dist": d2,
            "success_bonus": self.success_bonus if success else 0.0,
            "obstacle_penalty": self.obstacle_penalty if obstacle_hit else 0.0,
            "total_shaped": shaped,
        }

        return shaped

    def get_last_reward_log(self):
        return self._last_log
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = self.reward(reward)
        info.update(self._last_log)
        return obs, shaped_reward, terminated, truncated, info


def make_push_env(cfg: Dict[str, Any], seed: int | None = None, deterministic=False):
    env_id = str(cfg.get("id", "PandaPush-v3"))
    env_kwargs_raw = cfg.get("kwargs", {})
    env_kwargs = env_kwargs_raw if isinstance(env_kwargs_raw, dict) else {}
    env = gym.make(env_id, **env_kwargs)

    table_cfg = cfg.get("table", {})
    half_size = tuple(table_cfg.get("half_size", (0.5, 0.5)))
    z_th = table_cfg.get("obj_z_threshold", -0.01)

    # env = TerminateIfOffTableWrapper(env, table_half_size=half_size, obj_z_threshold=z_th)
    # env = MinObjectGoalDistanceWrapper(env, min_distance=cfg.get("min_obj_goal_dist", 0.05))

    obs_cfg = cfg.get("obstacle", {})
    n_obs = 0 if deterministic else int(obs_cfg.get("n", 0))
    if n_obs > 0:
        env = ObstacleWrapper(
            env,
            n_obstacles=n_obs,
            radius_range=tuple(obs_cfg.get("radius_range", (0.03, 0.07))),
            height_range=tuple(obs_cfg.get("height_range", (0.05, 0.10))),
            table_half_size=half_size,
            seed=seed,
        )

    if bool(cfg.get("dense_reward", False)):
        rw_cfg = cfg.get("reward", {})
        env = RewardShapingWrapper(
            env,
            w_ee_obj=rw_cfg.get("w_ee_obj", 1.0),
            w_obj_goal=rw_cfg.get("w_obj_goal", 1.0),
            success_bonus=rw_cfg.get("success_bonus", 5.0),
            obstacle_penalty=rw_cfg.get("obstacle_penalty", 2.0),
        )

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    env = RecordEpisodeStatistics(env)
    return env
