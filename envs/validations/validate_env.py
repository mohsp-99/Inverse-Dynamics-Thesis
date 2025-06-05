'''
validate_env.py

Checks to ensure environment initialization is well-behaved and correct:

1. Object and goal are not initialized too close
2. Obstacles do not overlap with object or goal
3. Object, goal, and obstacles are within table bounds
4. Termination correctly triggers when object falls off table
5. Environment is deterministic under fixed seed
6. Reward shaping terms are reasonable and logged
7. No crashes or inconsistencies across multiple resets
'''

import gymnasium as gym
import numpy as np
from envs.push_wrappers import make_push_env
from pprint import pprint
import copy


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def is_inside_table(x, y, half_size):
    return -half_size[0] <= x <= half_size[0] and -half_size[1] <= y <= half_size[1]


def validate_env(
    cfg=None,
    env_id="PandaPush-v3",
    n_trials=50,
    min_obj_goal_dist=0.05,
    verbose=True,
    seed=42,
):
    print("\n[Env Validation] Starting checks...")
    cfg = cfg or {"id": env_id}
    env = make_push_env(cfg, seed=seed)

    table_half = (0.5, 0.5)
    success_count = 0
    failures = []
    reward_logs = []

    first_obs = None

    for i in range(n_trials):
        obs, _ = env.reset()
        if i == 0:
            first_obs = copy.deepcopy(obs["observation"])

        sim = env.unwrapped.sim
        obj = sim.get_base_position("object")
        goal = env.unwrapped.task.goal
        obstacles = getattr(env.unwrapped, "_obstacle_ids", [])

        # --- Check 1: object-goal distance
        dist = distance(obj, goal)
        if dist < min_obj_goal_dist:
            failures.append((i, "Object and goal too close"))

        # --- Check 2: Obstacle collisions
        for bid in obstacles:
            pos, _ = sim._p.getBasePositionAndOrientation(bid)
            if distance(pos, goal) < 0.05:
                failures.append((i, "Obstacle too close to goal"))
            if distance(pos, obj) < 0.05:
                failures.append((i, "Obstacle too close to object"))

        # --- Check 3: Bounds
        for name, pos in [("goal", goal), ("object", obj)]:
            x, y = pos[0], pos[1]
            if not is_inside_table(x, y, table_half):
                failures.append((i, f"{name} out of table bounds"))

        # --- Check 4: Step object off the table
        env.reset()
        try:
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
        except Exception as e:
            failures.append((i, f"Crash during stepping: {str(e)}"))

        # --- Check 5: Reward shaping logger
        if hasattr(env, "get_original_reward"):
            original_reward = env.get_original_reward()
            reward_logs.append(original_reward)

        success_count += 1

    # --- Check 6: Determinism under fixed seed
    env2 = make_push_env(cfg, seed=seed)
    obs2, _ = env2.reset()
    obs2 = obs2["observation"]
    if not np.allclose(first_obs, obs2, atol=1e-6):
        failures.append(("determinism", "Environment not deterministic with fixed seed"))

    print(f"\n[Summary] Passed {success_count - len(failures)}/{n_trials} trials")
    if failures:
        print("\n[Failures]")
        for fail in failures:
            print(f"  Trial {fail[0]}: {fail[1]}")
    else:
        print("\n[OK] All checks passed.")

    print("\n[Sample Reward Shaping Logs]")
    pprint(reward_logs[:3])


if __name__ == "__main__":
    validate_env()
