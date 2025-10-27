# -*- coding: utf-8 -*-

import argparse
import os
import csv
import numpy as np
from stable_baselines3 import PPO
from tetris_env import TetrisEnv


def run_episode(model, reward_mode="hybrid", render=False, fps=60, drop_speed=10):
    """
    Run one episode; optionally render. Returns a metrics dict.
    - render: if True, env uses render_mode="human" and limits FPS via metadata["render_fps"].
    - drop_speed: smaller = faster gravity (env ticks). We also set it after reset to
      override any default the env assigns during reset().
    """
    env = TetrisEnv(
        render_mode="human" if render else None,
        reward_mode=reward_mode,
        drop_speed=drop_speed,
    )

    # Respect desired FPS for human render
    if render:
        env.metadata["render_fps"] = int(fps)

    obs, info = env.reset()

    # Some envs overwrite drop_speed in reset(); enforce again just in case.
    try:
        env.drop_speed = int(drop_speed)
    except Exception:
        pass

    done = False
    trunc = False
    ep_reward = 0.0
    steps = 0
    action_counts = {i: 0 for i in range(6)}  # keep if you want to inspect action usage

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_counts[action] += 1

        obs, r, done, trunc, info = env.step(action)
        ep_reward += float(r)
        steps += 1

    # Collect metrics (use .get to be robust)
    lines_cleared = int(info.get("lines_cleared", 0))
    pieces_placed = int(info.get("pieces_placed", 0))
    max_height = int(info.get("max_height", 0))
    holes = int(info.get("holes", 0))

    env.close()
    return {
        "reward": float(ep_reward),
        "lines_cleared": lines_cleared,
        "pieces_placed": pieces_placed,
        "max_height": max_height,
        "holes": holes,
        "steps": steps,
        "crashed": int(done and not trunc),
        "truncated": int(trunc),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="models/ppo_tetris_hybrid_final",
                   help="Saved model path (omit .zip)")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    p.add_argument("--render", type=int, default=0, help="1 = show window, 0 = headless")
    p.add_argument("--reward_mode", type=str, default="hybrid",
                   choices=["standard", "aggressive", "shaped", "hybrid"],
                   help="Reward setting used by the environment")
    p.add_argument("--csv_out", type=str, default="logs/eval_metrics_tetris.csv",
                   help="Where to save summary CSV")
    # Speed controls:
    p.add_argument("--fps", type=int, default=60, help="Window FPS cap when rendering")
    p.add_argument("--drop_speed", type=int, default=10,
                   help="Smaller = faster gravity (env ticks). Try 3–10 for speed.")
    args = p.parse_args()

    # Validate model
    zip_path = args.model_path if args.model_path.endswith(".zip") else args.model_path + ".zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Model not found: {zip_path}")

    # Ensure CSV directory exists
    out_dir = os.path.dirname(args.csv_out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Load on CPU to avoid the PPO GPU warning & keep inference snappy for MLP
    print(f"Loading model from: {args.model_path}")
    model = PPO.load(args.model_path, device="cpu")

    rows = []
    for ep in range(1, args.episodes + 1):
        print(f"Running episode {ep}...")
        metrics = run_episode(
            model,
            reward_mode=args.reward_mode,
            render=bool(args.render),
            fps=args.fps,
            drop_speed=args.drop_speed,
        )
        metrics["episode"] = ep
        rows.append(metrics)

    # Aggregate
    mean_reward = float(np.mean([r["reward"] for r in rows])) if rows else 0.0
    std_reward = float(np.std([r["reward"] for r in rows])) if rows else 0.0
    mean_lines = float(np.mean([r["lines_cleared"] for r in rows])) if rows else 0.0
    mean_pieces = float(np.mean([r["pieces_placed"] for r in rows])) if rows else 0.0
    crash_rate = float(np.mean([r["crashed"] for r in rows])) if rows else 0.0

    print(f"\n=== Results over {len(rows)} episodes ===")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean lines cleared: {mean_lines:.2f}")
    print(f"Mean pieces placed: {mean_pieces:.2f}")
    print(f"Crash rate: {crash_rate*100:.1f}%")

    # Write CSV
    fieldnames = [
        "episode", "reward", "lines_cleared", "pieces_placed",
        "max_height", "holes", "steps", "crashed", "truncated"
    ]
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Saved metrics to {args.csv_out}")


if __name__ == "__main__":
    main()
