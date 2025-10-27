import argparse
import time
import csv
from snake_env import SnakeEnv
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser(description='Test trained Snake agent')
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--width", type=int, default=20,
                        help="Game grid width")
    parser.add_argument("--height", type=int, default=15,
                        help="Game grid height")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay between steps (seconds)")
    parser.add_argument("--output", type=str, default="snake_eval_results.csv",
                        help="CSV file to save results")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model)

    # Create environment
    env = SnakeEnv(render_mode="human", width=args.width, height=args.height)

    results = []  # Store per-episode results

    print(f"\nTesting for {args.episodes} episodes...\n")

    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            env.render()
            time.sleep(args.delay)

        score = info.get('score', 0)
        results.append({
            "Episode": episode + 1,
            "Score": score,
            "Steps": steps,
            "Total_Reward": round(episode_reward, 2)
        })

        print(f"Episode {episode + 1}: Score = {score}, Steps = {steps}, Reward = {episode_reward:.2f}")

    env.close()

    # Calculate summary stats
    avg_score = sum(r["Score"] for r in results) / len(results)
    max_score = max(r["Score"] for r in results)
    min_score = min(r["Score"] for r in results)
    avg_steps = sum(r["Steps"] for r in results) / len(results)
    print(f"\nAverage Score: {avg_score:.2f}, Max Score: {max_score}, Min Score: {min_score}, Average Steps: {avg_steps:.2f}")

    # Save to CSV
    with open(args.output, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Episode", "Score", "Steps", "Total_Reward"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to '{args.output}'")

if __name__ == "__main__":
    main()
