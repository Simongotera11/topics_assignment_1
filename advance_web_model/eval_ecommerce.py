import argparse
import subprocess
import time
import sys
from stable_baselines3 import PPO
from ecommerce_env import EcommerceEnv


def start_http_server():
    """Start HTTP server."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8080))
        sock.close()
        if result == 0:
            print("Server running")
            return
    except:
        pass
    
    subprocess.Popen(
        [sys.executable, "-m", "http.server", "8080"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    print("Server started")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    print("="*70)
    print("Advanced E-Commerce Agent Evaluation")
    print("="*70 + "\n")

    start_http_server()

    print(f"?? Loading model...")
    model = PPO.load(args.model_path)
    print("? Model loaded\n")

    action_names = [
        "Type Username", "Type Password", "Click Login",
        "Add Laptop", "Add Headphones", "Add Mouse", "Add Keyboard",
        "Remove Item", "Proceed to Checkout",
        "Type Card Name", "Type Card Number", "Type Expiry", "Type CVV",
        "Type Address", "Type City", "Type ZIP", "Complete Payment"
    ]

    successes = 0
    total_rewards = []

    for ep in range(1, args.episodes + 1):
        print(f"\n{'='*70}")
        print(f"Episode {ep}/{args.episodes}")
        print('='*70)

        env = EcommerceEnv(headless=not args.render, max_steps=100)
        obs, info = env.reset()
        done = trunc = False
        total_reward = 0
        steps = 0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(int(action))
            total_reward += reward
            steps += 1

            print(f"  Step {steps:3d}: {action_names[action]:20s} | "
                  f"R: {reward:+7.2f} | Page: {info['current_page']:10s} | "
                  f"Cart: {info['items_in_cart']}")

        total_rewards.append(total_reward)
        if info['payment_completed']:
            successes += 1

        print(f"\n?? Episode Results:")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Logged In: {'?' if info['is_logged_in'] else '?'}")
        print(f"   Items in Cart: {info['items_in_cart']}")
        print(f"   Reached Checkout: {'?' if info['current_page'] == 'checkout' else '?'}")
        print(f"   Payment Complete: {'?' if info['payment_completed'] else '?'}")
        print(f"   Steps: {steps}")

        env.close()
        if ep < args.episodes:
            time.sleep(2)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Success Rate: {successes}/{args.episodes} ({successes/args.episodes*100:.1f}%)")
    print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Best Reward: {max(total_rewards):.2f}")


if __name__ == "__main__":
    main()