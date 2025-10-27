import argparse
import os
import subprocess
import time
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ecommerce_env import EcommerceEnv


def start_http_server():
    """Start HTTP server if not running."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8080))
        sock.close()
        
        if result == 0:
            print("HTTP server already running")
            return
    except:
        pass
    
    subprocess.Popen(
        [sys.executable, "-m", "http.server", "8080"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    print("HTTP server started at http://localhost:8080")


def make_env(headless=True):
    def _init():
        env = EcommerceEnv(
            app_url="http://localhost:8080",
            headless=headless,
            max_steps=100,
            target_items=2
        )
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent for advanced e-commerce with login and payment"
    )
    parser.add_argument("--timesteps", type=int, default=100_000,
                       help="Total training timesteps")
    parser.add_argument("--headless", action="store_true",
                       help="Run browser in headless mode")
    parser.add_argument("--modeldir", type=str, default="./models",
                       help="Directory to save models")
    parser.add_argument("--logdir", type=str, default="./logs",
                       help="Directory for logs")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=float, default=0.05,
                       help="Entropy coefficient for exploration")
    args = parser.parse_args()

    print("="*70)
    print("Advanced E-Commerce RL Training (Login + Shopping + Payment)")
    print("="*70 + "\n")

    os.makedirs(args.modeldir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    # Start HTTP server
    print("?? Starting HTTP server...")
    start_http_server()

    print("?? Creating environment...")
    print(f"   Headless: {args.headless}")
    print(f"   Max steps: 100")
    print(f"   Actions: 17 (login, shop, pay)\n")
    
    env = DummyVecEnv([make_env(headless=args.headless)])
    eval_env = DummyVecEnv([make_env(headless=True)])

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5_000,
        save_path=os.path.join(args.modeldir, "checkpoints"),
        name_prefix="advanced_ecommerce",
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.modeldir, "best_model"),
        log_path=args.logdir,
        eval_freq=5_000,
        deterministic=True,
        render=False,
        n_eval_episodes=3,
        verbose=1
    )

    print("Training PPO agent...")
    print(f"   Timesteps: {args.timesteps:,}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Entropy coef: {args.ent_coef} (exploration)\n")
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.logdir,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        clip_range=0.2,
        n_epochs=10,
        policy_kwargs=dict(net_arch=[256, 256])
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=True,
            callback=[checkpoint_callback, eval_callback]
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")

    save_path = os.path.join(args.modeldir, "ppo_advanced_ecommerce_final")
    model.save(save_path)
    print(f"\n? Training complete! Model saved to {save_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()