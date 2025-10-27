# train_tetris_hybrid.py
import argparse
import os
import signal
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from tetris_env import TetrisEnv

current_model = None
current_save_path = None

def signal_handler(sig, frame):
    print('\n\n??  Training interrupted!')
    if current_model is not None and current_save_path is not None:
        print(f'?? Saving model to {current_save_path}...')
        current_model.save(current_save_path)
        print('? Model saved!')
    sys.exit(0)

def make_env(reward_mode="hybrid", seed=7, drop_speed=30):
    def _init():
        env = TetrisEnv(reward_mode=reward_mode, seed=seed, drop_speed=drop_speed)
        env = Monitor(env)
        return env
    return _init

def main():
    global current_model, current_save_path
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--reward_mode", type=str, default="hybrid",
                       choices=["standard", "aggressive", "shaped", "hybrid"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modeldir", type=str, default="./models")
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--drop_speed", type=int, default=35)
    parser.add_argument("--ent_coef", type=float, default=0.02,
                       help="Entropy coefficient for exploration (higher=more exploration)")
    args = parser.parse_args()

    os.makedirs(args.modeldir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.join(args.modeldir, "checkpoints"), exist_ok=True)

    signal.signal(signal.SIGINT, signal_handler)

    env = DummyVecEnv([make_env(reward_mode=args.reward_mode, seed=args.seed, 
                                drop_speed=args.drop_speed)])

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(args.modeldir, "checkpoints"),
        name_prefix=f"tetris_{args.reward_mode}",
        verbose=1
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=args.ent_coef,  # Encourages trying different actions
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
    )

    current_model = model
    current_save_path = os.path.join(args.modeldir, f"interrupted_{args.reward_mode}")

    print(f"\n?? Training Tetris - {args.reward_mode.upper()} mode")
    print(f"??  Timesteps: {args.timesteps:,}")
    print(f"?? Entropy coef: {args.ent_coef} (exploration)")
    print(f"? Drop speed: {args.drop_speed}")
    print(f"?? Checkpoints every 50k steps\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=True,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        signal_handler(None, None)

    save_name = f"ppo_tetris_{args.reward_mode}_final"
    path = os.path.join(args.modeldir, save_name)
    model.save(path)
    print(f"\n? Training complete! Saved to {path}")

    env.close()

if __name__ == "__main__":
    main()