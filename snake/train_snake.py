import argparse
import os
import signal
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from snake_env import SnakeEnv

current_model = None
current_save_path = None

def signal_handler(sig, frame):
    print('\n\n??  Training interrupted!')
    if current_model is not None and current_save_path is not None:
        print(f'?? Saving model to {current_save_path}...')
        current_model.save(current_save_path)
        print('? Model saved!')
    sys.exit(0)

def make_env(width=20, height=15, seed=None):
    def _init():
        env = SnakeEnv(width=width, height=height)
        env = Monitor(env)
        return env
    return _init

def main():
    global current_model, current_save_path
    
    parser = argparse.ArgumentParser(description='Train PPO agent to play Snake')
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                       help="Total training timesteps")
    parser.add_argument("--width", type=int, default=20,
                       help="Game grid width")
    parser.add_argument("--height", type=int, default=15,
                       help="Game grid height")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--modeldir", type=str, default="./models",
                       help="Directory to save models")
    parser.add_argument("--logdir", type=str, default="./logs",
                       help="Directory for tensorboard logs")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                       help="Entropy coefficient for exploration (higher=more exploration)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--checkpoint_freq", type=int, default=100_000,
                       help="Save checkpoint every N steps")
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.modeldir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.join(args.modeldir, "checkpoints"), exist_ok=True)

    # Register interrupt handler
    signal.signal(signal.SIGINT, signal_handler)

    # Create vectorized environment
    env = DummyVecEnv([make_env(width=args.width, height=args.height, 
                                seed=args.seed)])

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(args.modeldir, "checkpoints"),
        name_prefix="snake_ppo",
        verbose=1
    )

    # Initialize PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
        n_steps=2048,           # Steps per environment per update
        batch_size=64,          # Minibatch size
        gamma=0.99,             # Discount factor
        gae_lambda=0.95,        # GAE lambda
        n_epochs=10,            # Epochs per update
        learning_rate=args.learning_rate,
        clip_range=0.2,         # PPO clip range
        ent_coef=args.ent_coef, # Entropy coefficient (exploration)
        vf_coef=0.5,            # Value function coefficient
        max_grad_norm=0.5,      # Gradient clipping
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])  # Neural network architecture
        )
    )

    current_model = model
    current_save_path = os.path.join(args.modeldir, "interrupted_snake")

    # Print training configuration
    print("\n" + "="*60)
    print("?? SNAKE RL TRAINING")
    print("="*60)
    print(f"Grid size:        {args.width}x{args.height}")
    print(f"Total timesteps:  {args.timesteps:,}")
    print(f"Learning rate:    {args.learning_rate}")
    print(f"Entropy coef:     {args.ent_coef} (exploration)")
    print(f"Checkpoint freq:  {args.checkpoint_freq:,} steps")
    print(f"Model dir:        {args.modeldir}")
    print(f"Log dir:          {args.logdir}")
    print("="*60 + "\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=True,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        signal_handler(None, None)

    # Save final model
    save_name = "ppo_snake_final"
    path = os.path.join(args.modeldir, save_name)
    model.save(path)
    print(f"\n? Training complete! Final model saved to {path}")

    env.close()

if __name__ == "__main__":
    main()