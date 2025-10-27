import argparse
from stable_baselines3 import PPO
from tetris_env import TetrisEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/ppo_tetris_standard")
    parser.add_argument("--reward_mode", type=str, default="standard")
    args = parser.parse_args()

    model = PPO.load(args.model_path)

    env = TetrisEnv( reward_mode=args.reward_mode)
    obs, info = env.reset()
    done, trunc = False, False

    print("Playing Tetris with trained model...")
    print("Close the window to exit.")

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(int(action))
    
    print(f"\nGame Over!")
    print(f"Lines cleared: {info['lines_cleared']}")
    print(f"Pieces placed: {info['pieces_placed']}")
    print(f"Score: {info['score']}")
    
    env.close()


if __name__ == "__main__":
    main()