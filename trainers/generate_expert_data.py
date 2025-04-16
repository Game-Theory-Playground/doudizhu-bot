
# generate_expert_data.py

import torch
import pickle
from bots import DMCBot
from rlcard.envs.doudizhu import DoudizhuEnv


def generate_expert_trajectories(model_path, output_path, num_games=1000, device='cpu'):
    env = DoudizhuEnv()
    agent = DMCBot(model_path, position=0, device=device)
    data = []

    for _ in range(num_games):
        state = env.reset()[0]['obs']
        done = False

        while not done:
            action = agent.act(state)
            data.append((torch.tensor(state), action))
            state, _, done, _ = env.step(action)
            state = state[0]['obs']

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Expert data saved to {output_path} with {len(data)} (state, action) pairs.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to DMCBot .pth model')
    parser.add_argument('--output_path', type=str, default='expert.pkl', help='Output file path for expert data')
    parser.add_argument('--num_games', type=int, default=1000, help='Number of expert games to generate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the bot')
    args = parser.parse_args()

    generate_expert_trajectories(
        model_path=args.model_path,
        output_path=args.output_path,
        num_games=args.num_games,
        device=args.device
    )
