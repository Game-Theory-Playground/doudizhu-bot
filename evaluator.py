import os
import argparse

import rlcard
from rlcard.utils import get_device, set_seed, tournament
from bots import DMCBot

def evaluate(args):

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make('doudizhu', config={'seed': args.seed})

    # Load different bot types 
    agents = []
    for position, model_path in enumerate(args.bots):
        parts = model_path.split(':')
        bot_type = parts[0]
        
        if bot_type == 'dmc':
            if len(parts) < 2:
                raise ValueError("DMC bot needs a model path")
            model_path = parts[1]
            agent = DMCBot(model_path, position)
        elif bot_type == 'random':
            from rlcard.agents import RandomAgent
            agent = RandomAgent(num_actions=env.num_actions)
        else:
            raise ValueError(f"Unknown bot type: {bot_type}")
            
        if hasattr(agent, 'set_device'):
            agent.set_device(device)
        agents.append(agent)
    
    env.set_agents(agents)

    # Evaluate
    rewards = tournament(env, args.num_games)
    for position, reward in enumerate(rewards):
        print(f"Position {position} ({args.bots[position]}): {reward}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Multi-bot evaluation for Doudizhu")
    
    parser.add_argument(
        '--bots',
        nargs='*',
        default=[
            'dmc:results/doudizhu/0_0.pth',
            'random',
            'random'
        ],
        help='Specify bots as type:model_path (model_path only needed for certain types)'
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)
