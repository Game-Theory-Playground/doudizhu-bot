import os
import argparse

parser = argparse.ArgumentParser("Multi-bot evaluation for Doudizhu")
parser.add_argument('--cuda', type=str, default='')
args, _ = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import rlcard
from rlcard.utils import get_device, set_seed, tournament
from bots import DMCBot, RARSMSBot

def evaluate(args):
    device = get_device()
    set_seed(args.seed)

    env = rlcard.make('doudizhu', config={'seed': args.seed})

    agents = []
    bot_labels = []
    for position, model_path in enumerate(args.bots):
        parts = model_path.split(':')
        bot_type = parts[0]

        if bot_type == 'dmc':
            if len(parts) < 2:
                raise ValueError("DMC bot needs a model path")
            model_file = parts[1]
            agent = DMCBot(model_file, position, device=device)
            bot_labels.append(f'dmc({os.path.basename(model_file)})')
        elif bot_type == 'random':
            from rlcard.agents import RandomAgent
            agent = RandomAgent(num_actions=env.num_actions)
            bot_labels.append('random')
        elif bot_type == 'rarsms':
            if len(parts) < 3:
                raise ValueError("rarsms bot needs both douzerox_path and actor_path. Format: rarsms:douzerox_path:actor_path")
            douzerox_path = parts[1]
            actor_path = parts[2]
            agent = RARSMSBot(douzerox_path=douzerox_path, actor_path=actor_path, position=position)
            bot_labels.append(f'rarsms({os.path.basename(actor_path)})')
        else:
            raise ValueError(f"Unknown bot type: {bot_type}")

        if hasattr(agent, 'set_device'):
            agent.set_device(device)

        print(f"[INFO] Loaded agent for position {position}: {bot_labels[-1]} (device: {device})")
        agents.append(agent)

    env.set_agents(agents)

    print(f"\n[INFO] Starting evaluation for {args.num_games} games...\n")
    rewards = tournament(env, args.num_games)

    print("\n[RESULTS]")
    for position, reward in enumerate(rewards):
        print(f"Position {position} ({bot_labels[position]}): {reward:.4f}")

if __name__ == '__main__':
    parser.add_argument('--bots', nargs='*', default=[
        'rarsms:trained_models/douzerox/0_0.pth:results/0_9990.pth',
        'dmc:trained_models/douzerox/1_0.pth',
        'dmc:trained_models/douzerox/2_0.pth'
    ], help='Bot specifications. Format depends on bot type:\n'
             'random\n'
             'dmc:model_path\n'
             'rarsms:douzerox_path:actor_path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_games', type=int, default=1000)

    args = parser.parse_args()
    evaluate(args)