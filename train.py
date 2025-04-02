import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='')
args, _ = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import rlcard
from trainers import DMCBotTrainer

def main():
    parser = argparse.ArgumentParser("Bot training for Doudizhu")
    parser.add_argument('algorithm', choices=['dmc'])
    parser.add_argument('--savedir', default='results/')
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--save_interval', type=int, default=30)
    parser.add_argument('--num_actor_devices', type=int, default=1)
    parser.add_argument('--num_actors', type=int, default=5)
    parser.add_argument('--training_device', type=str, default='0')
    args = parser.parse_args()

    env = rlcard.make('doudizhu')

    if args.algorithm == 'dmc':
        trainer = DMCBotTrainer(
            env,
            args.savedir,
            cuda=args.cuda,
            save_interval=args.save_interval,
            num_actor_devices=args.num_actor_devices,
            num_actors=args.num_actors,
            training_device=args.training_device,
        )
        trainer.train()

if __name__ == '__main__':
    main()
