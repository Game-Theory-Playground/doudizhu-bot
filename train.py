import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='')
args, _ = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import rlcard
from trainers.dmc_trainer import DMCBotTrainer
from trainers.rarsms_trainer import RARSMSBotTrainer

def main():
    parser = argparse.ArgumentParser("Bot training for Doudizhu")
    parser.add_argument('--algorithm', choices=['dmc', 'rarsms']) 
    
    # Update to have separate paths for each role
    parser.add_argument('--landlord_path', help='Path to DouZeroX weights for the landlord')
    parser.add_argument('--peasant_up_path', help='Path to DouZeroX weights for the peasant after landlord')
    parser.add_argument('--peasant_down_path', help='Path to DouZeroX weights for the peasant before landlord')
    
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
    elif args.algorithm == 'rarsms':
        # Collect all three paths in a list
        douzerox_paths = [
            args.landlord_path,
            args.peasant_up_path,
            args.peasant_down_path
        ]
        
        # Check if all required paths are provided
        if None in douzerox_paths:
            # If any path is missing, raise an error
            parser.error("When using 'rarsms' algorithm, you must provide all three paths: "
                         "--landlord_path, --peasant_up_path, and --peasant_down_path")
        
        trainer = RARSMSBotTrainer(
            env=env,
            douzerox_paths=douzerox_paths,  # Pass the list of paths
            savedir=args.savedir,
            cuda=args.cuda,
            save_interval=args.save_interval,
            num_actor_devices=args.num_actor_devices,
            num_actors=args.num_actors,
            training_device=args.training_device,
        )
        trainer.train()

if __name__ == '__main__':
    main()