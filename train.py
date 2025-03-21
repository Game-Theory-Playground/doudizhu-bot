import argparse
import rlcard
from trainers import DMCBotTrainer
# Import other trainers as needed

def main():
    parser = argparse.ArgumentParser("Bot training for Doudizhu")
    parser.add_argument(
        'algorithm',
        choices=['dmc'],
        help='Bot algorithm to train'
    )
    
    # Common parameters
    parser.add_argument(
        '--savedir',
        default='results/',
        help='Root dir where experiment data will be saved'
    )
    
    # DMC-specific parameters
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
        help='Cuda GPU numbers to run on. Ex/ `1,2,4`'
    )
    parser.add_argument(
        '--save_interval',
        default=30,
        type=int,
        help='Time interval (in minutes) at which to save the model',
    )
    parser.add_argument(
        '--num_actor_devices',
        default=1,
        type=int,
        help='The number of devices used for simulation',
    )
    parser.add_argument(
        '--num_actors',
        default=5,
        type=int,
        help='The number of actors for each simulation device',
    )
    parser.add_argument(
        '--training_device',
        default="0",
        type=str,
        help='The index of the GPU used for training models',
    )
    
    args = parser.parse_args()
    
    # Create environment
    env = rlcard.make('doudizhu')
    
    # Choose the appropriate trainer
    if args.algorithm == 'dmc':
        trainer = DMCBotTrainer(
            env, 
            args.savedir,
            cuda=args.cuda,
            save_interval=args.save_interval,
            num_actor_devices=args.num_actor_devices,
            num_actors=args.num_actors,
            training_device=args.training_device
        )
        trainer.train()
    else:
        print("Need more implementation")
    # Add more algorithms as needed
    
if __name__ == '__main__':
    main()