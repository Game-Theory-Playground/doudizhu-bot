''' An example of training a Deep Monte-Carlo (DMC) Agent on the environments in RLCard.
Adapted from: https://github.com/datamllab/rlcard/blob/master/examples/run_dmc.py
'''
import os
import argparse

import torch

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer

def train(args):

    # Make the environment
    env = rlcard.make('doudizhu')
    xpid = 'doudizhu'


    # Initialize the DMC trainer
    trainer = DMCTrainer(
        env,
        cuda=args.cuda,
        xpid=xpid,
        savedir=args.savedir,
        save_interval=args.save_interval,
        num_actor_devices=args.num_actor_devices,
        num_actors=args.num_actors,
        training_device=args.training_device,
    )

    # Train DMC Agents
    trainer.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DMC example in RLCard")
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
        help='Cuda GPU numbers to run on. Ex/ `1,2,4`'
    )
    parser.add_argument(
        '--savedir',
        default='results/',
        help='Root dir where experiment data will be saved'
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

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)