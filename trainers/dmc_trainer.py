import os
from rlcard.agents.dmc_agent import DMCTrainer
from .base_trainer import BaseTrainer

class DMCBotTrainer(BaseTrainer):
    def __init__(self, env, savedir, cuda='', save_interval=30, 
                 num_actor_devices=1, num_actors=5, training_device='0'):
        super().__init__(env, savedir)
        self.cuda = cuda
        self.save_interval = save_interval
        self.num_actor_devices = num_actor_devices
        self.num_actors = num_actors
        self.training_device = training_device
    
    def train(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda
        xpid = 'doudizhu'
        
        # Initialize the DMC trainer
        trainer = DMCTrainer(
            self.env,
            cuda=self.cuda,
            xpid=xpid,
            savedir=self.savedir,
            save_interval=self.save_interval,
            num_actor_devices=self.num_actor_devices,
            num_actors=self.num_actors,
            training_device=self.training_device,
        )
        
        # Train DMC Agents
        trainer.start()