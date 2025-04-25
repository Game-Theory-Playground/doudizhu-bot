import os
import torch

class BaseTrainer:
    """Base trainer interface for all training algorithms"""
    
    def __init__(self, env, savedir):
        self.env = env
        self.savedir = savedir
    
    def train(self):
        """Execute the training process"""
        raise NotImplementedError
    
    def _save_model(self, player_id, network, episode):
            os.makedirs(self.savedir, exist_ok=True) 
            path = os.path.join(self.savedir, f"{player_id}_{episode}.pth")
            print("Saved model to", path)

            torch.save(
                network.state_dict(),
                path
            )