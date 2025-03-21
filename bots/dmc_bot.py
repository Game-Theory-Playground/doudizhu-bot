# bots/dmc_bot.py
import torch
from .base_bot import BaseBot

class DMCBot(BaseBot):
    """Adapter for Deep Monte-Carlo bot"""
    
    def __init__(self, model_path, position=None):
        super().__init__(position)
        self.model_path = model_path
        self.agent = None
        self._load_model()
        
        if hasattr(self.agent, 'use_raw'):
            self.use_raw = self.agent.use_raw
    
    def _load_model(self):
        self.agent = torch.load(self.model_path, map_location=self.device, weights_only=False)
    
    def act(self, state):
        return self.agent.step(state)
    
    def eval_step(self, state):
        return self.agent.eval_step(state)
    
    def set_device(self, device):
        self.device = device
        if self.agent:
            self.agent.set_device(device)