import torch
from .base_bot import BaseBot

class DMCBot(BaseBot):
    """Adapter for Deep Monte-Carlo bot"""

    def __init__(self, model_path, position=None, device=None):
        super().__init__(position)
        self.model_path = model_path
        self.device = device or torch.device('cpu')
        self.agent = None
        self._load_model()

        if hasattr(self.agent, 'use_raw'):
            self.use_raw = self.agent.use_raw

    def _load_model(self):
        from rlcard.agents.dmc_agent.model import DMCAgent
        from torch.serialization import add_safe_globals
        add_safe_globals([DMCAgent])

        obj = torch.load(self.model_path, map_location=self.device, weights_only=False)
        print(f"[DMCBot] Loaded model from {self.model_path} ({type(obj)})")

        self.agent = obj
        if hasattr(self.agent, 'to'):
            self.agent.to(self.device)
        if hasattr(self.agent, 'eval'):
            self.agent.eval()

    def act(self, state):
        action = self.agent.step(state)
        return action

    def eval_step(self, state):
        output = self.agent.eval_step(state)
    
        if isinstance(output, tuple):
            action, info = output
        else:
            action = output
            info = {}

        return int(action), info

    def set_device(self, device):
        self.device = device
        if self.agent and hasattr(self.agent, 'to'):
            self.agent.to(device)
