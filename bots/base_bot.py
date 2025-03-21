class BaseBot:
    """Base interface for all Doudizhu bots"""
    
    def __init__(self, position=None):
        self.position = position
        self.device = 'gpu'
        self.use_raw = False 
    
    def act(self, state):
        """Take an action based on the current state"""
        raise NotImplementedError
    
    def eval_step(self, state):
        """Evaluation version of step (often the same as act)"""
        return self.act(state)
        
    def set_device(self, device):
        """Set the device for model computation"""
        self.device = device