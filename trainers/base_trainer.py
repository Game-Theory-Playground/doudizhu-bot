class BaseTrainer:
    """Base trainer interface for all training algorithms"""
    
    def __init__(self, env, savedir):
        self.env = env
        self.savedir = savedir
    
    def train(self):
        """Execute the training process"""
        raise NotImplementedError