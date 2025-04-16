
# irl/expert_dataset.py

import random
import pickle
import torch

class ExpertDataset:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        # Assume data format: List of (torch.Tensor state, int action)

    def sample(self, batch_size):
        return random.sample(self.data, batch_size)