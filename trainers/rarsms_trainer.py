from .base_trainer import BaseTrainer
from bots import RARSMSBot
import torch
import torch.optim as optim

class RARSMSBotTrainer(BaseTrainer):
    def __init__(self, env, savedir, learning_rate=0.001, batch_size=32,
                 num_episodes=10000, cuda='', training_device='0'):
        super().__init__(env, savedir)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.cuda = cuda
        self.training_device = training_device
        
    def train(self):
        # Set CUDA device
        device = torch.device(f"cuda:{self.training_device}" if torch.cuda.is_available() else "cpu")
        
        # Create bot instance
        
        bot = RARSMSBot()
        bot.build_networks()
        bot.set_device(device)
        
        # Create optimizers
        actor_optimizer = optim.Adam(bot.actor_network.parameters(), lr=self.learning_rate)
        critic_optimizer = optim.Adam(bot.critic_network.parameters(), lr=self.learning_rate)
        
        # Training loop
        for episode in range(self.num_episodes):
            # Reset environment
            state, player_id = self.env.reset()
            
            # Episode loop
            while not self.env.is_over():
                # Choose action
                action = bot.act(state)
                
                # Take action
                # next_state, reward, done, _ = self.env.step(action)
                next_state, next_player_id = self.env.step(action)

                # Give rewards
                # Environment reward only given at end of game (usually 0)
                environment_reward = self.env.get_payoffs()[player_id] if self.env.is_over() else 0
                intrinsic_reward = 


                
                # Store transition
                # ...
                
                # Update networks
                # ...
                
                state = next_state
                
            # Save model periodically
            if episode % 100 == 0:
                self._save_model(bot, episode)
                
            # Log progress
            # ...