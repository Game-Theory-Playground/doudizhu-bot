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

        # Hyperparameters
        self.gamma = 0.001
        self.lmda = 0.001 

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
        
        
        # Each Game (Episode)
        for episode in range(self.num_episodes):
            # Reset environment
            state, player_id = self.env.reset()

            temporal_difference_errors = []

            # Each round (frame)
            while not self.env.is_over():
                # Choose action
                action = bot.act(state)
                
                next_state, _ = self.env.step(action)
                
                reward = self._calculate_intrinsic_reward()
        
                temporal_difference_error = reward + self.gamma * self.bot.predict_state(state) - self.bot.predict_state(next_state)
                temporal_difference_errors.append(temporal_difference_error)
                state = next_state
            

            advantage_function = self._calculate_advantage(temporal_difference_errors)

            actor_loss = self._calculate_critic_loss(self, advantage_function)
            critic_loss = self._calculate_critic_loss(self)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
                
            # Save model periodically
            if episode % 100 == 0:
                self._save_model(bot, episode)
                
            # Log progress
            # ...


        def _calculate_intrinsic_reward(self):
            """
            Some of this may need to be moved to rarsms.
            This will also need more params
            This will need to:
            - Calculate minimum splits for each player
            - Some more stuff
            """

            return 0
        

        def _calculate_advantage_function(self, temporal_difference_errors):
            T = len(temporal_difference_errors)
            advantage_function = []
            for t in range(T):
                advtg = 0
                for i in range(t, T):
                    advtg += (self.gamma * self.lmda) ** (i-t) * temporal_difference_errors[i]
                advantage_function.append(advtg)

            return advantage_function         
                 

        def _calculate_actor_loss(self, advantage_function):
            """ 
                PPO Clipped Objective Function:
                This will also need more params
            """

            # TODO
        def _calculate_critic_loss(self):
            """ 
                Objective function. Not sure what loss function this is.
                This will also need more params
            """






