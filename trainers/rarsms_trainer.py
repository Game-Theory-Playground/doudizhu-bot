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
        self.epsilon = 0.1 

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
        
        old_states = []
        old_actions = []
        old_probs = []

        curr_states = []
        curr_actions = []
        curr_probs = []
        # Each Game (Episode)
        for episode in range(self.num_episodes):
            # Reset environment
            state, player_id = self.env.reset()

            temporal_difference_errors = []

            # Each round (frame)
            while not self.env.is_over():
                # Choose action
                action, prob = bot.act(state)
                
                curr_states.append(state)
                curr_actions.append(action)
                curr_probs.append(prob)
                
                next_state, _ = self.env.step(action)
                
                reward = self._calculate_intrinsic_reward()
        
                temporal_difference_error = reward + self.gamma * bot.predict_state(state) - bot.predict_state(next_state)
                temporal_difference_errors.append(temporal_difference_error)
                state = next_state
            

            advantage_function = self._calculate_advantage(temporal_difference_errors)

            actor_loss = self._calculate_actor_loss(bot, advantage_function, old_states, old_actions, old_probs)
            critic_loss = self._calculate_critic_loss()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            old_states = curr_states
            old_actions = curr_actions
            old_probs = curr_probs
            curr_states = []
            curr_actions = []
            curr_probs = []
            
                
            # Save model periodically
            if episode % 100 == 0:
                self._save_model(bot, episode)
                
    

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
                 

        def _calculate_actor_loss(self, bot, advantage_function, old_states, old_actions, old_probs):
            """ 
            PPO Clipped Objective Function.
            TODO may need logprob instead of just prob
            """
            
            old_states = torch.tensor(old_states, dtype=torch.float32)
            old_actions = torch.tensor(old_actions, dtype=torch.long)
            old_probs = torch.tensor(old_probs, dtype=torch.float32)
            advantages = torch.tensor(advantage_function, dtype=torch.float32)

            probs = []

            for state, action in zip(old_states, old_actions):
                prob = bot.get_log_prob(state, action)
                probs.append(prob)
            probs = torch.tensor(probs, dtype=torch.float32)

            ratios = torch.exp(probs - old_probs)
            clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)

            actor_loss = -torch.mean(torch.min(ratios * advantages, clipped_ratios * advantages))

            return actor_loss


        def _calculate_critic_loss(self):
            """ 
                Objective function. Not sure what loss function this is.
                This will also need more params
            """






