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
        self.beta = 0.01

        # Other required variables
        # Prior rewards
        self.r_landlord_prev = 0
        self.r_peasants_prev = 0


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

        # Each Game (Episode)
        for episode in range(self.num_episodes):
            curr_states = []
            curr_actions = []
            curr_probs = []
            curr_rewards = []
            self.r_landlord_prev = 0
            self.r_peasants_prev = 0
            
        
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
                
                next_state, player_id = self.env.step(action)
                
                environment_reward = self.env.get_payoffs()[player_id] if self.env.is_over() else 0
                reward = self._calculate_intrinsic_reward(state, environment_reward, player_id)
                curr_rewards.append(reward)
        
                temporal_difference_error = reward + self.gamma * bot.predict_state(state) - bot.predict_state(next_state)
                temporal_difference_errors.append(temporal_difference_error)
                state = next_state
            

            advantage_function = self._calculate_advantage(temporal_difference_errors)

            actor_loss = self._calculate_actor_loss(bot, advantage_function, old_states, old_actions, old_probs)
            critic_loss = self._calculate_critic_loss(bot, curr_states, curr_rewards)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            old_states = curr_states
            old_actions = curr_actions
            old_probs = curr_probs

            
                
            # Save model periodically
            if episode % 100 == 0:
                self._save_model(bot, episode)
                
    

        def _calculate_intrinsic_reward(self, state, environment_reward, player_id):
            """
            Some of this may need to be moved to rarsms.
            This will also need more params
            This will need to:
            - Calculate minimum splits for each player
            - Some more stuff
            """

            min_split = self._calculate_minimum_splits(state)

            if player_id == 0:  # Landlord
                k = 1
            else:  # Peasant
                k = -1/2

            r_landlord = 0
            r_peasant_down = 0
            r_peasant_up = 0

            r_peasants = max(r_peasant_down + self.beta*r_peasant_up, r_peasant_up + self.beta*r_peasant_down)
            r = torch.clamp(r_landlord - r_peasants - (self.r_landlord_prev - self.r_peasants_prev), -1, 1) * 2 * environment_reward * k

            self.r_landlord_prev = r_landlord
            self.r_peasants_prev = r_peasants



            return 0
        

        def _calculate_minimum_splits(self):
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


        def _calculate_critic_loss(self, bot, states, rewards):
            """ 
            Objective function using MSE.
            """

            states = torch.tensor(states, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)

            predicted_rewards = []
            for state in states:
                predicted_rewards.append(bot.critic_network(state))
            predicted_rewards = torch.tensor(predicted_rewards, dtype=torch.float32)


            loss = torch.mean((predicted_rewards - rewards) ** 2)

            return loss





