from .base_trainer import BaseTrainer
from bots import RARSMSBot
import torch
import torch.optim as optim
import sys

class RARSMSBotTrainer(BaseTrainer):
    def __init__(self, env, savedir, cuda, save_interval,  num_actor_devices,num_actors, training_device, 
                 learning_rate=0.001, batch_size=32, num_episodes=10000):
        super().__init__(env, savedir)
        self.learning_rate = learning_rate
        # self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.cuda = cuda
        self.training_device = training_device

        # Hyperparameters
        self.gamma = 0.001
        self.lmda = 0.001 
        self.epsilon = 0.1 
        self.beta = 0.5  # Weights cooperation of peasants (0 no cooperation -> 1 cooperation)

        # Other required variables
        # Prior rewards
        self.r_landlord_prev = 0
        self.r_peasants_prev = 0

        self.train()


    def train(self):
        # Set CUDA device
        device = torch.device(f"cuda:{self.training_device}" if torch.cuda.is_available() else "cpu")
        
        # Create bot instance 
        bot = RARSMSBot()
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
            print("EPISODE {episode}/{self.num_episodes} ACTOR LOSS: {actor_loss}")
            print("EPISODE {episode}/{self.num_episodes} CRITIC LOSS: {critic_loss}")

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
                
    

    def _calculate_intrinsic_reward(self, state, environment_reward:float, player_id:int, ):
        """
        Computes intrinsic reward based on progress in minimizing splits and reducing cards.
        """

        if player_id == 0:  # Landlord
            k = 1
        else:  # Peasant
            k = -1/2

        L, N = self._calculate_minimum_splits(0, True)
        Lt, Ct = self._calculate_minimum_splits(0, False)
        r_landlord = (L -Lt + N - Ct) /  (L + N)

        L, N = self._calculate_minimum_splits(1, True)
        Lt, Ct = self._calculate_minimum_splits(1, False)
        r_peasant_down = (L -Lt + N - Ct) /  (L + N)

        L, N = self._calculate_minimum_splits(2, True)
        Lt, Ct = self._calculate_minimum_splits(2, False)
        r_peasant_up = (L -Lt + N - Ct) /  (L + N)

        r_peasants = max(r_peasant_down + self.beta*r_peasant_up, r_peasant_up + self.beta*r_peasant_down)
        r = torch.clamp(r_landlord - r_peasants - (self.r_landlord_prev - self.r_peasants_prev), -1, 1) * 2 * environment_reward * k

        self.r_landlord_prev = r_landlord
        self.r_peasants_prev = r_peasants



        return r
    

    def _calculate_minimum_splits(self, player_id: int, initial_hand:bool):
        """
        Returns a tuple of the minimum split and the number of cards in the hand for current
        state. If initial_hand is true, then returns the minimum split, and the number of 
        cards in the hand for the players first hand at the beginning of the game.
        """

        RANK_ORDER = '3456789TJQKA2BR'  # 15 ranks
        RANK_INDEX = {rank: i for i, rank in enumerate(RANK_ORDER)}

        state = self.env.get_state(player_id)
        if initial_hand:
            hand = state['initial_hand']
        else:
            hand = state['current_hand']

        # Convert current_hand string to X vector (counts per rank)
        X = [0] * 15
        for card in hand:
            X[RANK_INDEX[card]] += 1

        memo = {}

        # Recursive function G
        def G(X):
            # Check for memoization (optional for speed)
            key = tuple(X)
            if key in memo:
                return memo[key]

            # Base case: no cards left
            if sum(X) == 0:
                return 0

            L = sys.maxsize

            # Try all chains and combos
            for p in range(12):  # up to 'A'
                for m in range(1, 4):  # only consider m=1,2,3 for chain
                    if X[p] < m:
                        continue
                    y = p
                    for f in range(p + 1, 12):
                        if X[f] >= m:
                            y = f
                        else:
                            break
                    for w in range(p + 4, y + 1):  # Chain-Solo (>=5)
                        if m == 1:
                            new_X = X[:]
                            valid = True
                            for i in range(p, w + 1):
                                if new_X[i] < 1:
                                    valid = False
                                    break
                                new_X[i] -= 1
                            if valid:
                                L = min(L, G(new_X) + 1)
                    for w in range(p + 2, y + 1):  # Chain-Pair (>=3)
                        if m == 2:
                            new_X = X[:]
                            valid = True
                            for i in range(p, w + 1):
                                if new_X[i] < 2:
                                    valid = False
                                    break
                                new_X[i] -= 2
                            if valid:
                                L = min(L, G(new_X) + 1)
                    for w in range(p + 1, y + 1):  # Plane (>=2)
                        if m == 3:
                            new_X = X[:]
                            valid = True
                            for i in range(p, w + 1):
                                if new_X[i] < 3:
                                    valid = False
                                    break
                                new_X[i] -= 3
                            if valid:
                                # Can optionally add solos/pairs for Plane-Solo or Plane-Pair
                                L = min(L, G(new_X) + 1)

            # Count Bombs, Trios, Pairs, Solos
            b, k, j, q = 0, 0, 0, 0
            for count in X:
                if count == 4:
                    b += 1
                elif count == 3:
                    k += 1
                elif count == 2:
                    j += 1
                elif count == 1:
                    q += 1

            # Add minimal combination of remaining cards
            L = min(L, b + k + j + q)

            memo[key] = L
            return L

        
        return G(X), len(hand)


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


