from .base_trainer import BaseTrainer
from bots import RARSMSBot
import torch
import torch.optim as optim
import sys
import copy


class RARSMSBotTrainer(BaseTrainer):
    def __init__(
        self,
        env,
        douzerox_paths,  # Now accepts a list of three paths
        savedir,
        cuda,
        save_interval=30,  # The number of episodes to save after
        num_actor_devices = None,
        num_actors = None,
        training_device = None,
        learning_rate=0.001,
        batch_size=32,
        num_episodes=10000,
    ):
        super().__init__(env, savedir)

        # Unpack paths for each role
        self.landlord_path = douzerox_paths[0]
        self.peasant_up_path = douzerox_paths[1]    # Peasant after landlord (landlord-up)
        self.peasant_down_path = douzerox_paths[2]  # Peasant before landlord
        
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.cuda = cuda
        self.training_device = training_device
        self.save_interval = save_interval
        self.device = None

        # Hyperparameters
        self.gamma = 0.001
        self.lmda = 0.001
        self.epsilon = 0.1
        self.beta = 0.5  # Weights cooperation of peasants (0 no cooperation -> 1 cooperation)

        # Prior rewards for each role
        self.r_landlord_prev = 0
        self.r_peasant_down_prev = 0
        self.r_peasant_up_prev = 0
        self.r_peasants_prev = 0

        self.initial_hands = {0: '', 1: '', 2: ''}  # Store initial hands for all players

    def train(self):
        # Set CUDA device
        self.device = torch.device(f"cuda:{self.training_device}" if torch.cuda.is_available() and self.cuda else "cpu")
        
        # Create bot instances for each role
        landlord_bot = RARSMSBot(self.landlord_path)
        peasant_up_bot = RARSMSBot(self.peasant_up_path)     # Peasant after landlord (landlord-up)
        peasant_down_bot = RARSMSBot(self.peasant_down_path) # Peasant before landlord
        
        # Set device for each bot
        landlord_bot.set_device(self.device)
        peasant_up_bot.set_device(self.device)
        peasant_down_bot.set_device(self.device)
        
        # Create optimizers for each bot
        optimizers = {
            # Landlord optimizers
            0: {
                'actor': optim.Adam(landlord_bot.actor_network.parameters(), lr=self.learning_rate),
                'critic': optim.Adam(landlord_bot.critic_network.parameters(), lr=self.learning_rate)
            },
            # Peasant down optimizers
            1: {
                'actor': optim.Adam(peasant_down_bot.actor_network.parameters(), lr=self.learning_rate),
                'critic': optim.Adam(peasant_down_bot.critic_network.parameters(), lr=self.learning_rate)
            },
            # Peasant up optimizers
            2: {
                'actor': optim.Adam(peasant_up_bot.actor_network.parameters(), lr=self.learning_rate),
                'critic': optim.Adam(peasant_up_bot.critic_network.parameters(), lr=self.learning_rate)
            }
        }
        
        # Create a mapping between player_id and bots
        bots = {
            0: landlord_bot,
            1: peasant_up_bot,   # Player 1 is now peasant up (after landlord)
            2: peasant_down_bot  # Player 2 is now peasant down (before landlord)
        }
        
        curr_player_data = {
                player_id: {
                    'states': [],
                    'perfect_states': [],
                    'actions': [],
                    'probs': [],
                    'rewards': [],
                    'td_errors': [],
                    'old_bot': None,
                }
                for player_id in range(3)
            }

        # Each Game (Episode)
        for episode in range(self.num_episodes):
            # Reset environment
            state, player_id = self.env.reset()

            # Clear data collection for the current episode except old actor network
            for p_id in range(3):
                curr_player_data[p_id]['states'] = []
                curr_player_data[p_id]['perfect_states'] = []
                curr_player_data[p_id]['actions'] = []
                curr_player_data[p_id]['probs'] = []
                curr_player_data[p_id]['rewards'] = []
                curr_player_data[p_id]['td_errors'] = []

            
            # Reset rewards
            self.r_landlord_prev = 0
            self.r_peasant_down_prev = 0
            self.r_peasant_up_prev = 0
            
            # Reset initial hands
            self.initial_hands = {0: '', 1: '', 2: ''}
            
            # Each round (frame)
            while not self.env.is_over():
                if self.initial_hands[player_id] == '':
                    self.initial_hands[player_id] = state['raw_obs']['current_hand']

                current_bot = bots[player_id]
                
                # Choose action using the appropriate bot
                action, prob = current_bot.act(state)
                perfect_state = self.env.get_perfect_information()
                
                # Store current state, action, and probability
                curr_player_data[player_id]['states'].append(state)
                curr_player_data[player_id]['perfect_states'].append(perfect_state)
                curr_player_data[player_id]['actions'].append(action)
                curr_player_data[player_id]['probs'].append(prob)
                
                # Take a step in the environment
                next_state, next_player_id = self.env.step(action)
                next_perfect_state = self.env.get_perfect_information()
                
                # Calculate rewards
                environment_reward = self.env.get_payoffs()[player_id] if self.env.is_over() else 0
                reward = self._calculate_intrinsic_reward(environment_reward, player_id)
                curr_player_data[player_id]['rewards'].append(reward)
                
                # Calculate temporal difference error
                td_error = (
                    reward + 
                    self.gamma * current_bot.predict_state(state, perfect_state) - 
                    current_bot.predict_state(next_state, next_perfect_state)
                )
                curr_player_data[player_id]['td_errors'].append(td_error)
                
                # Update state and player
                state = next_state
                player_id = next_player_id
            
            # After episode is over, update each bot
            for player_id in range(3):
                current_bot = bots[player_id]
    
                # Skip if no data collected for this player
                if not curr_player_data[player_id]['states']:
                    continue


                
                # Calculate advantage function
                advantage_function = self._calculate_advantage(curr_player_data[player_id]['td_errors'])
                
                # Calculate losses
                actor_loss = self._calculate_actor_loss(
                    curr_player_data[player_id]['old_bot'],
                    advantage_function,
                    curr_player_data[player_id]['states'],
                    curr_player_data[player_id]['actions'],
                    curr_player_data[player_id]['probs']
                )
                
                critic_loss = self._calculate_critic_loss(
                    current_bot,
                    curr_player_data[player_id]['states'],
                    curr_player_data[player_id]['perfect_states'],
                    curr_player_data[player_id]['rewards']
                )
                
                # Print losses for monitoring
                role = ["Landlord", "Peasant Up", "Peasant Down"][player_id]
                print(f"EPISODE {episode}/{self.num_episodes} {role} ACTOR LOSS: {actor_loss.item()}")
                print(f"EPISODE {episode}/{self.num_episodes} {role} CRITIC LOSS: {critic_loss.item()}")
                
                # Update actor network
                optimizers[player_id]['actor'].zero_grad()
                actor_loss.backward()
                optimizers[player_id]['actor'].step()
                
                # Update critic network
                optimizers[player_id]['critic'].zero_grad()
                critic_loss.backward()
                optimizers[player_id]['critic'].step()
                
                # Store current model as old data for next episode
                curr_player_data[player_id]['old_bot'] = copy.deepcopy(current_bot)
            
            # Save models periodically
            if episode % self.save_interval == 0:
                print("SAVE INTERVAL", self.save_interval, episode % self.save_interval)
                self._save_models(bots, episode)

    def _save_models(self, bots, episode):
        """Save all three models."""
        for player_id, _ in enumerate(["landlord", "peasant_up", "peasant_down"]):
            self._save_model("rarsms", player_id, bots[player_id].actor_network, episode)

    def _calculate_intrinsic_reward(self, environment_reward: float, player_id: int):
        """
        Computes intrinsic reward based on progress in minimizing splits and reducing cards.
        """
        
        # Set k based on player role
        if player_id == 0:  # Landlord
            k = 1
        else:  # Peasant
            k = -1/2
            
        # Calculate for landlord
        L, N = self._calculate_minimum_splits(0, True)
        Lt, Ct = self._calculate_minimum_splits(0, False)
        r_landlord = (L - Lt + N - Ct) / (L + N)
        
        # Calculate for peasant down
        L, N = self._calculate_minimum_splits(1, True)
        Lt, Ct = self._calculate_minimum_splits(1, False)
        r_peasant_down = (L - Lt + N - Ct) / (L + N)
        
        # Calculate for peasant up
        L, N = self._calculate_minimum_splits(2, True)
        Lt, Ct = self._calculate_minimum_splits(2, False)
        r_peasant_up = (L - Lt + N - Ct) / (L + N)
        
        # Calculate cooperative reward for peasants
        r_peasants = max(
            r_peasant_down + self.beta * r_peasant_up,
            r_peasant_up + self.beta * r_peasant_down
        )
        
        # Calculate previous rewards delta based on player role
        if player_id == 0:  # Landlord
            prev_delta = self.r_landlord_prev - self.r_peasants_prev
            r = self._clamp(r_landlord - r_peasants - prev_delta, -1, 1) * 2 * environment_reward * k
            
        elif player_id == 1:  # Peasant down
            prev_delta = self.r_peasants_prev - self.r_landlord_prev
            r = self._clamp(r_peasants - r_landlord - prev_delta, -1, 1) * 2 * environment_reward * k
            
        else:  # Peasant up
            prev_delta = self.r_peasants_prev - self.r_landlord_prev
            r = self._clamp(r_peasants - r_landlord - prev_delta, -1, 1) * 2 * environment_reward * k
        
        # Update previous rewards
        self.r_landlord_prev = r_landlord
        self.r_peasant_down_prev = r_peasant_down
        self.r_peasant_up_prev = r_peasant_up
        self.r_peasants_prev = r_peasants
        
        return r
    
    def _clamp(self, value, min_value, max_value):
        """Clamps value between min_value and max_value"""
        return max(min_value, min(value, max_value))

    def _calculate_minimum_splits(self, player_id: int, use_initial_hand: bool):
        """
        Returns a tuple of the minimum split and the number of cards in the hand for current
        state. If initial_hand is true, then returns the minimum split, and the number of 
        cards in the hand for the players first hand at the beginning of the game.
        """
        
        RANK_ORDER = '3456789TJQKA2BR'  # 15 ranks
        RANK_INDEX = {rank: i for i, rank in enumerate(RANK_ORDER)}
        
        state = self.env.get_state(player_id)
        
        if use_initial_hand:
            if self.initial_hands[player_id]:
                hand = self.initial_hands[player_id]
            else:
                hand = state['raw_obs']['current_hand']
        else:
            hand = state['raw_obs']['current_hand']
            
        # Convert current_hand string to X vector (counts per rank)
        X = [0] * 15
        for card in hand:
            X[RANK_INDEX[card]] += 1
            
        memo = {}
        
        # Recursive function G
        def G(X):
            # Check for memoization
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

    def _calculate_advantage(self, temporal_difference_errors):
        """Calculate advantage function from TD errors."""
        T = len(temporal_difference_errors)
        advantage_function = []
        for t in range(T):
            advtg = 0
            for i in range(t, T):
                advtg += (self.gamma * self.lmda) ** (i-t) * temporal_difference_errors[i]
            advantage_function.append(advtg)
            
        return advantage_function

    def _calculate_actor_loss(self, old_bot, advantage_function, states, actions, probs):
        """ 
        PPO Clipped Objective Function.
        """
        # Handle empty data case
        if not states or not old_bot:
            return torch.tensor(0.0, requires_grad=True)
            
        probs = torch.stack(probs)
        
        advantages = torch.tensor(advantage_function, dtype=torch.float32, device=self.device)
        
        old_probs = []
        
        for state, action in zip(states, actions):
            prob = old_bot.get_log_prob(state, action)
            old_probs.append(prob)
        old_probs = torch.tensor(old_probs, dtype=torch.float32, device=self.device)
        
        ratios = torch.exp(probs - old_probs)
        clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
        
        actor_loss = -torch.mean(torch.min(ratios * advantages, clipped_ratios * advantages))
        
        return actor_loss

    def _calculate_critic_loss(self, bot, states, perfect_states, rewards):
        """ 
        Objective function using MSE.
        """
        # Handle empty data case
        if not states:
            return torch.tensor(0.0, requires_grad=True)
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        predicted_rewards = []
        for state, perfect_state in zip(states, perfect_states):
            prediction = bot.forward_critic(state, perfect_state) 
            predicted_rewards.append(prediction)
        predicted_rewards = torch.stack(predicted_rewards)
        
        loss = torch.mean((predicted_rewards - rewards) ** 2)
        
        return loss