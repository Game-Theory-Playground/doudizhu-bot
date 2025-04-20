from .base_bot import BaseBot
from rlcard.games.doudizhu.utils.action_encoding import ID_TO_ACTION, ACTION_ID_DICT


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

# Card ordering for Dou Dizhu (Fighting the Landlord)
CARD_ORDER = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', '2', 'BJ', 'RJ']
SPECIFIC_MAP = {card: idx for idx, card in enumerate(CARD_ORDER)}

class ResidualBlock(nn.Module):
    """Basic ResNet-18 block with skip connections"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNetBackbone(nn.Module):
    """Shared ResNet-18 backbone implementation for both Actor and Critic networks"""
    def __init__(self, in_channels=64):
        super(ResNetBackbone, self).__init__()
        
        # Initial convolution layer
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Create ResNet backbone
        self.layers = self._create_layers()
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layers(x)
        return x
        
    def _create_layers(self):
        """Create a ResNet-18 backbone"""
        layers = []
        
        # Layer 1 (64 channels)
        in_channels = 64
        out_channels = 64
        for _ in range(2):
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
            
        # Layer 2 (128 channels)
        out_channels = 128
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        in_channels = out_channels
        layers.append(ResidualBlock(in_channels, out_channels))
        
        # Layer 3 (256 channels)
        out_channels = 256
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        in_channels = out_channels
        layers.append(ResidualBlock(in_channels, out_channels))
        
        # Layer 4 (512 channels)
        out_channels = 512
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        in_channels = out_channels
        layers.append(ResidualBlock(in_channels, out_channels))
        
        # Global average pooling and flatten
        backbone = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        return backbone

class ActorNetwork(nn.Module):
    """Actor network that outputs action probabilities"""
    def __init__(self):
        super().__init__()
        # Backbone network processing input features
        self.backbone = ResNetBackbone(in_channels=49)
        
        # FC layers
        self.fc1 = nn.Linear(512, 309)
        self.fc2 = nn.Linear(309, 1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, imperfect, history, perfect=None):
        """Process state features and output action probabilities"""
        # Concatenate available features
        if perfect is not None:
            x = torch.cat([imperfect, history, perfect], dim=0)
        else:
            x = torch.cat([imperfect, history], dim=0)
        
        # Forward pass
        x = x.unsqueeze(0)  # Add batch dimension
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.softmax(x)

class CriticNetwork(nn.Module):
    """Critic network that estimates state values"""
    def __init__(self):
        super().__init__()
        # Backbone network processing input features
        self.backbone = ResNetBackbone(in_channels=57)
        
        # FC layers for value output
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)  # Output state value V(s_t)
        
    def forward(self, imperfect, history, perfect):
        """Process state features and output state value"""
        x = torch.cat([imperfect, history, perfect], dim=0)
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # V(s_t)

class RARSMSBot(BaseBot):
    """Reinforcement Learning bot for Dou Dizhu using Actor-Critic architecture."""
    
    def __init__(self, position=None, device=None):
        super().__init__(position)

        # Recoding features from raw information from RLCard environment
        self.use_raw = True
        
        # Set device - use CUDA if available by default
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.actor_network = ActorNetwork()
        self.critic_network = CriticNetwork()
        
        # Move networks to the selected device
        self.actor_network.to(self.device)
        self.critic_network.to(self.device)
        
    def set_device(self, device):
        """Set the computation device (CPU/GPU)."""
        self.device = device
        self.actor_network.to(device)
        self.critic_network.to(device)
        
    def act(self, state):
        """Select an action based on current state."""
        
        masked_probs = self._get_action_probs(state)
        
        # Choose action with highest probability
        action_id = torch.argmax(masked_probs).item()
        
        # Add small value to avoid log(0)
        log_prog = torch.log(masked_probs[action_id] + 1e-8).item()

        return action_id, log_prog
    
    def get_log_prob(self, state, action):
        """
        Give a state and action, return the log probabilities
        of the action being played.
        """
        masked_probs = self._get_action_probs(state)
        
        # Add small value to avoid log(0)
        log_prog = torch.log(masked_probs[action] + 1e-8).item()

        return log_prog
    
    def predict_state(self, state):
        """
        Use the Critic Network to predict the expected reward using the
        current state
        """

        # Extract features
        imperfect_features = self._extract_imperfect_features(state)
        history_features = self._extract_history_features(state)
        perfect_features = self._extract_perfect_features(state)
        

        # Forward pass through actor network (without perfect features during play)
        with torch.no_grad():
            expected_reward = self.actor_network(imperfect_features, history_features, perfect_features)

            
        return expected_reward
    

    def _get_action_probs(self, state):
        """ Get the valid action probabilities of the current state """

        # Extract features
        imperfect_features = self._extract_imperfect_features(state)
        history_features = self._extract_history_features(state)
        
        # Forward pass through actor network (without perfect features during play)
        with torch.no_grad():
            action_probs = self.actor_network(imperfect_features, history_features)
            
        # Get legal actions mask and apply it
        legal_actions = self._get_legal_actions_mask(state)
        masked_probs = action_probs * legal_actions
        
        # Renormalize probabilities
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            # If all actions are masked, use uniform distribution
            masked_probs = legal_actions / legal_actions.sum()

        return masked_probs


    def _extract_imperfect_features(self, state):
        """
        Extract imperfect information features features (19x54).
        
        This includes information that is visible to the agent during gameplay
        without knowing other players' hands.
        """
        raw = state['raw_obs']
        role_id = raw['self']
        num_cards_left = raw['num_cards_left']
        trace = raw['trace']
        played_cards = raw['played_cards']
        seen_cards = list(raw.get('seen_cards', ''))
        current_hand = raw['current_hand']

        # Initialize feature tensor
        feat = torch.zeros(19, 54, device=self.device)
        
        # 0. Count bombs from trace
        bombs_played = self._count_bombs_in_trace(trace)
        feat[0][bombs_played - 1] = 1

        # 1. Other players' hand count
        total_other_cards = sum(num_cards_left[i] for i in range(3) if i != role_id)
        feat[1][total_other_cards - 1] = 1

        # 2. Seen cards
        feat[2] = self._cards_to_tensor(seen_cards)

        # 3. Role identity encoding
        feat[3] = self._encode_player_identity(role_id, num_cards_left[role_id])

        # 4. Self hand (up to 3 cards of each type)
        feat[4:7] = self._encode_cards_in_hand(list(current_hand))

        # 5. Played cards for each player
        feat[7:10] = self._encode_cards_in_hand(list(played_cards[0]))
        feat[10:13] = self._encode_cards_in_hand(list(played_cards[1]))
        feat[13:16] = self._encode_cards_in_hand(list(played_cards[2]))

        # 6. Card count per player
        feat[16] = self._encode_card_counts(num_cards_left)

        # Most recent action and cards played
        feat[17:19] = self._encode_last_action(trace, num_cards_left)
        
        return feat

    def _count_bombs_in_trace(self, trace):
        """Count the number of bombs played in the game trace."""
        bombs_played = 0
        for _, action in trace:
            if action == 'pass':
                continue
            # Regular bomb (4 of a kind)
            if len(action) == 4 and len(set(action)) == 1:
                bombs_played += 1
            # Rocket (joker pair)
            elif sorted(action) == ['BJ', 'RJ']:
                bombs_played += 1
        return bombs_played

    def _encode_player_identity(self, player_id, cards_left):
        """Encode player identity, role, and cards left."""
        identity = torch.zeros(54, device=self.device)
        
        # Player ID encoding
        if player_id == 0:
            base_bits = [0, 0]  # Landlord
        elif player_id == 1:
            base_bits = [0, 1]  # Farmer 1
        else:
            base_bits = [1, 0]  # Farmer 2
            
        identity[:18] = torch.tensor(base_bits * 9, dtype=torch.float, device=self.device)
        identity[18:34] = 1 if player_id == 0 else 0  # Is landlord?
        identity[34 + cards_left - 1] = 1  # Cards left
        
        return identity

    def _encode_card_counts(self, num_cards_left):
        """Encode card counts for all players."""
        counts = torch.zeros(54, device=self.device)
        counts[num_cards_left[0] - 1] = 1  # Player 0 cards
        counts[20 + num_cards_left[1] - 1] = 1  # Player 1 cards
        counts[37 + num_cards_left[2] - 1] = 1  # Player 2 cards
        return counts

    def _encode_last_action(self, trace, num_cards_left):
        """Encode the most recent non-pass action and its cards."""
        last_action = torch.zeros(2, 54, device=self.device)
        
        # Find the last non-pass action
        last = next(((pid, act) for pid, act in reversed(trace) if act != 'pass'), None)
        
        if last:
            pid, act = last
            # Encode player identity
            last_action[0] = self._encode_player_identity(pid, num_cards_left[pid])
            # Encode cards played
            last_action[1] = self._cards_to_tensor(list(act))
            
        return last_action

    def _extract_history_features(self, state):
        """
        Extract action history features (30x54).
        
        This captures the last 15 actions (player identity and cards played).
        """
        raw = state['raw_obs']
        num_cards_left = raw['num_cards_left']
        trace = raw['trace']

        feat = torch.zeros(30, 54, device=self.device)
        
        # Take the last 15 actions at most
        start = max(0, len(trace) - 15)
    
        for i, (pid, act) in enumerate(trace[start:]):
            idx = 2 * i
            
            # Encode player identity
            feat[idx] = self._encode_player_identity(pid, num_cards_left[pid])
            
            # Encode cards played (empty list if 'pass')
            cards = list(act) if act != 'pass' else []
            feat[idx + 1] = self._cards_to_tensor(cards)
            
        return feat

    def _extract_perfect_features(self, state, perfect_state):
        """
        Extract a 8x54 perfect feature tensor.
        
        This is the perfect information about
        other players' hands, which is only available during training.
        """
        num_cards_left = state['raw_obs']['num_cards_left']
        current_pid = perfect_state['current_player']
        prev_pid = (current_pid - 1) % 3
        next_pid = (current_pid + 1) % 3

        # Prepare an 8x54 block for perfect features
        perfect_feat = torch.zeros(8, 54, device=self.device)
        
        # Previous player's identity and cards
        perfect_feat[0] = self._encode_player_identity(prev_pid, num_cards_left[prev_pid])
        perfect_feat[1:4] = self._encode_cards_in_hand(list(perfect_state['hand_cards'][prev_pid]))
        
        # Next player's identity and cards
        perfect_feat[4] = self._encode_player_identity(next_pid, num_cards_left[next_pid])
        perfect_feat[5:8] = self._encode_cards_in_hand(list(perfect_state['hand_cards'][next_pid]))

        return perfect_feat

    def _get_legal_actions_mask(self, state):
        """Create a binary mask for legal actions."""
        mask = torch.zeros(309, device=self.device)
        for action_id in state['legal_actions']:
            mask[action_id] = 1.0
        return mask
    
    def _cards_to_tensor(self, cards):
        """Convert a list of cards into a 1x54 tensor."""
        vec = torch.zeros(54, device=self.device)
        for card in cards:
            if card in SPECIFIC_MAP:
                vec[SPECIFIC_MAP[card]] = 1
        return vec

    def _encode_cards_in_hand(self, cards):
        """
        Encode a hand of cards into a 3x54 feature tensor.
        
        This encoding includes:
        - Row 0: Card count encoding (4 bits per card)
        - Row 1: Pattern encoding (solo, pair, trio, bomb)
        - Row 2: Chain encoding (chain of solo, chain of pair, chain of trio, rocket)
        """
        # Initialize tensor
        feat = torch.zeros(3, 54, device=self.device)
        
        # Count occurrences of each card
        card_counter = Counter(cards)
        
        # 0-1: card count + pattern

        # Process regular cards (all except jokers)
        for card_idx, card in enumerate(CARD_ORDER[:-2]):
            count = card_counter.get(card, 0)
            
            if count > 0:
                # Card count encoding (first row)
                base_pos = card_idx * 4
                # Set bits based on count (more bits for higher counts)
                for i in range(min(count, 4)):
                    feat[0, base_pos + (3-i)] = 1
                
                # Card pattern encoding (second row)
                if count == 1:
                    feat[1, card_idx] = 1  # Solo
                elif count == 2:
                    feat[1, card_idx + 15] = 1  # Pair
                elif count == 3:
                    feat[1, card_idx + 28] = 1  # Trio
                elif count == 4:
                    feat[1, card_idx + 41] = 1  # Bomb
        
        # Handle jokers
        for joker, idx, solo_idx in [('BJ', 52, 13), ('RJ', 53, 14)]:
            if joker in card_counter and card_counter[joker] > 0:
                feat[0, idx] = 1       # Count encoding
                feat[1, solo_idx] = 1   # Solo pattern encoding
        
        # 2: chain detection

        # Count cards by rank for chain detection
        rank_counts = [card_counter.get(card, 0) for card in CARD_ORDER[:-2]]
        
        # Chain of Solo: 5+ consecutive solos
        for start_idx in range(8):  # Up to card 10 (need 5 consecutive cards)
            if all(rank_counts[i] >= 1 for i in range(start_idx, start_idx + 5)):
                for i in range(start_idx, start_idx + 5):
                    feat[2, i] = 1
        
        # Chain of Pair: 3+ consecutive pairs
        for start_idx in range(10):  # Up to card Q (need 3 consecutive pairs)
            if all(rank_counts[i] >= 2 for i in range(start_idx, start_idx + 3)):
                for i in range(start_idx, start_idx + 3):
                    feat[2, i + 15] = 1
        
        # Chain of Trio: 2+ consecutive trios
        for start_idx in range(11):  # Up to card K (need 2 consecutive trios)
            if all(rank_counts[i] >= 3 for i in range(start_idx, start_idx + 2)):
                for i in range(start_idx, start_idx + 2):
                    feat[2, i + 28] = 1
        
        # Rocket: Both jokers
        has_black_joker = 'BJ' in card_counter and card_counter['BJ'] > 0
        has_red_joker = 'RJ' in card_counter and card_counter['RJ'] > 0
        
        if has_black_joker and has_red_joker:
            feat[2, 41:] = 1
        
        return feat