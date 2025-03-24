from .base_bot import BaseBot

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

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
    
class ActorNetwork(nn.Module):
    """Actor network that outputs action probabilities"""
    def __init__(self):
        super().__init__()
        # CNN layer processing concatenated features [49 x 54]
        self.cnn = nn.Sequential(
            nn.Conv1d(49, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # ResNet-18 backbone
        self.resnet = self._create_resnet_backbone()
        
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
        x = self.cnn(x.unsqueeze(0))
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.softmax(x)
        
    def _create_resnet_backbone(self):
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

class CriticNetwork(nn.Module):
    """Critic network that estimates state values"""
    def __init__(self):
        super().__init__()
        # CNN layer processing concatenated features [57 x 54]
        self.cnn = nn.Sequential(
            nn.Conv1d(57, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # ResNet-18 backbone
        self.resnet = self._create_resnet_backbone()
        
        # FC layers for value output
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)  # Output state value V(s_t)
        
    def forward(self, state_features):
        """Process state features and output state value"""
        x = self.cnn(state_features)
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # V(s_t)
        
    def _create_resnet_backbone(self):
        """Create a ResNet-18 backbone (same as actor)"""
        layers = []
        
        # Layer 1
        in_channels = 64
        out_channels = 64
        for _ in range(2):
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
            
        # Layer 2
        out_channels = 128
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        in_channels = out_channels
        layers.append(ResidualBlock(in_channels, out_channels))
        
        # Layer 3
        out_channels = 256
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        in_channels = out_channels
        layers.append(ResidualBlock(in_channels, out_channels))
        
        # Layer 4
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

class RARSMSBot(BaseBot):
    """Reinforcement Learning bot for card games using Actor-Critic architecture"""
    def __init__(self, position=None):
        super().__init__(position)
        self.use_raw = False
        self.device = 'cpu'
        
        # Initialize networks
        self.actor_network = ActorNetwork()
        self.critic_network = CriticNetwork()
        
    def set_device(self, device):
        """Set the computation device (CPU/GPU)"""
        self.device = device
        self.actor_network.to(device)
        self.critic_network.to(device)
        
    def act(self, state):
        """Select an action based on current state"""
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
            # If all actions were masked, use uniform distribution
            masked_probs = legal_actions / legal_actions.sum()
            
        # Choose action with highest probability
        action_id = torch.argmax(masked_probs).item()
        
        return action_id
    
    def _extract_imperfect_features(self, state):
        """Extract imperfect information features (19x54)"""
        raw = state['raw_obs']
        role_id = raw['self']
        num_cards_left = raw['num_cards_left']
        trace = raw['trace']
        played_cards = raw['played_cards']
        seen_cards = list(raw.get('seen_cards', ''))

        # Count bombs from trace
        bombs_played = 0
        for _, action in trace:
            if action != 'pass' and len(action) == 4 and len(set(action)) == 1:
                bombs_played += 1
            elif action == 'BJ' or action == 'RJ' or sorted(action) == ['B', 'R']:
                bombs_played += 1

        feat = torch.zeros(49, 54, device=self.device)

        # 0. Bombs played
        feat[0][bombs_played - 1] = 1

        # 1. Other players' hand count
        total_other_cards = sum(num_cards_left[i] for i in range(3) if i != role_id)
        feat[1][total_other_cards - 1] = 1

        # 2. Seen cards
        feat[2] = self._cards_to_tensor(seen_cards)

        # 3. Role identity
        id_bits = ([0, 0] if role_id == 0 else [0, 1] if role_id == 1 else [1, 0]) * 9
        feat[3][:18] = torch.tensor(id_bits, dtype=torch.float, device=self.device)
        feat[3][18:34] = 1 if role_id == 0 else 0
        feat[3][34 + num_cards_left[role_id] - 1] = 1

        # 4-6. Self hand
        counter = Counter(raw['current_hand'])
        hand_feat = torch.zeros(3, 54, device=self.device)
        for card, count in counter.items():
            if card in SPECIFIC_MAP:
                col = SPECIFIC_MAP[card]
                for i in range(min(count, 3)):
                    hand_feat[i][col] = 1
        feat[4:7] = hand_feat

        # 7-15. Played cards
        feat[7:10] = self._get_played_card_tensor(played_cards[0])
        feat[10:13] = self._get_played_card_tensor(played_cards[1])
        feat[13:16] = self._get_played_card_tensor(played_cards[2])

        # 16. Card count per player
        feat[16][num_cards_left[0] - 1] = 1
        feat[16][20 + num_cards_left[1] - 1] = 1
        feat[16][37 + num_cards_left[2] - 1] = 1

        # 17. Most recent action (role, camp, card left)
        # 18. Most recent action cards
        last = next(((pid, act) for pid, act in reversed(trace) if act != 'pass'), None)
        if last:
            pid, act = last
            bits = ([0, 0] if pid == 0 else [0, 1] if pid == 1 else [1, 0]) * 9
            feat[17][:18] = torch.tensor(bits, dtype=torch.float, device=self.device)
            feat[17][18:34] = 1 if pid == 0 else 0
            feat[17][34 + num_cards_left[pid] - 1] = 1
            feat[18] = self._cards_to_tensor(list(act))

        # 19-48. Last 15 actions: [identity+camp+left], [cards]
        start = max(0, len(trace) - 15)
        for i, (pid, act) in enumerate(trace[start:]):
            idx = 19 + 2 * i
            bits = ([0, 0] if pid == 0 else [0, 1] if pid == 1 else [1, 0]) * 9
            hist_feat = torch.zeros(54, device=self.device)
            hist_feat[:18] = torch.tensor(bits, dtype=torch.float, device=self.device)
            hist_feat[18:34] = 1 if pid == 0 else 0
            hist_feat[34 + num_cards_left[pid] - 1] = 1
            feat[idx] = hist_feat
            feat[idx + 1] = self._cards_to_tensor(list(act) if act != 'pass' else [])

        return feat

    def _extract_history_features(self, state):
        """Extract action history features (30x54)"""
        features = torch.zeros(30, 54, device=self.device)
        # Implementation depends on game state format
        # ...
        return features

    def _extract_perfect_features(self, state):
        """Extract perfect information features (8x54) - only used in training"""
        features = torch.zeros(8, 54, device=self.device)
        # Implementation depends on game state format
        # ...
        return features
        
    def _get_legal_actions_mask(self, state):
        """Create a binary mask for legal actions"""
        mask = torch.zeros(309, device=self.device)
        for action_id in state['legal_actions']:
            mask[action_id] = 1.0
        return mask
    
    def _cards_to_tensor(cards):
        vec = torch.zeros(54)
        for c in cards:
            if c in SPECIFIC_MAP:
                vec[SPECIFIC_MAP[c]] = 1
        return vec

    def _get_played_card_tensor(played_cards_str):
        feat = torch.zeros((3, 54))
        count = Counter(played_cards_str)
        for card, times in count.items():
            if card not in SPECIFIC_MAP:
                continue
            col = SPECIFIC_MAP[card]
            for i in range(min(times, 3)):
                feat[i][col] = 1
        return feat