from .base_bot import BaseBot

import torch
import torch.nn as nn
import torch.nn.functional as F

class RARSMSBot(BaseBot):
    def __init__(self, position=None):
        super().__init__(position)
        self.use_raw = False
        self.device = 'cpu'
        self.actor_network = None
        self.critic_network = None
        self.dmc_module = None
        
    def build_networks(self):
        # Initialize the networks (will implement next)
        pass
        
    def act(self, state):
        #Extract features
        imperfect_features = self._extract_imperfect_features(state)
        history_features = self._extract_history_features(state)
        perfect_features = self._extract_perfect_features(state)
    
        # Get action probabilities from actor network
        action_probs = self.actor_network(imperfect_features, history_features, perfect_features)
    
        # Get highest probability action
        best_action = torch.argmax(action_probs).item()
    
        # Check if it's an abstraction action
        if self._is_abstraction_action(best_action):
            # Handle abstraction action
            legal_specific_actions = self._get_legal_specific_actions(state, best_action)
        
            # Use DMC to evaluate state-action values
            state_action_values = self.dmc_module(state, legal_specific_actions)
        
            # Choose the best specific action
            best_specific_action = torch.argmax(state_action_values).item()
            return legal_specific_actions[best_specific_action]
        else:
            # Return the action directly
            return best_action
        
    def _extract_imperfect_features(self, state):
        """Extract features from current game state"""
        # Create a tensor of shape [19 x 54]
        features = torch.zeros(19, 54)
    
        # Extract identity
        # Camp information
        # Face-down cards
        # Cards in hand
        # ... (detailed implementation)
    
        return features

    def _extract_history_features(self, state):
        """Extract features from action history"""
        # Create a tensor of shape [30 x 54]
        features = torch.zeros(30, 54)
    
        # Extract past actions
        # ... (detailed implementation)
    
        return features

    def _extract_perfect_features(self, state):
        """Extract known game information"""
        # Create a tensor of shape [8 x 54]
        features = torch.zeros(8, 54)
    
        # Extract perfect information
        # ... (detailed implementation)
    
        return features

class ResidualBlock(nn.Module):
    """Basic ResNet-18 block"""
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
        self.fc2 = nn.Linear(309, 1)  # Output action probabilities
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, imperfect, history, perfect=None):
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
        
        # Global average pooling and final layer
        backbone = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        return backbone

class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN layer processing concatenated features [57 x 54]
        self.cnn = nn.Sequential(
            nn.Conv1d(57, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # ResNet-18 backbone
        self.resnet = self._create_resnet_backbone()
        
        # FC layers - direct to 1×1 output as shown in diagram
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)  # Output state value V(s_t)
        
    def forward(self, state_features):
        # Process state features directly
        # Input shape should be [batch_size, 57, 54]
        x = self.cnn(state_features)
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # V(s_t)
        
    def _create_resnet_backbone(self):
        """Create a ResNet-18 backbone"""
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
        
        # Global average pooling and final layer
        backbone = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        return backbone
        
class DMCModule(nn.Module):
    def __init__(self):
        super().__init__()
        # DMC-specific layers
        # ...
        pass