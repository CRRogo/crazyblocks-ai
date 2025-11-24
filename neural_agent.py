#!/usr/bin/env python3
"""
Neural Network Agent for Crazy Blocks
Uses PyTorch to implement a Deep Q-Network (DQN) or Policy Network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import json
from pathlib import Path

# Game constants (must match train_ai.py)
COLUMNS = 5
ROWS = 17
COLORS = ['#6BA85A', '#A08FB8', '#D4A5A0', '#7DADB5']
COLOR_INDICES = {color: idx for idx, color in enumerate(COLORS)}


class CrazyBlocksNet(nn.Module):
    """
    Neural Network for Crazy Blocks
    Input: Game state (flattened grid + metadata)
    Output: Q-values for each possible action
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [256, 256, 128], output_size: int = None):
        super(CrazyBlocksNet, self).__init__()
        
        # Input size: grid (ROWS * COLUMNS * 5) + metadata (score, turn_count, etc.)
        # Grid encoding: each cell = 5 values (4 colors + empty)
        self.input_size = input_size
        self.output_size = output_size or (ROWS - 1) * COLUMNS  # All possible actions
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Small dropout for regularization
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, self.output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class NeuralAgent:
    """
    Neural Network-based agent using Deep Q-Network (DQN)
    """
    
    def __init__(
        self,
        game_engine_class,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        hidden_sizes: List[int] = [256, 256, 128],
        device: str = None,
        train_every_n_steps: int = 1
    ):
        self.game_engine_class = game_engine_class
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_every_n_steps = train_every_n_steps
        
        # Device (CPU or GPU)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculate input size
        # Grid: ROWS * COLUMNS * 5 (one-hot encoding: 4 colors + empty)
        # Metadata: score (normalized), turn_count (normalized), column_heights (5 values)
        self.input_size = (ROWS * COLUMNS * 5) + 2 + COLUMNS  # 85*5 + 2 + 5 = 432
        
        # Create networks
        self.q_network = CrazyBlocksNet(self.input_size, hidden_sizes).to(self.device)
        self.target_network = CrazyBlocksNet(self.input_size, hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Store output size for easy access
        self.output_size = self.q_network.output_size
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training stats
        self.total_games = 0
        self.total_score = 0
        self.best_score = 0
        self.fitness = 0.0
        self.games_played = 0
        self.update_count = 0
        
    def encode_state(self, game_engine) -> np.ndarray:
        """
        Encode game state as a feature vector
        Returns: numpy array of shape (input_size,)
        """
        features = []
        
        # One-hot encode grid: each cell = 5 values (4 colors + empty)
        for r in range(ROWS):
            for c in range(COLUMNS):
                cell = game_engine.grid[r][c]
                if cell is None:
                    features.extend([0, 0, 0, 0, 1])  # Empty
                else:
                    color_idx = COLOR_INDICES[cell]
                    one_hot = [0] * 4
                    one_hot[color_idx] = 1
                    features.extend(one_hot + [0])  # Not empty
        
        # Metadata
        features.append(game_engine.score / 10000.0)  # Normalized score
        features.append(game_engine.turn_count / 1000.0)  # Normalized turn count
        
        # Column heights (normalized)
        for c in range(COLUMNS):
            height = sum(1 for r in range(ROWS) if game_engine.grid[r][c] is not None)
            features.append(height / ROWS)
        
        return np.array(features, dtype=np.float32)
    
    def get_valid_action_mask(self, game_engine) -> np.ndarray:
        """
        Get a mask of valid actions (1 = valid, 0 = invalid)
        Returns: numpy array of shape (output_size,)
        """
        mask = np.zeros((ROWS - 1) * COLUMNS, dtype=np.float32)
        valid_actions = game_engine.get_valid_actions()
        for row, col in valid_actions:
            idx = row * COLUMNS + col
            mask[idx] = 1.0
        return mask
    
    def action_to_index(self, row: int, col: int) -> int:
        """Convert (row, col) to action index"""
        return row * COLUMNS + col
    
    def index_to_action(self, idx: int) -> Tuple[int, int]:
        """Convert action index to (row, col)"""
        row = idx // COLUMNS
        col = idx % COLUMNS
        return row, col
    
    def choose_action(self, game_engine, training: bool = True) -> Optional[Tuple[int, int]]:
        """
        Choose an action using epsilon-greedy policy
        """
        valid_actions = game_engine.get_valid_actions()
        if not valid_actions:
            return None
        
        # Epsilon-greedy: explore or exploit
        if training and random.random() < self.epsilon:
            # Explore: random valid action
            return random.choice(valid_actions)
        
        # Exploit: use neural network
        state = self.encode_state(game_engine)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            q_values = q_values.squeeze(0).cpu().numpy()
        
        # Mask invalid actions
        action_mask = self.get_valid_action_mask(game_engine)
        q_values = q_values * action_mask + (1 - action_mask) * (-1e9)  # Large negative for invalid
        
        # Choose best action
        best_idx = np.argmax(q_values)
        return self.index_to_action(best_idx)
    
    def remember(self, state, action_idx, reward, next_state, done, valid_actions_next):
        """Store experience in replay buffer"""
        # Store valid actions as a list for easier handling
        self.memory.append((state, action_idx, reward, next_state, done, list(valid_actions_next)))
    
    def replay(self):
        """Train on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, action_indices, rewards, next_states, dones, valid_actions_list = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        action_indices = torch.LongTensor(action_indices).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            # Mask invalid actions
            next_q_masked = []
            for i, valid_actions in enumerate(valid_actions_list):
                if valid_actions:  # If there are valid actions
                    mask = np.zeros((ROWS - 1) * COLUMNS, dtype=np.float32)
                    for row, col in valid_actions:
                        idx = self.action_to_index(row, col)
                        mask[idx] = 1.0
                    mask_tensor = torch.FloatTensor(mask).to(self.device)
                    masked_q = next_q_values[i] * mask_tensor + (1 - mask_tensor) * (-1e9)
                    next_q_masked.append(torch.max(masked_q).item())
                else:  # Terminal state or no valid actions
                    next_q_masked.append(0.0)
            
            next_q = torch.FloatTensor(next_q_masked).to(self.device)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def play_game(self, game_engine=None, training: bool = True) -> Dict:
        """Play a game and return results"""
        engine = game_engine if game_engine else self.game_engine_class()
        engine.reset()
        
        state = self.encode_state(engine)
        total_reward = 0
        steps = 0
        max_steps = 1000
        # Reset step counter for this episode
        episode_step_count = 0
        
        while not engine.game_over and steps < max_steps:
            # Choose action
            action = self.choose_action(engine, training=training)
            if not action:
                break
            
            row, col = action
            action_idx = self.action_to_index(row, col)
            
            # Take action
            result = engine.act(row, col)
            if not result['success']:
                break
            
            # Get next state
            next_state = self.encode_state(engine)
            reward = result['reward']
            done = engine.game_over
            valid_actions_next = engine.get_valid_actions()
            
            # Store experience
            if training:
                self.remember(state, action_idx, reward, next_state, done, valid_actions_next)
                
                # Train on batch (can skip some steps for speed)
                episode_step_count += 1
                if len(self.memory) >= self.batch_size and episode_step_count % self.train_every_n_steps == 0:
                    self.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # CRITICAL: Add massive terminal reward based on FINAL SCORE
        # This makes the network understand that overall score is what matters
        if engine.game_over or steps >= 1000:
            # Terminal reward: Final score is the ultimate goal
            # Scale it massively so network prioritizes high final scores
            final_score_reward = engine.score * 100.0  # MASSIVE reward for final score
            total_reward += final_score_reward
            
            # Store this as a terminal experience to reinforce final score importance
            if training:
                # Create terminal state
                terminal_state = self.encode_state(engine)
                # Store terminal reward experience (dummy action, but reward is what matters)
                terminal_action = 0
                self.remember(state, terminal_action, final_score_reward, terminal_state, True, [])
        
        # Update stats
        self.games_played += 1
        self.total_games += 1
        self.total_score += engine.score
        self.best_score = max(self.best_score, engine.score)
        self.fitness = self.total_score / self.games_played if self.games_played > 0 else 0
        
        return {
            'score': engine.score,
            'steps': steps,
            'reward': total_reward,
            'fitness': self.fitness
        }
    
    def save(self, filepath: str):
        """Save model and agent state"""
        # Extract hidden sizes from network architecture
        hidden_sizes = []
        for i in range(0, len(self.q_network.network) - 1, 3):  # Every 3rd module is a Linear layer
            if isinstance(self.q_network.network[i], nn.Linear):
                hidden_sizes.append(self.q_network.network[i].out_features)
        
        data = {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_games': self.total_games,
            'best_score': self.best_score,
            'fitness': self.fitness,
            'episode': self.total_games,  # Track episode count
            'hidden_sizes': hidden_sizes,  # Save architecture
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
            }
        }
        torch.save(data, filepath)
        print(f"Saved neural network to {filepath}")
    
    def load(self, filepath: str, hidden_sizes: List[int] = None):
        """Load model and agent state"""
        if not Path(filepath).exists():
            print(f"File not found: {filepath}")
            return False
        
        data = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Get architecture from saved file or use provided
        saved_hidden_sizes = data.get('hidden_sizes', None)
        if saved_hidden_sizes:
            hidden_sizes = saved_hidden_sizes
        elif hidden_sizes is None:
            # Try to infer from state dict (fallback)
            print("Warning: Architecture not found in saved file. Attempting to infer...")
            # This is tricky, so we'll recreate with default and let it fail if wrong
            hidden_sizes = [256, 256, 128]  # Default
        
        # Recreate networks with correct architecture if needed
        # Check current architecture
        current_sizes = []
        for i in range(0, len(self.q_network.network) - 1, 3):
            if isinstance(self.q_network.network[i], nn.Linear):
                current_sizes.append(self.q_network.network[i].out_features)
        
        if hidden_sizes and hidden_sizes != current_sizes:
            print(f"Recreating network with architecture: {hidden_sizes} (was {current_sizes})")
            self.q_network = CrazyBlocksNet(self.input_size, hidden_sizes).to(self.device)
            self.target_network = CrazyBlocksNet(self.input_size, hidden_sizes).to(self.device)
            self.output_size = self.q_network.output_size
        
        self.q_network.load_state_dict(data['q_network_state'])
        self.target_network.load_state_dict(data['target_network_state'])
        self.optimizer.load_state_dict(data['optimizer_state'])
        self.epsilon = data.get('epsilon', self.epsilon_min)
        self.total_games = data.get('total_games', 0)
        self.best_score = data.get('best_score', 0)
        self.fitness = data.get('fitness', 0.0)
        print(f"Loaded neural network from {filepath}")
        print(f"  Architecture: {hidden_sizes}")
        print(f"  Best score: {self.best_score}")
        print(f"  Episodes: {self.total_games}")
        return True
    
    def reset_fitness(self):
        """Reset fitness tracking"""
        self.fitness = 0.0
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0

