#!/usr/bin/env python3
"""
Diagnostic script to check if neural network training is working
"""

import torch
import numpy as np
from neural_agent import NeuralAgent
from train_ai import GameEngine

def main():
    print("Neural Network Training Diagnostics")
    print("=" * 60)
    
    # Create agent
    agent = NeuralAgent(
        game_engine_class=GameEngine,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=64,
        memory_size=10000,
        hidden_sizes=[256, 256, 128]
    )
    
    engine = GameEngine()
    
    print("\n[1] Testing Initial Network Behavior")
    print("-" * 60)
    
    # Play a few games with random actions (high epsilon)
    print("Playing 5 games with random actions (epsilon=1.0)...")
    random_scores = []
    for i in range(5):
        result = agent.play_game(engine, training=False)
        random_scores.append(result['score'])
        print(f"  Game {i+1}: Score = {result['score']:.0f}, Steps = {result['steps']}")
    
    avg_random = sum(random_scores) / len(random_scores)
    print(f"  Average random score: {avg_random:.1f}")
    
    print("\n[2] Checking Network Outputs")
    print("-" * 60)
    engine.reset()
    state = agent.encode_state(engine)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    
    with torch.no_grad():
        q_values = agent.q_network(state_tensor)
        q_values = q_values.squeeze(0).cpu().numpy()
    
    print(f"  Q-value stats: min={q_values.min():.3f}, max={q_values.max():.3f}, mean={q_values.mean():.3f}, std={q_values.std():.3f}")
    print(f"  Q-value range: {q_values.max() - q_values.min():.3f}")
    
    # Check if Q-values are all similar (bad sign)
    if q_values.std() < 0.01:
        print("  WARNING: Q-values have very low variance - network may not be learning!")
    
    print("\n[3] Testing Training Step")
    print("-" * 60)
    
    # Fill memory with some experiences
    print("Collecting experiences...")
    for _ in range(100):
        engine.reset()
        state = agent.encode_state(engine)
        action = agent.choose_action(engine, training=True)
        if action:
            row, col = action
            result = engine.act(row, col)
            if result['success']:
                next_state = agent.encode_state(engine)
                agent.remember(state, agent.action_to_index(row, col), result['reward'], 
                             next_state, engine.game_over, engine.get_valid_actions())
    
    print(f"  Memory size: {len(agent.memory)}/{agent.memory.maxlen}")
    
    if len(agent.memory) >= agent.batch_size:
        print("  Training on batch...")
        loss_before = None
        loss_after = None
        
        # Get a sample to check loss
        batch = list(agent.memory)[:agent.batch_size]
        states, action_indices, rewards, next_states, dones, valid_actions_list = zip(*batch)
        states_t = torch.FloatTensor(np.array(states)).to(agent.device)
        action_indices_t = torch.LongTensor(action_indices).to(agent.device)
        
        with torch.no_grad():
            current_q = agent.q_network(states_t)
            current_q_selected = current_q.gather(1, action_indices_t.unsqueeze(1)).squeeze(1)
            loss_before = current_q_selected.mean().item()
        
        # Train
        loss = agent.replay()
        
        with torch.no_grad():
            current_q = agent.q_network(states_t)
            current_q_selected = current_q.gather(1, action_indices_t.unsqueeze(1)).squeeze(1)
            loss_after = current_q_selected.mean().item()
        
        print(f"  Loss: {loss:.4f}")
        print(f"  Q-value before training: {loss_before:.3f}")
        print(f"  Q-value after training: {loss_after:.3f}")
        print(f"  Change: {loss_after - loss_before:.6f}")
        
        if abs(loss_after - loss_before) < 1e-6:
            print("  WARNING: Network weights may not be updating!")
    else:
        print("  WARNING: Not enough experiences in memory to train")
    
    print("\n[4] Checking Reward Structure")
    print("-" * 60)
    engine.reset()
    test_rewards = []
    for _ in range(10):
        valid_actions = engine.get_valid_actions()
        if not valid_actions:
            break
        action = valid_actions[0]
        result = engine.act(action[0], action[1])
        if result['success']:
            test_rewards.append(result['reward'])
    
    if test_rewards:
        print(f"  Sample rewards: {test_rewards[:5]}")
        print(f"  Reward stats: min={min(test_rewards):.2f}, max={max(test_rewards):.2f}, mean={sum(test_rewards)/len(test_rewards):.2f}")
        if max(test_rewards) < 10:
            print("  WARNING: Rewards seem very small!")
    
    print("\n[5] Checking Hyperparameters")
    print("-" * 60)
    print(f"  Learning rate: {agent.optimizer.param_groups[0]['lr']}")
    print(f"  Gamma (discount): {agent.gamma}")
    print(f"  Epsilon: {agent.epsilon:.3f} (decay: {agent.epsilon_decay}, min: {agent.epsilon_min})")
    print(f"  Batch size: {agent.batch_size}")
    print(f"  Memory size: {agent.memory.maxlen}")
    print(f"  Target update freq: {agent.target_update_freq}")
    print(f"  Train every N steps: {agent.train_every_n_steps}")
    
    print("\n[6] Network Architecture")
    print("-" * 60)
    total_params = sum(p.numel() for p in agent.q_network.parameters())
    trainable_params = sum(p.numel() for p in agent.q_network.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Input size: {agent.input_size}")
    print(f"  Output size: {agent.output_size}")
    
    # Check if gradients are flowing
    print("\n[7] Checking Gradient Flow")
    print("-" * 60)
    if len(agent.memory) >= agent.batch_size:
        agent.optimizer.zero_grad()
        loss = agent.replay()
        
        # Check gradients
        has_grad = False
        total_grad_norm = 0
        for name, param in agent.q_network.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
                if 'weight' in name and '0' in name:  # First layer
                    print(f"  First layer gradient norm: {grad_norm:.6f}")
        
        total_grad_norm = total_grad_norm ** 0.5
        print(f"  Has gradients: {has_grad}")
        print(f"  Total gradient norm: {total_grad_norm:.6f}")
        
        if not has_grad:
            print("  WARNING: No gradients detected - network not learning!")
        elif total_grad_norm < 1e-6:
            print("  WARNING: Gradients are extremely small - learning may be too slow!")
    
    print("\n" + "=" * 60)
    print("Diagnostics complete!")
    print("\nRecommendations:")
    print("  - If Q-values have low variance: Network may need more training")
    print("  - If gradients are tiny: Try increasing learning rate")
    print("  - If rewards are small: Check reward function")
    print("  - If network isn't updating: Check training loop")

if __name__ == '__main__':
    main()

