#!/usr/bin/env python3
"""
Fresh start training with optimized hyperparameters
Better for networks that are stuck
"""

import argparse
import time
from pathlib import Path
from neural_agent import NeuralAgent
from train_ai import GameEngine

def main():
    parser = argparse.ArgumentParser(description='Fresh start neural network training with optimized settings')
    parser.add_argument('--episodes', type=int, default=None, help='Number of training episodes (default: unlimited)')
    parser.add_argument('--output', type=str, default='crazyblocks-neural-fresh.pth', help='Output model file')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda), auto-detect if not specified')
    
    args = parser.parse_args()
    
    print("Neural Network Trainer (Fresh Start - Optimized)")
    print("=" * 60)
    
    # Optimized hyperparameters for better learning
    agent = NeuralAgent(
        game_engine_class=GameEngine,
        learning_rate=0.0003,  # Lower learning rate for stability
        gamma=0.99,  # High discount (value future)
        epsilon=1.0,  # Start with full exploration
        epsilon_decay=0.9999,  # Very slow decay (explore longer)
        epsilon_min=0.1,  # Keep 10% exploration minimum
        batch_size=128,  # Larger batches
        memory_size=100000,  # Much larger memory buffer
        target_update_freq=200,  # Update target network less frequently
        hidden_sizes=[512, 512, 256],  # Large network
        device=args.device,
        train_every_n_steps=1
    )
    
    print("Starting FRESH training (not loading previous model)")
    print(f"Network Architecture:")
    print(f"   Input size: {agent.input_size}")
    print(f"   Hidden layers: [512, 512, 256]")
    print(f"   Output size: {agent.output_size}")
    print(f"   Total parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print(f"Device: {agent.device}")
    print(f"Hyperparameters:")
    print(f"   Learning rate: 0.0003 (stable)")
    print(f"   Epsilon: 1.0 -> 0.1 (high exploration)")
    print(f"   Epsilon decay: 0.9999 (very slow)")
    print(f"   Memory buffer: 100,000 (large)")
    print(f"   Batch size: 128")
    if args.episodes:
        print(f"Training for {args.episodes} episodes")
    else:
        print(f"Training indefinitely (Press Ctrl+C to stop)")
    print(f"Saving every 500 episodes to {args.output}")
    print(f"Evaluating every 100 episodes (20 games)")
    print("=" * 60)
    print()
    
    start_time = time.time()
    engine = GameEngine()
    
    # Training statistics
    episode_scores = []
    episode_rewards = []
    eval_scores = []
    
    try:
        episode = 0
        while True:
            episode += 1
            
            if args.episodes and episode > args.episodes:
                print(f"\nReached target of {args.episodes} episodes")
                break
            
            # Train one episode
            result = agent.play_game(engine, training=True)
            episode_scores.append(result['score'])
            episode_rewards.append(result['reward'])
            
            # Print progress
            if episode % 10 == 0:
                avg_score = sum(episode_scores[-10:]) / 10
                avg_reward = sum(episode_rewards[-10:]) / 10
                print(
                    f"Episode {episode:5d} | "
                    f"Score: {result['score']:6.0f} (avg: {avg_score:6.1f}) | "
                    f"Reward: {result['reward']:8.1f} | "
                    f"ε: {agent.epsilon:.3f} | "
                    f"Mem: {len(agent.memory):6d}/{agent.memory.maxlen}"
                )
            
            # Evaluation
            if episode % 100 == 0:
                agent.q_network.eval()
                eval_scores_list = []
                for _ in range(20):
                    eval_result = agent.play_game(engine, training=False)
                    eval_scores_list.append(eval_result['score'])
                agent.q_network.train()
                
                avg_eval_score = sum(eval_scores_list) / len(eval_scores_list)
                best_eval_score = max(eval_scores_list)
                eval_scores.append(avg_eval_score)
                
                trend = ""
                if len(eval_scores) > 1:
                    if eval_scores[-1] > eval_scores[-2]:
                        trend = " (UP)"
                    elif eval_scores[-1] < eval_scores[-2]:
                        trend = " (DOWN)"
                
                print(
                    f"\nEvaluation (Episode {episode}): "
                    f"Avg: {avg_eval_score:.1f} | "
                    f"Best: {best_eval_score:.0f} | "
                    f"Best Overall: {agent.best_score:.0f}{trend}\n"
                )
            
            # Save checkpoint
            if episode % 500 == 0:
                agent.save(args.output)
                print(f"Checkpoint saved at episode {episode}")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        agent.save(args.output)
        print("Progress saved!")
    
    # Final save
    agent.save(args.output)
    
    total_time = time.time() - start_time
    
    if episode_scores:
        final_avg_score = sum(episode_scores[-100:]) / min(100, len(episode_scores))
    else:
        final_avg_score = 0
    
    print("=" * 60)
    print("Training stopped!")
    print(f"Episodes trained: {episode}")
    print(f"Best Score: {agent.best_score}")
    print(f"Final Average Score (last 100): {final_avg_score:.2f}")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Model saved to: {args.output}")


if __name__ == '__main__':
    main()

