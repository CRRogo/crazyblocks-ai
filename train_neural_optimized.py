#!/usr/bin/env python3
"""
Optimized Neural Network Training Script
Faster learning with better hyperparameters
"""

import argparse
import time
from pathlib import Path
from neural_agent import NeuralAgent
from train_ai import GameEngine

def main():
    parser = argparse.ArgumentParser(description='Train Crazy Blocks AI using Neural Network (DQN) - Optimized')
    parser.add_argument('--episodes', type=int, default=None, help='Number of training episodes (default: unlimited)')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate (lower = more stable)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (higher = values future more)')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon (exploration)')
    parser.add_argument('--epsilon-decay', type=float, default=0.9995, help='Epsilon decay rate (slower decay = more exploration)')
    parser.add_argument('--epsilon-min', type=float, default=0.05, help='Minimum epsilon (keep some exploration)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training (larger = more stable)')
    parser.add_argument('--memory-size', type=int, default=50000, help='Replay buffer size (larger = more diverse)')
    parser.add_argument('--train-freq', type=int, default=1, help='Train every N steps')
    parser.add_argument('--target-update-freq', type=int, default=100, help='Target network update frequency')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 256, 128], 
                        help='Hidden layer sizes')
    parser.add_argument('--save-freq', type=int, default=500, help='Save model every N episodes')
    parser.add_argument('--eval-freq', type=int, default=100, help='Evaluate every N episodes')
    parser.add_argument('--eval-games', type=int, default=20, help='Number of games for evaluation')
    parser.add_argument('--load', type=str, default=None, help='Load model from file')
    parser.add_argument('--output', type=str, default='crazyblocks-neural.pth', help='Output model file')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda), auto-detect if not specified')
    
    args = parser.parse_args()
    
    print("Neural Network Trainer (Optimized)")
    print("=" * 60)
    
    # Create agent
    agent = NeuralAgent(
        game_engine_class=GameEngine,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        target_update_freq=args.target_update_freq,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        train_every_n_steps=args.train_freq
    )
    
    # Load if specified
    if args.load:
        agent.load(args.load, hidden_sizes=args.hidden_sizes)
        print(f"Resuming training from {args.load}")
        print(f"Previous best: Score {agent.best_score}")
    else:
        print("Starting fresh training")
    
    print(f"Network Architecture:")
    print(f"   Input size: {agent.input_size}")
    print(f"   Hidden layers: {args.hidden_sizes}")
    print(f"   Output size: {agent.output_size}")
    print(f"   Total parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print(f"Device: {agent.device}")
    if args.episodes:
        print(f"Training for {args.episodes} episodes")
    else:
        print(f"Training indefinitely (Press Ctrl+C to stop)")
    print(f"Saving every {args.save_freq} episodes to {args.output}")
    print(f"Evaluating every {args.eval_freq} episodes ({args.eval_games} games)")
    print("=" * 60)
    print()
    
    start_time = time.time()
    engine = GameEngine()
    
    # Training statistics
    episode_scores = []
    episode_rewards = []
    eval_scores = []
    
    # Start from where we left off if loading
    start_episode = agent.total_games if args.load else 0
    
    try:
        episode = start_episode
        while True:
            episode += 1
            
            # Check if we've reached the episode limit (if specified)
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
                    f"Mem: {len(agent.memory):5d}/{agent.memory.maxlen}"
                )
            
            # Evaluation
            if episode % args.eval_freq == 0:
                agent.q_network.eval()
                eval_scores_list = []
                for _ in range(args.eval_games):
                    eval_result = agent.play_game(engine, training=False)
                    eval_scores_list.append(eval_result['score'])
                agent.q_network.train()
                
                avg_eval_score = sum(eval_scores_list) / len(eval_scores_list)
                best_eval_score = max(eval_scores_list)
                eval_scores.append(avg_eval_score)
                
                # Show trend
                trend = ""
                if len(eval_scores) > 1:
                    if eval_scores[-1] > eval_scores[-2]:
                        trend = " (UP)"
                    elif eval_scores[-1] < eval_scores[-2]:
                        trend = " (DOWN)"
                    else:
                        trend = " (SAME)"
                
                print(
                    f"\nEvaluation (Episode {episode}): "
                    f"Avg: {avg_eval_score:.1f} | "
                    f"Best: {best_eval_score:.0f} | "
                    f"Best Overall: {agent.best_score:.0f}{trend}\n"
                )
            
            # Save checkpoint
            if episode % args.save_freq == 0:
                agent.save(args.output)
                print(f"Checkpoint saved at episode {episode}")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        agent.save(args.output)
        print("Progress saved!")
    
    # Final save
    agent.save(args.output)
    
    total_time = time.time() - start_time
    
    # Final statistics
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
    print()
    print("Example usage to resume:")
    print(f"  python train_neural_optimized.py --load {args.output}")


if __name__ == '__main__':
    main()

