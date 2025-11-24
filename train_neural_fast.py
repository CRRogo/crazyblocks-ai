#!/usr/bin/env python3
"""
Fast Neural Network Training Script for Crazy Blocks
Optimized for speed while maintaining learning quality
"""

import argparse
import time
from pathlib import Path
from neural_agent import NeuralAgent
from train_ai import GameEngine

def main():
    parser = argparse.ArgumentParser(description='Train Crazy Blocks AI using Neural Network (Fast Mode)')
    parser.add_argument('--episodes', type=int, default=None, help='Number of training episodes (default: unlimited)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum epsilon')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (reduced for speed)')
    parser.add_argument('--memory-size', type=int, default=5000, help='Replay buffer size (reduced)')
    parser.add_argument('--target-update-freq', type=int, default=20, help='Target network update frequency (less frequent)')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[128, 64], 
                        help='Hidden layer sizes (smaller for speed, e.g., --hidden-sizes 128 64)')
    parser.add_argument('--save-freq', type=int, default=200, help='Save model every N episodes (less frequent)')
    parser.add_argument('--eval-freq', type=int, default=100, help='Evaluate every N episodes (less frequent)')
    parser.add_argument('--eval-games', type=int, default=5, help='Number of games for evaluation (reduced)')
    parser.add_argument('--train-freq', type=int, default=1, help='Train every N steps (1 = every step)')
    parser.add_argument('--load', type=str, default=None, help='Load model from file')
    parser.add_argument('--output', type=str, default='crazyblocks-neural-fast.pth', help='Output model file')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("⚡ Crazy Blocks Neural Network Trainer (Fast Mode)")
    print("=" * 60)
    
    # Create agent with smaller network
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
        agent.load(args.load)
        print(f"📂 Resuming training from {args.load}")
        print(f"🏆 Previous best: Score {agent.best_score}, Fitness {agent.fitness:.2f}")
    else:
        print("🆕 Starting fresh training")
    
    print(f"📊 Network Architecture (Optimized for Speed):")
    print(f"   Input size: {agent.input_size}")
    print(f"   Hidden layers: {args.hidden_sizes}")
    print(f"   Output size: {agent.output_size}")
    print(f"   Total parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print(f"⚡ Device: {agent.device}")
    print(f"⚡ Speed optimizations:")
    print(f"   - Smaller network: {args.hidden_sizes}")
    print(f"   - Smaller batch: {args.batch_size}")
    print(f"   - Less frequent evaluation: every {args.eval_freq} episodes")
    print(f"   - Fewer eval games: {args.eval_games}")
    if args.episodes:
        print(f"🎯 Training for {args.episodes} episodes")
    else:
        print(f"🎯 Training indefinitely (Press Ctrl+C to stop)")
    print(f"💾 Saving every {args.save_freq} episodes to {args.output}")
    print("=" * 60)
    print()
    
    start_time = time.time()
    engine = GameEngine()
    
    # Training statistics
    episode_scores = []
    episode_rewards = []
    eval_scores = []
    start_episode = agent.total_games if args.load else 0
    
    try:
        episode = start_episode
        step_count = 0
        while True:
            episode += 1
            step_count = 0
            
            # Check if we've reached the episode limit
            if args.episodes and episode > args.episodes:
                print(f"\n✅ Reached target of {args.episodes} episodes")
                break
            
            # Train one episode
            result = agent.play_game(engine, training=True)
            episode_scores.append(result['score'])
            episode_rewards.append(result['reward'])
            
            # Print progress (less frequently for speed)
            if episode % 20 == 0:
                avg_score = sum(episode_scores[-20:]) / 20
                avg_reward = sum(episode_rewards[-20:]) / 20
                elapsed = time.time() - start_time
                episodes_per_sec = episode / elapsed if elapsed > 0 else 0
                print(
                    f"Episode {episode:4d} | "
                    f"Score: {result['score']:6.0f} (avg: {avg_score:6.1f}) | "
                    f"ε: {agent.epsilon:.3f} | "
                    f"Memory: {len(agent.memory)}/{agent.memory.maxlen} | "
                    f"Speed: {episodes_per_sec:.1f} ep/s"
                )
            
            # Evaluation (less frequent)
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
                
                print(
                    f"\n📊 Evaluation (Episode {episode}): "
                    f"Avg: {avg_eval_score:.1f} | "
                    f"Best: {best_eval_score:.0f} | "
                    f"Overall Best: {agent.best_score:.0f}\n"
                )
            
            # Save checkpoint (less frequent)
            if episode % args.save_freq == 0:
                agent.save(args.output)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        agent.save(args.output)
        print("💾 Progress saved!")
    
    # Final save
    agent.save(args.output)
    
    total_time = time.time() - start_time
    
    # Final statistics
    if episode_scores:
        final_avg_score = sum(episode_scores[-100:]) / min(100, len(episode_scores))
    else:
        final_avg_score = 0
    
    print("=" * 60)
    print("✅ Training stopped!")
    print(f"📈 Episodes trained: {episode}")
    print(f"🏆 Best Score: {agent.best_score}")
    print(f"📊 Final Average Score (last 100): {final_avg_score:.2f}")
    print(f"⏱️  Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"⚡ Average Speed: {episode/total_time:.2f} episodes/second")
    print(f"💾 Model saved to: {args.output}")


if __name__ == '__main__':
    main()

