#!/usr/bin/env python3
"""
Play Crazy Blocks using a trained Neural Network
"""

import argparse
from neural_agent import NeuralAgent
from train_ai import GameEngine

def main():
    parser = argparse.ArgumentParser(description='Play Crazy Blocks with trained Neural Network')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("🎮 Playing Crazy Blocks with Neural Network")
    print("=" * 50)
    
    # Create agent
    agent = NeuralAgent(
        game_engine_class=GameEngine,
        device=args.device
    )
    
    # Load model
    if not agent.load(args.model):
        print("❌ Failed to load model")
        return
    
    print(f"✅ Loaded model from {args.model}")
    print(f"🏆 Model's best score during training: {agent.best_score}")
    print(f"📊 Playing {args.games} games...")
    print("=" * 50)
    print()
    
    # Play games
    engine = GameEngine()
    scores = []
    
    for game in range(1, args.games + 1):
        result = agent.play_game(engine, training=False)
        scores.append(result['score'])
        print(f"Game {game:3d}: Score = {result['score']:6.0f}")
    
    # Statistics
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    worst_score = min(scores)
    
    print()
    print("=" * 50)
    print("📊 Results:")
    print(f"   Average Score: {avg_score:.2f}")
    print(f"   Best Score: {best_score:.0f}")
    print(f"   Worst Score: {worst_score:.0f}")
    print("=" * 50)


if __name__ == '__main__':
    main()

