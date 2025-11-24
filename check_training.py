#!/usr/bin/env python3
"""
Check training status and diagnose issues
"""

import torch
import argparse
from neural_agent import NeuralAgent
from train_ai import GameEngine

def main():
    parser = argparse.ArgumentParser(description='Check neural network training status')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    
    args = parser.parse_args()
    
    print("Checking Training Status")
    print("=" * 60)
    
    # Create agent and load
    agent = NeuralAgent(GameEngine)
    if not agent.load(args.model):
        print("❌ Failed to load model")
        return
    
    print(f"Loaded model: {args.model}")
    print()
    
    # Check training stats
    print("Training Statistics:")
    print(f"   Total games/episodes: {agent.total_games}")
    print(f"   Best score: {agent.best_score}")
    print(f"   Current epsilon: {agent.epsilon:.4f}")
    print(f"   Memory buffer: {len(agent.memory)}/{agent.memory.maxlen}")
    print()
    
    # Test current performance
    print("Testing current performance (10 games)...")
    engine = GameEngine()
    scores = []
    
    for i in range(10):
        result = agent.play_game(engine, training=False)
        scores.append(result['score'])
        print(f"   Game {i+1}: {result['score']:.0f}")
    
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    worst_score = min(scores)
    
    print()
    print("Performance Summary:")
    print(f"   Average score: {avg_score:.1f}")
    print(f"   Best score: {best_score:.0f}")
    print(f"   Worst score: {worst_score:.0f}")
    print()
    
    # Diagnose issues
    print("Diagnosis:")
    if avg_score < 50:
        print("   WARNING: Scores are very low (< 50)")
        print("      - Network may not be learning effectively")
        print("      - Try: Lower learning rate (0.0005)")
        print("      - Try: Increase hidden layer sizes")
        print("      - Try: Train for more episodes")
    elif avg_score < 200:
        print("   WARNING: Scores are moderate (50-200)")
        print("      - Network is learning but needs more training")
        print("      - Continue training for more episodes")
    else:
        print("   OK: Scores are good (> 200)")
        print("      - Network is learning well!")
    
    if agent.epsilon > 0.1:
        print(f"   WARNING: Epsilon is still high ({agent.epsilon:.3f})")
        print("      - Network is still exploring a lot")
        print("      - This is normal early in training")
    
    if len(agent.memory) < agent.memory.maxlen * 0.5:
        print(f"   WARNING: Memory buffer not full ({len(agent.memory)}/{agent.memory.maxlen})")
        print("      - Network may need more diverse experiences")
    
    print()
    print("💡 Recommendations:")
    print("   1. If scores are low, try retraining with improved rewards:")
    print("      python train_neural.py --load model.pth --learning-rate 0.0005")
    print("   2. Or start fresh with better hyperparameters:")
    print("      python train_neural.py --learning-rate 0.0005 --hidden-sizes 512 256 128")

if __name__ == '__main__':
    main()

