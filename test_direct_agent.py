#!/usr/bin/env python3
"""
Quick test of DirectAgent with new bottom-touching avoidance heuristic
"""

from direct_agent import DirectAgent
from train_ai import GameEngine

def test_agent():
    """Test the DirectAgent with new heuristics"""
    agent = DirectAgent()
    engine = GameEngine()
    engine.reset()
    
    print("Testing DirectAgent with new bottom-touching avoidance strategy")
    print("=" * 60)
    print(f"Bottom-touching penalty weight: {agent.weights['bottom_touching_penalty']}")
    print(f"Isolated group bonus weight: {agent.weights['isolated_group_bonus']}")
    print()
    
    # Play a few games
    scores = []
    for game_num in range(5):
        engine.reset()
        score = 0
        steps = 0
        max_steps = 1000
        
        while not engine.game_over and steps < max_steps:
            move = agent.choose_action(engine)
            if not move:
                break
            
            result = engine.act(move[0], move[1])
            if result['success']:
                score += result['blocks_eliminated']
            steps += 1
        
        scores.append(score)
        print(f"Game {game_num + 1}: Score = {score}, Steps = {steps}")
    
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    print()
    print(f"Average score: {avg_score:.1f}")
    print(f"Best score: {best_score}")
    print()
    print("The agent now:")
    print("  - Avoids clearing groups that touch the bottom (they can grow!)")
    print("  - Prefers clearing large isolated groups (safe to eliminate)")

if __name__ == '__main__':
    test_agent()

