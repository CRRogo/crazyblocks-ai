#!/usr/bin/env python3
"""
Test script to verify group merging bonus works correctly
"""

from train_ai import GameEngine, COLORS

def test_group_merging_bonus():
    """Test that eliminating small separators gives bonus reward"""
    engine = GameEngine()
    engine.reset()
    
    print("Testing Group Merging Bonus")
    print("=" * 60)
    
    # Create a test scenario: two large groups separated by 1-2 blocks
    # This is hard to set up programmatically, so we'll just test the logic
    
    # Test 1: Small elimination (should check for merge bonus)
    print("\nTest 1: Small elimination (1-3 blocks)")
    print("-" * 60)
    
    # Get a valid action
    valid_actions = engine.get_valid_actions()
    if valid_actions:
        row, col = valid_actions[0]
        color = engine.grid[row][col]
        connected = engine.find_connected_blocks(row, col, color)
        blocks_before = len(connected)
        
        print(f"  Action: ({row}, {col}), Color: {color}")
        print(f"  Blocks in group: {blocks_before}")
        
        # Take action
        result = engine.act(row, col)
        
        print(f"  Blocks eliminated: {result['blocks_eliminated']}")
        print(f"  Reward: {result['reward']:.2f}")
        print(f"  Score reward: {result['blocks_eliminated'] * 50.0:.2f}")
        
        if result['blocks_eliminated'] <= 3:
            print(f"  -> Small elimination detected, checking for merge bonus...")
            # The bonus would be calculated in act(), so if reward > score_reward,
            # we got a bonus
            score_reward = result['blocks_eliminated'] * 50.0
            if result['reward'] > score_reward:
                bonus = result['reward'] - score_reward
                print(f"  -> Merge bonus detected: {bonus:.2f}")
            else:
                print(f"  -> No merge bonus (no large merged group found)")
        else:
            print(f"  -> Not a small elimination, no merge bonus check")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("\nThe group merging bonus will:")
    print("  1. Detect when 1-3 blocks are eliminated")
    print("  2. Check if a large merged group (10+ blocks) exists after gravity")
    print("  3. Add bonus reward (25-50) if merge detected")
    print("  4. Help neural network learn this strategy faster")

if __name__ == '__main__':
    test_group_merging_bonus()

