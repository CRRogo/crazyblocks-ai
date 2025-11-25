# Group Merging Enhancement for Neural Network

## What Was Added

A **group merging bonus** reward that explicitly rewards the neural network for eliminating small separators that merge large groups.

## How It Works

### Detection Logic

1. **Small Elimination Check**: When 1-3 blocks are eliminated
2. **Merge Detection**: After gravity, checks if a large merged group (10+ blocks) of the same color exists
3. **Bonus Calculation**: If merged group is 3x larger than eliminated blocks, adds bonus reward

### Reward Structure

```python
# Example scenarios:

# Scenario 1: Eliminate 2 blocks, creates 15-block merged group
# - Score reward: 2 * 50 = 100
# - Merge bonus: ~25-50 (based on merge quality)
# - Total: ~125-150

# Scenario 2: Eliminate 20 blocks directly
# - Score reward: 20 * 50 = 1000
# - No merge bonus (not a small elimination)
# - Total: 1000

# Scenario 3: Eliminate 2 blocks, no merge
# - Score reward: 2 * 50 = 100
# - No merge bonus (no large merged group)
# - Total: 100
```

### Bonus Formula

```python
if blocks_eliminated <= 3:  # Small elimination
    if largest_merged_group >= 10 and largest_merged_group > blocks_eliminated * 3:
        merge_ratio = (largest_merged_group - blocks_eliminated) / largest_merged_group
        group_merging_bonus = merge_ratio * 25.0  # Capped at 50.0
```

## Benefits for Neural Network

### Before Enhancement
- Network learns implicitly: "Small elimination → sometimes huge reward later"
- Takes many episodes to discover the pattern
- Less reliable recognition

### After Enhancement
- Network gets **immediate explicit signal**: "Small elimination with merge = bonus reward"
- Learns the pattern faster
- More reliable recognition of merge opportunities

## Impact on Training

### Expected Improvements

1. **Faster Learning**: Network will learn group merging strategy in fewer episodes
2. **Better Recognition**: More reliable detection of merge opportunities
3. **Strategic Play**: Network will prioritize eliminating small separators when large groups exist

### Reward Balance

- **Score reward still dominates**: 50 points per block eliminated
- **Merge bonus is moderate**: 25-50 points (helps but doesn't overshadow)
- **Total reward structure**: Score > Merge Bonus > Other bonuses

## Code Changes

### Added Function
- `_find_connected_in_grid()`: Helper to find connected blocks in a given grid state

### Modified Function
- `act()`: Added group merging bonus calculation in reward computation

## Testing

Run the test script to verify:
```bash
python test_group_merging_bonus.py
```

## Usage

The enhancement is **automatically active** for all training:
- Neural network training (`train_neural.py`)
- Genetic algorithm training (`train_ai.py`)
- Any agent using `GameEngine.act()`

No changes needed to your training commands - just continue training as normal!

## Example Training

```bash
# Continue training with enhanced rewards
python train_neural.py --load crazyblocks-neural.pth

# The network will now learn group merging faster!
```

## Expected Results

After training with this enhancement, you should see:
- Network learns to eliminate small separators when large groups exist
- Better strategic play (merging groups before eliminating)
- Higher scores as network discovers this pattern
- Faster convergence (fewer episodes needed)

The network will still learn this strategy without the bonus, but the bonus makes it **much faster and more reliable**.

