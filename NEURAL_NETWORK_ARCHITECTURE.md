# Neural Network Architecture Explanation

## How the Neural Network Works

### Input: Entire Board State (432 features)

The network receives the **complete board state** as input:

1. **Grid Encoding (425 features)**
   - Each of the 85 cells (17 rows × 5 columns) is one-hot encoded
   - Each cell = 5 values: [color1, color2, color3, color4, empty]
   - Total: 85 × 5 = 425 features
   - Example: If cell has color #6BA85A → [1, 0, 0, 0, 0]
   - Example: If cell is empty → [0, 0, 0, 0, 1]

2. **Metadata (7 features)**
   - Score (normalized): `score / 10000.0`
   - Turn count (normalized): `turn_count / 1000.0`
   - Column heights (5 values): Height of each column normalized by ROWS
   - Total: 2 + 5 = 7 features

**Total Input Size: 425 + 7 = 432 features**

### Output: Q-Values for All Possible Clicks (80 actions)

The network outputs Q-values (quality scores) for **every possible click position**:

- **80 possible actions**: (16 rows - 1) × 5 columns = 80
- Each output = Q-value for clicking that specific (row, col) position
- Example: `output[0]` = Q-value for clicking row 0, col 0
- Example: `output[15]` = Q-value for clicking row 3, col 0
- Example: `output[79]` = Q-value for clicking row 15, col 4

### How It Chooses Actions

1. **Encode current board state** → 432 features
2. **Pass through network** → Get 80 Q-values (one per possible click)
3. **Mask invalid actions** → Set invalid positions to -infinity
4. **Choose highest Q-value** → Click that position

### What the Network Learns

The network learns a function:
```
Q(board_state, action) = Expected future reward from clicking this position
```

Given any board state, it predicts:
- "Clicking position (0,0) will give me X reward"
- "Clicking position (5,2) will give me Y reward"
- etc. for all 80 positions

Then it chooses the position with the highest predicted reward.

## Example Flow

```
Current Board State:
[#6BA85A, #A08FB8, null, ...]  (85 cells)
Score: 150
Turn: 25
Column heights: [12, 11, 13, 10, 12]

↓ Encoded to 432 features ↓

Neural Network
(512 → 512 → 256 neurons)

↓ Outputs 80 Q-values ↓

Q-values: [2.3, 5.1, -1.2, 8.7, 3.4, ...]  (80 values)
          ↑    ↑    ↑     ↑    ↑
          (0,0)(0,1)(0,2) (0,3)(0,4) ... positions

↓ Mask invalid, choose best ↓

Best action: Position (5, 2) with Q-value 8.7
→ Click that block!
```

## Key Points

✅ **YES** - Network sees entire board state (all 85 cells)
✅ **YES** - Network outputs which block to click (80 possible positions)
✅ **YES** - Network learns the relationship between board state and best action
✅ **YES** - Network considers all positions simultaneously when choosing

The network is learning: "Given this complete board state, which position should I click to maximize my score?"

