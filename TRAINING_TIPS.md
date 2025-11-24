# Neural Network Training Tips

## If Training Isn't Working (Low Scores After 2000+ Episodes)

### 1. **Improved Reward Structure** ✅ (Already Updated)
The reward function now includes:
- **Efficiency bonus**: Rewards eliminating 5+ blocks (critical requirement)
- **Large group bonus**: Extra reward for big eliminations
- **Column balance bonus**: Rewards keeping columns similar height
- **Survival bonus**: Small reward for staying alive

### 2. **Better Hyperparameters**

Try these improved settings:

```bash
python train_neural.py \
    --learning-rate 0.0005 \
    --gamma 0.99 \
    --epsilon-decay 0.998 \
    --epsilon-min 0.05 \
    --batch-size 128 \
    --memory-size 20000 \
    --target-update-freq 20 \
    --hidden-sizes 512 256 128
```

**Why these changes:**
- **Lower learning rate (0.0005)**: More stable learning, less oscillation
- **Higher gamma (0.99)**: Values long-term rewards more
- **Slower epsilon decay (0.998)**: More exploration time
- **Larger batch size (128)**: More stable gradient estimates
- **Larger memory (20000)**: More diverse experiences
- **Larger network (512→256→128)**: More capacity to learn

### 3. **Start Fresh vs Resume**

If your current model isn't learning:
- **Option A**: Start fresh with better hyperparameters
  ```bash
  python train_neural.py --learning-rate 0.0005 --hidden-sizes 512 256 128
  ```

- **Option B**: Continue current training but with lower learning rate
  ```bash
  python train_neural.py --load crazyblocks-neural.pth --learning-rate 0.0005
  ```

### 4. **Check Training Status**

Use the diagnostic script:
```bash
python check_training.py --model crazyblocks-neural.pth
```

This will show:
- Current performance
- Training statistics
- Diagnosis and recommendations

### 5. **What to Expect**

**Good signs:**
- Average score increasing over time
- Epsilon decreasing (exploration → exploitation)
- Memory buffer filling up
- Evaluation scores improving

**Bad signs:**
- Scores stuck at same low level
- Epsilon stuck at high value
- No improvement after 1000+ episodes

### 6. **Alternative: Use Genetic Algorithm**

If neural network continues to struggle, the genetic algorithm approach might work better:
```bash
python train_ai.py --generations 100
```

The genetic algorithm uses explicit heuristics (like the ones you suggested) and often learns faster for this type of game.

### 7. **Debugging Steps**

1. **Check if rewards are being received:**
   - Look at episode rewards in training output
   - Should see positive rewards for good moves

2. **Check if network is updating:**
   - Loss should decrease over time
   - Q-values should change

3. **Check exploration:**
   - Epsilon should decay from 1.0 → 0.01
   - If stuck at 1.0, network isn't learning

4. **Try simpler architecture:**
   - Start with smaller network: `--hidden-sizes 128 64`
   - If that works, scale up

### 8. **Quick Fixes**

**If scores are < 50 after 2000 episodes:**
```bash
# Stop current training
# Start fresh with better settings
python train_neural.py \
    --learning-rate 0.0003 \
    --hidden-sizes 256 128 \
    --batch-size 64 \
    --save-freq 100
```

**If training is unstable (scores jumping around):**
```bash
# Lower learning rate
python train_neural.py --load model.pth --learning-rate 0.0001
```

### 9. **Comparison: Neural Network vs Genetic Algorithm**

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Neural Network** | Can learn complex patterns, generalizes | Slow to train, needs tuning | Long-term learning |
| **Genetic Algorithm** | Fast training, interpretable | Limited by heuristics | Quick results, strategic play |

For your game, **Genetic Algorithm might be better** because:
- You have clear strategic insights (column balance, 5+ blocks, etc.)
- These map well to heuristics
- Faster to train and tune

Consider training the genetic algorithm with your improved heuristics instead!

