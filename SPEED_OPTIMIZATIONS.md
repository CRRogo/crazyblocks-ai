# Speed Optimization Guide for Neural Network Training

## Quick Start - Fast Training

Use the optimized fast training script:

```bash
python train_neural_fast.py
```

This uses:
- **Smaller network**: 128→64 neurons (vs 256→256→128)
- **Smaller batch**: 32 (vs 64)
- **Less evaluation**: Every 100 episodes, 5 games (vs 50 episodes, 10 games)
- **Less saving**: Every 200 episodes (vs 100)

**Expected speedup: 2-3x faster**

## Manual Speed Optimizations

### 1. **Reduce Network Size** (Biggest Impact)

Smaller networks train much faster:

```bash
# Small network (fastest)
python train_neural.py --hidden-sizes 128 64

# Medium network (balanced)
python train_neural.py --hidden-sizes 256 128

# Large network (slowest, but may learn better)
python train_neural.py --hidden-sizes 512 256 128
```

**Speed impact**: 2-4x faster with smaller network

### 2. **Reduce Evaluation Frequency**

Evaluations slow down training significantly:

```bash
# Evaluate less often
python train_neural.py --eval-freq 100 --eval-games 5

# Or disable evaluation entirely during training
python train_neural.py --eval-freq 999999
```

**Speed impact**: 20-30% faster

### 3. **Reduce Batch Size**

Smaller batches = faster training steps:

```bash
python train_neural.py --batch-size 32
```

**Speed impact**: 10-20% faster (but may hurt learning slightly)

### 4. **Reduce Memory Buffer**

Smaller replay buffer = less memory, slightly faster:

```bash
python train_neural.py --memory-size 5000
```

**Speed impact**: 5-10% faster

### 5. **Less Frequent Saves**

Saving takes time:

```bash
python train_neural.py --save-freq 200
```

**Speed impact**: 2-5% faster

### 6. **Train Less Frequently**

Train every N steps instead of every step:

```bash
# Train every 2 steps (50% less training)
python train_neural.py --train-freq 2
```

**Speed impact**: 30-40% faster (but slower learning)

## Recommended Fast Configurations

### **Ultra Fast** (Fastest, but may learn slower)
```bash
python train_neural.py \
    --hidden-sizes 128 64 \
    --batch-size 32 \
    --memory-size 5000 \
    --eval-freq 200 \
    --eval-games 3 \
    --save-freq 200
```
**Speed**: ~3-4x faster | **Learning**: May be slower

### **Balanced Fast** (Recommended)
```bash
python train_neural.py \
    --hidden-sizes 256 128 \
    --batch-size 64 \
    --eval-freq 100 \
    --eval-games 5 \
    --save-freq 150
```
**Speed**: ~2x faster | **Learning**: Similar quality

### **Quality Focused** (Slower but better learning)
```bash
python train_neural.py \
    --hidden-sizes 512 256 128 \
    --batch-size 128 \
    --memory-size 20000 \
    --eval-freq 50 \
    --eval-games 10
```
**Speed**: Normal | **Learning**: Best quality

## Performance Comparison

| Configuration | Episodes/sec | Relative Speed | Learning Quality |
|--------------|--------------|----------------|------------------|
| Default | 1-2 | 1x | Good |
| Fast (128→64) | 3-4 | 2-3x | Good |
| Ultra Fast | 4-6 | 3-4x | Moderate |
| Quality (512→256→128) | 0.5-1 | 0.5x | Excellent |

## Tips

1. **Start fast, then refine**: Train quickly with small network, then fine-tune with larger network
2. **Monitor learning**: If scores aren't improving, network might be too small
3. **CPU vs GPU**: GPU would be 10-50x faster, but you're on CPU
4. **Parallel training**: Can't easily parallelize single network training

## Alternative: Use Genetic Algorithm

If speed is critical, the genetic algorithm is **much faster**:

```bash
python train_ai.py --generations 100
```

**Speed**: 10-20x faster than neural network
**Learning**: Often better for this type of game

## State Encoding Optimization (Advanced)

The state encoding could be optimized further:
- Sample fewer rows (top 12 instead of all 17)
- Use simpler encoding (color indices instead of one-hot)
- Reduce metadata

But this requires code changes. Current encoding is already reasonably optimized.

