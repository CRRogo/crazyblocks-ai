# Neural Network Training for Crazy Blocks

This directory contains a Deep Q-Network (DQN) implementation for training an AI to play Crazy Blocks.

## Architecture

The neural network has the following structure:

- **Input Layer**: 432 features
  - Grid encoding: 85 cells × 5 values (one-hot: 4 colors + empty) = 425 features
  - Metadata: score (normalized), turn count (normalized), 5 column heights = 7 features
  
- **Hidden Layers**: Configurable (default: 256 → 256 → 128 neurons)
  - Each layer uses ReLU activation
  - Dropout (0.1) for regularization
  
- **Output Layer**: 80 actions (16 rows × 5 columns, excluding bottom row)

**Total Parameters**: ~200,000-300,000 (depending on hidden layer sizes)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (for neural networks)
- NumPy (for numerical operations)

## Training

### Basic Training

Train a new model from scratch:
```bash
python train_neural.py --episodes 1000
```

### Advanced Training Options

```bash
python train_neural.py \
    --episodes 2000 \
    --learning-rate 0.0005 \
    --gamma 0.95 \
    --epsilon 1.0 \
    --epsilon-decay 0.995 \
    --epsilon-min 0.01 \
    --batch-size 64 \
    --memory-size 10000 \
    --target-update-freq 10 \
    --hidden-sizes 256 256 128 \
    --save-freq 100 \
    --eval-freq 50 \
    --output my_model.pth
```

### Resume Training

Continue training from a saved model:
```bash
python train_neural.py --load crazyblocks-neural.pth --episodes 500
```

### Parameters Explained

- `--episodes`: Number of training episodes (games)
- `--learning-rate`: How fast the network learns (default: 0.001)
- `--gamma`: Discount factor for future rewards (default: 0.95)
- `--epsilon`: Initial exploration rate (1.0 = 100% random)
- `--epsilon-decay`: How fast exploration decreases (default: 0.995)
- `--epsilon-min`: Minimum exploration rate (default: 0.01)
- `--batch-size`: Number of experiences to train on at once (default: 64)
- `--memory-size`: Size of experience replay buffer (default: 10000)
- `--target-update-freq`: How often to update target network (default: 10)
- `--hidden-sizes`: Hidden layer sizes, e.g., `256 256 128` (default: 256 256 128)
- `--save-freq`: Save model every N episodes (default: 100)
- `--eval-freq`: Evaluate performance every N episodes (default: 50)
- `--output`: Output file path (default: crazyblocks-neural.pth)

## Playing with Trained Model

Test your trained model:
```bash
python play_neural.py --model crazyblocks-neural.pth --games 10
```

## How It Works

### Deep Q-Network (DQN)

1. **Experience Replay**: Stores past experiences (state, action, reward, next_state) in a buffer
2. **Target Network**: Uses a separate "target" network for stable Q-value estimation
3. **Epsilon-Greedy**: Balances exploration (random actions) vs exploitation (best known actions)
4. **Q-Learning**: Updates Q-values using Bellman equation: `Q(s,a) = r + γ * max(Q(s',a'))`

### Training Process

1. Agent plays games and collects experiences
2. Experiences are stored in replay buffer
3. Periodically, agent samples a batch of experiences
4. Network is trained to predict Q-values using the batch
5. Target network is updated periodically for stability
6. Epsilon (exploration rate) decays over time

### State Encoding

The game state is encoded as:
- **Grid**: Each cell is one-hot encoded (5 values: 4 colors + empty)
- **Score**: Normalized by dividing by 10,000
- **Turn Count**: Normalized by dividing by 1,000
- **Column Heights**: Each column's height normalized by total rows

## Comparison with Genetic Algorithm

| Feature | Neural Network | Genetic Algorithm |
|---------|---------------|-------------------|
| **Approach** | Deep Reinforcement Learning | Evolutionary Algorithm |
| **Learning** | Learns from experience | Evolves heuristic weights |
| **Interpretability** | Black box | Transparent (heuristic weights) |
| **Training Time** | Moderate | Fast |
| **Memory** | Requires GPU for best performance | CPU-only, very fast |
| **Flexibility** | Can learn complex patterns | Limited by heuristics |

## Tips for Training

1. **Start Small**: Begin with 500-1000 episodes to see if it's learning
2. **Monitor Epsilon**: Watch how exploration decreases over time
3. **Check Memory**: Make sure replay buffer is filling up
4. **Evaluate Regularly**: Use `--eval-freq` to see how well it's performing
5. **Save Often**: Use `--save-freq` to save checkpoints
6. **GPU**: If available, PyTorch will automatically use GPU (much faster)

## Troubleshooting

**Low scores**: 
- Increase training episodes
- Adjust learning rate (try 0.0005 or 0.002)
- Increase hidden layer sizes

**Not learning**:
- Check that epsilon is decaying (should decrease over time)
- Verify replay buffer is filling (should reach memory_size)
- Try different learning rates

**Out of memory**:
- Reduce batch_size
- Reduce memory_size
- Use CPU instead of GPU

## Next Steps

- Try different network architectures (more/fewer layers)
- Experiment with hyperparameters
- Compare with genetic algorithm results
- Integrate into the main game UI

