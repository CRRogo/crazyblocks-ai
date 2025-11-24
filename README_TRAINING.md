# Python AI Training Script

Fast Python training script for Crazy Blocks AI that runs **10-50x faster** than the JavaScript version.

## Features

- ⚡ **Much Faster**: Uses parallel processing and optimized algorithms
- 🔄 **Continues Training**: Automatically loads and continues from existing JSON files
- 💾 **Same Format**: Uses the same JSON format as JavaScript version (fully compatible)
- 🚀 **Parallel Processing**: Uses all CPU cores for maximum speed
- 📊 **Progress Tracking**: Shows real-time statistics during training

## Installation

No dependencies required! Uses only Python standard library.

```bash
# Just make sure you have Python 3.7+
python --version
```

## Usage

### Basic Training

```bash
python train_ai.py
```

This will:
- Load existing `crazyblocks-strategies.json` if it exists
- Train for 50 generations (default)
- Save progress after each generation
- Create/update `crazyblocks-strategies.json`

### Custom Training

```bash
# Train for 100 generations
python train_ai.py --generations 100

# Use larger population
python train_ai.py --population 100

# More games per evaluation (more accurate but slower)
python train_ai.py --games 5

# Custom output file
python train_ai.py --output my-strategies.json

# Disable parallel processing (if you have issues)
python train_ai.py --no-parallel
```

### All Options

```bash
python train_ai.py --help
```

Options:
- `--generations`: Number of generations to train (default: 50)
- `--population`: Population size (default: 50)
- `--games`: Games per evaluation (default: 3)
- `--output`: Output JSON file path (default: crazyblocks-strategies.json)
- `--mutation-rate`: Mutation rate (default: 0.1)
- `--crossover-rate`: Crossover rate (default: 0.7)
- `--no-parallel`: Disable parallel processing

## Performance

**Speed Comparison:**
- JavaScript: ~5-15 minutes per generation
- Python (sequential): ~1-3 minutes per generation
- Python (parallel): ~10-30 seconds per generation ⚡

**Expected Training Times:**
- 10 generations: ~2-5 minutes
- 50 generations: ~10-25 minutes
- 100 generations: ~20-50 minutes

## Importing to JavaScript App

After training, the JSON file can be imported directly into the JavaScript app:

1. Train with Python: `python train_ai.py --generations 100`
2. Open the JavaScript app
3. Go to "AI Training" tab
4. Click "Import Strategies"
5. Select the `crazyblocks-strategies.json` file

The strategies will load and you can continue training in the browser or export/use them!

## Workflow Examples

### Continuous Training (Build on Previous)

```bash
# First run
python train_ai.py --generations 50

# Later, continue training (automatically loads previous)
python train_ai.py --generations 50

# Keep going...
python train_ai.py --generations 50
```

Each run continues from where the last one left off!

### Long Training Session

```bash
# Train for 200 generations (takes ~1-2 hours)
python train_ai.py --generations 200 --population 100
```

### Quick Test Run

```bash
# Quick 10 generation test
python train_ai.py --generations 10 --games 2
```

## Output Format

The JSON file contains:
- `generation`: Current generation number
- `bestFitness`: Best fitness score
- `bestScore`: Best game score achieved
- `fullPopulation`: All 50 evolved strategies
- `eliteStrategies`: Top 5 best strategies

This format is **100% compatible** with the JavaScript app!

## Tips

1. **Start Small**: Begin with 10-20 generations to see improvement
2. **Let it Run**: Longer training = better strategies
3. **Backup Files**: Copy the JSON file periodically
4. **Use Parallel**: Keep parallel processing enabled for maximum speed
5. **Continue Training**: Each run builds on the previous - no need to start over!

## Troubleshooting

**"No module named X"**: This script uses only Python standard library - no installation needed!

**Slow Performance**: Make sure parallel processing is enabled (default). Use `--no-parallel` only if you have issues.

**File Not Found**: The script will create a new file if one doesn't exist. Make sure you have write permissions.

**Keyboard Interrupt**: Press Ctrl+C to stop training. Progress is saved automatically!

