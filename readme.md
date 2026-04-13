# AlphaZero Checkers

A full AlphaZero-style reinforcement learning system for American checkers (8×8),
with Apple MPS acceleration and a Pygame GUI.

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    AlphaZero Training Loop                  │
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │ Self-Play│───▶│ Training │───▶│  Arena   │──▶ Best Model│
│   │  (MCTS)  │    │ (ResNet) │    │  (Eval)  │              │
│   └──────────┘    └──────────┘    └──────────┘              │
│        │                                                    │
│        ▼                                                    │
│   Replay Buffer                                             │
└─────────────────────────────────────────────────────────────┘
```

### Neural Network
- **Input**: 5 planes × 8×8 (own men, own kings, opponent men, opponent kings, turn)
- **Body**: 10 residual blocks, 128 filters each
- **Policy Head**: 1024 outputs (32×32 source-destination pairs)
- **Value Head**: single scalar in [-1, +1]

### MCTS
- PUCT exploration with neural network priors
- Dirichlet noise at root for exploration
- Temperature annealing for move selection

## Setup
```bash
# Create environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify MPS is available (macOS)
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

## Usage

### Train from scratch
```bash
python main.py train
```

### Resume training
```bash
python main.py train --resume
```

### Play against AI
```bash
# With latest trained model
python main.py play

# With specific checkpoint
python main.py play --model checkpoints/model_iter_0050.pt

# Adjust difficulty (MCTS simulations)
python main.py play --simulations 400
```

### Test self-play
```bash
python main.py selfplay --games 10
```

## Configuration

All hyperparameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| NUM_RES_BLOCKS | 10 | Residual blocks in network |
| NUM_FILTERS | 128 | Channels per conv layer |
| NUM_SIMULATIONS | 200 | MCTS sims per move |
| C_PUCT | 1.5 | Exploration constant |
| LEARNING_RATE | 0.001 | Adam LR |
| NUM_SELF_PLAY_GAMES | 100 | Games per iteration |
| REPLAY_BUFFER_SIZE | 50,000 | Max training examples |
| EVAL_GAMES | 20 | Arena evaluation games |

### Adjusting for your hardware

For **faster training** on Apple Silicon:
```python
# In config.py
NUM_RES_BLOCKS = 6       # Fewer blocks
NUM_FILTERS = 64          # Fewer filters
NUM_SIMULATIONS = 100     # Fewer MCTS sims
NUM_SELF_PLAY_GAMES = 50  # Fewer games per iter
```

For **stronger play** (longer training):
```python
NUM_RES_BLOCKS = 15
NUM_FILTERS = 256
NUM_SIMULATIONS = 400
NUM_SELF_PLAY_GAMES = 200
```

## Pygame Controls

| Key/Action | Effect |
|------------|--------|
| Click piece → click destination | Make a move |
| N | New game |
| U | Undo (human + AI move) |
| Q | Quit |
| Difficulty buttons | Easy/Medium/Hard/Expert |

## How It Works

1. **Self-Play**: The current best network plays games against itself using MCTS.
   Each position stores (board state, MCTS visit distribution, game outcome).

2. **Training**: The neural network trains on these examples:
   - Policy loss: cross-entropy between network policy and MCTS visits
   - Value loss: MSE between predicted value and actual game outcome

3. **Evaluation**: The newly trained network plays against the previous best.
   If it wins >55% of games, it becomes the new best.

4. **Repeat**: Each iteration generates more self-play data, trains, and evaluates.