# checkers-reinforcement-learning

for BIA Data Associate Programme

A self-learning checkers AI built from scratch, inspired by AlphaZero.
Combines Monte Carlo Tree Search with a residual neural network that learns purely
through self-play. No human game data. No hand-crafted heuristics.

## Architecture 

```

                  Training Loop                  
                                                             
   ┌──────────┐    ┌──────────┐    ┌──────────┐               
   │ Self-Play│───▶│ Training │───▶│  Arena   │──▶ Best Model 
   │  (MCTS)  │    │ (ResNet) │    │  (Eval)  │               
   └──────────┘    └──────────┘    └──────────┘               
        │                                                      
        ▼                                                      
   Replay Buffer                                               

```

### Neural Network
- **Input**: 5 planes × 8×8 (own men, own kings, opponent men, opponent kings, turn)
- **Body**: 10 residual blocks, 128 filters each
- **Policy Head**: 1024 outputs (32×32 source-destination pairs)
- **Value Head**: single scalar in [-1, +1]

### MCTS
- PUCT exploration with neural network priors
- Dirichlet noise at root for exploration (self-play only)
- Temperature annealing for move selection

## Project Structure

```
checkers_alphazero/
├── config.py            # All hyperparameters & settings
├── checkers_env.py      # Full checkers game engine
├── neural_network.py    # AlphaZero dual-head ResNet
├── mcts.py              # Monte Carlo Tree Search
├── self_play.py         # Self-play game generation
├── trainer.py           # Training loop with replay buffer
├── arena.py             # Model evaluation (new vs old)
├── pygame_gui.py        # Pygame interface to play vs AI
├── main.py              # Entry point (train or play)
├── test.py              # Evaluate model vs random
├── experiment.py        # MCTS correctness tests
└── requirements.txt     # Dependencies
```

## Setup

```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify MPS
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

### Prevent laptop sleep during training (macOS)
```bash
caffeinate python main.py train --resume
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
### Test model vs random
```bash
# Test latest checkpoint (100 games, 100 sims)
python test.py

# Test specific checkpoint
python test.py --model checkpoints/model_iter_0010.pt

# Test policy head only (no search)
python test.py --simulations 0 --games 200

# Test at high simulation count
python test.py --games 50 --simulations 200


```
### Test model vs other models
```bash
python -c "
from neural_network import NetworkWrapper
from arena import Arena

# Load the two models you want to compare
net_a = NetworkWrapper()
net_a.load('checkpoints/model_iter_0008.pt')

net_b = NetworkWrapper()
net_b.load('checkpoints/6th.pt')      

# Pit them against each other
arena = Arena(net_a, net_b, num_simulations=100)
arena.evaluate(num_games=40, verbose=True)
"
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
| EVAL_GAMES | 40 | Arena evaluation games |
| WIN_THRESHOLD | 0.55 | Win rate to accept new model |

### Lightweight settings for fast training

```python
# In config.py
class MCTSConfig:
    NUM_SIMULATIONS = 100

class SelfPlayConfig:
    NUM_SELF_PLAY_GAMES = 30

class TrainingConfig:
    EVAL_GAMES = 20
```

### Stronger but slow

```python
class MCTSConfig:
    NUM_SIMULATIONS = 400

class SelfPlayConfig:
    NUM_SELF_PLAY_GAMES = 150
```

## How It Works

1. **Self-Play**: The current best network plays games against itself using MCTS.
   Each position stores (board state, MCTS visit distribution, game outcome).

2. **Training**: The neural network trains on these examples:
   - Policy loss: cross-entropy between network policy and MCTS visits
   - Value loss: MSE between predicted value and actual game outcome

3. **Evaluation**: The newly trained network plays against the previous best.
   If it wins >55% of decisive games, it becomes new best.

4. **Repeat**: Each iteration generates more self-play data, trains, and evaluates.
