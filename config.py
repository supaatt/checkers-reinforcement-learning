"""
Config
"""

import torch

def get_device():
    """Select available device: MPS (Apple Silicon) > CUDA > CPU."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# Board
BOARD_SIZE = 8
NUM_SQUARES = 32

# Piece types
EMPTY     = 0
BLACK_MAN = 1
BLACK_KING= 2
WHITE_MAN = 3
WHITE_KING= 4

# Players
BLACK =  1   # moves first
WHITE = -1

#Neural Network
class NetworkConfig:
    NUM_INPUT_PLANES = 5   # black_men, black_kings, white_men, white_kings, turn
    BOARD_H = 8
    BOARD_W = 8
    NUM_RES_BLOCKS = 10
    NUM_FILTERS = 128
    POLICY_SIZE = 32 * 32   # (src_sq, dst_sq) pairs on 32 playable squares
    VALUE_HIDDEN = 256
    # LEARNING_RATE = 0.001 #original
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    LR_MILESTONES = [100, 200, 300]
    LR_GAMMA = 0.1

#MCTS
class MCTSConfig:
    NUM_SIMULATIONS = 150
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25
    TEMPERATURE_THRESHOLD = 15
    TEMPERATURE_INIT = 1.0

#Self-Play
class SelfPlayConfig:
    NUM_SELF_PLAY_GAMES = 50
    MAX_GAME_LENGTH = 200

#Training
class TrainingConfig:
    NUM_ITERATIONS = 100
    EPOCHS_PER_ITERATION = 10
    BATCH_SIZE = 64
    REPLAY_BUFFER_SIZE = 50_000
    MIN_REPLAY_SIZE = 512
    CHECKPOINT_INTERVAL = 5
    EVAL_GAMES = 40             # bumped from 20 for finer granularity
    WIN_THRESHOLD = 0.58
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"

#Pygame GUI
class GUIConfig:
    WINDOW_W = 1000
    BOARD_PX = 720
    SQUARE_SIZE = BOARD_PX // BOARD_SIZE
    INFO_PANEL_W = WINDOW_W - BOARD_PX
    WINDOW_H = BOARD_PX
    FPS = 60
    # Colors
    BG_COLOR         = (30, 30, 30)
    LIGHT_SQUARE     = (232, 213, 183)
    DARK_SQUARE      = (101, 67, 33)
    HIGHLIGHT_COLOR  = (50, 205, 50)
    VALID_MOVE_COLOR = (0, 180, 255)
    LAST_MOVE_FROM   = (200, 200, 50)
    LAST_MOVE_TO     = (255, 255, 80)
    BLACK_PIECE      = (20, 20, 20)
    BLACK_BORDER     = (80, 80, 80)
    WHITE_PIECE      = (240, 230, 210)
    WHITE_BORDER     = (180, 170, 150)
    KING_CROWN       = (255, 215, 0)
    PIECE_RADIUS     = SQUARE_SIZE // 2 - 8
    PIECE_BORDER_W   = 3
    FONT_SIZE        = 18
    TITLE_FONT_SIZE  = 28
