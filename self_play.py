"""
generate training data through self-play games using MCTS
"""

import numpy as np
from checkers_env import CheckersState
from mcts_fast import MCTS
from config import (
    MCTSConfig as MC, SelfPlayConfig as SP,
    BLACK, WHITE,
)


class SelfPlayWorker:
    def __init__(self, network):
        self.mcts = MCTS(network)

    def play_game(self, verbose=False):
        state = CheckersState()
        training_data = []
        move_num = 0

        while True:
            done, winner = state.is_terminal()
            if done:
                break

            if move_num >= SP.MAX_GAME_LENGTH:
                winner = 0
                break

            if move_num < MC.TEMPERATURE_THRESHOLD:
                temperature = MC.TEMPERATURE_INIT
            else:
                temperature = 0.1

            action_idx, move, pi = self.mcts.get_action(
                state, temperature=temperature
            )

            if move is None:
                winner = -state.current_player
                break

            encoded = state.encode()
            training_data.append((encoded, pi, state.current_player))

            if verbose:
                print(f"Move {move_num}: {'BLACK' if state.current_player == BLACK else 'WHITE'}")
                print(state)
                print(f"  Action: sq{action_idx // 32} -> sq{action_idx % 32}")
                print()

            state = state.apply_move(move)
            move_num += 1

        if verbose:
            result_str = {BLACK: "BLACK wins", WHITE: "WHITE wins", 0: "DRAW"}
            print(f"Game over after {move_num} moves: {result_str.get(winner, 'UNKNOWN')}")

        examples = []
        for encoded, pi, player in training_data:
            if winner == 0:
                value = -0.1 #oringinal 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            examples.append((encoded, pi, value))

        return examples, winner

    def generate_games(self, num_games, verbose=False):
        all_examples = []
        stats = {'black_wins': 0, 'white_wins': 0, 'draws': 0, 'total_moves': 0}

        for i in range(num_games):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Self-play game {i + 1}/{num_games}")
                print(f"{'='*50}")

            examples, winner = self.play_game(verbose=False)
            all_examples.extend(examples)
            stats['total_moves'] += len(examples)

            if winner == BLACK:
                stats['black_wins'] += 1
            elif winner == WHITE:
                stats['white_wins'] += 1
            else:
                stats['draws'] += 1

            if (i + 1) % 10 == 0 or verbose:
                print(f"  Game {i+1}/{num_games} done | "
                      f"B:{stats['black_wins']} W:{stats['white_wins']} "
                      f"D:{stats['draws']} | Examples: {len(all_examples)}")

        return all_examples, stats
