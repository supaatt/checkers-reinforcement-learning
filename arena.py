"""
arena (Model Evaluation)
set two models against each other to decide which is stronger

"""

import numpy as np
from checkers_env import CheckersState
from mcts import MCTS
from config import TrainingConfig as TC, SelfPlayConfig as SP, BLACK, WHITE


class Arena:
    def __init__(self, net1, net2, num_simulations=50):
        """
        net1: the challenger (new model)
        net2: the current best model
        """
        self.mcts1 = MCTS(net1)
        self.mcts2 = MCTS(net2)
        self.num_sims = num_simulations

    def play_game(self, net1_is_black=True):
        """
        Play one evaluation game.
        Returns: +1 if net1 wins, -1 if net2 wins, 0 for draw.
        """
        state = CheckersState()

        for move_num in range(SP.MAX_GAME_LENGTH):
            done, winner = state.is_terminal()
            if done:
                break

            if state.current_player == BLACK:
                mcts = self.mcts1 if net1_is_black else self.mcts2
            else:
                mcts = self.mcts2 if net1_is_black else self.mcts1

            _, move, _ = mcts.get_action(
                state, temperature=0, num_simulations=self.num_sims, add_noise = False
            )

            if move is None:
                winner = -state.current_player
                break

            state = state.apply_move(move)
        else:
            winner = 0

        if winner == 0:
            return 0
        if net1_is_black:
            return 1 if winner == BLACK else -1
        else:
            return 1 if winner == WHITE else -1

    def evaluate(self, num_games=None, verbose=True):
        """
        Play multiple games, alternating colors.
        Returns: (net1_wins, net2_wins, draws, win_rate)

        win_rate is computed as:
            wins / (wins + losses)   among decisive games only
        This avoids the 0.25 quantization bug when most games are draws.
        If ALL games are draws, win_rate defaults to 0.5 (no signal).
        """
        if num_games is None:
            num_games = TC.EVAL_GAMES

        net1_wins = 0
        net2_wins = 0
        draws = 0

        for i in range(num_games):
            net1_is_black = (i % 2 == 0)
            result = self.play_game(net1_is_black=net1_is_black)

            if result == 1:
                net1_wins += 1
            elif result == -1:
                net2_wins += 1
            else:
                draws += 1

            if verbose and (i + 1) % 5 == 0:
                print(f"  Eval game {i+1}/{num_games}: "
                      f"New={net1_wins} Old={net2_wins} Draw={draws}")

        #FIXED win rate: only count decisive games
        decisive = net1_wins + net2_wins
        if decisive > 0:
            win_rate = net1_wins / decisive
        else:
            # All draws → no signal, default to 0.5 (reject new model)
            win_rate = 0.5

        if verbose:
            print(f"  Final: New={net1_wins} Old={net2_wins} Draw={draws} "
                  f"Decisive={decisive} WinRate={win_rate:.3f}")

        return net1_wins, net2_wins, draws, win_rate
