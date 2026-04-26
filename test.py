"""
evaluate model vs random

loads a .pt checkpoint and plays it against a random-move opponent.
Reports win/loss/draw, Elo estimate, statistical significance, and verdict.

how to use:
    python test.py                                          # latest checkpoint
    python test.py --model checkpoints/model_iter_0050.pt   # specific checkpoint
    python test.py --games 200 --simulations 100            # more games / sims
    python test.py --quick                                  # fast 20-game check
"""

import argparse
import math
import os
import random
import time
import json

import numpy as np
import torch

from config import (
    DEVICE, BLACK, WHITE, BLACK_MAN, BLACK_KING,
    WHITE_MAN, WHITE_KING, TrainingConfig as TC,
    SelfPlayConfig as SP,
)
from checkers_env import CheckersState
from neural_network import NetworkWrapper
from mcts_fast import MCTS


#Players

class RandomPlayer:
    """Picks a uniformly random legal move."""
    def get_move(self, state):
        moves = state.get_legal_moves()
        return random.choice(moves) if moves else None


class AIPlayer:
    """Picks moves via MCTS backed by the neural network."""
    add_noise=False
    def __init__(self, network_wrapper, num_simulations=100):
        self.mcts = MCTS(network_wrapper)
        self.num_sims = num_simulations

    def get_move(self, state):
        _, move, _ = self.mcts.get_action(
            state, temperature=0.1, num_simulations=self.num_sims, add_noise=False
        )
        return move


#Single game

def play_game(ai_player, random_player, ai_color, max_moves=200):
    state = CheckersState()
    num_moves = 0

    while num_moves < max_moves:
        done, winner = state.is_terminal()
        if done:
            break

        if state.current_player == ai_color:
            move = ai_player.get_move(state)
        else:
            move = random_player.get_move(state)

        if move is None:
            winner = -state.current_player
            break

        state = state.apply_move(move)
        num_moves += 1
    else:
        done, winner = state.is_terminal()
        if not done:
            winner = 0

    if ai_color == BLACK:
        ai_pcs = int(np.sum(np.isin(state.board, [BLACK_MAN, BLACK_KING])))
        rnd_pcs = int(np.sum(np.isin(state.board, [WHITE_MAN, WHITE_KING])))
    else:
        ai_pcs = int(np.sum(np.isin(state.board, [WHITE_MAN, WHITE_KING])))
        rnd_pcs = int(np.sum(np.isin(state.board, [BLACK_MAN, BLACK_KING])))

    return {
        "winner": winner,
        "ai_won": winner == ai_color,
        "ai_lost": winner == -ai_color,
        "draw": winner == 0,
        "moves": num_moves,
        "ai_pieces": ai_pcs,
        "rnd_pieces": rnd_pcs,
    }


# Evaluation loop

def evaluate(network_wrapper, num_games, num_simulations):
    ai = AIPlayer(network_wrapper, num_simulations)
    rng = RandomPlayer()

    wins, losses, draws = 0, 0, 0
    lengths, ai_pcs, rnd_pcs = [], [], []

    t0 = time.time()
    for i in range(num_games):
        color = BLACK if i % 2 == 0 else WHITE
        result = play_game(ai, rng, color)

        if result["ai_won"]:
            wins += 1
        elif result["ai_lost"]:
            losses += 1
        else:
            draws += 1
        lengths.append(result["moves"])
        ai_pcs.append(result["ai_pieces"])
        rnd_pcs.append(result["rnd_pieces"])

        done = i + 1
        if done % max(1, num_games // 10) == 0 or done == num_games:
            elapsed = time.time() - t0
            eta = elapsed / done * (num_games - done)
            print(f"  [{done:4d}/{num_games}]  "
                  f"W:{wins}  L:{losses}  D:{draws}  "
                  f"WR:{wins/done:.1%}  "
                  f"ETA:{eta:.0f}s")

    total_time = time.time() - t0
    n = num_games
    win_rate = wins / n
    score = (wins + 0.5 * draws) / n

    # Elo vs random (random = 0)
    if 0 < score < 1:
        elo = -400 * math.log10(1.0 / score - 1.0)
    else:
        elo = float("inf") if score >= 1 else float("-inf")

    # Wilson 95% CI on win rate
    z = 1.96
    denom = 1 + z * z / n
    centre = (win_rate + z * z / (2 * n)) / denom
    margin = z * math.sqrt((win_rate * (1 - win_rate) + z * z / (4 * n)) / n) / denom
    ci_lo = max(0, centre - margin)
    ci_hi = min(1, centre + margin)

    # one-sided z-test: win_rate > 0.5?
    se = math.sqrt(0.25 / n)
    z_stat = (win_rate - 0.5) / se
    p_value = 0.5 * math.erfc(z_stat / math.sqrt(2))

    return {
        "games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "score": score,
        "elo": elo,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "avg_length": float(np.mean(lengths)),
        "avg_ai_pcs": float(np.mean(ai_pcs)),
        "avg_rnd_pcs": float(np.mean(rnd_pcs)),
        "total_time": total_time,
    }


#Pretty-print results
def print_results(r, model_label):
    elo_s = f"{r['elo']:+.0f}" if abs(r["elo"]) < 9999 else ("∞" if r["elo"] > 0 else "-∞")

    print(f"""
{'━' * 58}
  RESULTS  —  {model_label}
{'━' * 58}
  Games played   {r['games']}
  AI Wins        {r['wins']}   ({r['win_rate']:.1%})
  AI Losses      {r['losses']}   ({r['losses']/r['games']:.1%})
  Draws          {r['draws']}   ({r['draws']/r['games']:.1%})
  Score          {r['score']:.3f}
  Win-rate 95%CI [{r['ci_lo']:.1%} - {r['ci_hi']:.1%}]
  Elo vs random  {elo_s}
  Avg game len   {r['avg_length']:.1f} moves
  Avg AI pieces  {r['avg_ai_pcs']:.1f}
  Avg Rnd pieces {r['avg_rnd_pcs']:.1f}
  Time           {r['total_time']:.1f}s  ({r['total_time']/r['games']:.2f}s / game)

  Better than random (p<0.05)?  {'YES ✓' if r['significant'] else 'NO ✗'}   p={r['p_value']:.6f}""")

    wr = r["win_rate"]
    if wr >= 0.95:
        v = "DOMINANT — near-perfect vs random"
    elif wr >= 0.85:
        v = "STRONG — clearly superior"
    elif wr >= 0.70:
        v = "COMPETENT — basic strategy learned"
    elif wr >= 0.55:
        v = "LEARNING — slight edge over random"
    else:
        v = "UNTRAINED — no meaningful advantage"

    print(f"""
  VERDICT: {v}
{'━' * 58}""")


#Entry point

def main():
    ap = argparse.ArgumentParser(description="Evaluate AlphaZero Checkers vs random")
    ap.add_argument("--model", type=str, default=None,
                    help="Path to .pt checkpoint (default: checkpoints/model_latest.pt)")
    ap.add_argument("--games", type=int, default=100,
                    help="Number of games (default 100)")
    ap.add_argument("--simulations", type=int, default=100,
                    help="MCTS sims per move (default 100)")
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: 20 games, 25 sims")
    args = ap.parse_args()

    if args.quick:
        args.games = 20
        args.simulations = 25

    print("=" * 58)
    print("  AlphaZero Checkers — Eval vs Random")
    print(f"  Device : {DEVICE}")
    print(f"  Torch  : {torch.__version__}")
    if hasattr(torch.backends, "mps"):
        print(f"  MPS    : {torch.backends.mps.is_available()}")
    print(f"  Games  : {args.games}")
    print(f"  Sims   : {args.simulations}")
    print("=" * 58)

    # load model
    path = args.model or os.path.join(TC.CHECKPOINT_DIR, "model_latest.pt")
    nnet = NetworkWrapper()

    if os.path.exists(path):
        nnet.load(path)
        label = os.path.basename(path)
    else:
        print(f"\n  ⚠  No model at {path} — using untrained network\n")
        label = "UNTRAINED"

    # run evaluation
    r = evaluate(nnet, args.games, args.simulations)
    print_results(r, label)

    # save json
    os.makedirs(TC.LOG_DIR, exist_ok=True)
    out = os.path.join(TC.LOG_DIR, "eval_results.json")
    r_save = {k: v for k, v in r.items() if not isinstance(v, float) or math.isfinite(v)}
    r_save["model"] = label
    r_save["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out, "w") as f:
        json.dump(r_save, f, indent=2)
    print(f"\n  Results saved -> {out}")


if __name__ == "__main__":
    main()
