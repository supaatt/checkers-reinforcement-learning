"""
AlphaZero Checkers — Main Entry Point
=======================================
Usage:
    python main.py train          # Start training from scratch
    python main.py train --resume # Resume from latest checkpoint
    python main.py play           # Play against AI (loads latest model)
    python main.py play --model checkpoints/model_iter_0050.pt
    python main.py selfplay       # Run self-play only (for testing)
"""

import argparse
import os
import time
import sys

from config import (
    DEVICE, TrainingConfig as TC, SelfPlayConfig as SP,
    MCTSConfig as MC,
)
from neural_network import NetworkWrapper
from self_play import SelfPlayWorker
from trainer import Trainer
from arena import Arena


def train(args):
    print("=" * 60)
    print("  AlphaZero Checkers — Training")
    print(f"  Device: {DEVICE}")
    print(f"  Iterations: {TC.NUM_ITERATIONS}")
    print(f"  Self-play games/iter: {SP.NUM_SELF_PLAY_GAMES}")
    print(f"  MCTS simulations: {MC.NUM_SIMULATIONS}")
    print("=" * 60)

    trainer = Trainer()

    start_iter = 0
    if args.resume:
        if trainer.load_checkpoint():
            for i in range(TC.NUM_ITERATIONS, 0, -1):
                path = os.path.join(TC.CHECKPOINT_DIR, f"model_iter_{i:04d}.pt")
                if os.path.exists(path):
                    start_iter = i
                    break
            print(f"Resuming from iteration {start_iter}")

    best_nnet = NetworkWrapper()
    best_nnet.copy_weights_from(trainer.nnet)

    for iteration in range(start_iter, TC.NUM_ITERATIONS):
        iter_start = time.time()
        print(f"\n{'━' * 60}")
        print(f"  Iteration {iteration + 1}/{TC.NUM_ITERATIONS}")
        print(f"{'━' * 60}")

        # Phase 1: Self-Play
        print(f"\n[Phase 1] Self-play ({SP.NUM_SELF_PLAY_GAMES} games)...")
        worker = SelfPlayWorker(trainer.nnet)
        examples, stats = worker.generate_games(SP.NUM_SELF_PLAY_GAMES, verbose=False)
        print(f"  Generated {len(examples)} examples")
        print(f"  Stats: B={stats['black_wins']} W={stats['white_wins']} D={stats['draws']}")

        # Phase 2: Training
        print(f"\n[Phase 2] Training ({TC.EPOCHS_PER_ITERATION} epochs)...")
        loss_info = trainer.train_iteration(examples)
        if loss_info:
            print(f"  Avg losses: P={loss_info['policy_loss']:.4f} "
                  f"V={loss_info['value_loss']:.4f} "
                  f"T={loss_info['total_loss']:.4f}")

        # Phase 3: Evaluation
        print(f"\n[Phase 3] Evaluation ({TC.EVAL_GAMES} games)...")
        arena = Arena(trainer.nnet, best_nnet, num_simulations=50)
        n1_wins, n2_wins, draws, win_rate = arena.evaluate(
            num_games=TC.EVAL_GAMES, verbose=True
        )

        if win_rate >= TC.WIN_THRESHOLD:
            print(f"  ✓ New model accepted (win rate: {win_rate:.3f})")
            best_nnet.copy_weights_from(trainer.nnet)
        else:
            print(f"  ✗ New model rejected (win rate: {win_rate:.3f}). "
                  f"Keeping previous best.")
            trainer.nnet.copy_weights_from(best_nnet)

        # Checkpoint
        iter_time = time.time() - iter_start
        info = {
            'examples': len(examples),
            'stats': stats,
            'win_rate': win_rate,
            'time': iter_time,
        }
        if loss_info:
            info.update(loss_info)

        if (iteration + 1) % TC.CHECKPOINT_INTERVAL == 0:
            trainer.save_checkpoint(iteration + 1, extra_info=info)

        trainer.save_checkpoint(iteration + 1, extra_info=info)
        print(f"\n  Iteration time: {iter_time:.1f}s")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


def play(args):
    print("=" * 60)
    print("  AlphaZero Checkers — Play vs AI")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    model_path = args.model
    if model_path is None:
        model_path = os.path.join(TC.CHECKPOINT_DIR, "model_latest.pt")
        if not os.path.exists(model_path):
            print("No trained model found. Using random (untrained) network.")
            model_path = None

    sims = args.simulations if args.simulations else 100

    from pygame_gui import CheckersGUI
    gui = CheckersGUI(model_path=model_path, ai_simulations=sims)
    gui.run()


def selfplay_test(args):
    print("Running self-play test...")
    nnet = NetworkWrapper()
    if args.model:
        nnet.load(args.model)

    worker = SelfPlayWorker(nnet)
    num = args.games if args.games else 5
    examples, stats = worker.generate_games(num, verbose=True)
    print(f"\nTotal examples: {len(examples)}")
    print(f"Stats: {stats}")


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Checkers")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--resume', action='store_true',
                              help='Resume from checkpoint')

    play_parser = subparsers.add_parser('play', help='Play against AI')
    play_parser.add_argument('--model', type=str, default=None,
                             help='Model checkpoint path')
    play_parser.add_argument('--simulations', type=int, default=100,
                             help='MCTS simulations')

    sp_parser = subparsers.add_parser('selfplay', help='Run self-play test')
    sp_parser.add_argument('--model', type=str, default=None, help='Model path')
    sp_parser.add_argument('--games', type=int, default=5, help='Number of games')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'play':
        play(args)
    elif args.command == 'selfplay':
        selfplay_test(args)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python main.py train          # Train from scratch")
        print("  python main.py play           # Play vs AI")
        print("  python main.py selfplay       # Test self-play")


if __name__ == "__main__":
    main()
