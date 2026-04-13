"""
training pipeline

manage replay buffer, training epochs, and iteration loop.
"""

import numpy as np
import os
import time
import json
from collections import deque

from config import TrainingConfig as TC, DEVICE
from neural_network import NetworkWrapper


class ReplayBuffer:
    def __init__(self, max_size=TC.REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def add(self, examples):
        self.buffer.extend(examples)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([b[0] for b in batch])
        pis = np.array([b[1] for b in batch])
        values = np.array([b[2] for b in batch])
        return states, pis, values

    def __len__(self):
        return len(self.buffer)


class Trainer:
    def __init__(self, network_wrapper=None):
        self.nnet = network_wrapper or NetworkWrapper()
        self.replay_buffer = ReplayBuffer()
        self.training_log = []

        os.makedirs(TC.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(TC.LOG_DIR, exist_ok=True)

    def train_epoch(self):
        if len(self.replay_buffer) < TC.MIN_REPLAY_SIZE:
            return None

        num_batches = max(1, len(self.replay_buffer) // TC.BATCH_SIZE)
        total_p_loss = 0
        total_v_loss = 0
        total_loss = 0

        for _ in range(num_batches):
            batch_size = min(TC.BATCH_SIZE, len(self.replay_buffer))
            states, pis, values = self.replay_buffer.sample(batch_size)
            p_loss, v_loss, t_loss = self.nnet.train_batch(states, pis, values)
            total_p_loss += p_loss
            total_v_loss += v_loss
            total_loss += t_loss

        avg_p = total_p_loss / num_batches
        avg_v = total_v_loss / num_batches
        avg_t = total_loss / num_batches

        return {'policy_loss': avg_p, 'value_loss': avg_v, 'total_loss': avg_t}

    def train_iteration(self, examples):
        self.replay_buffer.add(examples)
        print(f"  Replay buffer size: {len(self.replay_buffer)}")

        if len(self.replay_buffer) < TC.MIN_REPLAY_SIZE:
            print(f"  Not enough data ({len(self.replay_buffer)}/{TC.MIN_REPLAY_SIZE}). Skipping training.")
            return None

        epoch_losses = []
        for epoch in range(TC.EPOCHS_PER_ITERATION):
            loss_info = self.train_epoch()
            if loss_info:
                epoch_losses.append(loss_info)
                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch {epoch+1}/{TC.EPOCHS_PER_ITERATION}: "
                          f"P={loss_info['policy_loss']:.4f} "
                          f"V={loss_info['value_loss']:.4f} "
                          f"T={loss_info['total_loss']:.4f}")

        self.nnet.scheduler.step()

        if epoch_losses:
            avg_losses = {
                'policy_loss': np.mean([e['policy_loss'] for e in epoch_losses]),
                'value_loss': np.mean([e['value_loss'] for e in epoch_losses]),
                'total_loss': np.mean([e['total_loss'] for e in epoch_losses]),
            }
            return avg_losses
        return None

    def save_checkpoint(self, iteration, extra_info=None):
        filepath = os.path.join(TC.CHECKPOINT_DIR, f"model_iter_{iteration:04d}.pt")
        self.nnet.save(filepath)

        latest_path = os.path.join(TC.CHECKPOINT_DIR, "model_latest.pt")
        self.nnet.save(latest_path)

        if extra_info:
            self.training_log.append({'iteration': iteration, **extra_info})
            log_path = os.path.join(TC.LOG_DIR, "training_log.json")
            with open(log_path, 'w') as f:
                json.dump(self.training_log, f, indent=2)

    def load_checkpoint(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(TC.CHECKPOINT_DIR, "model_latest.pt")
        if os.path.exists(filepath):
            self.nnet.load(filepath)
            return True
        print(f"no checkpoint found at {filepath}")
        return False
