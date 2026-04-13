"""
AlphaZero Checkers — Neural Network
=====================================
Dual-head residual network:
  - Shared residual tower
  - Policy head  -> probability distribution over moves
  - Value head   -> scalar evaluation [-1, +1]

Optimized for Apple MPS, CUDA, or CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from config import DEVICE, NetworkConfig as NC


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class AlphaZeroNet(nn.Module):
    """
    Input:  (batch, 5, 8, 8)
    Output: policy (batch, 1024), value (batch, 1)
    """
    def __init__(self):
        super().__init__()

        self.input_conv = ConvBlock(NC.NUM_INPUT_PLANES, NC.NUM_FILTERS)
        self.res_blocks = nn.ModuleList([
            ResBlock(NC.NUM_FILTERS) for _ in range(NC.NUM_RES_BLOCKS)
        ])

        # Policy head
        self.policy_conv = ConvBlock(NC.NUM_FILTERS, 32, kernel_size=1, padding=0)
        self.policy_fc = nn.Linear(32 * NC.BOARD_H * NC.BOARD_W, NC.POLICY_SIZE)

        # Value head
        self.value_conv = ConvBlock(NC.NUM_FILTERS, 1, kernel_size=1, padding=0)
        self.value_fc1 = nn.Linear(NC.BOARD_H * NC.BOARD_W, NC.VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(NC.VALUE_HIDDEN, 1)

    def forward(self, x):
        out = self.input_conv(x)
        for block in self.res_blocks:
            out = block(out)

        p = self.policy_conv(out)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        v = self.value_conv(out)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, state_tensor):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state_tensor).unsqueeze(0).to(DEVICE)
            p_logits, v = self(x)
            p_logits = p_logits.squeeze(0).cpu().numpy()
            v = v.item()
        return p_logits, v


class NetworkWrapper:
    """
    High-level wrapper: training, saving, loading, inference.
    """
    def __init__(self, lr=NC.LEARNING_RATE):
        self.net = AlphaZeroNet().to(DEVICE)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=lr,
            weight_decay=NC.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=NC.LR_MILESTONES,
            gamma=NC.LR_GAMMA,
        )

    def predict(self, encoded_state):
        return self.net.predict(encoded_state)

    def train_batch(self, states, target_pis, target_vs):
        self.net.train()

        states_t = torch.FloatTensor(states).to(DEVICE)
        pis_t = torch.FloatTensor(target_pis).to(DEVICE)
        vs_t = torch.FloatTensor(target_vs).unsqueeze(1).to(DEVICE)

        p_logits, v_pred = self.net(states_t)

        log_probs = F.log_softmax(p_logits, dim=1)
        policy_loss = -torch.sum(pis_t * log_probs, dim=1).mean()
        value_loss = F.mse_loss(v_pred, vs_t)
        total_loss = policy_loss + 0.5 * value_loss #original no 0.5
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), total_loss.item()

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Model loaded from {filepath}")

    def copy_weights_from(self, other):
        self.net.load_state_dict(other.net.state_dict())
