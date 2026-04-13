"""
AlphaZero Checkers — Monte Carlo Tree Search
==============================================
PUCT-based tree search with neural network evaluations.
"""

import math
import numpy as np
from config import MCTSConfig as MC, NetworkConfig as NC


class MCTSNode:
    __slots__ = ['state', 'parent', 'parent_action_idx', 'children',
                 'visit_count', 'value_sum', 'prior', 'is_expanded']

    def __init__(self, state, parent=None, parent_action_idx=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.parent_action_idx = parent_action_idx
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits, c_puct=MC.C_PUCT):
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def best_child(self):
        best_score = -float('inf')
        best_child_node = None
        for child in self.children.values():
            score = child.ucb_score(self.visit_count)
            if score > best_score:
                best_score = score
                best_child_node = child
        return best_child_node

    def is_leaf(self):
        return not self.is_expanded


class MCTS:
    def __init__(self, network, config=None):
        self.network = network
        self.config = config or MC()

    def search(self, state, num_simulations=None, add_noise=True):
        if num_simulations is None:
            num_simulations = MC.NUM_SIMULATIONS

        root = MCTSNode(state)
        self._expand(root)

        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            # SELECT
            while not node.is_leaf() and node.children:
                node = node.best_child()
                search_path.append(node)

            # EVALUATE
            done, winner = node.state.is_terminal()
            if done:
                if winner == 0:
                    value = 0.0
                else:
                    value = 1.0 if winner == node.state.current_player else -1.0
            else:
                # EXPAND
                self._expand(node)
                encoded = node.state.encode()
                _, value = self.network.predict(encoded)

            # BACKPROPAGATE
            self._backpropagate(search_path, value, node.state.current_player)

        return self._get_visit_distribution(root)

    def _expand(self, node):
        state = node.state
        done, _ = state.is_terminal()
        if done:
            node.is_expanded = True
            return

        legal_moves = state.get_legal_moves()
        if not legal_moves:
            node.is_expanded = True
            return

        encoded = state.encode()
        p_logits, _ = self.network.predict(encoded)

        legal_mask = state.get_legal_move_mask()
        p_logits[legal_mask == 0] = -1e9
        p_logits = p_logits - np.max(p_logits)
        exp_p = np.exp(p_logits)
        priors = exp_p / (np.sum(exp_p) + 1e-8)

        for move in legal_moves:
            idx = state.move_to_index(move)
            child_state = state.apply_move(move)
            child = MCTSNode(child_state, parent=node,
                             parent_action_idx=idx, prior=priors[idx])
            node.children[idx] = child

        node.is_expanded = True

    def _add_dirichlet_noise(self, root):
        actions = list(root.children.keys())
        noise = np.random.dirichlet([MC.DIRICHLET_ALPHA] * len(actions))
        eps = MC.DIRICHLET_EPSILON
        for i, action in enumerate(actions):
            root.children[action].prior = (
                (1 - eps) * root.children[action].prior + eps * noise[i]
            )

    def _backpropagate(self, search_path, value, eval_player):
        for node in reversed(search_path):
            if node.state.current_player != eval_player:
                node.value_sum += value
            else:
                node.value_sum -= value
            node.visit_count += 1

    def _get_visit_distribution(self, root):
        visits = np.zeros(NC.POLICY_SIZE, dtype=np.float32)
        for action_idx, child in root.children.items():
            visits[action_idx] = child.visit_count
        total = np.sum(visits)
        if total > 0:
            visits /= total
        return visits

    def get_action(self, state, temperature=1.0, num_simulations=None, add_noise = True):
        pi = self.search(state, num_simulations=num_simulations)

        if temperature == 0:
            action = np.argmax(pi)
        else:
            # Stable exponentiation: clamp to avoid overflow/underflow
            adjusted = np.zeros_like(pi)
            nonzero = pi > 0
            if not np.any(nonzero):
                # Fallback: pick uniformly from legal moves
                legal_mask = state.get_legal_move_mask()
                adjusted = legal_mask / (np.sum(legal_mask) + 1e-12)
            else:
                adjusted[nonzero] = pi[nonzero] ** (1.0 / temperature)
                total = np.sum(adjusted)
                if total <= 0:
                    legal_mask = state.get_legal_move_mask()
                    adjusted = legal_mask / (np.sum(legal_mask) + 1e-12)
                else:
                    adjusted /= total

            # Final safety: ensure exact sum to 1.0 for numpy
            adjusted = adjusted / adjusted.sum()
            action = np.random.choice(len(adjusted), p=adjusted)

        move = state.get_move_from_index(action)
        return action, move, pi