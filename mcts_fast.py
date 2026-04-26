"""
C++ MCTS Wrapper

replacement for mcts.py that uses the C++ backend.
Same interface: MCTS.search(), MCTS.get_action()
Usage:
    # Build first:
    #   pip install pybind11
    #   python setup_cpp.py build_ext --inplace
    #
    # change the import to change python MCTS to C++:
    #   from mcts_fast import MCTS 
    #  
"""

import numpy as np

try:
    import mcts_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("WARNING: mcts_cpp not found. Build it with:")
    print("  pip install pybind11")
    print("  python setup_cpp.py build_ext --inplace")
    print("Falling back to pure Python MCTS.")

from config import MCTSConfig as MC, NetworkConfig as NC


class StateRegistry:
    """
    Maps integer IDs to CheckersState objects.
    C++ refers to states by ID, Python looks them up here.
    Cleared after each search call.
    """

    def __init__(self):
        self.states = {}
        self.next_id = 0

    def register(self, state):
        sid = self.next_id
        self.states[sid] = state
        self.next_id += 1
        return sid

    def get(self, sid):
        return self.states[sid]

    def clear(self):
        self.states.clear()
        self.next_id = 0


class MCTS:
    """
    Drop-in replacement for the pure Python MCTS.
    Uses C++ for tree search, Python for game logic and neural network.
    """

    def __init__(self, network, config=None):
        self.network = network
        self.config = config or MC()
        self.registry = StateRegistry()

        if not CPP_AVAILABLE:
            # Fall back to pure Python
            from mcts import MCTS as PythonMCTS
            self._fallback = PythonMCTS(network, config)
        else:
            self._fallback = None

    def _make_expand_fn(self):
        """
        Create the callback that C++ calls to expand a node.
        Returns a function: state_id -> dict
        """
        registry = self.registry
        network = self.network

        def expand_fn(state_id):
            state = registry.get(state_id)

            # Check terminal
            done, winner = state.is_terminal()
            if done:
                if winner == 0:
                    terminal_value = 0.0
                else:
                    terminal_value = 1.0 if winner == state.current_player else -1.0
                return {
                    "is_terminal": True,
                    "terminal_value": terminal_value,
                    "action_indices": [],
                    "child_state_ids": [],
                    "priors": [],
                    "value": 0.0,
                }

            # Get legal moves
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                return {
                    "is_terminal": True,
                    "terminal_value": -1.0,  # no moves = loss
                    "action_indices": [],
                    "child_state_ids": [],
                    "priors": [],
                    "value": 0.0,
                }

            # Neural network evaluation
            encoded = state.encode()
            p_logits, value = network.predict(encoded)

            # Mask illegal moves and softmax
            legal_mask = state.get_legal_move_mask()
            p_logits[legal_mask == 0] = -1e9
            p_logits = p_logits - np.max(p_logits)
            exp_p = np.exp(p_logits)
            priors = exp_p / (np.sum(exp_p) + 1e-8)

            # Create child states and register them
            action_indices = []
            child_state_ids = []
            child_priors = []

            for move in legal_moves:
                idx = state.move_to_index(move)
                child_state = state.apply_move(move)
                child_sid = registry.register(child_state)

                action_indices.append(idx)
                child_state_ids.append(child_sid)
                child_priors.append(float(priors[idx]))

            return {
                "is_terminal": False,
                "terminal_value": 0.0,
                "action_indices": action_indices,
                "child_state_ids": child_state_ids,
                "priors": child_priors,
                "value": float(value),
            }

        return expand_fn

    def search(self, state, num_simulations=None, add_noise=True):
        """
        Run MCTS from the given state.
        Returns: numpy array of shape (1024,) — move visit probabilities.
        """
        if self._fallback:
            return self._fallback.search(state, num_simulations, add_noise)

        if num_simulations is None:
            num_simulations = MC.NUM_SIMULATIONS

        # Clear registry and register root state
        self.registry.clear()
        root_sid = self.registry.register(state)

        # Create expansion callback
        expand_fn = self._make_expand_fn()

        # Run C++ search
        pi = mcts_cpp.search(
            root_state_id=root_sid,
            num_simulations=num_simulations,
            add_noise=add_noise,
            expand_fn=expand_fn,
            c_puct=MC.C_PUCT,
            dirichlet_alpha=MC.DIRICHLET_ALPHA,
            dirichlet_epsilon=MC.DIRICHLET_EPSILON,
            policy_size=NC.POLICY_SIZE,
        )

        return np.array(pi, dtype=np.float32)

    def get_action(self, state, temperature=1.0, num_simulations=None, add_noise=True):
        """
        Run MCTS and select an action. Same interface as pure Python version.
        Returns: (action_index, move, pi_distribution)
        """
        if self._fallback:
            return self._fallback.get_action(state, temperature, num_simulations, add_noise)

        pi = self.search(state, num_simulations=num_simulations, add_noise=add_noise)

        if temperature == 0:
            action = np.argmax(pi)
        else:
            adjusted = np.zeros_like(pi)
            nonzero = pi > 0
            if not np.any(nonzero):
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

            adjusted = adjusted / adjusted.sum()
            action = np.random.choice(len(adjusted), p=adjusted)

        move = state.get_move_from_index(action)
        return action, move, pi
