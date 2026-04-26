#pragma once

#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <functional>
#include <algorithm>
#include <numeric>

//Node

struct MCTSNode {
    int state_id;
    int parent_action_idx;
    float prior;
    int visit_count;
    float value_sum;
    bool is_expanded;
    bool is_terminal;
    float terminal_value;  // only valid if is_terminal

    std::unordered_map<int, MCTSNode*> children;

    MCTSNode(int sid, int action_idx = -1, float p = 0.0f)
        : state_id(sid), parent_action_idx(action_idx), prior(p),
          visit_count(0), value_sum(0.0f), is_expanded(false),
          is_terminal(false), terminal_value(0.0f) {}

    ~MCTSNode() {
        for (auto& [idx, child] : children) {
            delete child;
        }
    }

    float q_value() const {
        if (visit_count == 0) return 0.0f;
        return value_sum / visit_count;
    }

    float ucb_score(int parent_visits, float c_puct) const {
        float exploration = c_puct * prior * std::sqrt((float)parent_visits) / (1.0f + visit_count);
        return q_value() + exploration;
    }

    MCTSNode* best_child(float c_puct) {
        MCTSNode* best = nullptr;
        float best_score = -1e9f;
        for (auto& [idx, child] : children) {
            float score = child->ucb_score(visit_count, c_puct);
            if (score > best_score) {
                best_score = score;
                best = child;
            }
        }
        return best;
    }
};

//Expansion result from Python

struct ExpansionResult {
    bool is_terminal;
    float terminal_value;  // +1 win for current player, -1 loss, 0 draw
    std::vector<int> action_indices;
    std::vector<int> child_state_ids;
    std::vector<float> priors;
    float value;  // neural net value estimate
};

// Callback type: Python function that expands a state
// Takes state_id, returns ExpansionResult
using ExpandFunc = std::function<ExpansionResult(int)>;

//Search

class MCTSSearch {
public:
    float c_puct;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    int policy_size;

    MCTSSearch(float c = 1.5f, float d_alpha = 0.3f, float d_eps = 0.25f, int psize = 1024)
        : c_puct(c), dirichlet_alpha(d_alpha), dirichlet_epsilon(d_eps),
          policy_size(psize), rng(std::random_device{}()) {}

    // Main search: returns visit distribution of shape
    std::vector<float> search(int root_state_id, int num_simulations,
                               bool add_noise, ExpandFunc expand_fn) {

        MCTSNode root(root_state_id);

        // Expand root
        expand_node(&root, expand_fn);

        // Add Dirichlet noise to root
        if (add_noise && !root.children.empty()) {
            add_dirichlet_noise(&root);
        }

        // Run simulations
        for (int sim = 0; sim < num_simulations; sim++) {
            MCTSNode* node = &root;
            std::vector<MCTSNode*> search_path;
            search_path.push_back(node);

            // SELECT: walk down the tree
            while (node->is_expanded && !node->children.empty()) {
                node = node->best_child(c_puct);
                search_path.push_back(node);
            }

            float value;

            if (node->is_terminal) {
                value = node->terminal_value;
            } else {
                // EXPAND
                expand_node(node, expand_fn);
                value = node->is_terminal ? node->terminal_value
                                          : last_expand_value;
            }

            // BACKPROPAGATE
            // value is from the perspective of node's current player
            // We use the != convention: positive for the parent who chose this node
            backpropagate(search_path, value);
        }

        // Extract visit distribution
        return get_visit_distribution(&root);
    }

private:
    std::mt19937 rng;
    float last_expand_value = 0.0f;

    void expand_node(MCTSNode* node, ExpandFunc& expand_fn) {
        if (node->is_expanded) return;

        ExpansionResult result = expand_fn(node->state_id);

        if (result.is_terminal) {
            node->is_terminal = true;
            node->terminal_value = result.terminal_value;
            node->is_expanded = true;
            last_expand_value = result.terminal_value;
            return;
        }

        // Create children
        for (size_t i = 0; i < result.action_indices.size(); i++) {
            int action_idx = result.action_indices[i];
            int child_sid = result.child_state_ids[i];
            float prior = result.priors[i];

            MCTSNode* child = new MCTSNode(child_sid, action_idx, prior);
            node->children[action_idx] = child;
        }

        node->is_expanded = true;
        last_expand_value = result.value;
    }

    void backpropagate(std::vector<MCTSNode*>& search_path, float value) {
        // We backpropagate with alternating signs.
        // The leaf's value is from its current_player's perspective.
        // Walking up: each node alternates perspective.
        // Using sign flip: start negative (leaf's parent benefits from opponent's loss)
        // then flip each step.

        // The node at the END of search_path is the leaf.
        // Its value is from its own player's perspective.
        // For the != convention: the leaf itself gets -value,
        // its parent gets +value, grandparent gets -value, etc.

        float v = -value;  // leaf node: current_player != eval_player for the leaf itself
        for (int i = (int)search_path.size() - 1; i >= 0; i--) {
            search_path[i]->value_sum += v;
            search_path[i]->visit_count += 1;
            v = -v;  // flip for alternating players
        }
    }

    void add_dirichlet_noise(MCTSNode* root) {
        std::vector<int> actions;
        for (auto& [idx, child] : root->children) {
            actions.push_back(idx);
        }

        // Sample Dirichlet noise
        std::vector<float> noise(actions.size());
        std::gamma_distribution<float> gamma(dirichlet_alpha, 1.0f);
        float noise_sum = 0.0f;
        for (size_t i = 0; i < noise.size(); i++) {
            noise[i] = gamma(rng);
            noise_sum += noise[i];
        }
        if (noise_sum > 0) {
            for (auto& n : noise) n /= noise_sum;
        }

        // Mix noise with priors
        float eps = dirichlet_epsilon;
        for (size_t i = 0; i < actions.size(); i++) {
            MCTSNode* child = root->children[actions[i]];
            child->prior = (1.0f - eps) * child->prior + eps * noise[i];
        }
    }

    std::vector<float> get_visit_distribution(MCTSNode* root) {
        std::vector<float> visits(policy_size, 0.0f);
        float total = 0.0f;
        for (auto& [action_idx, child] : root->children) {
            if (action_idx >= 0 && action_idx < policy_size) {
                visits[action_idx] = (float)child->visit_count;
                total += child->visit_count;
            }
        }
        if (total > 0) {
            for (auto& v : visits) v /= total;
        }
        return visits;
    }
};
