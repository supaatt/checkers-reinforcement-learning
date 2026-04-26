#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "mcts_engine.h"

namespace py = pybind11;

// Wrapper that converts Python callback to C++ ExpansionResult
// Python callback signature:
//   expand_fn(state_id: int) -> dict with keys:
//     is_terminal: bool
//     terminal_value: float
//     action_indices: list[int]
//     child_state_ids: list[int]
//     priors: list[float]
//     value: float

ExpansionResult py_expand_to_cpp(py::function py_fn, int state_id) {
    py::dict result = py_fn(state_id).cast<py::dict>();

    ExpansionResult er;
    er.is_terminal = result["is_terminal"].cast<bool>();
    er.terminal_value = result["terminal_value"].cast<float>();

    if (!er.is_terminal) {
        er.action_indices = result["action_indices"].cast<std::vector<int>>();
        er.child_state_ids = result["child_state_ids"].cast<std::vector<int>>();
        er.priors = result["priors"].cast<std::vector<float>>();
        er.value = result["value"].cast<float>();
    }

    return er;
}


// Main search function exposed to Python
py::array_t<float> mcts_search(
    int root_state_id,
    int num_simulations,
    bool add_noise,
    py::function expand_fn,
    float c_puct,
    float dirichlet_alpha,
    float dirichlet_epsilon,
    int policy_size
) {
    MCTSSearch searcher(c_puct, dirichlet_alpha, dirichlet_epsilon, policy_size);

    // Wrap Python callback
    ExpandFunc cpp_expand = [&expand_fn](int state_id) -> ExpansionResult {
        return py_expand_to_cpp(expand_fn, state_id);
    };

    std::vector<float> visits = searcher.search(
        root_state_id, num_simulations, add_noise, cpp_expand
    );

    // Convert to numpy array
    auto result = py::array_t<float>(visits.size());
    auto buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < visits.size(); i++) {
        buf(i) = visits[i];
    }
    return result;
}


PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "C++ MCTS engine for AlphaZero Checkers";

    m.def("search", &mcts_search,
          "Run MCTS search and return visit distribution",
          py::arg("root_state_id"),
          py::arg("num_simulations"),
          py::arg("add_noise"),
          py::arg("expand_fn"),
          py::arg("c_puct") = 1.5f,
          py::arg("dirichlet_alpha") = 0.3f,
          py::arg("dirichlet_epsilon") = 0.25f,
          py::arg("policy_size") = 1024);
}
