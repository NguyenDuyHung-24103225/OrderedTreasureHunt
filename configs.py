"""configs.py - Configuration of the experiments of De tai 08.

The module holds the dictionary ``CONFIGS`` with one entry per setting of the
assignment and the helper ``get_config(name)`` that returns a copy of a
configuration. Every configuration describes

    level            : key of the level inside ``envs.LEVELS``
    episodes         : number of training episodes of the model free methods
    eval_episodes    : number of episodes used to evaluate the learned policy
    seed             : seed of the training and of the evaluation
    reward_profile   : key of the reward profile inside ``REWARD_PROFILES``
    algorithms       : hyper parameters of the six algorithms

Keeping the configuration in one place makes the comparison reproducible and
lets the report generators read the same numbers that the training used.
"""

from __future__ import annotations

import copy


# ============================================================
# REWARD PROFILES
# ============================================================

#: Reward profiles used to show how the policy depends on the reward design.
#: The profiles only differ in the cost of one move, which decides whether
#: the detour of level 4 is worth it.
REWARD_PROFILES = {
    "default": {
        "step_penalty": -1.0,
        "item_reward": 10.0,
        "exit_bonus": 100.0,
        "wrong_item_penalty": -20.0,
    },
    "cheap_steps": {
        "step_penalty": -0.2,
        "item_reward": 10.0,
        "exit_bonus": 100.0,
        "wrong_item_penalty": -20.0,
    },
    "expensive_steps": {
        "step_penalty": -5.0,
        "item_reward": 10.0,
        "exit_bonus": 100.0,
        "wrong_item_penalty": -20.0,
    },
}


# ============================================================
# SHARED HYPER PARAMETERS
# ============================================================

#: Default hyper parameters of the six algorithms.
ALGO_PARAMS = {
    "vi": {"gamma": 0.99, "theta": 1e-8, "max_iterations": 10000},
    "pi": {"gamma": 0.99, "theta": 1e-8, "max_iterations": 1000},
    "qlearning": {
        "gamma": 0.99,
        "alpha": 0.1,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
    },
    "sarsa": {
        "gamma": 0.99,
        "alpha": 0.1,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
    },
    "mc": {
        "gamma": 0.99,
        "alpha": 0.1,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
    },
}

#: Names of the algorithms in the order they are reported.
ALGO_ORDER = ["vi", "pi", "qlearning", "sarsa", "mc"]


# ============================================================
# LEVEL CONFIGURATIONS
# ============================================================

CONFIGS = {
    "level1": {
        "level": "level1",
        "title": "Level 1 - chuỗi A -> B -> C",
        "description": (
            "Map 5x5 de, mot chuoi A -> B -> C. Cham vao mot vat pham sai thu tu "
            "bi phat, nen agent phai di vong quanh nhung vat pham con khoa."
        ),
        "episodes": 20000,
        "eval_episodes": 200,
        "seed": 42,
        "reward_profile": "default",
        "algorithms": {},
    },
    "level2": {
        "level": "level2",
        "title": "Level 2 - hai chuoi hop le khac nhau",
        "description": (
            "Map 7x7, bon vat pham A, B, C, D chap nhan hai chuoi hop le "
            "A -> B -> D va A -> C -> D. Cac vat pham da giai phai nam trong state, "
            "neu khong tien trinh se khong con tinh Markov."
        ),
        "episodes": 20000,
        "eval_episodes": 200,
        "seed": 42,
        "reward_profile": "default",
        "algorithms": {},
    },
    "level3": {
        "level": "level3",
        "title": "Level 3 - vat pham bi khoa va o thuong",
        "description": (
            "Map 8x8, chuoi A -> B -> C -> D. Vat pham C bi khoa cho toi khi bat "
            "cong tac W, va o thuong K bo qua duoc mot phan tu cua chuoi. Do do "
            "state can them bien phu switch_on va skip_used."
        ),
        "episodes": 20000,
        "eval_episodes": 200,
        "seed": 42,
        "reward_profile": "default",
        "algorithms": {},
    },
}


# ============================================================
# REWARD EXPERIMENT
# ============================================================

#: The reward experiment runs level 3 with two reward profiles. The only
#: difference is the cost of one move, and it is enough to flip the optimal
#: policy between "collect every item" and "use the bonus tile".
REWARD_EXPERIMENT = {
    "level": "level3",
    "profiles": ["cheap_steps", "expensive_steps"],
    "episodes": 20000,
    "eval_episodes": 200,
    "seed": 42,
    "algorithm": "qlearning",
}


# ============================================================
# SEED EXPERIMENT
# ============================================================

#: On the hard map the algorithms without a model can stop on the short path
#: and never try the long one again. This experiment repeats the training with
#: several random seeds, to see how often that happens.
SEED_EXPERIMENT = {
    "level": "level3",
    "seeds": [42, 7, 123],
    "episodes": 20000,
    "eval_episodes": 200,
    "algorithms": ["pi", "qlearning", "sarsa", "mc"],
}


# ============================================================
# HELPERS
# ============================================================

def get_config(name):
    """Return a copy of the configuration stored under ``name``."""
    if name not in CONFIGS:
        raise KeyError(
            "Unknown configuration %r. Available configurations: %s"
            % (name, ", ".join(sorted(CONFIGS)))
        )
    config = copy.deepcopy(CONFIGS[name])
    config["name"] = name
    config["rewards"] = copy.deepcopy(
        REWARD_PROFILES[config["reward_profile"]]
    )
    config["algorithms"] = {
        algo: copy.deepcopy(ALGO_PARAMS[algo]) for algo in ALGO_ORDER
    }
    return config


def get_reward_profile(name):
    """Return a copy of a reward profile."""
    if name not in REWARD_PROFILES:
        raise KeyError(
            "Unknown reward profile %r. Available profiles: %s"
            % (name, ", ".join(sorted(REWARD_PROFILES)))
        )
    return copy.deepcopy(REWARD_PROFILES[name])


def config_names():
    """Return the names of every configuration, ordered by difficulty."""
    return list(CONFIGS)
