"""envs.py - Ordered Treasure Hunt environment (De tai 08).

The environment is a custom ``gymnasium.Env`` built on a small ASCII grid.
The agent must resolve the items of a chain in the correct order and then
walk onto the exit tile.

MDP
---
State   : (row, col, task_progress, flags)
    row, col      : position of the agent on the grid
    task_progress : number of chain stages that are already resolved
    flags         : bitmask
        bit i (i < n_items)  -> item i has been resolved (collected or skipped)
        bit n_items + j      -> auxiliary flag j (switch_on, skip_used)
Action  : 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
Reward  : see DEFAULT_REWARDS below
Terminal: the agent steps on the exit tile while every chain stage is
          resolved. The episode is also truncated after ``max_steps`` moves.

The transition is deterministic, so the environment can expose an exact
transition model ``P[s][a]`` for the model based algorithms (Value Iteration
and Policy Iteration). A plain ``gymnasium.Env`` has no ``.P`` attribute, so
``build_transition_model()`` has to be used instead.

Only ASCII is used in the source code comments and docstrings; the markdown
reports keep the full Vietnamese text with diacritics.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ============================================================
# ACTION CONSTANTS
# ============================================================

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

#: Human readable name of every action.
ACTION_NAMES = {
    ACTION_UP: "up",
    ACTION_DOWN: "down",
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
}

#: Arrow glyph of every action, used when a policy grid is printed or drawn.
ACTION_SYMBOLS = {
    ACTION_UP: "^",
    ACTION_DOWN: "v",
    ACTION_LEFT: "<",
    ACTION_RIGHT: ">",
}

#: (row, col) displacement produced by every action.
ACTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

N_ACTIONS = 4


# ============================================================
# TILE LEGEND
# ============================================================

TILE_WALL = "#"
TILE_FLOOR = "."
TILE_START = "S"
TILE_EXIT = "X"
TILE_SWITCH = "W"
TILE_SKIP = "K"


# ============================================================
# DEFAULT REWARDS
# ============================================================

#: Default reward table. Every value can be overridden per level.
DEFAULT_REWARDS = {
    "step_penalty": -1.0,        # cost of a normal move
    "wall_penalty": -3.0,        # the move is blocked by a wall
    "wrong_item_penalty": -20.0, # the move touches an out of order item
    "locked_penalty": -8.0,      # the move touches a locked item
    "early_exit_penalty": -20.0, # the move touches the exit too early
    "item_reward": 10.0,         # reward of the correct next item
    "switch_reward": 0.0,       # reward of flipping the switch on
    "skip_reward": 0.0,          # reward of using the bonus skip tile
    "exit_bonus": 100.0,         # reward of leaving through the exit
}


# ============================================================
# LEVEL DEFINITIONS
# ============================================================

#: The four mandatory settings of De tai 08, from easy to hard.
#:
#: ``chain`` is a list of stages. Stage ``i`` lists the items that are
#: acceptable when ``task_progress == i``. A stage with two acceptable items
#: means the level has two different valid chains.
#:
#: ``aux`` lists the auxiliary boolean variables of the level. They are needed
#: to keep the process Markov, because they change the result of a move.
LEVELS = {
    "level1": {
        "title": "Level 1 - chuỗi A -> B -> C",
        "description": (
            "Map 5x5 de, mot chuoi A -> B -> C. Cham vao B hoac C truoc A se bi "
            "phat."
        ),
        "grid": [
            "#S...",
            "#.A..",
            "#..B.",
            "C....",
            "...X.",
        ],
        "chain": [["A"], ["B"], ["C"]],
        "aux": [],
        "locked": [],
    },
    "level2": {
        "title": "Level 2 - hai chuoi hop le khac nhau",
        "description": (
            "Map 7x7, bon vat pham A, B, C, D chap nhan hai chuoi hop le: "
            "A -> B -> D va A -> C -> D. Agent phai tu chon chuoi nao de di."
        ),
        "grid": [
            "#S....#",
            "#.A...#",
            "#..B..#",
            "#.....#",
            "#..C..#",
            "#.....#",
            "#...D.X",
        ],
        "chain": [["A"], ["B", "C"], ["D"]],
        "aux": [],
        "locked": [],
    },
    "level3": {
        "title": "Level 3 - vat pham bi khoa va o thuong",
        "description": (
            "Map 8x8, chuoi A -> B -> C -> D. Vat pham C bi khoa cho toi khi bat "
            "cong tac W, va o thuong K cho phep bo qua mot phan tu cua chuoi."
        ),
        "grid": [
            "########",
            "#S.....#",
            "#.A.K..#",
            "#..B...#",
            "#....W.#",
            "#....C.#",
            "#......#",
            "#D...X.#",
        ],
        "chain": [["A"], ["B"], ["C"], ["D"]],
        "aux": ["switch_on", "skip_used"],
        "locked": ["C"],
    },
}


def get_level(name):
    """Return the level specification stored under ``name``."""
    if name not in LEVELS:
        raise KeyError(
            "Unknown level %r. Available levels: %s"
            % (name, ", ".join(sorted(LEVELS)))
        )
    return LEVELS[name]


# ============================================================
# ENVIRONMENT
# ============================================================

class OrderedTreasureHuntEnv(gym.Env):
    """Grid environment where items have to be collected in a fixed order.

    Parameters
    ----------
    level : str
        Key of the level inside :data:`LEVELS`.
    rewards : dict, optional
        Overrides for the entries of :data:`DEFAULT_REWARDS`.
    max_steps : int
        Maximum number of moves per episode. Reaching the limit truncates the
        episode instead of terminating it.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, level="level1", rewards=None, max_steps=100):
        spec = get_level(level)
        self.level = level
        self.title = spec["title"]
        self.description = spec["description"]
        self.grid = [list(row) for row in spec["grid"]]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        if any(len(row) != self.width for row in self.grid):
            raise ValueError(
                "Level %r has rows of different lengths: %s"
                % (level, [len(row) for row in self.grid])
            )
        self.chain = [list(stage) for stage in spec["chain"]]
        self.aux_names = list(spec["aux"])
        #: Items that cannot be taken before the switch is turned on.
        self.locked_items = list(spec.get("locked", []))
        self.n_aux = len(self.aux_names)
        self.n_stages = len(self.chain)
        self.max_steps = int(max_steps)
        self.rewards = dict(DEFAULT_REWARDS)
        if rewards:
            self.rewards.update(rewards)

        # --- item table: the bit order follows the chain order ---
        self.item_list = []
        for stage in self.chain:
            for letter in stage:
                if letter not in self.item_list:
                    self.item_list.append(letter)
        self.n_items = len(self.item_list)
        self._item_bit = {letter: 1 << i for i, letter in enumerate(self.item_list)}

        # --- tile table ---
        self._cell_items = {}
        self.start = None
        self.exit = None
        self.switch = None
        self.skip = None
        for r in range(self.height):
            for c in range(self.width):
                cell = self.grid[r][c]
                if cell == TILE_START:
                    self.start = (r, c)
                elif cell == TILE_EXIT:
                    self.exit = (r, c)
                elif cell == TILE_SWITCH:
                    self.switch = (r, c)
                elif cell == TILE_SKIP:
                    self.skip = (r, c)
                elif cell.isalpha() and cell.isupper():
                    self._cell_items[(r, c)] = cell
        if self.start is None or self.exit is None:
            raise ValueError("Level %r needs a start 'S' and an exit 'X'" % level)

        # --- state space: only the reachable states are enumerated ---
        self._states, self._index = self._build_state_space()
        self.n_states = len(self._states)
        self.start_index = self._index[
            (self.start[0], self.start[1], 0, 0)
        ]

        # --- gymnasium spaces ---
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Discrete(self.n_states)

        self._state = self.start_index
        self._steps = 0

    # ------------------------------------------------------------
    # STATE ENCODING
    # ------------------------------------------------------------

    def _build_state_space(self):
        """Enumerate the reachable states with a breadth first search.

        Returns
        -------
        states : list
            The reachable state tuples, in discovery order.
        index : dict
            Maps a state tuple to its position inside ``states``.
        """
        start = (self.start[0], self.start[1], 0, 0)
        states = [start]
        index = {start: 0}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            for action in range(N_ACTIONS):
                nxt, _reward, _term, _event = self._transition(state, action)
                if nxt not in index:
                    index[nxt] = len(states)
                    states.append(nxt)
                    queue.append(nxt)
        return states, index

    def state_to_index(self, state):
        """Convert a state tuple into its flat index."""
        row, col, progress, flags = state
        key = (row, col, progress, flags)
        if key not in self._index:
            raise KeyError("State %r is not reachable" % (key,))
        return self._index[key]

    def index_to_state(self, index):
        """Convert a flat index back into its state tuple."""
        return self._states[index]

    def decode(self, index):
        """Return a readable description of a flat state index."""
        row, col, progress, flags = self._states[index]
        items = [
            letter
            for i, letter in enumerate(self.item_list)
            if flags & (1 << i)
        ]
        aux = [
            name
            for j, name in enumerate(self.aux_names)
            if flags & (1 << (self.n_items + j))
        ]
        return {
            "row": row,
            "col": col,
            "task_progress": progress,
            "resolved_items": items,
            "aux": aux,
        }

    # ------------------------------------------------------------
    # DYNAMICS
    # ------------------------------------------------------------

    def _item_at(self, row, col, flags):
        """Return the item letter still present on a cell, or None.

        A resolved item disappears from the grid, therefore the tile becomes
        an ordinary floor tile.
        """
        letter = self._cell_items.get((row, col))
        if letter is None:
            return None
        if flags & self._item_bit[letter]:
            return None
        return letter

    def _aux_flag(self, flags, name):
        """Return True when the auxiliary variable ``name`` is set."""
        j = self.aux_names.index(name)
        return bool(flags & (1 << (self.n_items + j)))

    def _set_aux(self, flags, name):
        """Return the flags with the auxiliary variable ``name`` switched on."""
        j = self.aux_names.index(name)
        return flags | (1 << (self.n_items + j))

    def _transition(self, state, action):
        """Compute one deterministic transition of the environment.

        Parameters
        ----------
        state : tuple
            The current state ``(row, col, task_progress, flags)``.
        action : int
            The action to apply.

        Returns
        -------
        next_state : tuple
            The state reached by the action.
        reward : float
            The reward of the transition.
        terminated : bool
            True when the episode ends on the exit tile.
        event : str
            A short tag describing what happened. It is used by the report
            generators and by the auto play demo.
        """
        row, col, progress, flags = state
        rewards = self.rewards
        if progress >= self.n_stages and (row, col) == self.exit:
            # the episode is already finished, the state is absorbing
            return state, 0.0, True, "terminal"
        dr, dc = ACTION_DELTAS[action]
        nr, nc = row + dr, col + dc

        # --- move outside of the grid or into a wall ---
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            return state, rewards["wall_penalty"], False, "wall"
        if self.grid[nr][nc] == TILE_WALL:
            return state, rewards["wall_penalty"], False, "wall"

        cell = self.grid[nr][nc]

        # --- an item tile ---
        letter = self._item_at(nr, nc, flags)
        if letter is not None:
            if progress >= self.n_stages:
                # every stage is resolved, no item can be left on the grid
                return state, rewards["wrong_item_penalty"], False, "wrong_item"
            if letter in self.chain[progress]:
                # correct next item of the chain
                new_flags = flags | self._item_bit[letter]
                new_state = (nr, nc, progress + 1, new_flags)
                return new_state, rewards["item_reward"], False, "collect"
            if letter in self.locked_items and not self._aux_flag(
                flags, "switch_on"
            ):
                # the item is locked until the switch is turned on
                return state, rewards["locked_penalty"], False, "locked"
            # out of order item, the move is blocked
            return state, rewards["wrong_item_penalty"], False, "wrong_item"

        # --- the exit tile ---
        if cell == TILE_EXIT:
            if progress >= self.n_stages:
                return (nr, nc, progress, flags), rewards["exit_bonus"], True, "exit"
            return state, rewards["early_exit_penalty"], False, "early_exit"

        # --- the switch tile ---
        if cell == TILE_SWITCH:
            new_flags = self._set_aux(flags, "switch_on")
            return (nr, nc, progress, new_flags), rewards["switch_reward"], False, "switch"

        # --- the bonus skip tile ---
        if cell == TILE_SKIP:
            if not self._aux_flag(flags, "skip_used") and progress < self.n_stages:
                # resolve the next chain element without collecting it
                letter = self.chain[progress][0]
                new_flags = self._set_aux(flags, "skip_used")
                new_flags = new_flags | self._item_bit[letter]
                new_state = (nr, nc, progress + 1, new_flags)
                return new_state, rewards["skip_reward"], False, "skip"
            # the tile has already been used, it is now an ordinary tile
            return (nr, nc, progress, flags), 0.0, False, "skip_used"

        # --- ordinary floor ---
        return (nr, nc, progress, flags), rewards["step_penalty"], False, "step"

    def build_transition_model(self):
        """Return the exact transition model of the environment.

        The result is a nested list ``P[state][action]`` where every entry is
        a list of tuples ``(probability, next_state, reward, terminated)``.
        The environment is deterministic, so every list holds exactly one
        tuple with probability 1.0.
        """
        model = []
        for state in self._states:
            row = []
            for action in range(N_ACTIONS):
                nxt, reward, terminated, _event = self._transition(state, action)
                row.append([(1.0, self.state_to_index(nxt), reward, terminated)])
            model.append(row)
        return model

    # ------------------------------------------------------------
    # GYMNASIUM API
    # ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Reset the environment and return the first observation."""
        super().reset(seed=seed)
        self._state = self.start_index
        self._steps = 0
        return self._state, {"event": "reset"}

    def step(self, action):
        """Apply one action and return ``(obs, reward, terminated, truncated, info)``."""
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError("Action %r is not part of the action space" % (action,))
        state = self._states[self._state]
        nxt, reward, terminated, event = self._transition(state, action)
        self._state = self.state_to_index(nxt)
        self._steps += 1
        truncated = (not terminated) and (self._steps >= self.max_steps)
        info = {"event": event, "steps": self._steps, "state": self._state}
        return self._state, reward, terminated, truncated, info

    # ------------------------------------------------------------
    # TEXT RENDERING
    # ------------------------------------------------------------

    def grid_text(self, state=None):
        """Return the current grid as a list of text lines."""
        if state is None:
            state = self._states[self._state]
        row, col, progress, flags = state
        lines = []
        for r in range(self.height):
            chars = []
            for c in range(self.width):
                cell = self.grid[r][c]
                if r == row and c == col:
                    chars.append("@")
                    continue
                letter = self._item_at(r, c, flags)
                if letter is not None:
                    chars.append(letter)
                elif cell == TILE_WALL:
                    chars.append(TILE_WALL)
                else:
                    chars.append(cell)
            lines.append("".join(chars))
        return lines

    def policy_text(self, policy):
        """Return the greedy policy of every state as a list of text lines.

        One block of ``height`` lines is printed per value of
        ``task_progress``; the arrow of a cell is the action of the agent in
        the state ``(row, col, task_progress)``. When several reachable states
        share the same position and progress, the action of the first one is
        used.
        """
        # group the reachable states by (row, col, task_progress)
        groups = {}
        for index, (row, col, progress, _flags) in enumerate(self._states):
            groups.setdefault((row, col, progress), index)
        lines = []
        for progress in range(self.n_stages + 1):
            for r in range(self.height):
                chars = []
                for c in range(self.width):
                    cell = self.grid[r][c]
                    if cell == TILE_WALL:
                        chars.append(TILE_WALL)
                        continue
                    index = groups.get((r, c, progress))
                    if index is not None:
                        action = int(policy[index])
                        chars.append(ACTION_SYMBOLS[action])
                    elif cell == TILE_EXIT:
                        chars.append(TILE_EXIT)
                    elif cell == TILE_START:
                        chars.append(TILE_START)
                    else:
                        letter = self._cell_items.get((r, c))
                        chars.append(letter if letter else cell)
                lines.append("".join(chars))
            lines.append("")
        return lines


# ============================================================
# QUICK SELF CHECK
# ============================================================

if __name__ == "__main__":
    for name in sorted(LEVELS):
        env = OrderedTreasureHuntEnv(level=name)
        print("%s -> %s" % (name, env.title))
        print("  grid %dx%d, states %d, actions %d"
              % (env.height, env.width, env.n_states, env.action_space.n))
        for line in env.grid_text():
            print("  " + line)
