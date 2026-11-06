"""plots.py - Figures of De tai 08.

The module draws the environment, the learned policies, the learning curves
and the comparison charts. Every figure is written as a PNG file into the
folder ``figures`` so that the notebook, the Word report and the PowerPoint
can reuse exactly the same pictures.

The text inside the figures is kept in English on purpose: the reports carry
the Vietnamese explanation, the figures stay readable for everybody.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow, Rectangle

from envs import ACTION_DELTAS, TILE_EXIT, TILE_FLOOR, TILE_SKIP, TILE_START, TILE_SWITCH, TILE_WALL


# ============================================================
# COLORS
# ============================================================

COLOR_WALL = "#37474F"
COLOR_FLOOR = "#ECEFF1"
COLOR_GRID = "#CFD8DC"
COLOR_START = "#1E88E5"
COLOR_EXIT = "#43A047"
COLOR_SWITCH = "#FB8C00"
COLOR_SKIP = "#8E24AA"
COLOR_ITEM = "#FDD835"
COLOR_ITEM_DONE = "#CFD8DC"
COLOR_AGENT = "#E53935"
COLOR_ARROW = "#37474F"
COLOR_TEXT = "#263238"


# ============================================================
# LOW LEVEL DRAWING
# ============================================================

def _cell_center(row, col):
    """Return the centre of a cell in data coordinates."""
    return col + 0.5, row + 0.5


def draw_grid(ax, env, state=None, policy=None, agent=None, title=None,
              show_policy=True, progress=None):
    """Draw one grid of the environment on the axis ``ax``.

    Parameters
    ----------
    state : tuple, optional
        State tuple whose tiles are drawn. By default the current state of
        the environment is used.
    policy : array, optional
        Greedy policy; an arrow is drawn on every cell of the grid.
    progress : int, optional
        Value of ``task_progress`` whose arrows are drawn. By default the
        progress of the drawn state is used.
    agent : tuple, optional
        Position ``(row, col)`` of the agent, drawn on top of the grid.
    """
    if state is None:
        state = env._states[env._state]
    state_row, state_col, state_progress, flags = state
    if progress is None:
        progress = state_progress

    for r in range(env.height):
        for c in range(env.width):
            cell = env.grid[r][c]
            if cell == TILE_WALL:
                face = COLOR_WALL
            elif cell == TILE_EXIT:
                face = COLOR_EXIT
            elif cell == TILE_SWITCH:
                face = COLOR_SWITCH
            elif cell == TILE_SKIP:
                face = COLOR_SKIP
            elif cell == TILE_START:
                face = COLOR_START
            else:
                face = COLOR_FLOOR
            ax.add_patch(
                Rectangle(
                    (c, r), 1, 1, facecolor=face,
                    edgecolor=COLOR_GRID, linewidth=0.8
                )
            )

    # items, the resolved ones are drawn in grey
    for (r, c), letter in env._cell_items.items():
        resolved = bool(flags & env._item_bit[letter])
        ax.add_patch(
            Circle(
                _cell_center(r, c), 0.28,
                facecolor=COLOR_ITEM_DONE if resolved else COLOR_ITEM,
                edgecolor=COLOR_TEXT, linewidth=1.2
            )
        )
        ax.text(
            c + 0.5, r + 0.5, letter,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=COLOR_TEXT if not resolved else "#78909C"
        )

    # labels of the special tiles
    special = [
        (env.exit, "X", "white"),
        (env.switch, "W", "white"),
        (env.skip, "K", "white"),
        (env.start, "S", "white"),
    ]
    for position, text, color in special:
        if position is None:
            continue
        r, c = position
        ax.text(
            c + 0.5, r + 0.5, text,
            ha="center", va="center", fontsize=11, fontweight="bold", color=color
        )

    # policy arrows
    if policy is not None and show_policy:
        groups = {}
        for index, (sr, sc, sprogress, _flags) in enumerate(env._states):
            groups.setdefault((sr, sc, sprogress), index)
        for r in range(env.height):
            for c in range(env.width):
                if env.grid[r][c] == TILE_WALL:
                    continue
                index = groups.get((r, c, progress))
                if index is None:
                    continue
                action = int(policy[index])
                dr, dc = ACTION_DELTAS[action]
                x0, y0 = _cell_center(r, c)
                ax.add_patch(
                    FancyArrow(
                        x0, y0, 0.34 * dc, 0.34 * dr,
                        width=0.05, head_width=0.16, head_length=0.16,
                        color=COLOR_ARROW, alpha=0.75,
                        length_includes_head=True
                    )
                )

    # the agent
    if agent is None:
        agent = (state_row, state_col)
    ax.add_patch(
        Circle(
            _cell_center(agent[0], agent[1]), 0.22,
            facecolor=COLOR_AGENT, edgecolor="white", linewidth=1.2, zorder=5
        )
    )

    ax.set_xlim(0, env.width)
    ax.set_ylim(env.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, color=COLOR_TEXT)
    return ax


def save_levels_figure(levels, path, title="Ba setting của đề tài 08"):
    """Save one picture that shows the three settings side by side.

    ``levels`` maps the name of a setting to the environment object.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = list(levels)
    figure, axes = plt.subplots(
        1, len(names), figsize=(4.2 * len(names), 4.6)
    )
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        env = levels[name]
        draw_grid(ax, env, policy=None, title="%s  (|S| = %d)"
                  % (name, env.n_states))
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_environment_figure(env, path, policy=None, title=None, state=None):
    """Save a picture of the environment, optionally with the policy arrows.

    When a policy is given, one panel is drawn for every value of
    ``task_progress``, because the optimal action depends on it.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if policy is None:
        figure, ax = plt.subplots(figsize=(0.95 * env.width, 0.95 * env.height))
        draw_grid(ax, env, state=state, policy=policy, title=title)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path
    panels = env.n_stages + 1
    figure, axes = plt.subplots(
        1, panels, figsize=(0.95 * env.width * panels, 0.95 * env.height + 1.1)
    )
    if panels == 1:
        axes = [axes]
    for index, ax in enumerate(axes):
        draw_grid(ax, env, policy=policy, title="task_progress = %d" % index,
                  progress=index)
    figure.suptitle(title or env.title, fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ============================================================
# LEARNING CURVES
# ============================================================

def moving_average(values, window):
    """Return the moving average of a sequence."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def save_learning_curves(curves, path, title, window=200,
                         ylabel="return của một episode"):
    """Save the learning curves of several algorithms on one figure.

    ``curves`` maps the name of an algorithm to its list of episode returns.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure, ax = plt.subplots(figsize=(8, 4.5))
    for name, values in curves.items():
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            continue
        smooth = moving_average(values, window)
        offset = values.size - smooth.size
        ax.plot(
            np.arange(offset, values.size), smooth,
            label=name, linewidth=1.6
        )
    ax.set_xlabel("episode huấn luyện")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_dp_convergence(curves, path, title):
    """Save the convergence curves of the dynamic programming algorithms."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure, ax = plt.subplots(figsize=(8, 4.5))
    for name, values in curves.items():
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            continue
        ax.plot(np.arange(1, values.size + 1), values, label=name, linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("vòng lặp")
    ax.set_ylabel("độ lớn nhất |V(k+1) - V(k)|")
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ============================================================
# COMPARISON CHARTS
# ============================================================

def save_comparison_chart(runs, path, metric, title, ylabel):
    """Save a grouped bar chart that compares the algorithms on every level.

    ``runs`` is a list of dictionaries with the keys ``config``, ``algo`` and
    ``metric``.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    levels = []
    for run in runs:
        if run["config"] not in levels:
            levels.append(run["config"])
    algos = []
    for run in runs:
        if run["algo"] not in algos:
            algos.append(run["algo"])
    labels = {}
    for run in runs:
        labels[run["algo"]] = run.get("algo_label", run["algo"])
    width = 0.8 / max(1, len(algos))
    figure, ax = plt.subplots(figsize=(10, 4.8))
    positions = np.arange(len(levels))
    for index, algo in enumerate(algos):
        values = []
        for level in levels:
            found = None
            for run in runs:
                if run["config"] == level and run["algo"] == algo:
                    found = run[metric]
                    break
            values.append(0.0 if found is None else found)
        ax.bar(
            positions + index * width, values, width,
            label=labels[algo], alpha=0.85
        )
    ax.set_xticks(positions + width * (len(algos) - 1) / 2.0)
    ax.set_xticklabels(levels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    highest = max([run[metric] for run in runs] + [1.0])
    ax.set_ylim(0, highest * 1.25)
    ax.legend(fontsize=9, ncol=len(algos), loc="upper center",
              bbox_to_anchor=(0.5, 1.0), framealpha=0.9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_seed_comparison(experiment, path, title):
    """Save a bar chart of the return of every algorithm for every seed.

    ``experiment`` is the dictionary of the seed experiment: it holds the list
    of seeds and the list of runs with the keys ``algo``, ``seed`` and
    ``eval_return_mean``.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seeds = experiment["seeds"]
    algos = []
    for run in experiment["runs"]:
        if run["algo"] not in algos:
            algos.append(run["algo"])
    labels = {}
    for run in experiment["runs"]:
        labels[run["algo"]] = run.get("algo_label", run["algo"])
    width = 0.8 / max(1, len(algos))
    figure, ax = plt.subplots(figsize=(10, 4.8))
    positions = np.arange(len(seeds))
    for index, algo in enumerate(algos):
        values = []
        for seed in seeds:
            found = None
            for run in experiment["runs"]:
                if run["algo"] == algo and run["seed"] == seed:
                    found = run["eval_return_mean"]
                    break
            values.append(0.0 if found is None else found)
        ax.bar(positions + index * width, values, width,
               label=labels[algo], alpha=0.85)
    ax.set_xticks(positions + width * (len(algos) - 1) / 2.0)
    ax.set_xticklabels(["seed %d" % seed for seed in seeds])
    ax.set_ylabel("mean return của policy tham lam")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    highest = max([run["eval_return_mean"] for run in experiment["runs"]] + [1.0])
    ax.set_ylim(0, highest * 1.3)
    ax.legend(fontsize=9, ncol=len(algos), loc="upper center",
              bbox_to_anchor=(0.5, 1.0), framealpha=0.9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_policy_comparison(env, entries, path, title):
    """Save a row of pictures that show one policy per entry.

    ``entries`` is a list of dictionaries with the keys ``name`` (the title of
    the panel), ``policy`` (the greedy policy), ``caption`` (the text that is
    printed under the picture) and ``progress`` (the value of
    ``task_progress`` whose arrows are drawn).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = len(entries)
    figure, axes = plt.subplots(
        1, count, figsize=(0.95 * env.width * count, 0.95 * env.height + 1.4)
    )
    if count == 1:
        axes = [axes]
    for ax, entry in zip(axes, entries):
        draw_grid(ax, env, policy=entry["policy"], title=entry["name"],
                  progress=entry.get("progress"))
        ax.set_xlabel(entry.get("caption", ""), fontsize=8, color=COLOR_TEXT)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ============================================================
# MARKOV PROPERTY FIGURE
# ============================================================

def save_markov_figure(env, path, level="level1", cell=(2, 1), item_cell=(2, 2)):
    """Save the figure that shows why the task progress belongs to the state.

    The left panel shows the cell with a task progress of zero, where the
    item is still on the grid, the right panel shows the same cell with a
    task progress of one, where the item was collected and disappeared. The
    action towards the item therefore gives two different rewards, so the two
    situations cannot be the same state.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    states = {}
    for index, state in enumerate(env._states):
        row, col, progress, flags = state
        if (row, col) == cell and progress in (0, 1):
            states.setdefault(progress, (index, state))
    if 0 not in states or 1 not in states:
        raise ValueError(
            "The cell %r is not reachable with two different task_progress "
            "values" % (cell,)
        )
    model = env.build_transition_model()
    captions = {}
    for progress, (index, _state) in states.items():
        entries = []
        for action in range(env.action_space.n):
            for prob, next_state, reward, terminated in model[index][action]:
                next_row, next_col, next_progress, _flags = env._states[next_state]
                if (next_row, next_col) == item_cell:
                    entries.append((action, reward, next_progress))
        if entries:
            action, reward, next_progress = entries[0]
            captions[progress] = (
                "bước vào ô vật phẩm cho %+.0f và tiến độ thành %d"
                % (reward, next_progress)
            )
        else:
            captions[progress] = "không tới được vật phẩm từ ô này"
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, progress in zip(axes, sorted(states)):
        index, state = states[progress]
        draw_grid(
            ax, env, state=state,
            title="trạng thái (%d, %d, %d)" % (cell[0], cell[1], progress)
        )
        ax.set_xlabel(captions[progress], fontsize=9, color=COLOR_TEXT)
    figure.suptitle(
        "Cùng một ô (%d, %d), task_progress khác nhau là hai state khác nhau"
        % (cell[0], cell[1]),
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ============================================================
# AUTO PLAY ANIMATION
# ============================================================

def save_episode_animation(env, policy, path, title, fps=6, seed=42, max_steps=100):
    """Play one episode of the learned policy and store it as a GIF file.

    The agent is drawn on the grid, the resolved items disappear and the
    current reward of the episode is printed in the title of the figure.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import matplotlib.animation as animation

    state, _info = env.reset(seed=seed)
    frames = []
    total = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated) and len(frames) < max_steps:
        action = int(policy[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        total += reward
        state = next_state
        frames.append((state, total, info.get("event", ""), terminated))
    # keep the last frame a little longer so that the result is visible
    frames.extend([frames[-1]] * (fps * 2))

    figure, ax = plt.subplots(figsize=(1.15 * env.width, 1.15 * env.height))

    def update(frame_index):
        ax.clear()
        current_state, running_total, event, done = frames[frame_index]
        draw_grid(ax, env, state=env._states[current_state], policy=policy,
                  show_policy=False)
        ax.set_title(
            "%s\nbước %d / %d   return = %.1f   sự kiện = %s%s"
            % (
                title, frame_index + 1, len(frames), running_total, event,
                "   (đã tới đích)" if done else "",
            ),
            fontsize=10,
        )
        return (ax,)

    anim = animation.FuncAnimation(figure, update, frames=len(frames), blit=False)
    writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer)
    plt.close(figure)
    return path


def save_trajectory_figure(env, policy, path, title, seed=42, max_steps=100):
    """Save the visited cells of one episode of the learned policy."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state, _info = env.reset(seed=seed)
    visited = []
    total = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated) and len(visited) < max_steps:
        row, col, progress, flags = env._states[state]
        visited.append((row, col))
        action = int(policy[state])
        state, reward, terminated, truncated, _info = env.step(action)
        total += reward
    figure, ax = plt.subplots(figsize=(1.1 * env.width, 1.1 * env.height))
    draw_grid(ax, env, state=env._states[state], policy=policy, show_policy=False)
    for order, (r, c) in enumerate(visited):
        ax.text(
            c + 0.5, r + 0.22, str(order + 1),
            ha="center", va="center", fontsize=7, color="#B71C1C"
        )
    ax.set_title("%s\nreturn = %.1f, %d bước" % (title, total, len(visited)), fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
