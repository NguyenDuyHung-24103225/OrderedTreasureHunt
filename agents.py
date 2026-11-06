
from __future__ import annotations

import time

import numpy as np


# ============================================================
# SMALL HELPERS
# ============================================================

def epsilon_greedy(q_values, state, epsilon, rng, n_actions):
    """Pick an action with the epsilon greedy rule.

    With probability ``1 - epsilon`` the action with the highest value is
    taken, otherwise a random action is drawn. Ties are broken randomly so
    that the exploration does not depend on the order of the actions.
    """
    best = np.flatnonzero(q_values[state] == q_values[state].max())
    if best.size > 1:
        greedy_action = int(rng.choice(best))
    else:
        greedy_action = int(best[0])
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    return greedy_action


def greedy_policy(q_values):
    """Turn a table of action values into a deterministic greedy policy."""
    return np.argmax(q_values, axis=1).astype(int)


def epsilon_schedule(episode, n_episodes, epsilon_start, epsilon_end):
    """Linear decay of the exploration rate over the training episodes."""
    if n_episodes <= 1:
        return epsilon_end
    frac = episode / float(n_episodes - 1)
    return epsilon_start + (epsilon_end - epsilon_start) * frac


def evaluate_policy(env, policy, n_episodes=200, seed=1234):
    """Run the deterministic policy on the environment and collect statistics.

    Returns a dictionary with the mean and the standard deviation of the
    return, the success rate and the mean number of steps per episode.
    """
    returns = []
    lengths = []
    successes = 0
    for episode in range(n_episodes):
        state, _info = env.reset(seed=seed + episode)
        total = 0.0
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = int(policy[state])
            state, reward, terminated, truncated, _info = env.step(action)
            total += reward
            steps += 1
        returns.append(total)
        lengths.append(steps)
        if terminated:
            successes += 1
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "success_rate": successes / float(n_episodes),
        "mean_steps": float(np.mean(lengths)),
        "min_steps": int(np.min(lengths)),
        "max_steps": int(np.max(lengths)),
    }


def rollout_events(env, policy, seed=0, max_steps=1000):
    """Play one episode and return the list of the events of the episode.

    The list tells which special tiles were used, for example ``skip`` when
    the bonus tile resolved a chain element.
    """
    state, _info = env.reset(seed=seed)
    events = []
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        action = int(policy[state])
        state, _reward, terminated, truncated, info = env.step(action)
        events.append(info.get("event", ""))
        steps += 1
    return events


# ============================================================
# BASE CLASS
# ============================================================

class BaseAgent:
    """Common interface of every algorithm of this project."""

    name = "base"
    #: True when the algorithm needs the transition model of the environment.
    needs_model = False

    def fit(self, env, seed=42, **kwargs):
        """Train the agent on ``env`` and return the result dictionary."""
        raise NotImplementedError


# ============================================================
# VALUE ITERATION
# ============================================================

class ValueIterationAgent(BaseAgent):
    """Value Iteration, a model based dynamic programming method.

    The Bellman optimality operator is applied until the maximum change of
    the value function drops below ``theta``:

        V(s) <- max_a sum_s' P(s'|s,a) [ r + gamma * V(s') ]
    """

    name = "vi"
    needs_model = True

    def fit(self, env, gamma=0.99, theta=1e-8, max_iterations=10000, **kwargs):
        started = time.perf_counter()
        model = env.build_transition_model()
        n_states = env.n_states
        n_actions = env.action_space.n
        value = np.zeros(n_states)
        deltas = []
        iterations = 0
        for iteration in range(max_iterations):
            new_value = np.zeros(n_states)
            for state in range(n_states):
                best = -np.inf
                for action in range(n_actions):
                    value_sum = 0.0
                    for prob, next_state, reward, terminated in model[state][action]:
                        future = 0.0 if terminated else gamma * value[next_state]
                        value_sum += prob * (reward + future)
                    if value_sum > best:
                        best = value_sum
                new_value[state] = best
            delta = float(np.max(np.abs(new_value - value)))
            deltas.append(delta)
            value = new_value
            iterations = iteration + 1
            if delta < theta:
                break
        q_values = self.q_from_value(model, value, gamma)
        policy = greedy_policy(q_values)
        return {
            "algo": self.name,
            "Q": q_values,
            "V": value,
            "policy": policy,
            "curve": deltas,
            "returns": [],
            "n_iterations": iterations,
            "time_sec": time.perf_counter() - started,
        }

    @staticmethod
    def q_from_value(model, value, gamma):
        """Compute the action values of every state from a value function."""
        n_states = len(model)
        n_actions = len(model[0])
        q_values = np.zeros((n_states, n_actions))
        for state in range(n_states):
            for action in range(n_actions):
                value_sum = 0.0
                for prob, next_state, reward, terminated in model[state][action]:
                    future = 0.0 if terminated else gamma * value[next_state]
                    value_sum += prob * (reward + future)
                q_values[state, action] = value_sum
        return q_values


# ============================================================
# POLICY ITERATION
# ============================================================

class PolicyIterationAgent(BaseAgent):
    """Policy Iteration, a model based dynamic programming method.

    Every outer iteration evaluates the current policy and then improves it.
    The loop stops as soon as the improved policy is identical to the old
    one.
    """

    name = "pi"
    needs_model = True

    def fit(self, env, gamma=0.99, theta=1e-8, max_iterations=1000, **kwargs):
        started = time.perf_counter()
        model = env.build_transition_model()
        n_states = env.n_states
        n_actions = env.action_space.n
        policy = np.zeros(n_states, dtype=int)
        value = np.zeros(n_states)
        deltas = []
        iterations = 0
        for iteration in range(max_iterations):
            value, evaluations = self.policy_evaluation(
                model, policy, gamma, theta
            )
            deltas.append(float(evaluations))
            q_values = ValueIterationAgent.q_from_value(model, value, gamma)
            new_policy = greedy_policy(q_values)
            iterations = iteration + 1
            if np.array_equal(policy, new_policy):
                break
            policy = new_policy
        q_values = ValueIterationAgent.q_from_value(model, value, gamma)
        policy = greedy_policy(q_values)
        return {
            "algo": self.name,
            "Q": q_values,
            "V": value,
            "policy": policy,
            "curve": deltas,
            "returns": [],
            "n_iterations": iterations,
            "time_sec": time.perf_counter() - started,
        }

    @staticmethod
    def policy_evaluation(model, policy, gamma, theta, max_iterations=10000):
        """Iterative evaluation of a fixed policy.

        Returns the value function and the number of sweeps that were
        needed to reach the requested precision.
        """
        n_states = len(model)
        value = np.zeros(n_states)
        for iteration in range(max_iterations):
            new_value = np.zeros(n_states)
            for state in range(n_states):
                value_sum = 0.0
                for prob, next_state, reward, terminated in model[state][policy[state]]:
                    future = 0.0 if terminated else gamma * value[next_state]
                    value_sum += prob * (reward + future)
                new_value[state] = value_sum
            delta = float(np.max(np.abs(new_value - value)))
            value = new_value
            if delta < theta:
                return value, iteration + 1
        return value, max_iterations


# ============================================================
# Q-LEARNING
# ============================================================

class QLearningAgent(BaseAgent):
    """Q-learning, an off policy temporal difference control method.

        Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]

    The target uses the best action of the next state, therefore the agent
    behaves greedily with respect to the current estimate of the optimal
    action values even while it explores.
    """

    name = "qlearning"

    def fit(self, env, episodes=20000, gamma=0.99, alpha=0.1,
            epsilon_start=1.0, epsilon_end=0.05, seed=42, **kwargs):
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        n_states = env.n_states
        n_actions = env.action_space.n
        q_values = np.zeros((n_states, n_actions))
        returns = []
        for episode in range(episodes):
            epsilon = epsilon_schedule(
                episode, episodes, epsilon_start, epsilon_end
            )
            state, _info = env.reset(seed=seed + episode)
            total = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = epsilon_greedy(q_values, state, epsilon, rng, n_actions)
                next_state, reward, terminated, truncated, _info = env.step(action)
                if terminated:
                    target = reward
                else:
                    target = reward + gamma * float(q_values[next_state].max())
                q_values[state, action] += alpha * (target - q_values[state, action])
                state = next_state
                total += reward
            returns.append(total)
        policy = greedy_policy(q_values)
        return {
            "algo": self.name,
            "Q": q_values,
            "V": q_values.max(axis=1),
            "policy": policy,
            "curve": returns,
            "returns": returns,
            "n_iterations": episodes,
            "time_sec": time.perf_counter() - started,
        }


# ============================================================
# SARSA
# ============================================================

class SarsaAgent(BaseAgent):
    """SARSA, an on policy temporal difference control method.

        Q(s,a) <- Q(s,a) + alpha [ r + gamma Q(s',a') - Q(s,a) ]

    The action of the next state is the action that is really taken by the
    behaviour policy, so the agent learns the value of the policy it is
    following instead of the value of the optimal policy.
    """

    name = "sarsa"

    def fit(self, env, episodes=20000, gamma=0.99, alpha=0.1,
            epsilon_start=1.0, epsilon_end=0.05, seed=42, **kwargs):
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        n_states = env.n_states
        n_actions = env.action_space.n
        q_values = np.zeros((n_states, n_actions))
        returns = []
        for episode in range(episodes):
            epsilon = epsilon_schedule(
                episode, episodes, epsilon_start, epsilon_end
            )
            state, _info = env.reset(seed=seed + episode)
            total = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = epsilon_greedy(q_values, state, epsilon, rng, n_actions)
                next_state, reward, terminated, truncated, _info = env.step(action)
                if terminated:
                    target = reward
                else:
                    next_action = epsilon_greedy(
                        q_values, next_state, epsilon, rng, n_actions
                    )
                    target = reward + gamma * q_values[next_state, next_action]
                q_values[state, action] += alpha * (target - q_values[state, action])
                state = next_state
                total += reward
            returns.append(total)
        policy = greedy_policy(q_values)
        return {
            "algo": self.name,
            "Q": q_values,
            "V": q_values.max(axis=1),
            "policy": policy,
            "curve": returns,
            "returns": returns,
            "n_iterations": episodes,
            "time_sec": time.perf_counter() - started,
        }


# ============================================================
# MONTE CARLO CONTROL
# ============================================================

class MonteCarloAgent(BaseAgent):
    """On policy first visit Monte Carlo control.

    The complete episode is generated with the epsilon greedy behaviour
    policy, the returns are computed backwards and the first visit of every
    state is updated with the incremental average of the observed returns.
    There is no bootstrapping, so the algorithm needs a finished episode.
    """

    name = "mc"

    def fit(self, env, episodes=20000, gamma=0.99, alpha=0.1,
            epsilon_start=1.0, epsilon_end=0.05, seed=42, **kwargs):
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        n_states = env.n_states
        n_actions = env.action_space.n
        q_values = np.zeros((n_states, n_actions))
        counts = np.zeros((n_states, n_actions))
        returns = []
        for episode in range(episodes):
            epsilon = epsilon_schedule(
                episode, episodes, epsilon_start, epsilon_end
            )
            state, _info = env.reset(seed=seed + episode)
            trajectory = []
            total = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = epsilon_greedy(q_values, state, epsilon, rng, n_actions)
                next_state, reward, terminated, truncated, _info = env.step(action)
                trajectory.append((state, action, reward))
                state = next_state
                total += reward
            returns.append(total)
            # compute the returns of the episode, backwards
            episode_returns = []
            accumulated = 0.0
            for _state, _action, reward in reversed(trajectory):
                accumulated = reward + gamma * accumulated
                episode_returns.append(accumulated)
            episode_returns.reverse()
            # first visit update with the incremental average
            visited = set()
            for (visited_state, visited_action, _reward), g in zip(
                trajectory, episode_returns
            ):
                if (visited_state, visited_action) in visited:
                    continue
                visited.add((visited_state, visited_action))
                counts[visited_state, visited_action] += 1.0
                q_values[visited_state, visited_action] += (
                    g - q_values[visited_state, visited_action]
                ) / counts[visited_state, visited_action]
        policy = greedy_policy(q_values)
        return {
            "algo": self.name,
            "Q": q_values,
            "V": q_values.max(axis=1),
            "policy": policy,
            "curve": returns,
            "returns": returns,
            "n_iterations": episodes,
            "time_sec": time.perf_counter() - started,
        }


# ============================================================
# REGISTRY
# ============================================================

#: All algorithms of the project, indexed by the name used on the command line.
AGENTS = {
    "vi": ValueIterationAgent,
    "pi": PolicyIterationAgent,
    "qlearning": QLearningAgent,
    "sarsa": SarsaAgent,
    "mc": MonteCarloAgent,
}

#: Pretty names used in the figures and in the reports.
ALGO_LABELS = {
    "vi": "Value Iteration",
    "pi": "Policy Iteration",
    "qlearning": "Q-learning",
    "sarsa": "SARSA",
    "mc": "Monte Carlo",
}

#: Short description of every algorithm, used by the report generators.
ALGO_NOTES = {
    "vi": "Model based, off policy, bootstraps with the Bellman optimality operator.",
    "pi": "Model based, off policy, alternates policy evaluation and policy improvement.",
    "qlearning": "Model free, off policy, bootstraps with max_a Q(s',a).",
    "sarsa": "Model free, on policy, bootstraps with Q(s',a') of the behaviour policy.",
    "mc": "Model free, on policy, no bootstrap, needs a complete episode.",
}


def build_agent(name):
    """Return a fresh agent instance for the given name."""
    if name not in AGENTS:
        raise KeyError(
            "Unknown algorithm %r. Available algorithms: %s"
            % (name, ", ".join(sorted(AGENTS)))
        )
    return AGENTS[name]()
