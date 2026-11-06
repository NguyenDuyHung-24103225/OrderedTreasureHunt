
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# The Windows console uses cp1252, so printing Vietnamese would raise a
# UnicodeEncodeError. The output is switched to utf-8 here.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from agents import (
    ALGO_LABELS,
    ALGO_NOTES,
    build_agent,
    evaluate_policy,
    rollout_events,
)
from configs import (
    ALGO_ORDER,
    ALGO_PARAMS,
    CONFIGS,
    REWARD_EXPERIMENT,
    REWARD_PROFILES,
    SEED_EXPERIMENT,
    config_names,
    get_config,
    get_reward_profile,
)
from envs import LEVELS, OrderedTreasureHuntEnv
import plots


# ============================================================
# CONSTANTS
# ============================================================

FIGURE_DIR = "figures"
RESULT_FILE = "results.json"
MAX_STEPS = 100
CURVE_POINTS = 200


# ============================================================
# HELPERS
# ============================================================

def make_env(config_name, reward_profile=None, max_steps=MAX_STEPS):
    """Build the environment of a configuration.

    Returns the environment, the configuration dictionary and the name of the
    reward profile that was used.
    """
    config = get_config(config_name)
    profile = reward_profile or config["reward_profile"]
    rewards = get_reward_profile(profile)
    env = OrderedTreasureHuntEnv(
        level=config["level"], rewards=rewards, max_steps=max_steps
    )
    return env, config, profile


def train_one(config_name, algo, episodes=None, seed=None,
              reward_profile=None, eval_episodes=None):
    """Train one algorithm and evaluate the policy it produced.

    Returns a dictionary with the metrics of the run. The keys that start
    with an underscore hold the objects that are needed by the figures and
    are removed before the results are written to the file.
    """
    env, config, profile = make_env(config_name, reward_profile)
    seed = config["seed"] if seed is None else seed
    episodes = config["episodes"] if episodes is None else episodes
    eval_episodes = config["eval_episodes"] if eval_episodes is None else eval_episodes
    params = dict(ALGO_PARAMS[algo])
    if algo in ("qlearning", "sarsa", "mc"):
        params["episodes"] = episodes
    agent = build_agent(algo)
    result = agent.fit(env, seed=seed, **params)
    metrics = evaluate_policy(
        env, result["policy"], n_episodes=eval_episodes, seed=seed + 100000
    )
    returns = np.asarray(result["returns"], dtype=float)
    events = rollout_events(env, result["policy"], seed=seed)
    summary = {
        "config": config_name,
        "level": config["level"],
        "algo": algo,
        "algo_label": ALGO_LABELS[algo],
        "algo_note": ALGO_NOTES[algo],
        "reward_profile": profile,
        "n_states": int(env.n_states),
        "n_actions": int(env.action_space.n),
        "n_stages": int(env.n_stages),
        "grid": "%dx%d" % (env.height, env.width),
        "episodes": int(episodes),
        "eval_episodes": int(eval_episodes),
        "seed": int(seed),
        "gamma": float(params.get("gamma", 0.99)),
        "train_return_mean": float(returns.mean()) if returns.size else None,
        "train_return_std": float(returns.std()) if returns.size else None,
        "n_iterations": int(result["n_iterations"]),
        "time_sec": float(result["time_sec"]),
        "uses_skip_tile": "skip" in events,
        "reaches_exit": "exit" in events,
    }
    summary.update(metrics)
    summary["curve"] = downsample(result["curve"], CURVE_POINTS)
    # objects that are only needed while the figures are built
    summary["_policy"] = result["policy"]
    summary["_curve_full"] = result["curve"]
    summary["_policy_text"] = env.policy_text(result["policy"])
    return summary


def downsample(values, points):
    """Reduce a long curve to at most ``points`` values."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return []
    if values.size <= points:
        return [float(v) for v in values]
    step = values.size / float(points)
    picked = [values[int(i * step)] for i in range(points)]
    return [float(v) for v in picked]


def strip_artifacts(summary):
    """Remove the private objects of a run so that it can be serialised."""
    clean = {key: value for key, value in summary.items()
             if not key.startswith("_")}
    return clean


# ============================================================
# MODES
# ============================================================

def mode_train(args):
    """Train one algorithm and print the metrics of the run."""
    summary = train_one(
        args.config, args.algo, episodes=args.episodes, seed=args.seed
    )
    print("Thuật toán     : %s (%s)" % (summary["algo_label"], summary["algo"]))
    print("Cấu hình       : %s" % summary["config"])
    print("Không gian state: %d state, %d hành động" % (
        summary["n_states"], summary["n_actions"]))
    print("Số episode     : %d" % summary["episodes"])
    print("Return đánh giá: %.2f +/- %.2f" % (
        summary["eval_return_mean"], summary["eval_return_std"]))
    print("Tỷ lệ thành công: %.1f %%" % (100.0 * summary["success_rate"]))
    print("Số bước trung bình: %.1f" % summary["mean_steps"])
    print("Thời gian huấn luyện: %.2f s" % summary["time_sec"])
    if args.policy:
        env, _config, _profile = make_env(args.config)
        print()
        print("Policy tham lam (mỗi khối ứng với một giá trị của task_progress):")
        for line in summary["_policy_text"]:
            print("   " + line)
    return summary


def mode_compare(args):
    """Run every algorithm on every setting and write the results file."""
    results = run_comparison(
        episodes=args.episodes,
        seed=args.seed,
        out=args.out,
        figures=not args.no_figures,
    )
    return results


def run_comparison(episodes=None, seed=None, out=None, figures=True):
    """Train every algorithm on every setting and collect the results.

    The function is used by the command line and by the notebook, so both
    produce exactly the same numbers.

    Returns the dictionary that is written to the results file.
    """
    all_runs = []
    artifacts = {}
    levels_info = {}
    for config_name in config_names():
        env, config, _profile = make_env(config_name)
        levels_info[config_name] = {
            "title": config["title"],
            "description": config["description"],
            "level": config["level"],
            "height": int(env.height),
            "width": int(env.width),
            "n_states": int(env.n_states),
            "n_actions": int(env.action_space.n),
            "n_stages": int(env.n_stages),
            "chain": env.chain,
            "aux": env.aux_names,
        }
        print("=" * 74)
        print("Cấu hình %s : %s" % (config_name, config["title"]))
        print("  |S| = %d, |A| = %d, lưới %dx%d"
              % (env.n_states, env.action_space.n, env.height, env.width))
        artifacts[config_name] = {}
        for algo in ALGO_ORDER:
            started = time.perf_counter()
            summary = train_one(
                config_name, algo, episodes=episodes, seed=seed
            )
            artifacts[config_name][algo] = summary
            all_runs.append(strip_artifacts(summary))
            print(
                "  %-18s return %8.2f +/- %6.2f   success %5.1f %%   "
                "steps %5.1f   %6.2f s"
                % (
                    summary["algo_label"],
                    summary["eval_return_mean"],
                    summary["eval_return_std"],
                    100.0 * summary["success_rate"],
                    summary["mean_steps"],
                    time.perf_counter() - started,
                )
            )
    print("=" * 74)
    print("Thí nghiệm reward trên %s" % REWARD_EXPERIMENT["level"])
    reward_data = run_reward_experiment(episodes, seed, artifacts)
    print("=" * 74)
    print("Thí nghiệm seed trên %s" % SEED_EXPERIMENT["level"])
    seed_data = run_seed_experiment(episodes)
    results = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "episodes": episodes,
            "seed": seed,
            "max_steps": MAX_STEPS,
            "algorithms": list(ALGO_ORDER),
            "algorithm_labels": dict(ALGO_LABELS),
            "algorithm_notes": dict(ALGO_NOTES),
            "reward_profiles": REWARD_PROFILES,
        },
        "levels": levels_info,
        "runs": all_runs,
        "reward_experiment": reward_data,
        "seed_experiment": seed_data,
    }
    out = out or RESULT_FILE
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print("=" * 74)
    print("Kết quả đã lưu vào %s" % out)
    if figures:
        build_figures(results, artifacts)
    return results


def run_reward_experiment(episodes, seed, artifacts=None):
    """Compare the two reward profiles on level 4.

    For every profile the optimal policy (Policy Iteration) and the policy
    that Q-learning learns are measured, so that the effect of the reward on
    the behaviour of the agent is visible.
    """
    data = {
        "level": REWARD_EXPERIMENT["level"],
        "algorithm": REWARD_EXPERIMENT["algorithm"],
        "profiles": [],
    }
    for profile in REWARD_EXPERIMENT["profiles"]:
        entry = {
            "profile": profile,
            "rewards": REWARD_PROFILES[profile],
        }
        for tag, algo in (("optimal", "pi"), ("learned", REWARD_EXPERIMENT["algorithm"])):
            summary = train_one(
                REWARD_EXPERIMENT["level"],
                algo,
                episodes=episodes,
                seed=seed,
                reward_profile=profile,
                eval_episodes=REWARD_EXPERIMENT["eval_episodes"],
            )
            if artifacts is not None:
                artifacts.setdefault("reward_" + profile, {})[algo] = summary
            entry[tag] = {
                "algo": algo,
                "eval_return_mean": summary["eval_return_mean"],
                "eval_return_std": summary["eval_return_std"],
                "success_rate": summary["success_rate"],
                "mean_steps": summary["mean_steps"],
                "uses_skip_tile": summary["uses_skip_tile"],
            }
        data["profiles"].append(entry)
        print("  hồ sơ reward %-16s tối ưu: return %8.2f bước %5.1f dùng ô K %-5s"
              "   học được: return %8.2f bước %5.1f dùng ô K %-5s"
              % (
                  profile,
                  entry["optimal"]["eval_return_mean"],
                  entry["optimal"]["mean_steps"],
                  entry["optimal"]["uses_skip_tile"],
                  entry["learned"]["eval_return_mean"],
                  entry["learned"]["mean_steps"],
                  entry["learned"]["uses_skip_tile"],
              ))
    return data


def run_seed_experiment(episodes):
    """Train the algorithms of the hard map with several random seeds.

    On the hard map the algorithms without a model can stop on the short path
    and never try the long one again. This experiment shows how often that
    happens, and if it depends on the random seed.
    """
    data = {
        "level": SEED_EXPERIMENT["level"],
        "seeds": list(SEED_EXPERIMENT["seeds"]),
        "runs": [],
    }
    for algo in SEED_EXPERIMENT["algorithms"]:
        for seed in SEED_EXPERIMENT["seeds"]:
            summary = train_one(
                SEED_EXPERIMENT["level"], algo, episodes=episodes, seed=seed,
                eval_episodes=SEED_EXPERIMENT["eval_episodes"],
            )
            data["runs"].append({
                "algo": algo,
                "algo_label": summary["algo_label"],
                "seed": seed,
                "eval_return_mean": summary["eval_return_mean"],
                "mean_steps": summary["mean_steps"],
                "uses_skip_tile": summary["uses_skip_tile"],
            })
            print("  seed %-4d %-18s return %8.2f  steps %5.1f  dùng ô thưởng: %s"
                  % (seed, summary["algo_label"],
                     summary["eval_return_mean"], summary["mean_steps"],
                     summary["uses_skip_tile"]))
    return data


def build_figures(results, artifacts):
    """Create every figure of the report from the results of the comparison."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    runs = results["runs"]
    levels = {}
    for config_name in config_names():
        env, _config, _profile = make_env(config_name)
        levels[config_name] = env
    plots.save_levels_figure(
        levels, os.path.join(FIGURE_DIR, "notebook_levels.png")
    )
    for config_name in config_names():
        env = levels[config_name]
        subset = artifacts[config_name]
        # the policy of Policy Iteration is the optimal one
        plots.save_environment_figure(
            env,
            os.path.join(FIGURE_DIR, "env_%s.png" % config_name),
            policy=subset["pi"]["_policy"],
            title="%s  (Policy Iteration)" % env.title,
        )
        # learning curves of the model free algorithms
        curves = {}
        for algo in ("qlearning", "sarsa", "mc"):
            curves[ALGO_LABELS[algo]] = subset[algo]["_curve_full"]
        plots.save_learning_curves(
            curves,
            os.path.join(FIGURE_DIR, "learning_curves_%s.png" % config_name),
            "Đường học - %s" % env.title,
        )
        # convergence of the dynamic programming methods
        dp_curves = {}
        for algo in ("vi", "pi"):
            dp_curves[ALGO_LABELS[algo]] = subset[algo]["_curve_full"]
        plots.save_dp_convergence(
            dp_curves,
            os.path.join(FIGURE_DIR, "dp_convergence_%s.png" % config_name),
            "Hội tụ của các phương pháp quy hoạch động - %s" % env.title,
        )
    # comparison charts
    plots.save_comparison_chart(
        runs,
        os.path.join(FIGURE_DIR, "comparison_return.png"),
        "eval_return_mean",
        "Mean return của mọi thuật toán trên mọi setting",
        "mean return của policy tham lam",
    )
    plots.save_comparison_chart(
        runs,
        os.path.join(FIGURE_DIR, "comparison_success_rate.png"),
        "success_rate",
        "Tỷ lệ thành công của mọi thuật toán trên mọi setting",
        "tỷ lệ thành công",
    )
    # markov property
    env1, _c, _p = make_env("level1")
    plots.save_markov_figure(
        env1, os.path.join(FIGURE_DIR, "markov_proof.png"), level="level1"
    )
    # reward profiles
    env3, _c, _p = make_env("level3")
    entries = []
    for entry in results["reward_experiment"]["profiles"]:
        for tag, algo in (("optimal", "pi"), ("learned", REWARD_EXPERIMENT["algorithm"])):
            summary = artifacts["reward_" + entry["profile"]][algo]
            entries.append({
                "name": "%s - %s" % (entry["profile"], tag),
                "policy": summary["_policy"],
                "progress": 1,
                "caption": "step penalty = %.1f\nreturn = %.1f, steps = %.1f\nbonus tile used: %s"
                           % (
                               entry["rewards"]["step_penalty"],
                               entry[tag]["eval_return_mean"],
                               entry[tag]["mean_steps"],
                               entry[tag]["uses_skip_tile"],
                           ),
            })
    plots.save_policy_comparison(
        env3,
        entries,
        os.path.join(FIGURE_DIR, "reward_profiles_level3.png"),
        "Level 3 - policy sau khi lấy vật phẩm A phụ thuộc chi phí của một bước",
    )
    # the failure case of Monte Carlo on level 3
    mc_summary = artifacts["level3"]["mc"]
    pi_summary = artifacts["level3"]["pi"]
    plots.save_policy_comparison(
        env3,
        [
            {
                "name": "Policy Iteration (tối ưu)",
                "policy": pi_summary["_policy"],
                "progress": 1,
                "caption": "return = %.1f, steps = %.1f\ndùng ô thưởng: %s"
                           % (
                               pi_summary["eval_return_mean"],
                               pi_summary["mean_steps"],
                               pi_summary["uses_skip_tile"],
                           ),
            },
            {
                "name": "Monte Carlo (học được)",
                "policy": mc_summary["_policy"],
                "progress": 1,
                "caption": "return = %.1f, steps = %.1f\ndùng ô thưởng: %s"
                           % (
                               mc_summary["eval_return_mean"],
                               mc_summary["mean_steps"],
                               mc_summary["uses_skip_tile"],
                           ),
            },
        ],
        os.path.join(FIGURE_DIR, "local_optimum_level3.png"),
        "Level 3 - một cực điểm cục bộ của Monte Carlo control",
    )
    # the result of the seed experiment
    plots.save_seed_comparison(
        results["seed_experiment"],
        os.path.join(FIGURE_DIR, "seed_experiment_level3.png"),
        "Level 3 - mean return của mỗi thuật toán theo từng seed",
    )
    # the trajectory of the auto play, used by the report and by the slides
    for config_name in ("level1", "level3"):
        env = levels[config_name]
        plots.save_trajectory_figure(
            env,
            artifacts[config_name]["qlearning"]["_policy"],
            os.path.join(
                FIGURE_DIR, "trajectory_%s_qlearning.png" % config_name
            ),
            env.title,
            seed=artifacts[config_name]["qlearning"]["seed"],
        )
    print("Các hình đã lưu vào %s" % FIGURE_DIR)


def mode_render(args):
    """Draw the environment and the learned policy."""
    env, config, _profile = make_env(args.config)
    summary = train_one(
        args.config, args.algo, episodes=args.episodes, seed=args.seed
    )
    out = args.out or os.path.join(FIGURE_DIR, "policy_%s_%s.png" % (
        args.config, args.algo))
    plots.save_environment_figure(
        env, out, policy=summary["_policy"],
        title="%s  (%s)" % (env.title, summary["algo_label"]),
    )
    print("Hình đã lưu vào %s" % out)
    return summary


def mode_autoplay(args):
    """Play one episode with the learned policy."""
    env, config, _profile = make_env(args.config)
    summary = train_one(
        args.config, args.algo, episodes=args.episodes, seed=args.seed
    )
    policy = summary["_policy"]
    state, _info = env.reset(seed=args.seed)
    total = 0.0
    steps = 0
    terminated = False
    truncated = False
    print("Tự chơi: %s trên %s" % (summary["algo_label"], config["title"]))
    print("-" * 74)
    while not (terminated or truncated) and steps < MAX_STEPS:
        action = int(policy[state])
        row, col, progress, flags = env._states[state]
        state, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        print(
            "  bước %2d  ô (%d,%d) tiến độ %d  hành động %-5s  thưởng %7.2f  "
            "sự kiện %s"
            % (steps, row, col, progress, action, reward, info.get("event", ""))
        )
    print("-" * 74)
    print("  return = %.2f, %d bước, kết thúc = %s" % (
        total, steps, terminated))
    if args.gif:
        plots.save_episode_animation(
            env, policy, args.gif, config["title"], fps=args.fps, seed=args.seed
        )
        print("Hình động đã lưu vào %s" % args.gif)
    trajectory = os.path.join(
        FIGURE_DIR, "trajectory_%s_%s.png" % (args.config, args.algo)
    )
    plots.save_trajectory_figure(
        env, policy, trajectory, config["title"], seed=args.seed
    )
    print("Hình quỹ đạo đã lưu vào %s" % trajectory)
    return summary


# ============================================================
# COMMAND LINE
# ============================================================

def build_parser():
    """Build the argument parser of the program."""
    parser = argparse.ArgumentParser(
        description="Ordered Treasure Hunt - De tai 08 (Reinforcement Learning)"
    )
    parser.add_argument(
        "--algo", choices=sorted(ALGO_ORDER),
        help="algorithm that is trained or played"
    )
    parser.add_argument(
        "--config", default="level1", choices=sorted(CONFIGS),
        help="setting of the assignment"
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="number of training episodes"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--policy", action="store_true",
                        help="print the greedy policy on the console")
    parser.add_argument(
        "--compare", action="store_true",
        help="train every algorithm on every setting and write the results file"
    )
    parser.add_argument(
        "--out", default=None,
        help="path of the results file written by --compare"
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="do not create the figures when --compare is used"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="draw the environment with the learned policy"
    )
    parser.add_argument(
        "--autoplay", action="store_true",
        help="play one episode with the learned policy"
    )
    parser.add_argument("--gif", default=None,
                        help="file the auto play animation is written to")
    parser.add_argument("--fps", type=int, default=6,
                        help="frames per second of the animation")
    return parser


def main(argv=None):
    """Entry point of the program."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = get_config(args.config)["seed"]
    if args.compare:
        mode_compare(args)
    elif args.render:
        if args.algo is None:
            parser.error("--render needs --algo")
        mode_render(args)
    elif args.autoplay:
        if args.algo is None:
            parser.error("--autoplay needs --algo")
        mode_autoplay(args)
    else:
        if args.algo is None:
            parser.error("an algorithm is needed, use --algo or --compare")
        mode_train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
