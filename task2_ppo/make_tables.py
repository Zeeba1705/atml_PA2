from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json
from common.metrics import paired_bootstrap_ci

RUNS= ["sft", "midpoint", "standard", "clip_0.05", "clip_0.2", "clip_0.5", "kl_0.0", "kl_0.1", "kl_0.2"]
CLIP_RUNS= ["clip_0.05", "clip_0.2", "clip_0.5"]
KL_RUNS= ["kl_0.0", "kl_0.1", "kl_0.2"]
BASELINE= "midpoint"

HELDOUT_KEYS= ["reward_mean", "kl_token_mean", "kl_sequence_mean", "entropy", "entropy_exact", "length_mean", "no_eos_fraction"]
HELDOUT_EXTRA= ["reward_std", "length_std", "length_median", "length_iqr", "truncated_fraction", "generated_tokens", "n_responses"]
TRAIN_KEYS= ["clip_epsilon", "kl_beta", "updates", "generated_tokens", "step_retries_total", "wall_clock_sec", "peak_vram_bytes", "gpu_name", "dtype", "git_commit"]
PAIRED_KEYS= ["reward", "n_tokens"]

#one panel per logged quantity of the standard run
TRAJECTORY_PANELS= [
    ("reward", "Learned reward"),
    ("kl", "KL from reference (sampled)"),
    ("policy_loss", "Policy loss"),
    ("value_loss", "Value loss"),
    ("entropy", "Entropy (sampled)"),
    ("clip_fraction", "Clip fraction"),
    ("grad_norm", "Policy gradient norm"),
    ("response_length", "Response length (tokens)"),
    ("explained_variance", "Critic explained variance"),
    ("max_ratio_deviation", "Largest |ratio - 1|"),
]
FORK_PANELS= [
    ("reward", "Learned reward"),
    ("kl", "KL from reference (sampled)"),
    ("delta_kl", "Policy movement per update (delta_kl)"),
    ("entropy", "Entropy (sampled)"),
    ("response_length", "Response length (tokens)"),
    ("grad_norm", "Policy gradient norm"),
]
HELDOUT_PANELS= [
    ("reward_mean", "Held-out reward"),
    ("kl_token_mean", "Held-out KL per token"),
    ("entropy", "Held-out entropy (sampled)"),
    ("length_mean", "Held-out length (tokens)"),
]

#fixed categorical order, light surface
SERIES= ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE= "#fcfcfb"
INK= "#0b0b0b"
INK_SECONDARY= "#52514e"
MUTED= "#898781"
GRID= "#e1e0d9"
AXIS= "#c3c2b7"


def style_axis(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)


def save_figure(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)

    print("saved", path)


def load_runs(results_dir):
    #whatever exists is used, a run without an evaluation or a training log keeps empty columns
    runs= {}

    for name in RUNS:
        run_dir= results_dir / name
        run= {"eval": None, "train": None, "log": None, "generations": None}

        if (run_dir / "eval_metrics.json").exists():
            run["eval"]= load_json(run_dir / "eval_metrics.json")
            run["generations"]= {r["prompt_id"]: r for r in read_jsonl(run_dir / "generations.jsonl")}

        if (run_dir / "train_metrics.json").exists():
            run["train"]= load_json(run_dir / "train_metrics.json")
            run["log"]= read_jsonl(run_dir / "train_log.jsonl")

        runs[name]= run

        print(
            f"{name}: "
            f"evaluation={'yes' if run['eval'] is not None else 'MISSING'}, "
            f"training log={'yes' if run['log'] is not None else 'none'}"
        )

    return runs


def summary_table(runs, results_dir):
    stability= {}

    if (results_dir / "clipping_forks.json").exists():
        forks= load_json(results_dir / "clipping_forks.json")

        for fork in forks["forks"]:
            stability[fork["run_name"]]= fork

    rows= []

    for name in RUNS:
        run= runs[name]
        row= {"run": name}

        if run["train"] is not None:
            for key in TRAIN_KEYS:
                row["train_" + key]= run["train"].get(key)

        if run["eval"] is not None:
            heldout= run["eval"]["heldout"]

            for key in HELDOUT_KEYS:
                row[key]= heldout[key]
                row[key + "_ci_low"]= heldout[key + "_ci95"][0]
                row[key + "_ci_high"]= heldout[key + "_ci95"][1]

            for key in HELDOUT_EXTRA:
                row[key]= heldout[key]

            row["eval_gpu_name"]= run["eval"]["gpu_name"]
            row["eval_git_commit"]= run["eval"]["git_commit"]

        if name in stability:
            row["s1_std_delta_kl"]= stability[name]["s1_std_delta_kl"]
            row["s2_updates_above_threshold"]= stability[name]["s2_updates_above_threshold"]

        rows.append(row)

    return pd.DataFrame(rows)


def paired_table(runs, seed):
    #same held-out prompts for every run, so each run is compared with the baseline prompt by prompt
    rows= []

    comparisons= [(name, BASELINE) for name in RUNS if name != BASELINE]
    comparisons += [("kl_0.0", "kl_0.1"), ("kl_0.2", "kl_0.1")]

    for name, base in comparisons:
        if runs[name]["generations"] is None or runs[base]["generations"] is None:
            continue

        a= runs[name]["generations"]
        b= runs[base]["generations"]

        prompt_ids= [p for p in a if p in b]

        row= {
            "run": name,
            "baseline": base,
            "n_prompts": len(prompt_ids),
            "identical_responses": sum(a[p]["response"] == b[p]["response"] for p in prompt_ids),
        }

        for key in PAIRED_KEYS:
            values_a= [float(a[p][key]) for p in prompt_ids]
            values_b= [float(b[p][key]) for p in prompt_ids]

            low, high = paired_bootstrap_ci(values_a, values_b, seed)

            row[key + "_diff"]= float(np.mean(values_a) - np.mean(values_b))
            row[key + "_diff_ci_low"]= low
            row[key + "_diff_ci_high"]= high

        rows.append(row)

    return pd.DataFrame(rows)


def cached_table(results_dir):
    rows= []

    for name in ["midpoint", "standard"]:
        path= results_dir / f"clipping_cached_{name}.json"

        if not path.exists():
            continue

        cached= load_json(path)

        for eps in cached["epsilons"]:
            rows.append({
                "policy": name,
                "clip_epsilon": eps["clip_epsilon"],
                "clipped_surrogate": eps["clipped_surrogate"],
                "unclipped_surrogate": cached["unclipped_surrogate"],
                "clip_fraction": eps["clip_fraction"],
                "clip_fraction_ci_low": eps["clip_fraction_ci95"][0],
                "clip_fraction_ci_high": eps["clip_fraction_ci95"][1],
                "binding_fraction": eps["binding_fraction"],
                "binding_fraction_ci_low": eps["binding_fraction_ci95"][0],
                "binding_fraction_ci_high": eps["binding_fraction_ci95"][1],
                "n_rows": cached["n_rows"],
                "n_tokens": cached["n_tokens"],
            })

    return pd.DataFrame(rows)


def plot_standard(log, path):
    updates= [r["update"] for r in log]

    fig, axes = plt.subplots(2, 5, figsize=(17, 6))

    for ax, (key, title) in zip(axes.flat, TRAJECTORY_PANELS):
        style_axis(ax, title)

        ax.plot(updates, [r[key] for r in log], color=SERIES[0], linewidth=2, marker="o", markersize=4)
        ax.set_xticks([1, 5, 10, 15, 20])

    for ax in axes[1]:
        ax.set_xlabel("update", fontsize=8, color=INK_SECONDARY)

    fig.suptitle("Standard PPO continuation: 20 updates from the midpoint, one prompt per update", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def plot_forks(runs, names, labels, title, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))

    for ax, (key, panel_title) in zip(axes.flat, FORK_PANELS):
        style_axis(ax, panel_title)

        for i in range(len(names)):
            log= runs[names[i]]["log"]

            if log is None:
                continue

            ax.plot([r["update"] for r in log], [r[key] for r in log], color=SERIES[i], linewidth=2, marker="o", markersize=4, label=labels[i])

    for ax in axes[1]:
        ax.set_xlabel("update", fontsize=8, color=INK_SECONDARY)

    axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def plot_heldout(summary, path):
    table= summary.dropna(subset=["reward_mean"])
    names= list(table["run"])
    positions= list(range(len(names)))[::-1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)

    for ax, (key, title) in zip(axes, HELDOUT_PANELS):
        style_axis(ax, title)

        values= np.array(table[key])
        low= np.array(table[key + "_ci_low"])
        high= np.array(table[key + "_ci_high"])

        ax.errorbar(values, positions, xerr=[values - low, high - values], fmt="o", color=SERIES[0], ecolor=SERIES[0], elinewidth=2, markersize=6, capsize=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=9, color=INK_SECONDARY)

    fig.suptitle("Held-out evaluation, 200 prompts: mean and 95% bootstrap interval", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def plot_paired(paired, path):
    table= paired[paired["baseline"] == BASELINE]
    names= list(table["run"])
    positions= list(range(len(names)))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, (key, title) in zip(axes, [("reward", "Reward difference"), ("n_tokens", "Length difference (tokens)")]):
        style_axis(ax, title)

        values= np.array(table[key + "_diff"])
        low= np.array(table[key + "_diff_ci_low"])
        high= np.array(table[key + "_diff_ci_high"])

        ax.axvline(0.0, color=AXIS, linewidth=1.5)
        ax.errorbar(values, positions, xerr=[values - low, high - values], fmt="o", color=SERIES[0], ecolor=SERIES[0], elinewidth=2, markersize=6, capsize=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=9, color=INK_SECONDARY)

    fig.suptitle(f"Per-prompt difference from the {BASELINE} on the same 200 prompts: mean and 95% paired bootstrap interval", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def plot_cached(cached, path):
    policies= list(dict.fromkeys(cached["policy"]))
    eps_values= sorted(set(cached["clip_epsilon"]))
    width= 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (key, title) in zip(axes, [("clip_fraction", "Clip fraction (ratio outside the range)"), ("binding_fraction", "Binding fraction (clipped term used)")]):
        style_axis(ax, title)

        for i in range(len(policies)):
            rows= cached[cached["policy"] == policies[i]].sort_values("clip_epsilon")

            x= np.arange(len(eps_values)) + (i - (len(policies) - 1) / 2) * (width + 0.03)
            values= np.array(rows[key])
            low= np.array(rows[key + "_ci_low"])
            high= np.array(rows[key + "_ci_high"])

            ax.bar(x, values, width=width, color=SERIES[i], label=policies[i])
            ax.errorbar(x, values, yerr=[values - low, high - values], fmt="none", ecolor=INK_SECONDARY, elinewidth=1.5, capsize=0)

            for k in range(len(values)):
                ax.text(x[k], high[k], f" {values[k]:.3f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)

        ax.set_xticks(np.arange(len(eps_values)))
        ax.set_xticklabels([f"eps = {e:g}" for e in eps_values], fontsize=9, color=INK_SECONDARY)

    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle("Cached rollout batch (32 rows): fraction of response tokens, 95% bootstrap interval over prompts", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def main(args):
    cfg= load_yaml(args.config)

    results_dir= repo_path(cfg["results_dir"])
    figures_dir= results_dir / "figures"

    os.makedirs(figures_dir, exist_ok=True)

    seed= int(cfg["seed"])

    runs= load_runs(results_dir)

    summary= summary_table(runs, results_dir)
    summary.to_csv(results_dir / "summary.csv", index=False)
    print("saved", results_dir / "summary.csv")

    paired= paired_table(runs, seed)
    paired.to_csv(results_dir / "paired_differences.csv", index=False)
    print("saved", results_dir / "paired_differences.csv")

    cached= cached_table(results_dir)

    if len(cached) > 0:
        cached.to_csv(results_dir / "clipping_cached.csv", index=False)
        print("saved", results_dir / "clipping_cached.csv")

        plot_cached(cached, figures_dir / "clipping_cached.png")

    if runs["standard"]["log"] is not None:
        pd.DataFrame(runs["standard"]["log"]).drop(columns=["prompt_rows"]).to_csv(results_dir / "standard_trajectory.csv", index=False)
        print("saved", results_dir / "standard_trajectory.csv")

        plot_standard(runs["standard"]["log"], figures_dir / "standard_trajectories.png")

    plot_forks(
        runs,
        CLIP_RUNS,
        ["eps = 0.05", "eps = 0.2", "eps = 0.5"],
        "Clipping forks: 8 updates from the midpoint, same prompt sequence (the three lines coincide where the runs are identical)",
        figures_dir / "clipping_fork_trajectories.png"
    )

    plot_forks(
        runs,
        KL_RUNS,
        ["beta_KL = 0", "beta_KL = 0.1", "beta_KL = 0.2"],
        "KL-pressure forks: 8 updates from the midpoint, same prompt sequence",
        figures_dir / "kl_fork_trajectories.png"
    )

    if "reward_mean" in summary.columns:
        plot_heldout(summary, figures_dir / "heldout_comparison.png")

    if len(paired) > 0:
        plot_paired(paired, figures_dir / "paired_differences.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo.yaml"
    )

    args = parser.parse_args()

    main(args)
