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
from task2_ppo.make_tables import AXIS, INK, INK_SECONDARY, MUTED, SERIES, plot_heldout, plot_paired, save_figure, style_axis

RUNS= ["sft", "midpoint", "standard", "norm_grpo", "norm_dr_grpo"]
FORK_RUNS= ["norm_grpo", "norm_dr_grpo"]
BASELINE= "midpoint"
#the base policy under the identical held-out protocol was evaluated once, in task 2
SFT_SOURCE= "results/task2_ppo/sft"

HELDOUT_KEYS= ["reward_mean", "kl_token_mean", "kl_sequence_mean", "entropy", "entropy_exact", "length_mean", "no_eos_fraction"]
HELDOUT_EXTRA= ["reward_std", "length_std", "length_median", "length_iqr", "truncated_fraction", "generated_tokens", "n_responses"]
TRAIN_KEYS= ["loss_type", "clip_epsilon", "kl_beta", "updates", "optimizer_steps", "updates_without_gradient", "num_generations", "generated_tokens", "step_retries_total", "wall_clock_sec", "peak_vram_bytes", "gpu_name", "dtype", "git_commit"]
PAIRED_KEYS= ["reward", "n_tokens"]

TRAJECTORY_PANELS= [
    ("reward", "Mean reward of the group"),
    ("kl", "KL from reference (sampled)"),
    ("reward_std_within_group", "Within-group reward std"),
    ("uninformative_group_fraction", "Uninformative group (1 = yes)"),
    ("policy_loss", "Policy loss"),
    ("grad_norm", "Gradient norm"),
    ("entropy", "Entropy (sampled)"),
    ("response_length", "Completion length (tokens)"),
    ("masked_completions", "Masked completions (of K)"),
    ("max_ratio_deviation_after_update", "Largest |ratio - 1| after the update"),
]
FORK_PANELS= [
    ("reward", "Mean reward of the group"),
    ("kl", "KL from reference (sampled)"),
    ("reward_std_within_group", "Within-group reward std"),
    ("grad_norm", "Gradient norm"),
    ("response_length", "Completion length (tokens)"),
    ("masked_completions", "Masked completions (of K)"),
]
GROUP_PANELS= [
    ("informative_rate", "Informative-group rate"),
    ("reward_std", "Mean within-group reward std"),
    ("centred_reward_variance", "Variance of the centred reward"),
]
GROUP_BINS= ["hard", "medium", "easy"]


def load_runs(results_dir):
    #whatever exists is used, a run without an evaluation or a training log keeps empty columns
    runs= {}

    for name in RUNS:
        run_dir= results_dir / name
        source= str(run_dir)

        if name == "sft" and not (run_dir / "eval_metrics.json").exists():
            run_dir= repo_path(SFT_SOURCE)
            source= SFT_SOURCE

        run= {"eval": None, "train": None, "log": None, "generations": None, "source": source}

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


def summary_table(runs):
    rows= []

    for name in RUNS:
        run= runs[name]
        row= {"run": name}

        if run["train"] is not None:
            for key in TRAIN_KEYS:
                row["train_" + key]= run["train"].get(key)

        if run["log"] is not None:
            log= run["log"]

            row["train_reward_std_within_group_mean"]= float(np.mean([r["reward_std_within_group"] for r in log]))
            row["train_uninformative_group_fraction"]= float(np.mean([r["uninformative_group_fraction"] for r in log]))
            row["train_truncated_fraction_mean"]= float(np.mean([r["truncated_fraction"] for r in log]))
            row["train_masked_completions_total"]= int(sum(r["masked_completions"] for r in log))

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
            row["eval_source"]= run["source"]

        rows.append(row)

    return pd.DataFrame(rows)


def paired_table(runs, seed):
    #same held-out prompts for every run, so runs are compared prompt by prompt
    rows= []

    comparisons= [(name, BASELINE) for name in RUNS if name != BASELINE]
    comparisons.append(("norm_dr_grpo", "norm_grpo"))

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


def group_size_table(study):
    rows= []

    for result in study["results"]:
        for name in ["all"] + GROUP_BINS:
            part= result[name]

            row= {
                "group_size": result["group_size"],
                "groups_per_prompt": result["groups_per_prompt"],
                "total_generations": result["total_generations"],
                "difficulty": name,
                "n_prompts": part["n_prompts"],
            }

            for key in ["informative_rate", "reward_std", "advantage_variance", "centred_reward_variance"]:
                row[key]= part[key]
                row[key + "_ci_low"]= part[key + "_ci95"][0]
                row[key + "_ci_high"]= part[key + "_ci95"][1]

            rows.append(row)

    return pd.DataFrame(rows)


def length_bin_table(study):
    rows= []

    for run in study["runs"]:
        for b in run["length_bins"]:
            row= {"run": run["run_name"], "loss_type": run["loss_type"]}
            row.update(b)

            rows.append(row)

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

    fig.suptitle("Standard GRPO continuation: 20 updates from the midpoint, one prompt with K = 4 completions per update", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def plot_forks(runs, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))

    for ax, (key, title) in zip(axes.flat, FORK_PANELS):
        style_axis(ax, title)

        for i in range(len(FORK_RUNS)):
            log= runs[FORK_RUNS[i]]["log"]

            if log is None:
                continue

            ax.plot([r["update"] for r in log], [r[key] for r in log], color=SERIES[i], linewidth=2, marker="o", markersize=4, label=FORK_RUNS[i])

    for ax in axes[1]:
        ax.set_xlabel("update", fontsize=8, color=INK_SECONDARY)

    axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle("Normalisation forks: 8 updates from the midpoint, same prompt sequence, each fork samples its own completions", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def plot_group_size(table, path):
    sizes= sorted(set(table["group_size"]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, (key, title) in zip(axes, GROUP_PANELS):
        style_axis(ax, title)

        #overall in muted ink, the three difficulty thirds in the categorical colours
        for i, name in enumerate(["all"] + GROUP_BINS):
            rows= table[table["difficulty"] == name].sort_values("group_size")

            color= MUTED if name == "all" else SERIES[i - 1]
            x= np.arange(len(sizes)) + (i - 1.5) * 0.08

            values= np.array(rows[key])
            low= np.array(rows[key + "_ci_low"])
            high= np.array(rows[key + "_ci_high"])

            ax.errorbar(x, values, yerr=[values - low, high - values], fmt="o-", color=color, ecolor=color, elinewidth=1.5, linewidth=2, markersize=6, capsize=0, label=name)

        ax.set_xticks(np.arange(len(sizes)))
        ax.set_xticklabels([f"K = {k}" for k in sizes], fontsize=9, color=INK_SECONDARY)

    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle("Group-size study on the cached completions (24 prompts x 8, equal generations): mean and 95% bootstrap interval over prompts", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def plot_length_bins(table, path):
    runs= list(dict.fromkeys(table["run"]))
    bins= list(dict.fromkeys(table["bin"]))
    width= 0.36

    panels= [
        ("per_token_weight_mean", "Mean per-token weight |A| x w"),
        ("per_token_weight_share", "Share of total per-token weight"),
        ("per_sequence_weight_share", "Share of total per-sequence weight"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, (key, title) in zip(axes, panels):
        style_axis(ax, title)

        for i in range(len(runs)):
            rows= table[table["run"] == runs[i]]

            x= np.arange(len(bins)) + (i - (len(runs) - 1) / 2) * (width + 0.03)

            ax.bar(x, np.array(rows[key], dtype=float), width=width, color=SERIES[i], label=runs[i])

        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels(bins, fontsize=8, color=INK_SECONDARY)

    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle("Policy-term weight by completion-length quartile (8 training completions per quartile per fork, masked completions count as 0)", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def main(args):
    cfg= load_yaml(args.config)

    results_dir= repo_path(cfg["results_dir"])
    figures_dir= results_dir / "figures"

    os.makedirs(figures_dir, exist_ok=True)

    seed= int(cfg["seed"])

    runs= load_runs(results_dir)

    summary= summary_table(runs)
    summary.to_csv(results_dir / "summary.csv", index=False)
    print("saved", results_dir / "summary.csv")

    paired= paired_table(runs, seed)
    paired.to_csv(results_dir / "paired_differences.csv", index=False)
    print("saved", results_dir / "paired_differences.csv")

    if runs["standard"]["log"] is not None:
        pd.DataFrame(runs["standard"]["log"]).drop(columns=["prompt_rows"]).to_csv(results_dir / "standard_trajectory.csv", index=False)
        print("saved", results_dir / "standard_trajectory.csv")

        plot_standard(runs["standard"]["log"], figures_dir / "standard_trajectories.png")

    if (results_dir / "group_size_study.json").exists():
        group= group_size_table(load_json(results_dir / "group_size_study.json"))
        group.to_csv(results_dir / "group_size.csv", index=False)
        print("saved", results_dir / "group_size.csv")

        plot_group_size(group, figures_dir / "group_size.png")

    if (results_dir / "normalization_study.json").exists():
        bins= length_bin_table(load_json(results_dir / "normalization_study.json"))
        bins.to_csv(results_dir / "length_bins.csv", index=False)
        print("saved", results_dir / "length_bins.csv")

        plot_length_bins(bins, figures_dir / "length_bins.png")
        plot_forks(runs, figures_dir / "normalization_fork_trajectories.png")

    if "reward_mean" in summary.columns:
        plot_heldout(summary, figures_dir / "heldout_comparison.png")

    if len(paired) > 0:
        plot_paired(paired, figures_dir / "paired_differences.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo.yaml"
    )

    args = parser.parse_args()

    main(args)
