from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json
from common.metrics import bootstrap_ci, paired_bootstrap_ci
from task1_dpo.evaluate import STRATA
from task2_ppo.make_tables import INK, INK_SECONDARY, SERIES, save_figure, style_axis

RUNS= ["reference", "standard", "beta_0.03", "beta_0.1", "beta_0.3", "length_balanced"]
BETA_RUNS= ["beta_0.03", "beta_0.1", "beta_0.3"]
LENGTH_RUNS= ["reference", "standard", "length_balanced"]
#the one-epoch runs and the short forks were trained on different budgets and are never ranked against each other
BUDGETS= {
    "reference": "no training",
    "standard": "one epoch",
    "length_balanced": "one epoch (length-balanced subset)",
    "beta_0.03": "short fork",
    "beta_0.1": "short fork",
    "beta_0.3": "short fork",
}
TRAIN_PANELS= [
    ("loss", "Training DPO loss"),
    ("margin", "Training margin"),
    ("reward_accuracy", "Training preference accuracy"),
    ("grad_norm", "Gradient norm"),
]
BETA_PANELS= [
    ("preference_accuracy", "Held-out preference accuracy"),
    ("dpo_loss", "Held-out DPO loss (at the run's beta)"),
    ("kl_token_mean", "KL from reference per token"),
    ("reward_mean", "Reward-model score"),
    ("length_mean", "Response length (tokens)"),
]


def dpo_losses(margins, beta):
    #per pair: -log sigmoid(beta * margin), written so a large negative margin cannot overflow
    out= []

    for m in margins:
        x= beta * m
        out.append(math.log1p(math.exp(-x)) if x > 0 else -x + math.log1p(math.exp(x)))

    return out


def word_limit_stats(rows, seed):
    #each prompt was sampled several times, so compliance is resampled by prompt: compliant samples / samples
    by_prompt= {}

    for r in rows:
        by_prompt.setdefault(r["prompt_id"], []).append(float(r["word_limit_ok"]))

    ok= [sum(v) for v in by_prompt.values()]
    n= [len(v) for v in by_prompt.values()]

    return sum(ok) / sum(n), bootstrap_ci(ok, n, seed)


def load_runs(results_dir):
    runs= {}

    for name in RUNS:
        run_dir= results_dir / name
        run= {"eval": None, "train": None, "log": None, "pairs": None, "heldout": None, "word_limit": None, "strat": None, "strat_pairs": None}

        if (run_dir / "eval_metrics.json").exists():
            generations= read_jsonl(run_dir / "generations.jsonl")

            run["eval"]= load_json(run_dir / "eval_metrics.json")
            run["pairs"]= read_jsonl(run_dir / "pairs.jsonl")
            run["heldout"]= [r for r in generations if r["set"] == "heldout"]
            run["word_limit"]= [r for r in generations if r["set"] == "word_limit"]

        if (run_dir / "train_metrics.json").exists():
            run["train"]= load_json(run_dir / "train_metrics.json")
            run["log"]= read_jsonl(run_dir / "train_log.jsonl")

        if (run_dir / "stratified_metrics.json").exists():
            run["strat"]= load_json(run_dir / "stratified_metrics.json")
            run["strat_pairs"]= read_jsonl(run_dir / "stratified_pairs.jsonl")

        runs[name]= run

        print(
            f"{name}: "
            f"evaluation={'yes' if run['eval'] is not None else 'MISSING'}, "
            f"training log={'yes' if run['log'] is not None else 'none'}, "
            f"stratified={'yes' if run['strat'] is not None else 'none'}"
        )

    return runs


def summary_table(runs, seed):
    rows= []

    for name in RUNS:
        run= runs[name]

        if run["eval"] is None:
            continue

        e= run["eval"]
        beta= float(e["beta"])
        margins= [p["margin"] for p in run["pairs"]]

        accuracy= [float(m > 0) for m in margins]
        losses= dpo_losses(margins, beta)
        rewards= [r["reward"] for r in run["heldout"]]
        lengths= [float(r["n_tokens"]) for r in run["heldout"]]

        row= {
            "run": name,
            "budget": BUDGETS[name],
            "beta": beta,
            "train_examples": e.get("train_examples"),
            "train_optimizer_steps": e.get("train_optimizer_steps"),
            "n_pairs": len(margins),
            "dpo_loss": float(np.mean(losses)),
            "dpo_loss_ci_low": bootstrap_ci(losses, None, seed)[0],
            "dpo_loss_ci_high": bootstrap_ci(losses, None, seed)[1],
            #fraction of held-out pairs with margin > 0
            "preference_accuracy": float(np.mean(accuracy)),
            "preference_accuracy_ci_low": bootstrap_ci(accuracy, None, seed)[0],
            "preference_accuracy_ci_high": bootstrap_ci(accuracy, None, seed)[1],
            "margin_mean": float(np.mean(margins)),
            "margin_mean_ci_low": bootstrap_ci(margins, None, seed)[0],
            "margin_mean_ci_high": bootstrap_ci(margins, None, seed)[1],
            #per-response kl was not saved for task 1, so these two have no interval
            "kl_token_mean": e["heldout_generation"]["kl_token_mean"],
            "kl_sequence_mean": e["heldout_generation"]["kl_sequence_mean"],
            "reward_mean": float(np.mean(rewards)),
            "reward_mean_ci_low": bootstrap_ci(rewards, None, seed)[0],
            "reward_mean_ci_high": bootstrap_ci(rewards, None, seed)[1],
            "length_mean": float(np.mean(lengths)),
            "length_mean_ci_low": bootstrap_ci(lengths, None, seed)[0],
            "length_mean_ci_high": bootstrap_ci(lengths, None, seed)[1],
            "length_std": e["heldout_generation"]["length_std"],
            "length_q25": e["heldout_generation"]["length_q25"],
            "length_median": e["heldout_generation"]["length_median"],
            "length_q75": e["heldout_generation"]["length_q75"],
            "length_iqr": e["heldout_generation"]["length_iqr"],
            "truncated_fraction": e["heldout_generation"]["truncated_fraction"],
        }

        compliance, ci = word_limit_stats(run["word_limit"], seed)

        row["word_limit_compliance"]= compliance
        row["word_limit_compliance_ci_low"]= ci[0]
        row["word_limit_compliance_ci_high"]= ci[1]
        row["word_limit_length_mean"]= e["word_limit"]["length_mean"]
        row["word_limit_words_mean"]= e["word_limit"]["words_mean"]

        if run["train"] is not None:
            t= run["train"]

            row["train_skipped_optimizer_steps"]= t.get("skipped_optimizer_steps")
            row["train_step_retries_total"]= t.get("step_retries_total")
            row["train_wall_clock_sec"]= t.get("wall_clock_sec")
            row["train_peak_vram_bytes"]= t.get("peak_vram_bytes")
            row["train_git_commit"]= t.get("git_commit")

        row["gpu_name"]= e["gpu_name"]
        row["eval_git_commit"]= e["git_commit"]

        rows.append(row)

    return pd.DataFrame(rows)


def stratified_table(runs, seed):
    rows= []

    for name in LENGTH_RUNS:
        if runs[name]["strat_pairs"] is None:
            continue

        for stratum in STRATA + ["all"]:
            pairs= [p for p in runs[name]["strat_pairs"] if stratum == "all" or p["length_stratum"] == stratum]

            accuracy= [float(p["margin"] > 0) for p in pairs]
            raw= [float(p["policy_chosen_logp"] > p["policy_rejected_logp"]) for p in pairs]
            margins= [p["margin"] for p in pairs]

            rows.append({
                "run": name,
                "stratum": stratum,
                "n_pairs": len(pairs),
                #margin > 0: the policy moved towards the preferred response relative to the reference
                "preference_accuracy": float(np.mean(accuracy)),
                "preference_accuracy_ci_low": bootstrap_ci(accuracy, None, seed)[0],
                "preference_accuracy_ci_high": bootstrap_ci(accuracy, None, seed)[1],
                #the policy alone gives the preferred response the higher log-probability
                "raw_preference_accuracy": float(np.mean(raw)),
                "raw_preference_accuracy_ci_low": bootstrap_ci(raw, None, seed)[0],
                "raw_preference_accuracy_ci_high": bootstrap_ci(raw, None, seed)[1],
                "margin_mean": float(np.mean(margins)),
                "margin_mean_ci_low": bootstrap_ci(margins, None, seed)[0],
                "margin_mean_ci_high": bootstrap_ci(margins, None, seed)[1],
            })

    return pd.DataFrame(rows)


def paired_table(runs, seed):
    #runs evaluated on the same held-out pairs and prompts are compared item by item
    rows= []

    comparisons= [("standard", "reference"), ("length_balanced", "standard"), ("beta_0.03", "beta_0.1"), ("beta_0.3", "beta_0.1")]

    for name, base in comparisons:
        if runs[name]["eval"] is None or runs[base]["eval"] is None:
            continue

        row= {"run": name, "baseline": base}

        a= {p["prompt_id"]: p for p in runs[name]["pairs"]}
        b= {p["prompt_id"]: p for p in runs[base]["pairs"]}
        ids= [i for i in a if i in b]

        acc_a= [float(a[i]["margin"] > 0) for i in ids]
        acc_b= [float(b[i]["margin"] > 0) for i in ids]

        row["n_pairs"]= len(ids)
        row["preference_accuracy_diff"]= float(np.mean(acc_a) - np.mean(acc_b))
        row["preference_accuracy_diff_ci_low"], row["preference_accuracy_diff_ci_high"] = paired_bootstrap_ci(acc_a, acc_b, seed)

        ga= {r["prompt_id"]: r for r in runs[name]["heldout"]}
        gb= {r["prompt_id"]: r for r in runs[base]["heldout"]}
        ids= [i for i in ga if i in gb]

        for key in ["reward", "n_tokens"]:
            va= [float(ga[i][key]) for i in ids]
            vb= [float(gb[i][key]) for i in ids]

            row[key + "_diff"]= float(np.mean(va) - np.mean(vb))
            row[key + "_diff_ci_low"], row[key + "_diff_ci_high"] = paired_bootstrap_ci(va, vb, seed)

        rows.append(row)

    #the two one-epoch models on the same length-stratified pairs, per stratum
    if runs["standard"]["strat_pairs"] is not None and runs["length_balanced"]["strat_pairs"] is not None:
        a= {p["prompt_id"]: p for p in runs["length_balanced"]["strat_pairs"]}
        b= {p["prompt_id"]: p for p in runs["standard"]["strat_pairs"]}

        for stratum in STRATA:
            ids= [i for i in a if i in b and a[i]["length_stratum"] == stratum]

            acc_a= [float(a[i]["margin"] > 0) for i in ids]
            acc_b= [float(b[i]["margin"] > 0) for i in ids]

            row= {"run": "length_balanced", "baseline": "standard", "stratum": stratum, "n_pairs": len(ids)}
            row["preference_accuracy_diff"]= float(np.mean(acc_a) - np.mean(acc_b))
            row["preference_accuracy_diff_ci_low"], row["preference_accuracy_diff_ci_high"] = paired_bootstrap_ci(acc_a, acc_b, seed)

            rows.append(row)

    return pd.DataFrame(rows)


def plot_training(runs, names, title, path):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))

    for ax, (key, panel_title) in zip(axes, TRAIN_PANELS):
        style_axis(ax, panel_title)

        for i in range(len(names)):
            log= runs[names[i]]["log"]

            if log is None:
                continue

            ax.plot([r["step"] for r in log], [r[key] for r in log], color=SERIES[i], linewidth=2, label=names[i])

        ax.set_xlabel("optimizer step", fontsize=8, color=INK_SECONDARY)

    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def plot_beta(summary, path):
    table= summary[summary["run"].isin(BETA_RUNS)].set_index("run").reindex(BETA_RUNS)
    x= np.arange(len(BETA_RUNS))

    fig, axes = plt.subplots(1, 5, figsize=(19, 4))

    for ax, (key, title) in zip(axes, BETA_PANELS):
        style_axis(ax, title)

        values= np.array(table[key], dtype=float)

        if key + "_ci_low" in table.columns:
            low= np.array(table[key + "_ci_low"], dtype=float)
            high= np.array(table[key + "_ci_high"], dtype=float)

            ax.errorbar(x, values, yerr=[values - low, high - values], fmt="o-", color=SERIES[0], ecolor=SERIES[0], elinewidth=2, linewidth=2, markersize=6, capsize=0)
        else:
            ax.plot(x, values, "o-", color=SERIES[0], linewidth=2, markersize=6)

        ax.set_xticks(x)
        ax.set_xticklabels([f"beta = {b:g}" for b in table["beta"]], fontsize=9, color=INK_SECONDARY)

    fig.suptitle("Beta study: short forks of 600 examples from the same initialisation, same held-out pairs and decoding (95% bootstrap interval where per-item values were saved)", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def plot_stratified(strat, path):
    names= [n for n in ["standard", "length_balanced"] if n in set(strat["run"])]
    width= 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for ax, (key, title) in zip(axes, [("preference_accuracy", "Preference accuracy (margin > 0)"), ("margin_mean", "Mean margin")]):
        style_axis(ax, title)

        for i in range(len(names)):
            rows= strat[strat["run"] == names[i]].set_index("stratum").reindex(STRATA)

            x= np.arange(len(STRATA)) + (i - (len(names) - 1) / 2) * (width + 0.03)
            values= np.array(rows[key], dtype=float)
            low= np.array(rows[key + "_ci_low"], dtype=float)
            high= np.array(rows[key + "_ci_high"], dtype=float)

            ax.bar(x, values, width=width, color=SERIES[i], label=names[i])
            ax.errorbar(x, values, yerr=[values - low, high - values], fmt="none", ecolor=INK_SECONDARY, elinewidth=1.5, capsize=0)

        ax.set_xticks(np.arange(len(STRATA)))
        ax.set_xticklabels(STRATA, fontsize=9, color=INK_SECONDARY)

    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle("Length-stratified held-out pairs: standard vs length-balanced DPO, 95% bootstrap interval over pairs", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def plot_length_behaviour(summary, path):
    table= summary[summary["run"].isin(LENGTH_RUNS)].set_index("run").reindex(LENGTH_RUNS)
    x= np.arange(len(LENGTH_RUNS))

    panels= [
        ("word_limit_compliance", "Word-limit compliance (10 prompts x 5 samples)"),
        ("length_mean", "Held-out response length (tokens)"),
        ("reward_mean", "Held-out reward-model score"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, (key, title) in zip(axes, panels):
        style_axis(ax, title)

        values= np.array(table[key], dtype=float)
        low= np.array(table[key + "_ci_low"], dtype=float)
        high= np.array(table[key + "_ci_high"], dtype=float)

        ax.errorbar(x, values, yerr=[values - low, high - values], fmt="o", color=SERIES[0], ecolor=SERIES[0], elinewidth=2, markersize=7, capsize=0)
        ax.set_xticks(x)
        ax.set_xticklabels(LENGTH_RUNS, fontsize=9, color=INK_SECONDARY)
        ax.set_xlim(-0.5, len(LENGTH_RUNS) - 0.5)

    fig.suptitle("Generated length and word-limit behaviour: reference, standard and length-balanced DPO (95% bootstrap interval)", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def main(args):
    cfg= load_yaml(args.config)

    results_dir= repo_path(cfg["results_dir"])
    figures_dir= results_dir / "figures"

    os.makedirs(figures_dir, exist_ok=True)

    seed= int(cfg["seed"])

    runs= load_runs(results_dir)

    summary= summary_table(runs, seed)
    summary.to_csv(results_dir / "summary.csv", index=False)
    print("saved", results_dir / "summary.csv")

    strat= stratified_table(runs, seed)
    strat.to_csv(results_dir / "stratified.csv", index=False)
    print("saved", results_dir / "stratified.csv")

    paired= paired_table(runs, seed)
    paired.to_csv(results_dir / "paired_differences.csv", index=False)
    print("saved", results_dir / "paired_differences.csv")

    plot_training(
        runs,
        ["standard", "length_balanced"],
        "One-epoch DPO training: standard and length-balanced subsets",
        figures_dir / "training_one_epoch.png"
    )

    plot_training(
        runs,
        BETA_RUNS,
        "Short DPO forks (600 examples): training curves for each beta",
        figures_dir / "training_beta_forks.png"
    )

    plot_beta(summary, figures_dir / "beta_study.png")
    plot_stratified(strat, figures_dir / "length_strata.png")
    plot_length_behaviour(summary, figures_dir / "length_behaviour.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dpo.yaml"
    )

    args = parser.parse_args()

    main(args)
