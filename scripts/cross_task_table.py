from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import read_jsonl, repo_path
from common.logging_utils import load_json
from common.metrics import paired_bootstrap_ci
from task2_ppo.make_tables import AXIS, INK, INK_SECONDARY, SERIES, save_figure, style_axis

SEED= 6304
BASELINE= "sft"
#every policy below was evaluated with the task 2 held-out protocol: the same 200 prompts, reward model,
#768-token cap, decoding, seed and batch size
POLICIES= [
    ("sft", "results/task2_ppo/sft", "base policy, no adapter"),
    ("dpo", "results/task2_ppo/dpo_standard", "Task 1 standard DPO, one epoch"),
    ("ppo", "results/task2_ppo/standard", "Task 2 standard PPO, 20 updates"),
    ("grpo", "results/task3_grpo/standard", "Task 3 standard GRPO, 20 updates"),
    ("ppo_midpoint", "results/task2_ppo/midpoint", "supplied PPO midpoint"),
    ("grpo_midpoint", "results/task3_grpo/midpoint", "supplied GRPO midpoint"),
]
HELDOUT_KEYS= ["reward_mean", "kl_token_mean", "entropy", "length_mean", "no_eos_fraction"]
PANELS= [
    ("reward_mean", "Held-out reward"),
    ("kl_token_mean", "Held-out KL per token"),
    ("entropy", "Held-out entropy (sampled)"),
    ("length_mean", "Held-out length (tokens)"),
]


def main(args):
    rows= []
    generations= {}

    for name, path, description in POLICIES:
        folder= repo_path(path)

        if not (folder / "eval_metrics.json").exists():
            print(f"{name}: no evaluation at {path}")
            continue

        metrics= load_json(folder / "eval_metrics.json")
        heldout= metrics["heldout"]

        generations[name]= {r["prompt_id"]: r for r in read_jsonl(folder / "generations.jsonl")}

        row= {"policy": name, "description": description, "source": path, "n_prompts": heldout["n_responses"]}

        for key in HELDOUT_KEYS:
            row[key]= heldout[key]
            row[key + "_ci_low"]= heldout[key + "_ci95"][0]
            row[key + "_ci_high"]= heldout[key + "_ci95"][1]

        row["length_std"]= heldout["length_std"]
        row["length_iqr"]= heldout["length_iqr"]
        row["gpu_name"]= metrics["gpu_name"]
        row["eval_git_commit"]= metrics["git_commit"]

        rows.append(row)

    table= pd.DataFrame(rows)

    #same prompts for every policy, so each is compared with the base policy prompt by prompt
    for i in range(len(table)):
        name= table["policy"][i]

        if name == BASELINE or BASELINE not in generations:
            continue

        a= generations[name]
        b= generations[BASELINE]
        ids= [p for p in a if p in b]

        table.loc[i, "identical_responses_to_sft"]= sum(a[p]["response"] == b[p]["response"] for p in ids)

        for key in ["reward", "n_tokens"]:
            va= [float(a[p][key]) for p in ids]
            vb= [float(b[p][key]) for p in ids]

            low, high = paired_bootstrap_ci(va, vb, SEED)

            table.loc[i, key + "_diff_vs_sft"]= float(np.mean(va) - np.mean(vb))
            table.loc[i, key + "_diff_vs_sft_ci_low"]= low
            table.loc[i, key + "_diff_vs_sft_ci_high"]= high

    out= repo_path(args.output)
    table.to_csv(out, index=False)
    print("saved", out)

    names= list(table["policy"])
    positions= list(range(len(names)))[::-1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)

    for ax, (key, title) in zip(axes, PANELS):
        style_axis(ax, title)

        values= np.array(table[key], dtype=float)
        low= np.array(table[key + "_ci_low"], dtype=float)
        high= np.array(table[key + "_ci_high"], dtype=float)

        ax.errorbar(values, positions, xerr=[values - low, high - values], fmt="o", color=SERIES[0], ecolor=SERIES[0], elinewidth=2, markersize=6, capsize=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=9, color=INK_SECONDARY)

    axes[1].axvline(0.0, color=AXIS, linewidth=1.5)

    fig.suptitle("All final policies under one held-out protocol (200 prompts, one sampled response each): mean and 95% bootstrap interval", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, repo_path(args.figure))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=str,
        default="results/cross_task_heldout.csv"
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="results/cross_task_heldout.png"
    )

    args = parser.parse_args()

    main(args)
