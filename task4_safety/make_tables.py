from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl
from task2_ppo.make_tables import INK, INK_SECONDARY, MUTED, SERIES, save_figure, style_axis
from task4_safety.evaluate_safety import LABEL_ORDER, RATES
from task4_safety.generate_responses import POLICIES, adjust_config

#one colour per judge label, the same in every panel
LABEL_COLORS= {
    "SAFE_ANSWER": SERIES[0],
    "JUSTIFIED_REFUSAL": SERIES[2],
    "UNSAFE_COMPLIANCE": "#e34948",
    "OVER_REFUSAL": SERIES[1],
    "AMBIGUOUS": MUTED,
}
RATE_TITLES= {
    "safe_answer_rate": "Safe prompts: SAFE_ANSWER",
    "safe_over_refusal_rate": "Safe prompts: OVER_REFUSAL",
    "unsafe_compliance_rate": "Unsafe prompts: UNSAFE_COMPLIANCE",
    "unsafe_justified_refusal_rate": "Unsafe prompts: JUSTIFIED_REFUSAL",
    "ambiguous_rate": "All prompts: AMBIGUOUS",
    "response_tokens_mean": "Response length (tokens)",
}


def candidate_table(outdir):
    #responses worth reading by hand, picked by rule from the judge labels. which ones are used and what they show is decided by hand
    generated= {name: {r["xstest_id"]: r for r in read_jsonl(outdir / f"generated_{name}.jsonl")} for name in POLICIES}
    judged= {name: {r["xstest_id"]: r for r in read_jsonl(outdir / f"judged_{name}.jsonl")} for name in POLICIES}

    rows= []

    for xstest_id in sorted(generated["sft"]):
        labels= {name: judged[name][xstest_id]["label"] for name in POLICIES}
        benchmark_class= generated["sft"][xstest_id]["benchmark_class"]

        for name in POLICIES:
            label= labels[name]
            reasons= []

            if benchmark_class == "UNSAFE" and label in ["SAFE_ANSWER", "UNSAFE_COMPLIANCE"]:
                reasons.append("answered_an_unsafe_prompt")

            if benchmark_class == "SAFE" and label in ["JUSTIFIED_REFUSAL", "OVER_REFUSAL"]:
                reasons.append("refused_a_safe_prompt")

            if benchmark_class == "SAFE" and label == "UNSAFE_COMPLIANCE":
                reasons.append("unsafe_compliance_label_on_a_safe_prompt")

            if len(set(labels.values())) > 1:
                reasons.append("policies_get_different_labels")

            if len(reasons) == 0:
                continue

            row= generated[name][xstest_id]

            rows.append({
                "xstest_id": xstest_id,
                "benchmark_class": benchmark_class,
                "category": row["type"],
                "policy": name,
                "judge_label": label,
                "reasons": ";".join(reasons),
                "labels_of_all_policies": " ".join(f"{n}={labels[n]}" for n in POLICIES),
                "identical_to_sft_response": row["response"] == generated["sft"][xstest_id]["response"],
                "response_tokens": row["response_tokens"],
                "prompt": row["prompt"],
                "response": row["response"],
            })

    return pd.DataFrame(rows)


def plot_rates(summary, path):
    table= summary.set_index("policy").reindex(POLICIES)
    x= np.arange(len(POLICIES))

    keys= [name for name, _, _ in RATES] + ["ambiguous_rate", "response_tokens_mean"]

    fig, axes = plt.subplots(1, 6, figsize=(21, 3.8))

    for ax, key in zip(axes, keys):
        style_axis(ax, RATE_TITLES[key])

        values= np.array(table[key], dtype=float)
        low= np.array(table[key + "_ci_low"], dtype=float)
        high= np.array(table[key + "_ci_high"], dtype=float)

        ax.errorbar(x, values, yerr=[values - low, high - values], fmt="o", color=SERIES[0], ecolor=SERIES[0], elinewidth=2, markersize=7, capsize=0)
        ax.set_xticks(x)
        ax.set_xticklabels(POLICIES, fontsize=9, color=INK_SECONDARY)
        ax.set_xlim(-0.5, len(POLICIES) - 0.5)

        if key != "response_tokens_mean":
            ax.set_ylim(-0.03, 1.03)

    fig.suptitle("Judge-label rates per policy on XSTest (250 safe, 200 unsafe prompts, greedy decoding): 95% bootstrap interval over prompts", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def plot_categories(categories, path):
    types= list(dict.fromkeys(categories[categories["policy"] == POLICIES[0]]["type"]))
    positions= np.arange(len(types))[::-1]

    fig, axes = plt.subplots(1, 4, figsize=(19, 7), sharey=True)

    for ax, name in zip(axes, POLICIES):
        style_axis(ax, name)

        rows= categories[categories["policy"] == name].set_index("type").reindex(types)
        left= np.zeros(len(types))

        for label in LABEL_ORDER:
            values= np.array(rows[label + "_fraction"], dtype=float)

            ax.barh(positions, values, left=left, height=0.7, color=LABEL_COLORS[label], edgecolor="#fcfcfb", linewidth=1.5, label=label)

            left += values

        ax.set_yticks(positions)
        ax.set_yticklabels(types, fontsize=8, color=INK_SECONDARY)
        ax.set_xlim(0, 1)

    axes[-1].legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, loc="lower right")

    fig.suptitle("Judge-label distribution per XSTest category (25 prompts each; categories starting with contrast_ are the unsafe ones)", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def main(args):
    cfg= adjust_config(load_yaml(args.config), False)

    outdir= Path(cfg["task4_dir"])
    figures_dir= outdir / "figures"

    os.makedirs(figures_dir, exist_ok=True)

    if not (outdir / "safety_summary.csv").exists():
        raise SystemExit(f"No safety tables in {outdir}. Run task4_safety.evaluate_safety first.")

    plot_rates(pd.read_csv(outdir / "safety_summary.csv"), figures_dir / "safety_rates.png")
    plot_categories(pd.read_csv(outdir / "category_label_distribution.csv"), figures_dir / "category_labels.png")

    candidates= candidate_table(outdir)
    candidates.to_csv(outdir / "qualitative_candidates.csv", index=False)

    print("saved", outdir / "qualitative_candidates.csv")
    print(candidates.groupby(["policy", "reasons"]).size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/feedback.yaml"
    )

    args = parser.parse_args()

    main(args)
