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
from common.logging_utils import load_json, save_json
from task1_dpo.train import git_commit
from task2_ppo.make_tables import INK, INK_SECONDARY, MUTED, SERIES, save_figure, style_axis
from task5_feedback.evaluate_math import FAILURE_TYPES, POLICIES, adjust_config
from task5_feedback.score_perturbations import MECHANISMS, OUTCOMES, PERTURBATIONS

DATASETS= ["gsm", "transfer"]
DATASET_LABELS= {"gsm": "GSM8K (in-domain)", "transfer": "SVAMP (transfer)"}
DROP_KEYS= ["correct", "format_ok", "response_tokens"]
#better / tie / wrong in the stacked bars
OUTCOME_COLORS= {"better": SERIES[0], "tie": MUTED, "wrong": SERIES[1]}


def drop_table(generated, seed, n_resamples=2000):
    #transfer minus in-domain for each policy. the two sets hold different problems, so each is resampled on its own
    rows= []

    for name in POLICIES:
        row= {"policy": name}

        for key in DROP_KEYS:
            a= np.array([float(r[key]) for r in generated["gsm"][name]])
            b= np.array([float(r[key]) for r in generated["transfer"][name]])

            rng= np.random.default_rng(seed)

            means_a= a[rng.integers(0, len(a), size=(n_resamples, len(a)))].mean(axis=1)
            means_b= b[rng.integers(0, len(b), size=(n_resamples, len(b)))].mean(axis=1)

            low, high = np.percentile(means_b - means_a, [2.5, 97.5])

            row[key + "_gsm"]= float(a.mean())
            row[key + "_transfer"]= float(b.mean())
            row[key + "_transfer_minus_gsm"]= float(b.mean() - a.mean())
            row[key + "_transfer_minus_gsm_ci_low"]= float(low)
            row[key + "_transfer_minus_gsm_ci_high"]= float(high)

        rows.append(row)

    return pd.DataFrame(rows)


def failure_table(sheet_path):
    #counts of the hand-assigned failure types per policy, only once every row has a label
    sheet= pd.read_csv(sheet_path, dtype=str).fillna("")
    labels= sheet["failure_type"].str.strip().str.lower()

    if (labels == "").all():
        print("the failure-type sheet has no labels yet, nothing to tabulate")
        return None

    missing= int((labels == "").sum())
    unknown= sorted(set(x for x in labels if x != "" and x not in FAILURE_TYPES))

    if missing > 0 or len(unknown) > 0:
        raise SystemExit(
            f"The failure-type sheet is not finished: {missing} rows without a label, unknown labels {unknown}. "
            f"Valid labels: {', '.join(FAILURE_TYPES)}"
        )

    rows= []

    for name in POLICIES:
        mine= labels[sheet["policy"] == name]
        row= {"policy": name, "n_labelled": len(mine)}

        for failure in FAILURE_TYPES:
            row[failure]= int((mine == failure).sum())
            row[failure + "_fraction"]= float((mine == failure).mean()) if len(mine) > 0 else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def plot_policies(summary, path):
    panels= [
        ("exact_accuracy", "Exact final-answer accuracy"),
        ("format_compliance", "Format compliance"),
        ("response_tokens_mean", "Response length (tokens)"),
        ("win_rate_vs_sft", "Judge win rate against SFT (tie = 0.5)"),
    ]
    width= 0.36

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

    for ax, (key, title) in zip(axes, panels):
        style_axis(ax, title)

        for i in range(len(DATASETS)):
            rows= summary[summary["dataset"] == DATASETS[i]].set_index("policy").reindex(POLICIES)

            x= np.arange(len(POLICIES)) + (i - 0.5) * (width + 0.03)
            values= np.array(rows[key], dtype=float)
            low= np.array(rows[key + "_ci_low"], dtype=float)
            high= np.array(rows[key + "_ci_high"], dtype=float)

            #the base policy has no win rate against itself
            shown= ~np.isnan(values)

            ax.bar(x[shown], values[shown], width=width, color=SERIES[i], label=DATASET_LABELS[DATASETS[i]])
            ax.errorbar(x[shown], values[shown], yerr=[values[shown] - low[shown], high[shown] - values[shown]], fmt="none", ecolor=INK_SECONDARY, elinewidth=1.5, capsize=0)

        ax.set_xticks(np.arange(len(POLICIES)))
        ax.set_xticklabels(POLICIES, fontsize=9, color=INK_SECONDARY)

    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle("SFT, RLVR and RLAIF with greedy decoding: mean and 95% bootstrap interval over problems", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def plot_diagnostics(rates, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)

    for ax, mechanism in zip(axes, MECHANISMS):
        style_axis(ax, "Exact verifier" if mechanism == "verifier" else "Pairwise AI judge")

        rows= rates[rates["mechanism"] == mechanism].set_index("perturbation").reindex(PERTURBATIONS)
        left= np.zeros(len(PERTURBATIONS))
        positions= np.arange(len(PERTURBATIONS))[::-1]

        for outcome in OUTCOMES:
            values= np.array(rows[outcome + "_rate"], dtype=float)

            #a thin gap in the surface colour separates the stacked segments
            ax.barh(positions, values, left=left, height=0.6, color=OUTCOME_COLORS[outcome], edgecolor="#fcfcfb", linewidth=2, label=outcome)

            for k in range(len(values)):
                if values[k] >= 0.08:
                    ax.text(left[k] + values[k] / 2, positions[k], f"{values[k]:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if outcome != "tie" else INK)

            left += values

        ax.set_yticks(positions)
        ax.set_yticklabels(PERTURBATIONS, fontsize=8, color=INK_SECONDARY)
        ax.set_xlim(0, 1)

    axes[1].legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, loc="lower right")

    fig.suptitle("Controlled diagnostic pairs, clean response vs one perturbation (20 problems each): which response the mechanism prefers", fontsize=11, color=INK, x=0.01, ha="left")

    save_figure(fig, path)


def main(args):
    cfg= adjust_config(load_yaml(args.config), args.smoke, args.max_examples)

    outdir= Path(cfg["task5_dir"])
    figures_dir= outdir / "figures"
    seed= int(cfg["seed"])

    os.makedirs(figures_dir, exist_ok=True)

    for dataset in DATASETS:
        if not (outdir / dataset / "summary.csv").exists():
            raise SystemExit(f"No results for '{dataset}' in {outdir}. Run task5_feedback.evaluate_math --dataset {dataset} first.")

    summary= pd.concat([pd.read_csv(outdir / dataset / "summary.csv") for dataset in DATASETS])
    summary.to_csv(outdir / "policy_summary.csv", index=False)
    print("saved", outdir / "policy_summary.csv")

    generated= {d: {name: read_jsonl(outdir / d / f"generated_{name}.jsonl") for name in POLICIES} for d in DATASETS}

    drops= drop_table(generated, seed)
    drops.to_csv(outdir / "transfer_drop.csv", index=False)
    print("saved", outdir / "transfer_drop.csv")

    #pairwise results and verifier-judge agreement of both datasets side by side
    rows= []

    for dataset in DATASETS:
        metrics= load_json(outdir / dataset / "metrics.json")

        for key in metrics["pairwise"]:
            row= {"dataset": dataset, "comparison": key}
            row.update({k: v for k, v in metrics["pairwise"][key].items() if not k.endswith("_ci95")})
            row["win_rate_ci_low"]= metrics["pairwise"][key]["win_rate_ci95"][0]
            row["win_rate_ci_high"]= metrics["pairwise"][key]["win_rate_ci95"][1]

            for k, v in metrics["verifier_judge_agreement"][key].items():
                row["agreement_" + k]= v

            rows.append(row)

        pooled= {"dataset": dataset, "comparison": "pooled_vs_sft"}

        for k, v in metrics["verifier_judge_agreement"]["pooled_vs_sft"].items():
            pooled["agreement_" + k]= v

        rows.append(pooled)

    pd.DataFrame(rows).to_csv(outdir / "pairwise_and_agreement.csv", index=False)
    print("saved", outdir / "pairwise_and_agreement.csv")

    plot_policies(summary, figures_dir / "policy_comparison.png")

    info= {"git_commit": git_commit(), "seed": seed, "smoke": args.smoke, "diagnostics": None, "failure_types": None}

    if (outdir / "diagnostics" / "perturbation_rates.csv").exists():
        rates= pd.read_csv(outdir / "diagnostics" / "perturbation_rates.csv")

        plot_diagnostics(rates, figures_dir / "diagnostic_pairs.png")

        sensitivity= load_json(outdir / "diagnostics" / "summary.json")["summary"]["sensitivity"]

        pd.DataFrame([{"mechanism": m, **{k: v for k, v in sensitivity[m].items() if not k.endswith("_ci95")}} for m in MECHANISMS]).to_csv(outdir / "sensitivity.csv", index=False)
        print("saved", outdir / "sensitivity.csv")

        info["diagnostics"]= sensitivity
    else:
        print("no diagnostic study yet, run task5_feedback.score_perturbations")

    sheet_path= outdir / "transfer" / "failure_types_sheet.csv"

    if sheet_path.exists():
        failures= failure_table(sheet_path)

        if failures is not None:
            failures.to_csv(outdir / "failure_types.csv", index=False)
            print("saved", outdir / "failure_types.csv")

            info["failure_types"]= failures.to_dict(orient="records")

    #what each mechanism costs to run, from the evaluation sessions that actually called it
    info["judge_cost"]= {}

    for dataset in DATASETS:
        metrics= load_json(outdir / dataset / "metrics.json")

        info["judge_cost"][dataset]= {
            "judge_comparisons": metrics["judge_comparisons"],
            "judge_seconds_this_session": metrics["judge_seconds_this_session"],
        }

    info["note"]= "the verifier is a regular expression and a number comparison and takes no measurable time; cached judge answers make judge_seconds smaller than a first run"

    save_json(outdir / "comparison.json", info)

    print(f"saved the combined comparison to {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/feedback.yaml"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args = parser.parse_args()

    main(args)
