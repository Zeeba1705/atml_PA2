from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.data import load_yaml
from common.logging_utils import save_json
from task1_dpo.train import git_commit
from task3_grpo.continue_train import adjust_config
from task3_grpo.evaluate import train_and_evaluate_fork

RUNS= [("norm_grpo", "grpo"), ("norm_dr_grpo", "dr_grpo")]
HELDOUT_KEYS= ["reward_mean", "kl_token_mean", "kl_sequence_mean", "entropy", "length_mean", "no_eos_fraction"]
TRAIN_KEYS= ["reward", "kl", "reward_std_within_group", "uninformative_group_fraction", "response_length", "truncated_fraction", "masked_completions", "grad_norm"]
BIN_NAMES= ["q1_shortest", "q2", "q3", "q4_longest"]


def length_bin_edges(lengths):
    #quartiles of the completion length, pooled over both runs so both use the same bins
    return [float(e) for e in np.percentile(lengths, [25, 50, 75])]


def length_bin(n_tokens, edges):
    #0= shortest quartile ... 3= longest quartile, a length equal to an edge goes to the lower bin
    for i in range(len(edges)):
        if n_tokens <= edges[i]:
            return i

    return len(edges)


def effective_weight(row):
    #|A_k| x w_k, the weight each token of this completion gets in the policy term
    #a masked completion is not in the loss, so its effective weight is 0
    if row["masked"]:
        return 0.0

    return row["abs_advantage_times_weight"]


def length_conditioned_stats(rollout_rows, edges):
    bins= []

    for name in BIN_NAMES:
        bins.append({"bin": name, "n_completions": 0, "n_masked": 0, "lengths": [], "token_weights": [], "sequence_weights": []})

    for row in rollout_rows:
        b= bins[length_bin(row["n_tokens"], edges)]
        weight= effective_weight(row)

        b["n_completions"] += 1
        b["n_masked"] += int(row["masked"])
        b["lengths"].append(row["n_tokens"])
        b["token_weights"].append(weight)
        #summed over the completion's tokens: its whole weight in the policy term
        b["sequence_weights"].append(weight * row["n_tokens"])

    total_token_weight= sum(sum(b["token_weights"]) for b in bins)
    total_sequence_weight= sum(sum(b["sequence_weights"]) for b in bins)

    out= []

    for b in bins:
        n= b["n_completions"]

        out.append({
            "bin": b["bin"],
            "n_completions": n,
            "n_masked": b["n_masked"],
            "length_mean": float(np.mean(b["lengths"])) if n > 0 else None,
            "per_token_weight_mean": float(np.mean(b["token_weights"])) if n > 0 else None,
            "per_token_weight_share": sum(b["token_weights"]) / total_token_weight if total_token_weight > 0 else None,
            "per_sequence_weight_mean": float(np.mean(b["sequence_weights"])) if n > 0 else None,
            "per_sequence_weight_share": sum(b["sequence_weights"]) / total_sequence_weight if total_sequence_weight > 0 else None,
        })

    return out


def run_normalization_study(config_path: str, smoke: bool = False, allow_cpu: bool = False, gen_batch_size: int = 4):
    cfg= adjust_config(load_yaml(config_path), smoke)

    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")

    runs= []

    for run_name, loss_type in RUNS:
        #only the sequence normalisation changes: same midpoint, prompts, seed, reward, beta, epsilon and update budget
        log_rows, rollout_rows, train, heldout = train_and_evaluate_fork(
            config_path,
            run_name,
            loss_type,
            smoke,
            allow_cpu,
            gen_batch_size
        )

        run= {
            "run_name": run_name,
            "loss_type": loss_type,
            "updates": train["updates"],
            "updates_without_gradient": train["updates_without_gradient"],
            "generated_tokens_train": train["generated_tokens"],
            "generated_tokens_heldout": heldout["generated_tokens"],
            "rollout_rows": rollout_rows,
        }

        for key in HELDOUT_KEYS:
            run["heldout_" + key]= heldout[key]
            run["heldout_" + key + "_ci95"]= heldout[key + "_ci95"]

        run["heldout_length_std"]= heldout["length_std"]
        run["heldout_length_iqr"]= heldout["length_iqr"]

        for key in TRAIN_KEYS:
            run["train_" + key]= [r[key] for r in log_rows]

        runs.append(run)

    pooled_lengths= [row["n_tokens"] for run in runs for row in run["rollout_rows"]]
    edges= length_bin_edges(pooled_lengths)

    for run in runs:
        stats= length_conditioned_stats(run.pop("rollout_rows"), edges)

        run["length_bins"]= stats
        run["per_token_weight_share_shortest"]= stats[0]["per_token_weight_share"]
        run["per_token_weight_share_longest"]= stats[-1]["per_token_weight_share"]
        run["per_sequence_weight_share_shortest"]= stats[0]["per_sequence_weight_share"]
        run["per_sequence_weight_share_longest"]= stats[-1]["per_sequence_weight_share"]

        print(
            f"{run['run_name']}: "
            f"reward={run['heldout_reward_mean']:.4f}, "
            f"kl_token={run['heldout_kl_token_mean']:.6f}, "
            f"length={run['heldout_length_mean']:.1f}, "
            f"token weight share shortest/longest="
            f"{run['per_token_weight_share_shortest']}/{run['per_token_weight_share_longest']}"
        )

    out= {
        "git_commit": git_commit(),
        "seed": int(cfg["seed"]),
        "fork_updates": int(cfg["fork_updates"]),
        "num_generations": int(cfg["num_generations"]),
        "max_completion_length": int(cfg["max_completion_length"]),
        "length_bin_edges": edges,
        "definitions": {
            "length_bins": "quartiles of the training completion length T_k, pooled over both runs; a length equal to an edge goes to the lower bin",
            "per_token_weight": "|A_k| x w_k with w_k = 1/T_k for grpo and 1/max_completion_length for dr_grpo, 0 for a completion masked for hitting the length cap",
            "per_token_weight_share": "the bin's sum of per-token weights over completions, divided by the sum over all bins",
            "per_sequence_weight": "per-token weight x T_k, the completion's whole weight in the policy term",
            "per_sequence_weight_share": "the bin's sum of per-sequence weights, divided by the sum over all bins",
        },
        "note": "each update uses one prompt with K completions, so there are few completions per bin; both forks see the same prompt sequence but sample their own completions",
        "budget_note": f"forks run {cfg['fork_updates']} updates from the midpoint, the standard run has {cfg['updates']}",
        "smoke": smoke,
        "runs": runs,
    }

    path= Path(cfg["results_dir"]) / "normalization_study.json"
    save_json(path, out)

    print(f"saved normalisation comparison to {path}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--gen-batch-size", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    run_normalization_study(args.config, args.smoke, args.allow_cpu, args.gen_batch_size)


if __name__ == "__main__":
    main()
