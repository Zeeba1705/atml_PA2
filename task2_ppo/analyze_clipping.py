from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import load_json, save_json
from common.metrics import bootstrap_ci, masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer
from task1_dpo.train import check_setup, git_commit, model_dtype
from task2_ppo.continue_train import adjust_config, run_ppo
from task2_ppo.evaluate import train_and_evaluate_fork
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards

SMOKE_ROWS= 3
THRESHOLD_FACTOR= 3.0
PROMPT_NOTE= "each update uses prompts_per_update prompt(s), so per-update values are noisy; all forks see the same prompt sequence"


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def run_name_for(eps):
    return f"clip_{eps:g}"


def pad_rows(tensors, length):
    #right padded to the longest cached response, the mask marks the real tokens
    out= torch.zeros(len(tensors), length)

    for i in range(len(tensors)):
        out[i, :len(tensors[i])]= tensors[i].float()

    return out


def new_logprobs(model, tokenizer, row, messages, cfg):
    #rebuilds the token sequence of one cached response and scores it under the given policy
    rendered= tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    #same prompt truncation as batch_generate
    prompt_ids= tokenizer(
        rendered,
        truncation=True,
        max_length=int(cfg["max_prompt_length"])
    )["input_ids"]

    response_ids= tokenizer(row["response"], add_special_tokens=False)["input_ids"]

    if row["terminated_with_eos"]:
        response_ids= response_ids + [tokenizer.eos_token_id]

    if len(response_ids) != len(row["old_logprobs"]):
        raise SystemExit(
            f"cached response {row['prompt_id']} re-tokenises to {len(response_ids)} tokens, "
            f"the cache has {len(row['old_logprobs'])} log-probs"
        )

    device= next(model.parameters()).device

    sequences= torch.tensor([prompt_ids + response_ids], device=device)
    response= torch.tensor([response_ids], device=device)

    with torch.no_grad():
        logp, _ = response_token_logprobs(
            model,
            sequences,
            torch.ones_like(sequences),
            len(prompt_ids),
            response
        )

    return logp[0].float().cpu()


def analyze_cached_batch(config_path: str, adapter: str | None, name: str, smoke: bool = False, allow_cpu: bool = False):
    #immediate geometric effect of epsilon: one fixed batch, one fixed policy, only the clip range changes
    cfg= adjust_config(load_yaml(config_path), smoke)

    check_setup(
        [cfg["cached_rollouts"], cfg["paths"]["rl_prompt_eval"]],
        smoke,
        allow_cpu
    )

    if adapter is not None and not (repo_path(adapter) / "adapter_config.json").exists():
        raise SystemExit(f"No adapter at {adapter}.")

    rows= load_cached_rollouts(cfg["cached_rollouts"])

    if smoke:
        rows= rows[:SMOKE_ROWS]

    prompts= {r["prompt_id"]: r["messages"] for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}

    tokenizer= load_tokenizer(cfg["base_model"])
    model= load_policy(cfg, adapter_path=adapter, trainable=False)

    new_rows= []

    for i in range(len(rows)):
        new_rows.append(new_logprobs(model, tokenizer, rows[i], prompts[rows[i]["prompt_id"]], cfg))

        if (i + 1) % 8 == 0 or i + 1 == len(rows):
            print(f"scored {i + 1}/{len(rows)} cached responses")

    n_tokens= [len(r["old_logprobs"]) for r in rows]
    max_len= max(n_tokens)

    mask= pad_rows([torch.ones(n) for n in n_tokens], max_len)
    new_logp= pad_rows(new_rows, max_len)
    old_logp= pad_rows([r["old_logprobs"] for r in rows], max_len)
    ref_logp= pad_rows([r["ref_logprobs"] for r in rows], max_len)
    values= pad_rows([r["values"] for r in rows], max_len)

    #the cache already holds the terminal reward after the missing-eos penalty
    task_reward= torch.tensor([float(r["effective_terminal_reward"]) for r in rows])

    rewards= shaped_rewards(
        task_reward,
        old_logp,
        ref_logp,
        mask,
        float(cfg["kl_beta"])
    )

    advantages, returns = compute_gae(
        rewards,
        values,
        mask,
        float(cfg["gamma"]),
        float(cfg["gae_lambda"])
    )

    #normalised once over the whole cached batch, the same advantages are used for every epsilon
    advantages= normalize_advantages(advantages, mask)

    ratio= torch.exp(new_logp - old_logp)
    ratio_dev= (ratio - 1.0).abs()[mask.bool()].numpy()

    seed= int(cfg["seed"])

    out= {
        "git_commit": git_commit(),
        "name": name,
        "adapter": adapter,
        "seed": seed,
        "base_model": cfg["base_model"],
        "dtype": model_dtype(model),
        "cache": cfg["cached_rollouts"],
        "n_rows": len(rows),
        "n_tokens": int(sum(n_tokens)),
        "kl_beta": float(cfg["kl_beta"]),
        "gamma": float(cfg["gamma"]),
        "gae_lambda": float(cfg["gae_lambda"]),
        "smoke": smoke,
        "unclipped_surrogate": masked_mean(ratio * advantages, mask).item(),
        "ratio_abs_deviation": {
            "mean": float(ratio_dev.mean()),
            "median": float(np.percentile(ratio_dev, 50)),
            "p90": float(np.percentile(ratio_dev, 90)),
            "p99": float(np.percentile(ratio_dev, 99)),
            "max": float(ratio_dev.max()),
        },
        "epsilons": [],
        "rows": [],
    }

    #per cached response, to see whether the tokens outside the clip range are spread out or sit in a few rows
    for i in range(len(rows)):
        row_dev= (ratio[i] - 1.0).abs()[:n_tokens[i]]

        out["rows"].append({
            "row_index": i,
            "prompt_id": rows[i]["prompt_id"],
            "n_tokens": n_tokens[i],
            "terminated_with_eos": bool(rows[i]["terminated_with_eos"]),
            "ratio_abs_deviation_mean": row_dev.mean().item(),
            "ratio_abs_deviation_max": row_dev.max().item(),
            "position_of_max": int(row_dev.argmax().item()),
        })

    for eps in cfg["clip_values"]:
        eps= float(eps)

        loss, _, clip_fraction = ppo_policy_loss(
            new_logp,
            old_logp,
            advantages,
            mask,
            eps
        )

        outside= ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).float() * mask

        #clipping only changes the objective when the min picks the clipped term
        binding= (((ratio > 1.0 + eps) & (advantages > 0)) | ((ratio < 1.0 - eps) & (advantages < 0))).float() * mask

        for i in range(len(rows)):
            out["rows"][i][f"tokens_outside_{eps:g}"]= int(outside[i].sum().item())

        out["epsilons"].append({
            "clip_epsilon": eps,
            "clipped_surrogate": -loss.item(),
            #fraction of valid response tokens with the ratio outside [1-eps, 1+eps] before clipping
            "clip_fraction": clip_fraction.item(),
            "clip_fraction_ci95": bootstrap_ci(outside.sum(-1).tolist(), n_tokens, seed),
            #fraction of valid response tokens where the clipped term is the one the objective uses
            "binding_fraction": masked_mean(binding, mask).item(),
            "binding_fraction_ci95": bootstrap_ci(binding.sum(-1).tolist(), n_tokens, seed),
        })

        print(
            f"{name}, eps={eps:g}: "
            f"clipped_surrogate={-loss.item():.6f}, "
            f"clip_fraction={clip_fraction.item():.4f}, "
            f"binding_fraction={out['epsilons'][-1]['binding_fraction']:.4f}"
        )

    for row in sorted(out["rows"], key=lambda r: -r["ratio_abs_deviation_max"])[:3]:
        print(
            f"row {row['row_index']}: "
            f"tokens={row['n_tokens']}, "
            f"mean |ratio-1|={row['ratio_abs_deviation_mean']:.4f}, "
            f"max |ratio-1|={row['ratio_abs_deviation_max']:.4f} at token {row['position_of_max']}"
        )

    path= Path(cfg["results_dir"]) / f"clipping_cached_{name}.json"
    save_json(path, out)

    print(f"saved cached-batch analysis to {path}")

    del model
    clear_gpu()

    return out


def delta_kls(log_rows):
    #sampled kl after minus before each update on that update's own rollout tokens
    #the reference log-probs cancel, so this is the mean log ratio new/old: how far the update moved the policy
    if len(log_rows) == 0 or "delta_kl" not in log_rows[0]:
        raise SystemExit(
            "this train_log.jsonl has no delta_kl. It was written before the post-update kl was logged, "
            "rerun that run with the current code."
        )

    return [r["delta_kl"] for r in log_rows]


def stability_stats(deltas, threshold):
    #S1= std over updates of delta kl (population std), S2= number of updates with |delta kl| above the threshold
    return {
        "s1_std_delta_kl": float(np.std(deltas)),
        "s2_updates_above_threshold": int(sum(abs(d) > threshold for d in deltas)),
    }


def stability_threshold(cfg):
    #computed once from the standard run and never recomputed, so no fork can move it
    path= Path(cfg["results_dir"]) / "stability_threshold.json"

    if repo_path(path).exists():
        return load_json(path)["threshold"]

    log_path= Path(cfg["results_dir"]) / "standard" / "train_log.jsonl"

    if not repo_path(log_path).exists():
        raise SystemExit(f"No standard run log at {log_path}. Run the standard continuation first.")

    deltas= delta_kls(read_jsonl(log_path))

    if len(deltas) != int(cfg["updates"]):
        raise SystemExit(f"standard run has {len(deltas)} updates in its log, expected {cfg['updates']}")

    threshold= THRESHOLD_FACTOR * float(np.median(np.abs(deltas)))

    save_json(path, {
        "threshold": threshold,
        "definition": "3 x median |delta_kl| over the updates of the standard run",
        "delta_kl": "sampled kl estimate after the update minus before it, on the same rollout tokens. the reference term cancels, so this is the mean change in log-prob of the sampled tokens: a measure of how far the update moved the policy on its own rollout, not a fresh estimate of kl from the reference",
        "source": str(log_path),
        "n_updates": len(deltas),
        "git_commit": git_commit(),
    })

    print(f"saved stability threshold {threshold:.3e} to {path}")

    return threshold


def run_clipping_forks(config_path: str, smoke: bool = False, allow_cpu: bool = False, gen_batch_size: int = 4):
    cfg= adjust_config(load_yaml(config_path), smoke)

    results_dir= Path(cfg["results_dir"])
    fork_updates= int(cfg["fork_updates"])

    if smoke and not repo_path(results_dir / "standard" / "train_metrics.json").exists():
        #a smoke run of the study needs a smoke standard run for the threshold
        run_ppo(config_path, run_name="standard", smoke=True, resume=True, allow_cpu=allow_cpu)
        clear_gpu()

    #fixed before any fork is looked at
    threshold= stability_threshold(cfg)

    forks= []

    for eps in cfg["clip_values"]:
        eps= float(eps)
        run_name= run_name_for(eps)

        #only epsilon changes: same midpoint, prompts, seed, kl beta and update budget
        log_rows, train, heldout = train_and_evaluate_fork(
            config_path,
            run_name,
            clip_epsilon=eps,
            smoke=smoke,
            allow_cpu=allow_cpu,
            gen_batch_size=gen_batch_size
        )

        deltas= delta_kls(log_rows)

        fork= {
            "run_name": run_name,
            "clip_epsilon": eps,
            "kl_beta": train["kl_beta"],
            "updates": train["updates"],
            "generated_tokens_train": train["generated_tokens"],
            "delta_kl": deltas,
            "train_clip_fraction": [r["clip_fraction"] for r in log_rows],
            "train_max_ratio_deviation": [r["max_ratio_deviation"] for r in log_rows],
        }

        fork.update(stability_stats(deltas, threshold))

        for key in ["reward_mean", "kl_token_mean", "kl_sequence_mean", "length_mean", "no_eos_fraction"]:
            fork["heldout_" + key]= heldout[key]
            fork["heldout_" + key + "_ci95"]= heldout[key + "_ci95"]

        fork["heldout_length_std"]= heldout["length_std"]

        forks.append(fork)

        print(
            f"{run_name}: "
            f"S1={fork['s1_std_delta_kl']:.3e}, "
            f"S2={fork['s2_updates_above_threshold']}, "
            f"reward={fork['heldout_reward_mean']:.4f}, "
            f"kl_token={fork['heldout_kl_token_mean']:.6f}, "
            f"length={fork['heldout_length_mean']:.1f}"
        )

    out= {
        "git_commit": git_commit(),
        "seed": int(cfg["seed"]),
        "fork_updates": fork_updates,
        "prompts_per_update": int(cfg["prompts_per_update"]),
        "stability_threshold": threshold,
        "s1": "population std over updates of delta_kl",
        "s2": "number of updates with |delta_kl| above the stability threshold",
        "note": PROMPT_NOTE,
        "budget_note": f"forks run {fork_updates} updates from the midpoint, the standard run has {cfg['updates']}",
        "smoke": smoke,
        "forks": forks,
    }

    path= results_dir / "clipping_forks.json"
    save_json(path, out)

    print(f"saved fork comparison to {path}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--part", choices=["all", "cached", "forks"], default="all")
    ap.add_argument("--gen-batch-size", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    cfg = adjust_config(load_yaml(args.config), args.smoke)

    if args.part in ["all", "cached"]:
        if args.smoke:
            #the tiny smoke model has no midpoint adapter, the base model stands in
            analyze_cached_batch(args.config, None, "base", True, args.allow_cpu)
        else:
            analyze_cached_batch(args.config, cfg["paths"]["ppo_midpoint_policy"], "midpoint", False, args.allow_cpu)

            #the standard run has moved away from the midpoint, so its ratios on the cached batch are not all 1
            if (repo_path(cfg["output"]) / "adapter_config.json").exists():
                analyze_cached_batch(args.config, cfg["output"], "standard", False, args.allow_cpu)
            else:
                print("no standard adapter yet, cached batch analysed for the midpoint only")

    if args.part in ["all", "forks"]:
        run_clipping_forks(args.config, args.smoke, args.allow_cpu, args.gen_batch_size)


if __name__ == "__main__":
    main()
