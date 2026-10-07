from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, read_jsonl
from common.logging_utils import save_json
from common.metrics import bootstrap_ci
from task1_dpo.train import check_setup, git_commit
from task3_grpo.continue_train import INFORMATIVE_STD_TOL, adjust_config
from task3_grpo.grpo import group_relative_advantages

CACHED_PER_PROMPT= 8
REPEATS= 200
SMOKE_REPEATS= 5
BINS= ["hard", "medium", "easy"]
METRICS= ["informative_rate", "reward_std", "advantage_variance", "centred_reward_variance"]


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, rng=None):
    """Return K-sized groups while keeping total cached completions fixed.

    Every prompt keeps its 8 cached completions and they are split into 8 / K groups of K, so each K uses
    the same completions. With rng the 8 completions are shuffled first, without it the cached order is kept.
    Returns a list of (prompt key, list of K rewards).
    """
    groups= []

    for key in by_prompt:
        rewards= [float(row["reward"]) for row in by_prompt[key][:CACHED_PER_PROMPT]]

        order= list(range(CACHED_PER_PROMPT))

        if rng is not None:
            order= rng.permutation(CACHED_PER_PROMPT).tolist()

        for start in range(0, CACHED_PER_PROMPT, k):
            groups.append((key, [rewards[i] for i in order[start:start + k]]))

    return groups


def difficulty_bins(by_prompt):
    #fixed rule: prompts ranked by their mean reward over all 8 cached completions, split into thirds
    #bottom third= hard, middle third= medium, top third= easy
    keys= list(by_prompt)
    means= [np.mean([float(row["reward"]) for row in by_prompt[key][:CACHED_PER_PROMPT]]) for key in keys]

    ranked= [keys[i] for i in np.argsort(means, kind="stable")]
    third= len(ranked) // 3

    bins= {}

    for i in range(len(ranked)):
        if i < third:
            bins[ranked[i]]= "hard"
        elif i < 2 * third:
            bins[ranked[i]]= "medium"
        else:
            bins[ranked[i]]= "easy"

    return bins


def partition_metrics(groups):
    #per prompt, for one way of splitting its completions into groups
    group_ids= []
    rewards= []

    for g in range(len(groups)):
        for r in groups[g][1]:
            group_ids.append(g)
            rewards.append(r)

    #the same function the training loop uses
    advantages= group_relative_advantages(
        torch.tensor(rewards, dtype=torch.float64),
        torch.tensor(group_ids)
    ).tolist()

    sums= defaultdict(lambda: defaultdict(float))
    position= 0

    for key, group in groups:
        std= float(np.std(group))
        mean= float(np.mean(group))

        sums[key]["groups"] += 1
        sums[key]["informative"] += float(std > INFORMATIVE_STD_TOL)
        sums[key]["std"] += std

        for r in group:
            sums[key]["completions"] += 1
            #advantages and centred rewards have mean 0 inside a group, so their variance is the mean square
            sums[key]["adv_sq"] += advantages[position] ** 2
            sums[key]["centred_sq"] += (r - mean) ** 2
            position += 1

    out= {}

    for key in sums:
        s= sums[key]

        out[key]= {
            "informative_rate": s["informative"] / s["groups"],
            "reward_std": s["std"] / s["groups"],
            "advantage_variance": s["adv_sq"] / s["completions"],
            "centred_reward_variance": s["centred_sq"] / s["completions"],
        }

    return out


def summarize(per_prompt, keys, seed):
    #mean over prompts, with a bootstrap interval from resampling prompts
    out= {"n_prompts": len(keys)}

    for metric in METRICS:
        values= [per_prompt[key][metric] for key in keys]

        out[metric]= float(np.mean(values))
        out[metric + "_ci95"]= bootstrap_ci(values, None, seed)

    return out


def run_group_size_study(config_path: str, smoke: bool = False):
    cfg= adjust_config(load_yaml(config_path), smoke)

    check_setup([cfg["group_cache"]], True, True)

    seed= int(cfg["seed"])
    repeats= SMOKE_REPEATS if smoke else REPEATS

    by_prompt= load_k8_cache(cfg["group_cache"])
    bins= difficulty_bins(by_prompt)
    keys= list(by_prompt)

    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])

    results= []

    for k in cfg["group_sizes"]:
        k= int(k)

        #same shuffles for every K, so the group sizes are compared on the same random orders
        rng= np.random.default_rng(seed)

        totals= {key: {metric: 0.0 for metric in METRICS} for key in keys}

        for _ in range(repeats):
            metrics= partition_metrics(regroup_equal_generation_budget(by_prompt, k, rng))

            for key in keys:
                for metric in METRICS:
                    totals[key][metric] += metrics[key][metric] / repeats

        row= {
            "group_size": k,
            "groups_per_prompt": CACHED_PER_PROMPT // k,
            "total_generations": CACHED_PER_PROMPT * len(keys),
            "all": summarize(totals, keys, seed),
        }

        for name in BINS:
            row[name]= summarize(totals, [key for key in keys if bins[key] == name], seed)

        results.append(row)

        print(
            f"K={k}: "
            f"informative_rate={row['all']['informative_rate']:.4f}, "
            f"reward_std={row['all']['reward_std']:.4f}, "
            f"advantage_variance={row['all']['advantage_variance']:.4f}, "
            f"centred_reward_variance={row['all']['centred_reward_variance']:.4f}"
        )

        for name in BINS:
            print(
                f"  {name}: "
                f"informative_rate={row[name]['informative_rate']:.4f}, "
                f"reward_std={row[name]['reward_std']:.4f}, "
                f"advantage_variance={row[name]['advantage_variance']:.4f}"
            )

    out= {
        "git_commit": git_commit(),
        "seed": seed,
        "cache": cfg["group_cache"],
        "n_prompts": len(keys),
        "completions_per_prompt": CACHED_PER_PROMPT,
        "repeats": repeats,
        "informative_std_tolerance": INFORMATIVE_STD_TOL,
        "smoke": smoke,
        "definitions": {
            "regrouping": "each prompt's 8 cached completions are shuffled and split into 8/K groups of K, so every K uses the same 192 completions; metrics are averaged over the random shuffles",
            "informative_rate": "fraction of groups whose reward std (population) is above the tolerance",
            "reward_std": "mean within-group reward std (population)",
            "advantage_variance": "variance over completions of the group-relative advantage (r - group mean) / (group std + eps)",
            "centred_reward_variance": "variance over completions of r - group mean, the group-relative signal before dividing by the std",
            "difficulty_bins": "prompts ranked by mean reward over all 8 cached completions: bottom third hard, middle third medium, top third easy",
            "ci95": "bootstrap over prompts",
        },
        "prompt_bins": bins,
        "results": results,
    }

    path= Path(cfg["results_dir"]) / "group_size_study.json"
    save_json(path, out)

    print(f"saved group-size study to {path}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_group_size_study(args.config, args.smoke)


if __name__ == "__main__":
    main()
