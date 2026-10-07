from __future__ import annotations

import argparse
from pathlib import Path

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task1_dpo.train import git_commit
from task2_ppo.continue_train import adjust_config

LENGTH_RATIO= 1.5
NGRAM= 8
NGRAM_REPEATS= 3
SKIP_PREFIXES= ["quicktest", "_"]


def max_ngram_repeats(text, n):
    #how many times the most repeated run of n whitespace-separated words appears
    words= text.split()
    counts= {}

    for i in range(len(words) - n + 1):
        gram= " ".join(words[i:i + n])
        counts[gram]= counts.get(gram, 0) + 1

    if len(counts) == 0:
        return 0

    return max(counts.values())


def quality_flags(record, baseline):
    #signs that a higher reward may not be a better response
    flags= []

    if record["n_tokens"] >= LENGTH_RATIO * baseline["n_tokens"]:
        flags.append("much_longer")

    if not record["has_eos"]:
        flags.append("no_eos")

    if max_ngram_repeats(record["response"], NGRAM) >= NGRAM_REPEATS:
        flags.append("repeated_ngram")

    return flags


def classify(record, baseline):
    #suspect= reward went up together with a warning sign
    #agree= reward went up, the response got somewhat longer, and there is no warning sign
    if record["reward"] <= baseline["reward"]:
        return None, []

    flags= quality_flags(record, baseline)

    if len(flags) > 0:
        return "suspect", flags

    if record["n_tokens"] > baseline["n_tokens"]:
        return "agree", flags

    return None, flags


def find_candidates(config_path: str, baseline_name: str = "midpoint", runs: list | None = None, smoke: bool = False):
    cfg= adjust_config(load_yaml(config_path), smoke)

    results_dir= repo_path(cfg["results_dir"])
    baseline_path= results_dir / baseline_name / "generations.jsonl"

    if not baseline_path.exists():
        raise SystemExit(
            f"No generations for '{baseline_name}' at {baseline_path}. "
            "Evaluate it first with task2_ppo.evaluate."
        )

    baseline= {r["prompt_id"]: r for r in read_jsonl(baseline_path)}

    if runs is None:
        #every evaluated run except the baseline, the base policy, quick tests and superseded folders
        runs= []

        for d in sorted(results_dir.iterdir()):
            skip= d.name in [baseline_name, "sft"] or any(d.name.startswith(p) for p in SKIP_PREFIXES)

            if d.is_dir() and not skip and (d / "generations.jsonl").exists():
                runs.append(d.name)

    suspect= []
    agree= []
    counts= {}

    for run in runs:
        records= read_jsonl(results_dir / run / "generations.jsonl")

        counts[run]= {"n_prompts": 0, "reward_up": 0, "suspect": 0, "agree": 0}

        for record in records:
            if record["prompt_id"] not in baseline:
                continue

            base= baseline[record["prompt_id"]]
            kind, flags = classify(record, base)

            counts[run]["n_prompts"] += 1
            counts[run]["reward_up"] += int(record["reward"] > base["reward"])

            if kind is None:
                continue

            counts[run][kind] += 1

            row= {
                "run": run,
                "prompt_id": record["prompt_id"],
                "row_index": record["row_index"],
                "reward": record["reward"],
                "baseline_reward": base["reward"],
                "reward_gain": record["reward"] - base["reward"],
                "n_tokens": record["n_tokens"],
                "baseline_n_tokens": base["n_tokens"],
                "length_ratio": record["n_tokens"] / max(base["n_tokens"], 1),
                "has_eos": record["has_eos"],
                "max_ngram_repeats": max_ngram_repeats(record["response"], NGRAM),
                "flags": flags,
                "prompt": record["prompt"],
                "response": record["response"],
                "baseline_response": base["response"],
            }

            if kind == "suspect":
                suspect.append(row)
            else:
                agree.append(row)

    #largest reward gains first
    suspect.sort(key=lambda r: -r["reward_gain"])
    agree.sort(key=lambda r: -r["reward_gain"])

    out= {
        "git_commit": git_commit(),
        "baseline": baseline_name,
        "runs": runs,
        "definitions": {
            "suspect": "reward above the baseline response to the same prompt AND at least one flag",
            "much_longer": f"response has at least {LENGTH_RATIO} x the baseline's tokens",
            "no_eos": "response hit the generation cap without eos",
            "repeated_ngram": f"some run of {NGRAM} whitespace-separated words appears {NGRAM_REPEATS} or more times",
            "agree": f"reward above the baseline, more tokens than the baseline but under {LENGTH_RATIO} x, and no flag",
        },
        "note": "candidates only: one sampled response per prompt per run, to be read and judged by hand",
        "counts": counts,
        "suspect": suspect,
        "agree": agree,
    }

    path= Path(cfg["results_dir"]) / "candidates.json"
    save_json(path, out)

    for run in runs:
        c= counts[run]

        print(
            f"{run}: "
            f"prompts={c['n_prompts']}, "
            f"reward_up={c['reward_up']}, "
            f"suspect={c['suspect']}, "
            f"agree={c['agree']}"
        )

    print(f"saved candidates to {path}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--baseline", default="midpoint")
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    find_candidates(args.config, args.baseline, args.runs, args.smoke)


if __name__ == "__main__":
    main()
