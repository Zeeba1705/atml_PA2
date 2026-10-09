from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, write_jsonl
from common.logging_utils import save_json, wall_timer
from common.metrics import bootstrap_ci
from task1_dpo.train import check_setup, git_commit
from task5_feedback.evaluate_math import adjust_config
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

CLEAN= "clean_correct"
#every controlled pair is the clean response against one perturbed response, the clean one is the better one
PERTURBATIONS= [
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
]
MECHANISMS= ["verifier", "judge"]
OUTCOMES= ["better", "tie", "wrong"]
#S_reason: same correct final answer, reasoning degraded
REASON_PAIRS= ["corrupt_reasoning_correct_final"]
#S_outcome: reasoning held roughly fixed, designated final answer changed
OUTCOME_PAIRS= ["good_reasoning_wrong_final"]
#every pair where the perturbed response ends in a wrong final answer
WRONG_FINAL_PAIRS= ["good_reasoning_wrong_final", "gold_distractor_wrong_final"]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def verifier_outcome(reward_clean, reward_other):
    #binary rewards: the verifier prefers the response with the higher one and ties when they are equal
    if reward_clean > reward_other:
        return "better"

    if reward_clean < reward_other:
        return "wrong"

    return "tie"


def judge_outcome(preference):
    #the judge was shown the clean response as candidate A
    return {"A": "better", "TIE": "tie", "B": "wrong"}[preference]


def score_pairs(groups, judge):
    pairs= []

    for pid in groups:
        clean= groups[pid][CLEAN]
        gold= str(clean["gold_final"])
        reward_clean= exact_reward(clean["response"], gold)

        for variant in PERTURBATIONS:
            other= groups[pid][variant]
            reward_other= exact_reward(other["response"], gold)

            preference= judge.compare(clean["question"], clean["response"], other["response"])

            pairs.append({
                "problem_id": pid,
                "perturbation": variant,
                "verifier_reward_clean": reward_clean,
                "verifier_reward_perturbed": reward_other,
                "verifier_outcome": verifier_outcome(reward_clean, reward_other),
                "judge_preference": preference,
                "judge_outcome": judge_outcome(preference),
                "perturbed_reasoning_quality": other["reasoning_quality"],
                "perturbed_style": other["style"],
            })

    return pairs


def rates(pairs, mechanism, perturbations, seed):
    #better / tie / wrong fractions over the pairs of these perturbation types, intervals from resampling problems
    selected= [p for p in pairs if p["perturbation"] in perturbations]

    by_problem= defaultdict(list)

    for p in selected:
        by_problem[p["problem_id"]].append(p[mechanism + "_outcome"])

    out= {"n_pairs": len(selected), "n_problems": len(by_problem)}

    for outcome in OUTCOMES:
        #sum over problems / number of pairs, resampled as a ratio so problems stay whole
        counts= [sum(x == outcome for x in by_problem[pid]) for pid in by_problem]
        sizes= [len(by_problem[pid]) for pid in by_problem]

        out[outcome + "_rate"]= sum(counts) / max(sum(sizes), 1)
        out[outcome + "_rate_ci95"]= bootstrap_ci(counts, sizes, seed)

    return out


def summarize(pairs, seed):
    out= {"per_perturbation": {}, "sensitivity": {}}

    for variant in PERTURBATIONS:
        out["per_perturbation"][variant]= {m: rates(pairs, m, [variant], seed) for m in MECHANISMS}

    out["all_pairs"]= {m: rates(pairs, m, PERTURBATIONS, seed) for m in MECHANISMS}

    for mechanism in MECHANISMS:
        reason= rates(pairs, mechanism, REASON_PAIRS, seed)
        outcome= rates(pairs, mechanism, OUTCOME_PAIRS, seed)
        wrong_final= rates(pairs, mechanism, WRONG_FINAL_PAIRS, seed)

        out["sensitivity"][mechanism]= {
            #Pr[R(clean) > R(reasoning corrupted, same correct final)]
            "s_reason": reason["better_rate"],
            "s_reason_ci95": reason["better_rate_ci95"],
            "s_reason_tie_rate": reason["tie_rate"],
            "s_reason_wrong_rate": reason["wrong_rate"],
            #Pr[R(correct final) > R(wrong final)] with the reasoning held roughly fixed
            "s_outcome": outcome["better_rate"],
            "s_outcome_ci95": outcome["better_rate_ci95"],
            "s_outcome_tie_rate": outcome["tie_rate"],
            "s_outcome_wrong_rate": outcome["wrong_rate"],
            #the same over both wrong-final variants, gold-distractor pairs included
            "s_outcome_all_wrong_final": wrong_final["better_rate"],
            "s_outcome_all_wrong_final_ci95": wrong_final["better_rate_ci95"],
        }

    return out


def run_perturbation_study(config_path: str, smoke: bool = False, allow_cpu: bool = False):
    cfg = adjust_config(load_yaml(config_path), smoke)

    check_setup([cfg["paths"]["task5_diagnostics"]], smoke, allow_cpu)

    seed= int(cfg["seed"])
    outdir= Path(cfg["task5_dir"]) / "diagnostics"

    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])

    if smoke:
        groups= {pid: groups[pid] for pid in list(groups)[:3]}

    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))

    #the supplied verifier should reproduce the reward the staff recorded for every diagnostic response
    mismatches= 0

    for pid in groups:
        for variant in groups[pid]:
            row= groups[pid][variant]
            mismatches += int(exact_reward(row["response"], str(row["gold_final"])) != float(row["expected_exact_reward"]))

    print("verifier vs expected_exact_reward mismatches:", mismatches)

    judge= PairwiseAIJudge(cfg, Path(cfg["task5_dir"]) / "judge_cache.json")

    timer= wall_timer()
    pairs= score_pairs(groups, judge)
    judge_seconds= timer()

    summary= summarize(pairs, seed)

    table= []

    for variant in PERTURBATIONS + ["all_pairs"]:
        part= summary["all_pairs"] if variant == "all_pairs" else summary["per_perturbation"][variant]

        for mechanism in MECHANISMS:
            row= {"perturbation": variant, "mechanism": mechanism, "n_pairs": part[mechanism]["n_pairs"]}

            for outcome in OUTCOMES:
                row[outcome + "_rate"]= part[mechanism][outcome + "_rate"]
                row[outcome + "_rate_ci_low"]= part[mechanism][outcome + "_rate_ci95"][0]
                row[outcome + "_rate_ci_high"]= part[mechanism][outcome + "_rate_ci95"][1]

            table.append(row)

            print(
                f"{variant} / {mechanism}: "
                f"better={row['better_rate']:.4f}, "
                f"tie={row['tie_rate']:.4f}, "
                f"wrong={row['wrong_rate']:.4f}"
            )

    for mechanism in MECHANISMS:
        s= summary["sensitivity"][mechanism]

        print(
            f"{mechanism}: "
            f"S_reason={s['s_reason']:.4f}, "
            f"S_outcome={s['s_outcome']:.4f}, "
            f"S_outcome over all wrong-final pairs={s['s_outcome_all_wrong_final']:.4f}"
        )

    write_jsonl(outdir / "pairs.jsonl", pairs)
    pd.DataFrame(table).to_csv(outdir / "perturbation_rates.csv", index=False)

    save_json(outdir / "summary.json", {
        "git_commit": git_commit(),
        "seed": seed,
        "dataset_file": cfg["paths"]["task5_diagnostics"],
        "n_problems": len(groups),
        "n_pairs": len(pairs),
        "judge_model": cfg["ai_judge_model"],
        "smoke": smoke,
        "definitions": {
            "controlled_pair": "the clean correct response against one perturbed response of the same problem; the clean response is the diagnostically better one in every pair",
            "verifier": "better when its binary reward is higher for the clean response, tie when the two rewards are equal, wrong when it is higher for the perturbed response",
            "judge": "the supplied pairwise judge with the clean response as candidate A: better = A, tie = TIE, wrong = B",
            "s_reason": "better rate on clean vs corrupt_reasoning_correct_final",
            "s_outcome": "better rate on clean vs good_reasoning_wrong_final",
            "s_outcome_all_wrong_final": "better rate on clean vs good_reasoning_wrong_final and clean vs gold_distractor_wrong_final together",
            "ci95": "bootstrap over problems",
        },
        "note": "a verifier tie on a reasoning-only or filler pair is expected: both responses end in the correct final answer",
        "verifier_vs_expected_exact_reward_mismatches": mismatches,
        "judge_seconds_this_session": judge_seconds,
        "summary": summary,
    })

    print(f"saved diagnostic study to {outdir}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    run_perturbation_study(args.config, args.smoke, args.allow_cpu)


if __name__ == "__main__":
    main()
