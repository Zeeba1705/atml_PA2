from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed, wall_timer
from common.metrics import bootstrap_ci, paired_bootstrap_ci
from common.models import clear_gpu, load_policy, load_tokenizer
from task1_dpo.train import SMOKE_MODEL, check_setup, git_commit
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

POLICIES= ["sft", "rlvr", "rlaif"]
BASELINE= "sft"
#(a, b): the judge is asked to compare a's response with b's response to the same problem
COMPARISONS= [("rlvr", "sft"), ("rlaif", "sft"), ("rlaif", "rlvr")]
MAX_PROMPT_LENGTH= 512
FAILURE_SAMPLE= 30
FAILURE_TYPES= ["format_failure", "arithmetic_slip", "misread_problem", "right_method_wrong_final", "other"]
SMOKE_EXAMPLES= 4
SMOKE_NEW_TOKENS= 32


def adjust_config(cfg, smoke, max_examples=None):
    #shared by every task 5 script so they all read and write the same folder
    cfg["task5_dir"]= str(repo_path(cfg["results_dir"]) / "task5_feedback")

    if max_examples is not None:
        #a partial run goes to its own folder so it can never be mistaken for the full evaluation
        cfg["max_examples"]= int(max_examples)
        cfg["task5_dir"]= str(repo_path(cfg["results_dir"]) / "task5_feedback_quicktest")

    if smoke:
        #tiny model for policy and judge, no adapters (they were trained on the 1.5B model), a few problems
        cfg["base_model"]= SMOKE_MODEL
        cfg["ai_judge_model"]= SMOKE_MODEL
        cfg["math_max_new_tokens"]= SMOKE_NEW_TOKENS
        cfg["max_examples"]= SMOKE_EXAMPLES
        cfg["task5_dir"]= str(repo_path(cfg["results_dir"]) / "smoke" / "task5_feedback")

        for name in ["rlvr", "rlaif"]:
            cfg["policies"][name]= None

    if not torch.cuda.is_available():
        #float16 on cpu is very slow, the gpu runs keep the config dtype
        cfg["dtype"]= "float32"

    return cfg


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str, smoke: bool = False, max_examples: int | None = None):
    cfg = adjust_config(load_yaml(config_path), smoke, max_examples)
    rows = read_jsonl(dataset_path(cfg, dataset))

    if cfg.get("max_examples") is not None:
        rows= rows[: int(cfg["max_examples"])]

    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def problem_id(row):
    return str(row.get("prompt_id", row.get("source_index")))


def generate_for_policy(cfg, rows, tokenizer, name, batch_size):
    #one greedy response per problem, then the supplied verifier on each
    model= load_frozen_policy(cfg, name)

    records= []

    for start in range(0, len(rows), batch_size):
        chunk= rows[start:start + batch_size]

        gen= batch_generate(
            model,
            tokenizer,
            [row["messages"] for row in chunk],
            max_prompt_length=MAX_PROMPT_LENGTH,
            max_new_tokens=int(cfg["math_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False
        )

        for k in range(len(chunk)):
            row= chunk[k]
            response= gen["responses"][k]
            gold= str(row["gold_final"])
            predicted= extract_designated_final(response)

            records.append({
                "problem_id": problem_id(row),
                "policy": name,
                "question": row["question"],
                "gold_final": gold,
                "response": response,
                "response_tokens": gen["response_lengths"][k],
                "has_eos": gen["terminated_with_eos"][k],
                "truncated": gen["truncated"][k],
                "predicted_final": predicted,
                #format compliance= the designated '#### <number>' final answer could be parsed
                "format_ok": predicted is not None,
                "correct": exact_reward(response, gold) == 1.0,
            })

        if (start // batch_size + 1) % 10 == 0:
            print(f"{name}: generated {min(start + batch_size, len(rows))}/{len(rows)}")

    del model
    clear_gpu()

    return records


def policy_metrics(records, seed):
    correct= [float(r["correct"]) for r in records]
    format_ok= [float(r["format_ok"]) for r in records]
    lengths= [float(r["response_tokens"]) for r in records]

    return {
        "n_problems": len(records),
        "exact_accuracy": float(np.mean(correct)),
        "exact_accuracy_ci95": bootstrap_ci(correct, None, seed),
        "format_compliance": float(np.mean(format_ok)),
        "format_compliance_ci95": bootstrap_ci(format_ok, None, seed),
        "response_tokens_mean": float(np.mean(lengths)),
        "response_tokens_std": float(np.std(lengths)),
        "response_tokens_mean_ci95": bootstrap_ci(lengths, None, seed),
        "truncated_fraction": float(np.mean([float(r["truncated"]) for r in records])),
        #correct among the responses whose final answer could be parsed
        "accuracy_given_format_ok": float(np.mean([float(r["correct"]) for r in records if r["format_ok"]])) if sum(format_ok) > 0 else float("nan"),
    }


def paired_metrics(records, base_records, seed):
    #same problems in the same order, so the two policies are compared problem by problem
    out= {}

    for key in ["correct", "format_ok", "response_tokens"]:
        a= [float(r[key]) for r in records]
        b= [float(r[key]) for r in base_records]

        out[key + "_diff"]= float(np.mean(a) - np.mean(b))
        out[key + "_diff_ci95"]= paired_bootstrap_ci(a, b, seed)

    return out


def judge_pairs(judge, records_a, records_b, name_a, name_b):
    #the supplied judge on a's and b's response to the same problem: "A" means it prefers a's response
    rows= []
    timer= wall_timer()

    for a, b in zip(records_a, records_b):
        if a["response"] == b["response"]:
            #two identical texts cannot be ranked, asking the judge would only measure its position bias
            preference= "TIE"
        else:
            preference= judge.compare(a["question"], a["response"], b["response"])

        rows.append({
            "problem_id": a["problem_id"],
            "a": name_a,
            "b": name_b,
            "preference": preference,
            "a_correct": a["correct"],
            "b_correct": b["correct"],
            "identical_responses": a["response"] == b["response"],
        })

    return rows, timer()


def pairwise_summary(pair_rows, seed):
    #win 1, tie 0.5, loss 0 from a's side, ties also reported on their own
    scores= []

    for r in pair_rows:
        scores.append({"A": 1.0, "TIE": 0.5, "B": 0.0}[r["preference"]])

    n= len(pair_rows)

    return {
        "n_pairs": n,
        "wins": sum(r["preference"] == "A" for r in pair_rows),
        "ties": sum(r["preference"] == "TIE" for r in pair_rows),
        "losses": sum(r["preference"] == "B" for r in pair_rows),
        "tie_rate": sum(r["preference"] == "TIE" for r in pair_rows) / n,
        "win_rate": float(np.mean(scores)),
        "win_rate_ci95": bootstrap_ci(scores, None, seed),
        "identical_responses": sum(r["identical_responses"] for r in pair_rows),
    }


def verifier_judge_agreement(pair_rows):
    #fixed definition: over pairs where exactly one response is verifier-correct,
    #the fraction where the judge prefers the correct one, with judge ties counted separately
    decisive= [r for r in pair_rows if r["a_correct"] != r["b_correct"]]

    prefers_correct= 0
    prefers_wrong= 0
    ties= 0

    for r in decisive:
        correct_side= "A" if r["a_correct"] else "B"

        if r["preference"] == "TIE":
            ties += 1
        elif r["preference"] == correct_side:
            prefers_correct += 1
        else:
            prefers_wrong += 1

    n= len(decisive)

    return {
        "n_pairs_exactly_one_correct": n,
        "judge_prefers_correct": prefers_correct,
        "judge_ties": ties,
        "judge_prefers_wrong": prefers_wrong,
        "agreement": prefers_correct / n if n > 0 else float("nan"),
        "tie_rate": ties / n if n > 0 else float("nan"),
        "wrong_rate": prefers_wrong / n if n > 0 else float("nan"),
    }


def failure_sheet(generated, seed):
    #a fixed random sample of wrong responses per policy, the failure_type column is left empty for hand labelling
    rows= []

    for name in POLICIES:
        wrong= [r for r in generated[name] if not r["correct"]]

        rng= np.random.default_rng(seed)
        picked= sorted(rng.choice(len(wrong), size=min(FAILURE_SAMPLE, len(wrong)), replace=False).tolist())

        for i in picked:
            r= wrong[i]

            rows.append({
                "policy": name,
                "problem_id": r["problem_id"],
                "question": r["question"],
                "gold_final": r["gold_final"],
                "predicted_final": r["predicted_final"],
                "format_ok": r["format_ok"],
                "response": r["response"],
                "failure_type": "",
            })

    return pd.DataFrame(rows)


def run_math_evaluation(config_path: str, dataset: str, batch_size: int = 8, smoke: bool = False, max_examples: int | None = None, allow_cpu: bool = False):
    #checked before anything is read, so a missing file or gpu stops with the fix spelled out
    check_setup(
        [dataset_path(adjust_config(load_yaml(config_path), smoke, max_examples), dataset)],
        smoke,
        allow_cpu
    )

    cfg, rows, tokenizer = load_math_evaluation(config_path, dataset, smoke, max_examples)

    specs= policy_specs(cfg)

    for name in POLICIES:
        if specs[name] is not None and not (repo_path(specs[name]) / "adapter_config.json").exists():
            raise SystemExit(f"No adapter for '{name}' at {specs[name]}. Run the course asset step first.")

    seed= int(cfg["seed"])
    outdir= Path(cfg["task5_dir"]) / dataset

    print("Rows:", len(rows))
    print("Policies:", list(specs))

    #a prompt longer than the generation limit would be cut, so this is counted and reported
    n_long= 0

    for row in rows:
        rendered= tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        n_long += int(len(tokenizer(rendered)["input_ids"]) > MAX_PROMPT_LENGTH)

    generated= {}
    generation_seconds= {}

    for name in POLICIES:
        path= outdir / f"generated_{name}.jsonl"

        if path.exists() and len(read_jsonl(path)) == len(rows):
            print(f"responses of '{name}' are already saved")
            generated[name]= read_jsonl(path)
            continue

        timer= wall_timer()

        #greedy decoding has no sampling, the seed is set anyway so every policy starts from the same state
        set_seed(seed)

        generated[name]= generate_for_policy(cfg, rows, tokenizer, name, batch_size)
        generation_seconds[name]= timer()

        write_jsonl(path, generated[name])

        print(f"saved {len(generated[name])} responses of '{name}' to {path}")

    #one cache for both datasets and the diagnostic set, so no pair is ever judged twice
    judge= PairwiseAIJudge(cfg, Path(cfg["task5_dir"]) / "judge_cache.json")

    pair_rows= []
    pairwise= {}
    agreement= {}
    judge_seconds= 0.0

    for name_a, name_b in COMPARISONS:
        rows_ab, seconds = judge_pairs(judge, generated[name_a], generated[name_b], name_a, name_b)

        pair_rows += rows_ab
        judge_seconds += seconds

        key= f"{name_a}_vs_{name_b}"

        pairwise[key]= pairwise_summary(rows_ab, seed)
        agreement[key]= verifier_judge_agreement(rows_ab)

        print(
            f"{key}: "
            f"win_rate={pairwise[key]['win_rate']:.4f}, "
            f"wins={pairwise[key]['wins']}, "
            f"ties={pairwise[key]['ties']}, "
            f"losses={pairwise[key]['losses']}"
        )

    #both trained policies against the base policy, pooled
    agreement["pooled_vs_sft"]= verifier_judge_agreement([r for r in pair_rows if r["b"] == BASELINE])

    write_jsonl(outdir / "pairwise.jsonl", pair_rows)

    metrics= {name: policy_metrics(generated[name], seed) for name in POLICIES}
    paired= {name: paired_metrics(generated[name], generated[BASELINE], seed) for name in POLICIES if name != BASELINE}

    summary= []

    for name in POLICIES:
        row= {"dataset": dataset, "policy": name}

        for key, value in metrics[name].items():
            if key.endswith("_ci95"):
                row[key[:-5] + "_ci_low"]= value[0]
                row[key[:-5] + "_ci_high"]= value[1]
            else:
                row[key]= value

        key= f"{name}_vs_{BASELINE}"

        if key in pairwise:
            row["win_rate_vs_sft"]= pairwise[key]["win_rate"]
            row["win_rate_vs_sft_ci_low"]= pairwise[key]["win_rate_ci95"][0]
            row["win_rate_vs_sft_ci_high"]= pairwise[key]["win_rate_ci95"][1]
            row["tie_rate_vs_sft"]= pairwise[key]["tie_rate"]
            row["verifier_judge_agreement_vs_sft"]= agreement[key]["agreement"]
            row["verifier_judge_pairs_vs_sft"]= agreement[key]["n_pairs_exactly_one_correct"]

        summary.append(row)

        print(
            f"{name}: "
            f"exact_accuracy={metrics[name]['exact_accuracy']:.4f}, "
            f"format_compliance={metrics[name]['format_compliance']:.4f}, "
            f"length={metrics[name]['response_tokens_mean']:.1f} +- {metrics[name]['response_tokens_std']:.1f}"
        )

    pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False)

    gpu_name= "cpu"

    if torch.cuda.is_available():
        gpu_name= torch.cuda.get_device_name(0)

    save_json(outdir / "metrics.json", {
        "git_commit": git_commit(),
        "dataset": dataset,
        "dataset_file": dataset_path(cfg, dataset),
        "n_problems": len(rows),
        "seed": seed,
        "base_model": cfg["base_model"],
        "judge_model": cfg["ai_judge_model"],
        "decoding": "greedy (do_sample=False), one response per problem",
        "math_max_new_tokens": int(cfg["math_max_new_tokens"]),
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "prompts_over_max_prompt_length": n_long,
        "batch_size": batch_size,
        "gpu_name": gpu_name,
        "dtype": cfg["dtype"],
        "smoke": smoke,
        "definitions": {
            "exact_accuracy": "supplied verifier: the last '#### <number>' in the response equals the gold final answer",
            "format_compliance": "a '#### <number>' final answer could be parsed from the response",
            "win_rate": "supplied pairwise judge, a's response vs b's response to the same problem: win 1, tie 0.5, loss 0",
            "verifier_judge_agreement": "over pairs where exactly one response is verifier-correct, the fraction where the judge prefers the correct one; judge ties on those pairs are counted separately",
            "ci95": "bootstrap over problems, paired bootstrap for differences",
        },
        "policies": metrics,
        "paired_vs_sft": paired,
        "pairwise": pairwise,
        "verifier_judge_agreement": agreement,
        #this session only: cached judge answers and already saved responses cost nothing
        "generation_seconds_this_session": generation_seconds,
        "judge_seconds_this_session": judge_seconds,
        "judge_comparisons": len(pair_rows),
    })

    if dataset == "transfer":
        sheet_path= outdir / "failure_types_sheet.csv"
        labelled= False

        if sheet_path.exists():
            existing= pd.read_csv(sheet_path, dtype=str).fillna("")
            labelled= (existing["failure_type"].str.strip() != "").any()

        if labelled:
            #never overwrite labels that were already typed in
            print(f"{sheet_path} already has failure types in it and was left untouched")
        else:
            failure_sheet(generated, seed).to_csv(sheet_path, index=False)

            print("wrote the failure-type sheet to label:", sheet_path)
            print("fill the failure_type column with one of:", ", ".join(FAILURE_TYPES))

    print(f"saved math evaluation to {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    run_math_evaluation(args.config, args.dataset, args.batch_size, args.smoke, args.max_examples, args.allow_cpu)


if __name__ == "__main__":
    main()
