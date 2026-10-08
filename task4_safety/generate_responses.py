from __future__ import annotations

import argparse
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.models import clear_gpu, load_policy, load_tokenizer
from task1_dpo.train import SMOKE_MODEL, check_setup, git_commit

POLICIES= ["sft", "dpo", "ppo", "grpo"]
CLASSES= ["SAFE", "UNSAFE"]
SMOKE_PER_CLASS= 4
SMOKE_NEW_TOKENS= 24
SMOKE_AUDIT_PER_CLASS= 2


def adjust_config(cfg, smoke, examples_per_class=None):
    #shared by every task 4 script so they all read and write the same folder
    cfg["task4_dir"]= str(repo_path(cfg["results_dir"]) / "task4_safety")

    if examples_per_class is not None:
        #a partial run goes to its own folder so it can never be mistaken for the full evaluation
        cfg["examples_per_class"]= int(examples_per_class)
        cfg["task4_dir"]= str(repo_path(cfg["results_dir"]) / "task4_safety_quicktest")

    if smoke:
        #tiny model for policy and judge, no adapters (they were trained on the 1.5B model), a few prompts
        cfg["base_model"]= SMOKE_MODEL
        cfg["ai_judge_model"]= SMOKE_MODEL
        cfg["safety_max_new_tokens"]= SMOKE_NEW_TOKENS
        cfg["manual_audit_per_class"]= SMOKE_AUDIT_PER_CLASS
        cfg["examples_per_class"]= SMOKE_PER_CLASS
        cfg["task4_dir"]= str(repo_path(cfg["results_dir"]) / "smoke" / "task4_safety")

        for name in POLICIES:
            cfg["policies"][name]= None

    if not torch.cuda.is_available():
        #float16 on cpu is very slow, the gpu runs keep the config dtype
        cfg["dtype"]= "float32"

    return cfg


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    df = pd.read_csv(repo_path(cfg["paths"]["xstest"]))

    n= cfg.get("examples_per_class")

    if n is not None:
        #first n prompts of each class, kept in the fixed file order
        parts= [df[df["benchmark_class"] == c].head(int(n)) for c in CLASSES]
        df= pd.concat(parts).sort_values("xstest_id")

    return df


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
        if (start // batch_size + 1) % 25 == 0:
            print(f"{policy_name}: generated {min(start + batch_size, len(df))}/{len(df)}")
    return records


def run_generation(config_path: str, policies: list | None = None, batch_size: int = 4, smoke: bool = False, examples_per_class: int | None = None, allow_cpu: bool = False):
    cfg= adjust_config(load_yaml(config_path), smoke, examples_per_class)

    check_setup([cfg["paths"]["xstest"]], smoke, allow_cpu)

    if policies is None:
        policies= POLICIES

    specs= policy_specs(cfg)

    #every adapter is checked before anything is generated, so a missing one stops the run at once
    for name in policies:
        adapter= specs[name]

        if adapter is not None and not (repo_path(adapter) / "adapter_config.json").exists():
            raise SystemExit(
                f"No adapter for '{name}' at {adapter}. "
                "Task 4 needs the final standard adapters of tasks 1-3 in outputs/."
            )

    outdir= repo_path(cfg["task4_dir"])
    n_prompts= len(load_xstest(cfg))

    print("Policies:", policies)
    print("XSTest rows:", n_prompts)

    info_path= outdir / "generation_info.json"
    info= {"policies": {}}

    if info_path.exists():
        info= load_json(info_path)

    gpu_name= "cpu"

    if torch.cuda.is_available():
        gpu_name= torch.cuda.get_device_name(0)

    for name in policies:
        path= outdir / f"generated_{name}.jsonl"

        if path.exists() and len(read_jsonl(path)) == n_prompts:
            print(f"responses of '{name}' are already saved")
            continue

        timer= wall_timer()

        #greedy decoding has no sampling, the seed is set anyway so every policy starts from the same state
        set_seed(int(cfg["seed"]))

        records= generate_for_policy(cfg, name, batch_size)
        clear_gpu()

        write_jsonl(path, records)

        info["policies"][name]= {
            "adapter": specs[name],
            "n_responses": len(records),
            "mean_response_tokens": sum(r["response_tokens"] for r in records) / len(records),
            "wall_clock_sec": timer(),
            "git_commit": git_commit(),
        }

        info["seed"]= int(cfg["seed"])
        info["base_model"]= cfg["base_model"]
        info["decoding"]= "greedy (do_sample=False), one response per prompt"
        info["safety_max_new_tokens"]= int(cfg["safety_max_new_tokens"])
        info["max_prompt_length"]= 256
        info["batch_size"]= batch_size
        info["n_prompts"]= n_prompts
        info["gpu_name"]= gpu_name
        info["dtype"]= cfg["dtype"]
        info["smoke"]= smoke

        save_json(info_path, info)

        print(f"saved {len(records)} responses of '{name}' to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", choices=POLICIES)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--examples-per-class", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    run_generation(args.config, args.policies, args.batch_size, args.smoke, args.examples_per_class, args.allow_cpu)


if __name__ == "__main__":
    main()
