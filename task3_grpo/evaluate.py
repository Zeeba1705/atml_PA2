from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.metrics import BOOTSTRAP_RESAMPLES
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from task1_dpo.train import check_setup, git_commit, model_dtype
from task2_ppo.evaluate import generate_and_score, summarize_generations
from task3_grpo.continue_train import adjust_config, run_grpo

SMOKE_EXAMPLES= 3
#the held-out protocol is the task 2 one, so its generation cap and reward-model input length are read from there
PROTOCOL_CONFIG= "configs/ppo.yaml"
PROTOCOL_KEYS= ["eval_max_response_length", "reward_max_length"]


def load_evaluation_bundle(config_path: str, adapter: str | None, smoke: bool = False, max_examples: int | None = None, allow_cpu: bool = False):
    cfg = adjust_config(load_yaml(config_path), smoke)

    protocol= load_yaml(PROTOCOL_CONFIG)

    for key in PROTOCOL_KEYS:
        cfg[key]= protocol[key]

    if smoke:
        cfg["eval_max_response_length"]= cfg["max_completion_length"]

        if max_examples is None:
            max_examples= SMOKE_EXAMPLES

    dataset_path= cfg["paths"]["rl_prompt_eval"]

    if smoke and not repo_path(dataset_path).exists():
        #no course assets on this machine, the tracked word-limit prompts stand in
        dataset_path= cfg["paths"]["word_limit_prompts"]

    check_setup([dataset_path], smoke, allow_cpu)

    if adapter is not None and not (repo_path(adapter) / "adapter_config.json").exists():
        raise SystemExit(f"No adapter at {adapter}. Train that run first.")

    rows= read_jsonl(dataset_path)

    if max_examples is not None:
        rows= rows[: int(max_examples)]

    return {
        "cfg": cfg,
        "rows": rows,
        "dataset_path": str(dataset_path),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        #no adapter= the untouched base policy
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def run_evaluation(config_path: str, adapter: str | None, name: str, smoke: bool = False, max_examples: int | None = None, gen_batch_size: int = 4, allow_cpu: bool = False):
    if max_examples is not None and not smoke and not name.startswith("quicktest"):
        raise SystemExit(
            "--max-examples gives partial metrics. "
            "Use a --name starting with 'quicktest' so they cannot overwrite a real run's results."
        )

    timer= wall_timer()

    bundle= load_evaluation_bundle(
        config_path,
        adapter,
        smoke,
        max_examples,
        allow_cpu
    )

    cfg= bundle["cfg"]
    rows= bundle["rows"]
    model= bundle["policy"]

    seed= int(cfg["seed"])

    device= next(model.parameters()).device
    print("Using device:", device)

    results_dir= Path(cfg["results_dir"]) / name

    gpu_name= "cpu"

    if torch.cuda.is_available():
        gpu_name= torch.cuda.get_device_name(0)

    #same seed, prompts, order, batch size and decoding for every policy that is compared
    set_seed(seed)

    #the task 2 function, so both tasks score held-out responses in exactly the same way
    records= generate_and_score(
        model,
        bundle["tokenizer"],
        bundle["reward"],
        rows,
        cfg,
        gen_batch_size
    )

    metrics= {
        "git_commit": git_commit(),
        "name": name,
        "adapter": adapter,
        "seed": seed,
        "base_model": cfg["base_model"],
        "dataset": bundle["dataset_path"],
        "n_examples": len(rows),
        "max_examples": max_examples,
        "gen_batch_size": gen_batch_size,
        "eval_max_response_length": int(cfg["eval_max_response_length"]),
        "generation": cfg["generation"],
        "gpu_name": gpu_name,
        "dtype": model_dtype(model),
        "smoke": smoke,
        "bootstrap": f"95% percentile interval, {BOOTSTRAP_RESAMPLES} resamples of prompts, seed {seed}",
        "heldout": summarize_generations(records, seed),
    }

    if torch.cuda.is_available():
        metrics["peak_vram_bytes"]= torch.cuda.max_memory_allocated()

    metrics["wall_clock_sec"]= timer()

    write_jsonl(results_dir / "generations.jsonl", records)
    save_json(results_dir / "eval_metrics.json", metrics)

    heldout= metrics["heldout"]

    print(
        f"{name}: "
        f"reward={heldout['reward_mean']:.4f}, "
        f"kl_token={heldout['kl_token_mean']:.6f}, "
        f"entropy={heldout['entropy']:.4f}, "
        f"length={heldout['length_mean']:.1f} +- {heldout['length_std']:.1f}, "
        f"no_eos={heldout['no_eos_fraction']:.2f}"
    )
    print(f"saved eval metrics to {results_dir}")

    return metrics


def train_and_evaluate_fork(config_path: str, run_name: str, loss_type: str = "grpo", smoke: bool = False, allow_cpu: bool = False, gen_batch_size: int = 4):
    #one short fork from the supplied midpoint, then the common held-out protocol
    #whatever is already finished is skipped, so a study can be rerun after a dead session
    cfg= adjust_config(load_yaml(config_path), smoke)

    results_dir= Path(cfg["results_dir"]) / run_name
    adapter= Path(cfg["output"]).parent / run_name

    #same midpoint, prompt sequence, seed and update budget for every fork
    run_grpo(
        config_path,
        updates=int(cfg["fork_updates"]),
        loss_type=loss_type,
        run_name=run_name,
        smoke=smoke,
        resume=True,
        allow_cpu=allow_cpu
    )
    clear_gpu()

    eval_path= results_dir / "eval_metrics.json"

    if repo_path(eval_path).exists():
        print(f"evaluation of '{run_name}' is already saved")
    else:
        run_evaluation(
            config_path,
            str(adapter),
            run_name,
            smoke,
            None,
            gen_batch_size,
            allow_cpu
        )
        clear_gpu()

    log_rows= read_jsonl(results_dir / "train_log.jsonl")
    rollout_rows= read_jsonl(results_dir / "rollouts.jsonl")
    train= load_json(results_dir / "train_metrics.json")
    heldout= load_json(eval_path)["heldout"]

    return log_rows, rollout_rows, train, heldout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--gen-batch-size", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name, args.smoke, args.max_examples, args.gen_batch_size, args.allow_cpu)


if __name__ == "__main__":
    main()
