from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task1_dpo.train import adjust_config


def run_module(module, args, smoke):
    #each run is its own process: fresh adapter from the same seed, and the gpu is freed afterwards
    command= [sys.executable, "-m", module] + args

    if smoke:
        command.append("--smoke")

    #flush so this line lands before the child process output in a notebook
    print("\n--- " + " ".join(command) + " ---", flush=True)

    subprocess.run(
        command,
        cwd=str(repo_path(".")),
        check=True
    )


def evaluate_once(config_path, cfg, name, adapter, eval_set, smoke):
    #skips an evaluation whose metrics file already exists, delete the file to redo it
    filename= "eval_metrics.json"

    if eval_set == "length":
        filename= "stratified_metrics.json"

    metrics_path= Path(cfg["results_dir"]) / name / filename

    if repo_path(metrics_path).exists():
        print(f"{metrics_path} exists, skipping evaluation of {name} [{eval_set}]")
        return load_json(metrics_path)

    args= ["--config", config_path, "--name", name, "--eval-set", eval_set]

    if adapter is not None:
        args += ["--adapter", str(adapter)]

    run_module("task1_dpo.evaluate", args, smoke)

    return load_json(metrics_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = adjust_config(load_yaml(args.config), args.smoke)
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", cfg["short_ablation_examples"])

    out_root= Path(cfg["standard_output"]).parent
    summary= []

    for beta in cfg["betas"]:
        beta= float(beta)
        run_name= f"beta_{beta}"
        adapter= out_root / run_name

        #only --beta and the run name change between the three forks
        train_args= [
            "--config", args.config,
            "--run-name", run_name,
            "--beta", str(beta),
            "--resume",
        ]

        if not args.smoke:
            train_args += ["--max-examples", str(cfg["short_ablation_examples"])]

        run_module("task1_dpo.train", train_args, args.smoke)

        metrics= evaluate_once(
            args.config,
            cfg,
            run_name,
            adapter,
            "standard",
            args.smoke
        )

        summary.append({
            "run_name": run_name,
            "beta": beta,
            #short fork: fewer examples and steps than the one-epoch standard run
            "budget": "short_fork",
            "train_examples": metrics["train_examples"],
            "train_optimizer_steps": metrics["train_optimizer_steps"],
            "dpo_loss": metrics["pairs"]["dpo_loss"],
            "dpo_loss_at_config_beta": metrics["pairs"]["dpo_loss_at_config_beta"],
            "preference_accuracy": metrics["pairs"]["preference_accuracy"],
            "margin_mean": metrics["pairs"]["margin_mean"],
            "kl_token_mean": metrics["heldout_generation"]["kl_token_mean"],
            "kl_sequence_mean": metrics["heldout_generation"]["kl_sequence_mean"],
            "reward_mean": metrics["heldout_generation"]["reward_mean"],
            "length_mean": metrics["heldout_generation"]["length_mean"],
            "length_std": metrics["heldout_generation"]["length_std"],
            "length_iqr": metrics["heldout_generation"]["length_iqr"],
        })

    save_json(Path(cfg["results_dir"]) / "beta_ablation.json", summary)

    print("\n--- beta study ---")

    for row in summary:
        print(
            f"{row['run_name']}: "
            f"examples={row['train_examples']}, "
            f"steps={row['train_optimizer_steps']}, "
            f"pref_acc={row['preference_accuracy']:.4f}, "
            f"kl_token={row['kl_token_mean']:.4f}, "
            f"reward={row['reward_mean']:.4f}, "
            f"length={row['length_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
