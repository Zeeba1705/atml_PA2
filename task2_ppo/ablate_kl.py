from __future__ import annotations

import argparse
from pathlib import Path

from common.data import load_yaml
from common.logging_utils import save_json
from task1_dpo.train import git_commit
from task2_ppo.continue_train import adjust_config
from task2_ppo.evaluate import train_and_evaluate_fork

HELDOUT_KEYS= ["reward_mean", "kl_token_mean", "kl_sequence_mean", "entropy", "entropy_exact", "length_mean", "no_eos_fraction"]
TRAIN_KEYS= ["reward", "kl", "delta_kl", "entropy", "entropy_exact", "response_length", "no_eos_fraction", "clip_fraction"]


def run_name_for(beta):
    return f"kl_{beta:.1f}"


def run_kl_forks(config_path: str, smoke: bool = False, allow_cpu: bool = False, gen_batch_size: int = 4):
    cfg= adjust_config(load_yaml(config_path), smoke)

    forks= []

    for beta in cfg["kl_values"]:
        beta= float(beta)
        run_name= run_name_for(beta)

        #only the kl coefficient changes: same midpoint, prompts, seed, epsilon and update budget
        log_rows, train, heldout = train_and_evaluate_fork(
            config_path,
            run_name,
            kl_beta=beta,
            smoke=smoke,
            allow_cpu=allow_cpu,
            gen_batch_size=gen_batch_size
        )

        fork= {
            "run_name": run_name,
            "kl_beta": beta,
            "clip_epsilon": train["clip_epsilon"],
            "updates": train["updates"],
            "generated_tokens_train": train["generated_tokens"],
            "generated_tokens_heldout": heldout["generated_tokens"],
        }

        for key in HELDOUT_KEYS:
            fork["heldout_" + key]= heldout[key]
            fork["heldout_" + key + "_ci95"]= heldout[key + "_ci95"]

        fork["heldout_length_std"]= heldout["length_std"]
        fork["heldout_length_iqr"]= heldout["length_iqr"]

        #per-update trajectories, to see which quantity moves first when the kl pressure changes
        for key in TRAIN_KEYS:
            fork["train_" + key]= [r[key] for r in log_rows]

        forks.append(fork)

        print(
            f"{run_name}: "
            f"reward={fork['heldout_reward_mean']:.4f}, "
            f"kl_token={fork['heldout_kl_token_mean']:.6f}, "
            f"entropy={fork['heldout_entropy']:.4f}, "
            f"length={fork['heldout_length_mean']:.1f}"
        )

    out= {
        "git_commit": git_commit(),
        "seed": int(cfg["seed"]),
        "fork_updates": int(cfg["fork_updates"]),
        "prompts_per_update": int(cfg["prompts_per_update"]),
        "note": "each update uses prompts_per_update prompt(s), so per-update values are noisy; all forks see the same prompt sequence",
        "budget_note": f"forks run {cfg['fork_updates']} updates from the midpoint, the standard run has {cfg['updates']}",
        "smoke": smoke,
        "forks": forks,
    }

    path= Path(cfg["results_dir"]) / "kl_ablation.json"
    save_json(path, out)

    print(f"saved kl-pressure comparison to {path}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--gen-batch-size", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    run_kl_forks(args.config, args.smoke, args.allow_cpu, args.gen_batch_size)


if __name__ == "__main__":
    main()
