from __future__ import annotations

import argparse
from pathlib import Path

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task1_dpo.ablate_beta import evaluate_once, run_module
from task1_dpo.evaluate import STRATA
from task1_dpo.train import adjust_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = adjust_config(load_yaml(args.config), args.smoke)
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))

    standard_adapter= Path(cfg["standard_output"])
    length_adapter= Path(cfg["length_output"])

    if not repo_path(standard_adapter / "adapter_config.json").exists():
        raise SystemExit(
            f"no standard adapter at {standard_adapter}. "
            "Run the standard DPO training first, this study compares against it."
        )

    #same config as the standard run, only the training file changes
    run_module(
        "task1_dpo.train",
        [
            "--config", args.config,
            "--run-name", "length_balanced",
            "--dataset", cfg["paths"]["dpo_length_train"],
            "--resume",
        ],
        args.smoke
    )

    #no adapter= the untrained reference, kept as a baseline for the same strata
    models= [
        ("reference", None),
        ("standard", standard_adapter),
        ("length_balanced", length_adapter),
    ]

    summary= []

    for name, adapter in models:
        #preference accuracy per stratum on the length-stratified held-out set
        stratified_metrics= evaluate_once(
            args.config,
            cfg,
            name,
            adapter,
            "length",
            args.smoke
        )

        #generated length and word-limit compliance come from the standard evaluation
        metrics= evaluate_once(
            args.config,
            cfg,
            name,
            adapter,
            "standard",
            args.smoke
        )

        row= {
            "run_name": name,
            "train_dataset": metrics["train_dataset"],
            "train_examples": metrics["train_examples"],
            "stratified_preference_accuracy": stratified_metrics["pairs"]["preference_accuracy"],
            "heldout_length_mean": metrics["heldout_generation"]["length_mean"],
            "heldout_length_std": metrics["heldout_generation"]["length_std"],
            "heldout_length_iqr": metrics["heldout_generation"]["length_iqr"],
            "word_limit_compliance": metrics["word_limit"]["compliance"],
            "word_limit_length_mean": metrics["word_limit"]["length_mean"],
            "word_limit_length_std": metrics["word_limit"]["length_std"],
            "word_limit_words_mean": metrics["word_limit"]["words_mean"],
            "word_limit_n_responses": metrics["word_limit"]["n_responses"],
        }

        for stratum in STRATA:
            row[f"{stratum}_n"]= stratified_metrics["by_stratum"][stratum]["n_pairs"]
            row[f"{stratum}_preference_accuracy"]= stratified_metrics["by_stratum"][stratum]["preference_accuracy"]
            row[f"{stratum}_raw_preference_accuracy"]= stratified_metrics["by_stratum"][stratum]["raw_preference_accuracy"]

        summary.append(row)

    save_json(Path(cfg["results_dir"]) / "length_analysis.json", summary)

    print("\n--- length study ---")

    for row in summary:
        print(
            f"{row['run_name']}: "
            f"preferred_longer={row['preferred_longer_preference_accuracy']:.4f}, "
            f"length_matched={row['length_matched_preference_accuracy']:.4f}, "
            f"rejected_longer={row['rejected_longer_preference_accuracy']:.4f}, "
            f"word_limit_compliance={row['word_limit_compliance']:.4f}, "
            f"heldout_length={row['heldout_length_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
