from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.metrics import BOOTSTRAP_RESAMPLES, bootstrap_ci, word_count
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.evaluate import length_stats
from task1_dpo.train import check_setup, git_commit, model_dtype
from task2_ppo.continue_train import adjust_config

SMOKE_EXAMPLES= 3


def load_evaluation_bundle(config_path: str, adapter: str | None, smoke: bool = False, max_examples: int | None = None, allow_cpu: bool = False):
    cfg = adjust_config(load_yaml(config_path), smoke)

    if smoke and max_examples is None:
        max_examples= SMOKE_EXAMPLES

    dataset_path= cfg["paths"]["rl_prompt_eval"]

    if smoke and not repo_path(dataset_path).exists():
        #no course assets on this machine, the tracked word-limit prompts stand in
        dataset_path= cfg["paths"]["word_limit_prompts"]

    check_setup([dataset_path], smoke, allow_cpu)

    if adapter is not None and not (repo_path(adapter) / "adapter_config.json").exists():
        raise SystemExit(
            f"No adapter at {adapter}. Train that run first, "
            "and on Colab check that Drive is mounted and linked (section 4)."
        )

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


def generate_and_score(model, tokenizer, reward, rows, cfg, gen_batch_size):
    #samples one response per held-out prompt, then scores it: reward model, token log-probs, entropy, length
    rm_model, rm_tokenizer = reward
    gen= cfg["generation"]

    records= []

    for start in range(0, len(rows), gen_batch_size):
        batch_rows= rows[start:start + gen_batch_size]
        prompts= [prompt_messages(row) for row in batch_rows]

        out= batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=int(cfg["eval_max_response_length"]),
            temperature=float(gen["temperature"]),
            top_p=float(gen["top_p"]),
            do_sample=bool(gen["do_sample"])
        )

        mask= out["response_mask"].float()

        with torch.no_grad():
            pol_logp, pol_logits = response_token_logprobs(
                model,
                out["sequences"],
                out["attention_mask"],
                out["prompt_width"],
                out["response_ids"]
            )

            #exact token entropy, one row at a time so the full-vocabulary tensors stay small
            entropy_sums= []

            for k in range(len(batch_rows)):
                token_logp= F.log_softmax(pol_logits[k].float(), dim=-1)
                token_entropy= -(token_logp.exp() * token_logp).sum(-1)

                entropy_sums.append((token_entropy * mask[k]).sum().item())

            del pol_logits, token_logp

            #reference= the same model with the lora adapter switched off
            with reference_mode(model):
                ref_logp, _ = response_token_logprobs(
                    model,
                    out["sequences"],
                    out["attention_mask"],
                    out["prompt_width"],
                    out["response_ids"]
                )

        pol_logp= pol_logp.float()
        ref_logp= ref_logp.float()

        kl_sums= ((pol_logp - ref_logp) * mask).sum(-1).tolist()
        logp_sums= (pol_logp * mask).sum(-1).tolist()

        rewards= score_reward_pairs(
            rm_model,
            rm_tokenizer,
            prompts,
            out["responses"],
            max_length=int(cfg["reward_max_length"])
        ).tolist()

        for k in range(len(batch_rows)):
            row= batch_rows[k]
            response= out["responses"][k]

            records.append({
                "prompt_id": row.get("prompt_id", row.get("source_index", start + k)),
                "row_index": start + k,
                "prompt": prompts[k][-1]["content"],
                "response": response,
                "n_tokens": out["response_lengths"][k],
                "n_words": word_count(response),
                "reward": rewards[k],
                "has_eos": out["terminated_with_eos"][k],
                "truncated": out["truncated"][k],
                #sums over the response tokens, so token-level means can be rebuilt for any subset of prompts
                "kl_sum": kl_sums[k],
                "logp_sum": logp_sums[k],
                "entropy_exact_sum": entropy_sums[k],
            })

        print(f"generated {min(start + gen_batch_size, len(rows))}/{len(rows)}")

    return records


def summarize_generations(records, seed):
    #every interval resamples prompts, token-level means are resampled as sum over tokens / number of tokens
    n_tokens= [r["n_tokens"] for r in records]
    rewards= [r["reward"] for r in records]
    kl_sums= [r["kl_sum"] for r in records]
    neg_logp_sums= [-r["logp_sum"] for r in records]
    entropy_sums= [r["entropy_exact_sum"] for r in records]
    no_eos= [0.0 if r["has_eos"] else 1.0 for r in records]

    total_tokens= max(sum(n_tokens), 1)

    out= length_stats(n_tokens)

    out["n_responses"]= len(records)
    out["generated_tokens"]= sum(n_tokens)
    out["length_mean_ci95"]= bootstrap_ci(n_tokens, None, seed)

    out["reward_mean"]= float(np.mean(rewards))
    out["reward_std"]= float(np.std(rewards))
    out["reward_mean_ci95"]= bootstrap_ci(rewards, None, seed)

    #released estimator: mean over all response tokens of log pi_policy - log pi_ref
    out["kl_token_mean"]= sum(kl_sums) / total_tokens
    out["kl_token_mean_ci95"]= bootstrap_ci(kl_sums, n_tokens, seed)

    #same differences summed per response, then averaged over responses
    out["kl_sequence_mean"]= float(np.mean(kl_sums))
    out["kl_sequence_mean_ci95"]= bootstrap_ci(kl_sums, None, seed)

    #released sampled estimate: minus the mean log-prob of the sampled tokens
    out["entropy"]= sum(neg_logp_sums) / total_tokens
    out["entropy_ci95"]= bootstrap_ci(neg_logp_sums, n_tokens, seed)

    out["entropy_exact"]= sum(entropy_sums) / total_tokens
    out["entropy_exact_ci95"]= bootstrap_ci(entropy_sums, n_tokens, seed)

    out["no_eos_fraction"]= float(np.mean(no_eos))
    out["no_eos_fraction_ci95"]= bootstrap_ci(no_eos, None, seed)

    out["truncated_fraction"]= float(np.mean([r["truncated"] for r in records]))
    out["words_mean"]= float(np.mean([r["n_words"] for r in records]))

    return out


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
        f"kl_token={heldout['kl_token_mean']:.4f}, "
        f"entropy={heldout['entropy']:.4f}, "
        f"length={heldout['length_mean']:.1f} +- {heldout['length_std']:.1f}, "
        f"no_eos={heldout['no_eos_fraction']:.2f}"
    )
    print(f"saved eval metrics to {results_dir}")

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
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
