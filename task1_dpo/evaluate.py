from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.data import load_yaml, prompt_messages, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.metrics import parse_word_limit, preference_accuracy, sampled_kl, word_count, word_limit_compliance
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import (
    SMOKE_EXAMPLES,
    adjust_config,
    filter_long_prompts,
    make_collate,
    reference_logps,
    run_info,
    to_device,
)

STRATA= ["preferred_longer", "length_matched", "rejected_longer"]


def load_evaluation_bundle(config_path: str, adapter: str | None, eval_set: str = "standard", smoke: bool = False, max_examples: int | None = None):
    cfg = adjust_config(load_yaml(config_path), smoke)

    if smoke and max_examples is None:
        max_examples= SMOKE_EXAMPLES

    #standard= fixed held-out pairs, length= length-stratified held-out pairs
    dataset_path= cfg["paths"]["dpo_standard_eval"]

    if eval_set == "length":
        dataset_path= cfg["paths"]["dpo_length_eval"]

    rows= read_jsonl(dataset_path)

    if max_examples is not None:
        rows= rows[: int(max_examples)]

    #the length set is only scored for preference accuracy, so it needs no reward model
    reward= None

    if eval_set == "standard":
        reward= load_reward_model(cfg)

    return {
        "cfg": cfg,
        "rows": rows,
        "dataset_path": dataset_path,
        "tokenizer": load_tokenizer(cfg["base_model"]),
        #no adapter= the untrained reference policy
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": reward,
    }


def last_user_text(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))

    return ""


def score_pairs(model, tokenizer, rows, indices, ref, cfg, is_reference):
    #sequence log-probs of chosen and rejected under the policy, same batches as the reference cache
    max_length= int(cfg["max_sequence_length"])
    batch_size= int(cfg["batch_size"])
    device= next(model.parameters()).device

    collate= make_collate(tokenizer, max_length)
    pairs= []

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx= indices[start:start + batch_size]

            if is_reference:
                #the policy is the reference, reuse the cached numbers so the margin is exactly 0
                pol_chosen= [ref["chosen"][str(i)] for i in idx]
                pol_rejected= [ref["rejected"][str(i)] for i in idx]
            else:
                chosen_batch, rejected_batch = collate([rows[i] for i in idx])

                pol_chosen, _, _ = response_sequence_logprobs(model, to_device(chosen_batch, device))
                pol_rejected, _, _ = response_sequence_logprobs(model, to_device(rejected_batch, device))

                pol_chosen= pol_chosen.tolist()
                pol_rejected= pol_rejected.tolist()

            for k, i in enumerate(idx):
                ref_chosen= ref["chosen"][str(i)]
                ref_rejected= ref["rejected"][str(i)]

                pairs.append({
                    "row_index": i,
                    "prompt_id": rows[i].get("prompt_id"),
                    "length_stratum": rows[i].get("length_stratum"),
                    "chosen_tokens": rows[i].get("chosen_tokens"),
                    "rejected_tokens": rows[i].get("rejected_tokens"),
                    "policy_chosen_logp": pol_chosen[k],
                    "policy_rejected_logp": pol_rejected[k],
                    "ref_chosen_logp": ref_chosen,
                    "ref_rejected_logp": ref_rejected,
                    "margin": (pol_chosen[k] - ref_chosen) - (pol_rejected[k] - ref_rejected),
                })

    return pairs


def summarize_pairs(pairs, beta, default_beta):
    pol_chosen= torch.tensor([p["policy_chosen_logp"] for p in pairs])
    pol_rejected= torch.tensor([p["policy_rejected_logp"] for p in pairs])
    ref_chosen= torch.tensor([p["ref_chosen_logp"] for p in pairs])
    ref_rejected= torch.tensor([p["ref_rejected_logp"] for p in pairs])

    loss, diag = dpo_loss(
        pol_chosen,
        pol_rejected,
        ref_chosen,
        ref_rejected,
        beta
    )

    #the loss value depends on beta, so also report it at the config beta for every run
    default_loss, _ = dpo_loss(
        pol_chosen,
        pol_rejected,
        ref_chosen,
        ref_rejected,
        default_beta
    )

    margins= [p["margin"] for p in pairs]

    return {
        "n_pairs": len(pairs),
        "dpo_loss": loss.item(),
        "dpo_loss_at_config_beta": default_loss.item(),
        #handout definition: fraction of pairs with margin m > 0
        "preference_accuracy": diag["preference_accuracy"].item(),
        #released helper: fraction with log pi(chosen) > log pi(rejected), no reference term
        "raw_preference_accuracy": preference_accuracy(pol_chosen, pol_rejected),
        "margin_mean": float(np.mean(margins)),
        "margin_std": float(np.std(margins)),
    }


def length_stats(lengths):
    q25, q50, q75 = np.percentile(lengths, [25, 50, 75])

    return {
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "length_q25": float(q25),
        "length_median": float(q50),
        "length_q75": float(q75),
        "length_iqr": float(q75 - q25),
    }


def generate_and_score(model, tokenizer, reward, prompts, meta, cfg, gen_batch_size):
    #samples one response per prompt, then scores it: token log-probs for kl, reward model, length
    rm_model, rm_tokenizer = reward
    max_new= int(cfg["max_generation_tokens"])
    gen= cfg["generation"]

    records= []
    pol_logps= []
    ref_logps= []
    masks= []

    for start in range(0, len(prompts), gen_batch_size):
        batch_prompts= prompts[start:start + gen_batch_size]

        out= batch_generate(
            model,
            tokenizer,
            batch_prompts,
            max_prompt_length=int(cfg["max_sequence_length"]),
            max_new_tokens=max_new,
            temperature=float(gen["temperature"]),
            top_p=float(gen["top_p"]),
            do_sample=bool(gen["do_sample"])
        )

        with torch.no_grad():
            pol_logp, _ = response_token_logprobs(
                model,
                out["sequences"],
                out["attention_mask"],
                out["prompt_width"],
                out["response_ids"]
            )

            with reference_mode(model):
                ref_logp, _ = response_token_logprobs(
                    model,
                    out["sequences"],
                    out["attention_mask"],
                    out["prompt_width"],
                    out["response_ids"]
                )

        #batches stop at different lengths, pad to the cap so they can be stacked for the kl helper
        pad= max_new - pol_logp.shape[1]

        pol_logps.append(F.pad(pol_logp.float().cpu(), (0, pad)))
        ref_logps.append(F.pad(ref_logp.float().cpu(), (0, pad)))
        masks.append(F.pad(out["response_mask"].float().cpu(), (0, pad)))

        rewards= score_reward_pairs(
            rm_model,
            rm_tokenizer,
            batch_prompts,
            out["responses"]
        ).tolist()

        for k in range(len(batch_prompts)):
            prompt_text= last_user_text(batch_prompts[k])
            response= out["responses"][k]

            record= dict(meta[start + k])
            record["prompt"]= prompt_text
            record["response"]= response
            record["n_tokens"]= out["response_lengths"][k]
            record["n_words"]= word_count(response)
            record["reward"]= rewards[k]
            record["word_limit"]= parse_word_limit(prompt_text)
            record["word_limit_ok"]= word_limit_compliance(prompt_text, response)
            record["truncated"]= out["truncated"][k]
            record["terminated_with_eos"]= out["terminated_with_eos"][k]

            records.append(record)

        print(f"generated {min(start + gen_batch_size, len(prompts))}/{len(prompts)}")

    pol_logps= torch.cat(pol_logps)
    ref_logps= torch.cat(ref_logps)
    masks= torch.cat(masks)

    #released estimator: mean over all response tokens of log pi_policy - log pi_ref
    kl_token= sampled_kl(pol_logps, ref_logps, masks).item()

    #same differences summed per response, then averaged over responses
    kl_sequence= ((pol_logps - ref_logps) * masks).sum(-1).mean().item()

    return records, kl_token, kl_sequence


def summarize_generations(records):
    out= length_stats([r["n_tokens"] for r in records])

    rewards= [r["reward"] for r in records]

    out["n_responses"]= len(records)
    out["reward_mean"]= float(np.mean(rewards))
    out["reward_std"]= float(np.std(rewards))
    out["truncated_fraction"]= float(np.mean([r["truncated"] for r in records]))
    out["words_mean"]= float(np.mean([r["n_words"] for r in records]))

    return out


def run_evaluation(config_path: str, adapter: str | None, name: str, eval_set: str = "standard", beta: float | None = None, smoke: bool = False, max_examples: int | None = None, gen_batch_size: int = 4, word_limit_samples: int = 5):
    if max_examples is not None and not smoke and not name.startswith("quicktest"):
        raise SystemExit(
            "--max-examples gives partial metrics. "
            "Use a --name starting with 'quicktest' so they cannot overwrite a real run's results."
        )

    timer= wall_timer()

    bundle= load_evaluation_bundle(
        config_path,
        adapter,
        eval_set,
        smoke,
        max_examples
    )

    cfg= bundle["cfg"]
    rows= bundle["rows"]
    tokenizer= bundle["tokenizer"]
    model= bundle["policy"]
    dataset_path= bundle["dataset_path"]

    seed= int(cfg["seed"])
    is_reference= adapter is None

    device= next(model.parameters()).device
    print("Using device:", device)

    results_dir= Path(cfg["results_dir"]) / name
    cache_dir= Path(cfg["standard_output"]).parent / "ref_cache"

    #beta only changes the loss value: use the one the run was trained with unless told otherwise
    train_metrics= {}
    beta_source= "config"

    if repo_path(results_dir / "train_metrics.json").exists():
        train_metrics= load_json(results_dir / "train_metrics.json")

    if beta is not None:
        beta_source= "argument"
    elif "beta" in train_metrics:
        beta= train_metrics["beta"]
        beta_source= "train_metrics"
    else:
        beta= float(cfg["beta"])

    #same filter as training, so every metric below uses the same pairs for every model
    indices= filter_long_prompts(
        tokenizer,
        rows,
        cfg,
        dataset_path
    )

    ref= reference_logps(
        model,
        tokenizer,
        rows,
        indices,
        cfg,
        dataset_path,
        cache_dir
    )

    pairs= score_pairs(
        model,
        tokenizer,
        rows,
        indices,
        ref,
        cfg,
        is_reference
    )

    metrics= run_info(cfg, model, beta, dataset_path, len(indices))
    metrics["run_name"]= name
    metrics["adapter"]= adapter
    metrics["eval_set"]= eval_set
    metrics["beta_source"]= beta_source
    metrics["n_examples_loaded"]= len(rows)
    metrics["smoke"]= smoke

    #training budget of this run, the short beta forks and the one-epoch runs differ here
    metrics["train_examples"]= train_metrics.get("n_examples")
    metrics["train_optimizer_steps"]= train_metrics.get("optimizer_steps")
    metrics["train_dataset"]= train_metrics.get("dataset")

    metrics["pairs"]= summarize_pairs(pairs, beta, float(cfg["beta"]))

    print(
        f"{name} [{eval_set}]: "
        f"pairs={metrics['pairs']['n_pairs']}, "
        f"dpo_loss={metrics['pairs']['dpo_loss']:.4f}, "
        f"pref_acc={metrics['pairs']['preference_accuracy']:.4f}, "
        f"margin={metrics['pairs']['margin_mean']:.4f}"
    )

    if eval_set == "length":
        by_stratum= {}

        for stratum in STRATA:
            subset= [p for p in pairs if p["length_stratum"] == stratum]
            by_stratum[stratum]= summarize_pairs(subset, beta, float(cfg["beta"]))

            print(
                f"{stratum}: "
                f"n={by_stratum[stratum]['n_pairs']}, "
                f"pref_acc={by_stratum[stratum]['preference_accuracy']:.4f}, "
                f"raw_acc={by_stratum[stratum]['raw_preference_accuracy']:.4f}"
            )

        metrics["by_stratum"]= by_stratum

        if torch.cuda.is_available():
            metrics["peak_vram_bytes"]= torch.cuda.max_memory_allocated()

        metrics["wall_clock_sec"]= timer()

        write_jsonl(results_dir / "stratified_pairs.jsonl", pairs)
        save_json(results_dir / "stratified_metrics.json", metrics)

        print(f"saved stratified metrics to {results_dir}")
        return metrics

    gen= cfg["generation"]

    metrics["decoding"]= {
        "temperature": float(gen["temperature"]),
        "top_p": float(gen["top_p"]),
        "do_sample": bool(gen["do_sample"]),
        "max_generation_tokens": int(cfg["max_generation_tokens"]),
        "gen_batch_size": gen_batch_size,
        "word_limit_samples": word_limit_samples,
    }

    #one sampled response per held-out prompt, the same prompts the pair metrics use
    prompts= [prompt_messages_from_preference(rows[i]) for i in indices]
    meta= [{"set": "heldout", "prompt_id": rows[i].get("prompt_id"), "row_index": i, "sample": 0} for i in indices]

    set_seed(seed)

    heldout_records, kl_token, kl_sequence = generate_and_score(
        model,
        tokenizer,
        bundle["reward"],
        prompts,
        meta,
        cfg,
        gen_batch_size
    )

    metrics["heldout_generation"]= summarize_generations(heldout_records)
    metrics["heldout_generation"]["kl_token_mean"]= kl_token
    metrics["heldout_generation"]["kl_sequence_mean"]= kl_sequence

    #common word-limit prompt set, every prompt sampled word_limit_samples times
    wl_rows= read_jsonl(cfg["paths"]["word_limit_prompts"])
    wl_prompts= []
    wl_meta= []

    for sample in range(word_limit_samples):
        for row in wl_rows:
            wl_prompts.append(prompt_messages(row))
            wl_meta.append({"set": "word_limit", "prompt_id": row["prompt_id"], "row_index": None, "sample": sample})

    #reseeded so these samples do not depend on how many held-out prompts came before
    set_seed(seed)

    wl_records, wl_kl_token, wl_kl_sequence = generate_and_score(
        model,
        tokenizer,
        bundle["reward"],
        wl_prompts,
        wl_meta,
        cfg,
        gen_batch_size
    )

    metrics["word_limit"]= summarize_generations(wl_records)
    metrics["word_limit"]["kl_token_mean"]= wl_kl_token
    metrics["word_limit"]["kl_sequence_mean"]= wl_kl_sequence
    metrics["word_limit"]["n_prompts"]= len(wl_rows)
    metrics["word_limit"]["compliance"]= float(np.mean([r["word_limit_ok"] for r in wl_records]))

    if torch.cuda.is_available():
        metrics["peak_vram_bytes"]= torch.cuda.max_memory_allocated()

    metrics["wall_clock_sec"]= timer()

    write_jsonl(results_dir / "pairs.jsonl", pairs)
    write_jsonl(results_dir / "generations.jsonl", heldout_records + wl_records)
    save_json(results_dir / "eval_metrics.json", metrics)

    print(
        f"{name}: "
        f"kl_token={kl_token:.4f}, "
        f"reward={metrics['heldout_generation']['reward_mean']:.4f}, "
        f"length={metrics['heldout_generation']['length_mean']:.1f} "
        f"+- {metrics['heldout_generation']['length_std']:.1f}, "
        f"word_limit_compliance={metrics['word_limit']['compliance']:.4f}"
    )
    print(f"saved eval metrics to {results_dir}")

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--eval-set", choices=["standard", "length"], default="standard")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--gen-batch-size", type=int, default=4)
    ap.add_argument("--word-limit-samples", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name, args.eval_set, args.beta, args.smoke, args.max_examples, args.gen_batch_size, args.word_limit_samples)


if __name__ == "__main__":
    main()
