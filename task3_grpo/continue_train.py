from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import sample_entropy, sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.train import SMOKE_MODEL, check_setup, git_commit, latest_checkpoint, model_dtype
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

SMOKE_UPDATES= 2
SMOKE_COMPLETION_TOKENS= 16
SMOKE_OUTPUT= "outputs/smoke/task3_grpo/standard"
SMOKE_RESULTS= "results/smoke/task3_grpo"
MAX_STEP_RETRIES= 10
RATIO_TOL= 1e-2
#a group is informative when its reward std is above this, the eps default of group_relative_advantages
INFORMATIVE_STD_TOL= 1e-6


def adjust_config(cfg, smoke):
    #shared by training and evaluation so both see the same model, folders and dtype
    if smoke:
        #tiny model, short completions and separate folders so a smoke run can never touch a real run
        cfg["base_model"]= SMOKE_MODEL
        cfg["updates"]= SMOKE_UPDATES
        cfg["fork_updates"]= SMOKE_UPDATES
        cfg["max_completion_length"]= SMOKE_COMPLETION_TOKENS
        #at this length every completion hits the cap, masking them would leave nothing to step on
        cfg["mask_truncated_completions"]= False
        cfg["output"]= SMOKE_OUTPUT
        cfg["results_dir"]= SMOKE_RESULTS

    if not torch.cuda.is_available():
        #float16 on cpu is very slow, the gpu runs keep the config dtype
        cfg["dtype"]= "float32"

    return cfg


def prepare_grpo_continuation(config_path: str, smoke: bool = False, allow_cpu: bool = False):
    cfg = adjust_config(load_yaml(config_path), smoke)
    set_seed(int(cfg["seed"]))

    prompt_path= cfg["paths"]["rl_prompt_train"]
    policy_path= cfg["paths"]["grpo_midpoint_policy"]

    if smoke:
        #the midpoint adapter was trained on the 1.5B model, so the tiny smoke policy gets a fresh lora
        policy_path= None

        if not repo_path(prompt_path).exists():
            #no course assets on this machine, the tracked word-limit prompts stand in
            prompt_path= cfg["paths"]["word_limit_prompts"]

    check_setup([prompt_path], smoke, allow_cpu)

    if policy_path is not None and not repo_path(policy_path).exists():
        raise SystemExit(
            f"Missing checkpoint {policy_path}. "
            "Run the course asset step first (python -m scripts.download_assets)."
        )

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=policy_path,
        trainable=True,
        fresh_lora=smoke,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(prompt_path)
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "prompt_path": str(prompt_path),
        "policy_path": policy_path,
        "optimizer": optimizer,
    }


def sequence_logprobs(policy, rollout, k):
    #one completion on its own, cut at its last valid token, so the old and the new pass see the same tensors
    width= rollout["prompt_width"]
    n= rollout["lengths"][k]

    return response_token_logprobs(
        policy,
        rollout["sequences"][k:k + 1, :width + n],
        rollout["attention_mask"][k:k + 1, :width + n],
        width,
        rollout["response_ids"][k:k + 1, :n]
    )


def sequence_loss(new_logp, old_logp, adv, mask, ref_logp, eps, beta, loss_type, max_completion_length, n_sequences, total_tokens):
    #this completion's part of the released batch loss, so the batch can go through the model one row at a time
    #beta= 0 leaves only the policy term, which the batch loss averages over sequences
    policy_loss, diag = grpo_policy_loss(
        new_logp,
        old_logp,
        adv,
        mask,
        ref_logp,
        eps,
        0.0,
        loss_type,
        max_completion_length
    )

    #a zero advantage leaves only the kl term, which the batch loss averages over all valid tokens
    kl, _ = grpo_policy_loss(
        new_logp,
        old_logp,
        torch.zeros_like(adv),
        mask,
        ref_logp,
        eps,
        1.0,
        loss_type,
        max_completion_length
    )

    token_share= mask.sum() / total_tokens

    loss= policy_loss / n_sequences + beta * token_share * kl

    return loss, policy_loss / n_sequences, token_share * kl, diag["clip_fraction"]


def collect_rollout(policy, tokenizer, reward_model, reward_tokenizer, prompts, cfg):
    #K completions per prompt with everything that stays fixed during the update
    gen= cfg["generation"]
    num_generations= int(cfg["num_generations"])

    #each prompt repeated K times in a row, so completions of one prompt are neighbours
    group_prompts= [p for p in prompts for _ in range(num_generations)]
    group_ids= [g for g in range(len(prompts)) for _ in range(num_generations)]

    out= batch_generate(
        policy,
        tokenizer,
        group_prompts,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg["max_completion_length"]),
        temperature=float(gen["temperature"]),
        top_p=float(gen["top_p"]),
        do_sample=bool(gen["do_sample"])
    )

    #generate runs in inference mode, those tensors cant go through a forward pass that needs gradients
    rollout= {
        "sequences": out["sequences"].clone(),
        "attention_mask": out["attention_mask"].clone(),
        "prompt_width": out["prompt_width"],
        "response_ids": out["response_ids"].clone(),
        "response_mask": out["response_mask"].clone().float(),
        "responses": out["responses"],
        "lengths": out["response_lengths"],
        "truncated": out["truncated"],
        "has_eos": out["terminated_with_eos"],
        "prompts": group_prompts,
        "group_ids": group_ids,
    }

    device= rollout["response_mask"].device
    n_rows= len(group_prompts)

    old_logps= []
    ref_logps= []
    entropy_sum= 0.0

    with torch.no_grad():
        for k in range(n_rows):
            old_logp, old_logits = sequence_logprobs(policy, rollout, k)

            #exact token entropy of the rollout policy, next to the released sampled estimate
            token_logp= F.log_softmax(old_logits.float(), dim=-1)
            entropy_sum += -(token_logp.exp() * token_logp).sum().item()

            del old_logits, token_logp

            #reference= the same model with the lora adapter switched off
            with reference_mode(policy):
                ref_logp, _ = sequence_logprobs(policy, rollout, k)

            old_logps.append(old_logp[0].float())
            ref_logps.append(ref_logp[0].float())

    #padded back to one tensor per batch for the released metric helpers
    width= rollout["response_ids"].shape[1]
    old_all= torch.zeros(n_rows, width, device=device)
    ref_all= torch.zeros(n_rows, width, device=device)

    for k in range(n_rows):
        old_all[k, :len(old_logps[k])]= old_logps[k]
        ref_all[k, :len(ref_logps[k])]= ref_logps[k]

    rewards= score_reward_pairs(
        reward_model,
        reward_tokenizer,
        group_prompts,
        rollout["responses"]
    ).to(device)

    group_tensor= torch.tensor(group_ids, device=device)

    #every completion counts in its group's mean and std, truncated ones are only kept out of the loss
    advantages= group_relative_advantages(rewards, group_tensor)

    train_mask= rollout["response_mask"]

    if bool(cfg["mask_truncated_completions"]):
        train_mask= mask_truncated_sequences(train_mask, rollout["truncated"])

    group_stds= []

    for g in range(len(prompts)):
        group_stds.append(rewards[group_tensor == g].std(unbiased=False).item())

    rollout["old_logp"]= old_logps
    rollout["ref_logp"]= ref_logps
    rollout["rewards"]= rewards
    rollout["advantages"]= advantages
    rollout["train_mask"]= train_mask
    rollout["group_stds"]= group_stds
    rollout["kl"]= sampled_kl(old_all, ref_all, rollout["response_mask"]).item()
    rollout["entropy_sampled"]= sample_entropy(old_all, rollout["response_mask"]).item()
    rollout["entropy_exact"]= entropy_sum / max(sum(rollout["lengths"]), 1)

    return rollout


def grpo_train(policy, rollout, optimizer, scaler, clip_epsilon, kl_beta, loss_type, max_completion_length, max_grad_norm):
    #one optimisation step on the fixed rollout, redone at a lower scale if float16 gradients overflow
    #completions go through the model one at a time and their gradients add up to the released batch loss
    params= trainable_parameters(policy)

    n_rows= len(rollout["lengths"])
    train_mask= rollout["train_mask"]
    total_tokens= train_mask.sum().clamp_min(1.0)

    retries= 0

    while True:
        optimizer.zero_grad()

        tot_loss= 0.0
        policy_term= 0.0
        kl_term= 0.0
        clipped_tokens= 0.0
        max_ratio_dev= 0.0

        for k in range(n_rows):
            n= rollout["lengths"][k]
            mask= train_mask[k:k + 1, :n]

            if mask.sum().item() == 0:
                #a masked completion has zero loss and zero gradient
                continue

            new_logp, _ = sequence_logprobs(policy, rollout, k)
            new_logp= new_logp.float()

            loss, policy_part, kl_part, clip_fraction = sequence_loss(
                new_logp,
                rollout["old_logp"][k][None, :],
                rollout["advantages"][k:k + 1],
                mask,
                rollout["ref_logp"][k][None, :],
                clip_epsilon,
                kl_beta,
                loss_type,
                max_completion_length,
                n_rows,
                total_tokens
            )

            if not torch.isfinite(loss):
                raise RuntimeError("non-finite grpo loss")

            scaler.scale(loss).backward()

            ratio_dev= (torch.exp(new_logp.detach() - rollout["old_logp"][k][None, :]) - 1.0).abs()

            tot_loss += loss.item()
            policy_term += policy_part.item()
            kl_term += kl_part.item()
            clipped_tokens += clip_fraction.item() * n
            max_ratio_dev= max(max_ratio_dev, ratio_dev.max().item())

        #back to the true gradient scale before clipping
        scaler.unscale_(optimizer)
        grad_norm= torch.nn.utils.clip_grad_norm_(params, max_grad_norm).item()

        if math.isfinite(grad_norm):
            break

        #inf/nan gradients= the scaled float16 backward overflowed
        if not scaler.is_enabled():
            raise RuntimeError("non-finite gradient")

        if retries >= MAX_STEP_RETRIES:
            raise RuntimeError(f"still non-finite gradients after {retries} retries")

        #update() halves the scale because unscale_ saw the inf/nan, then the same step is redone
        scaler.update()

        retries += 1

        print(f"non-finite gradient, retry {retries} with scale {scaler.get_scale():.0f}")

    scaler.step(optimizer)
    scaler.update()

    return (
        tot_loss,
        policy_term,
        kl_term,
        clipped_tokens / total_tokens.item(),
        grad_norm,
        max_ratio_dev,
        retries
    )


def save_checkpoint(policy, optimizer, scaler, ckpt_dir, state):
    os.makedirs(ckpt_dir, exist_ok=True)

    policy.save_pretrained(str(ckpt_dir))

    state= dict(state)
    state["optimizer"]= optimizer.state_dict()
    state["scaler"]= scaler.state_dict()

    #written last and renamed into place, so its presence marks a complete checkpoint
    tmp_path= ckpt_dir / "trainer_state.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, ckpt_dir / "trainer_state.pt")


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard", smoke: bool = False, resume: bool = False, allow_cpu: bool = False):
    if run_name == "standard" and not smoke and (updates is not None or loss_type != "grpo"):
        raise SystemExit(
            "run name 'standard' is the 20-update canonical GRPO run reused by Task 4. "
            "Use another --run-name together with --updates or --loss-type dr_grpo."
        )

    bundle = prepare_grpo_continuation(config_path, smoke, allow_cpu)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)

    if output:
        out = repo_path(output)
    elif run_name == "standard":
        out = repo_path(cfg["output"])
    else:
        #the released default sent every run to the standard folder, which would overwrite it
        out = repo_path(Path(cfg["output"]).parent / run_name)

    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer= bundle["tokenizer"]
    policy= bundle["policy"]
    reward_model= bundle["reward_model"]
    reward_tokenizer= bundle["reward_tokenizer"]
    prompt_rows= bundle["prompt_rows"]
    optimizer= bundle["optimizer"]

    seed= int(cfg["seed"])
    total_updates= int(cfg["updates"])
    prompts_per_update= int(cfg["prompts_per_update"])
    policy_epochs= int(cfg["policy_epochs"])
    num_generations= int(cfg["num_generations"])
    clip_epsilon= float(cfg["clip_epsilon"])
    kl_beta= float(cfg["kl_beta"])
    max_completion_length= int(cfg["max_completion_length"])
    max_grad_norm= float(cfg["max_grad_norm"])

    device= next(policy.parameters()).device
    print("Using device:", device)

    results_dir= Path(cfg["results_dir"]) / run_name
    log_path= results_dir / "train_log.jsonl"
    rollouts_path= results_dir / "rollouts.jsonl"
    metrics_path= results_dir / "train_metrics.json"
    ckpt_root= out / "checkpoints"

    finished= (out / "adapter_config.json").exists() and repo_path(metrics_path).exists()
    ckpt_dir= latest_checkpoint(ckpt_root)

    if finished and resume:
        print(f"run '{run_name}' is already finished, adapter at {out}")
        return

    if (finished or ckpt_dir is not None) and not resume:
        raise SystemExit(
            f"run '{run_name}' already has saved state in {out}. "
            "Pass --resume to continue it, or delete that folder to start again."
        )

    if total_updates * prompts_per_update > len(prompt_rows):
        raise SystemExit(
            f"{total_updates} updates x {prompts_per_update} prompts need more than "
            f"the {len(prompt_rows)} prompts in {bundle['prompt_path']}"
        )

    #fixed prompt order from the seed, so every fork and every resumed run sees the same prompt sequence
    g= torch.Generator()
    g.manual_seed(seed)
    order= torch.randperm(len(prompt_rows), generator=g).tolist()

    gpu_name= "cpu"

    if torch.cuda.is_available():
        gpu_name= torch.cuda.get_device_name(0)

    info= {
        "git_commit": git_commit(),
        "run_name": run_name,
        "seed": seed,
        "loss_type": loss_type,
        "clip_epsilon": clip_epsilon,
        "kl_beta": kl_beta,
        "updates": total_updates,
        "prompts_per_update": prompts_per_update,
        "policy_epochs": policy_epochs,
        "num_generations": num_generations,
        "max_completion_length": max_completion_length,
        "mask_truncated_completions": bool(cfg["mask_truncated_completions"]),
        "informative_std_tolerance": INFORMATIVE_STD_TOL,
        "base_model": cfg["base_model"],
        "policy_init": bundle["policy_path"],
        "dataset": bundle["prompt_path"],
        "n_prompts_in_pool": len(prompt_rows),
        "gpu_name": gpu_name,
        "dtype": model_dtype(policy),
        "smoke": smoke,
    }

    save_json(results_dir / "config.json", {"config": cfg, "run": info})

    #float16 gradients can underflow to zero, the scaler multiplies the loss up before backward
    scaler= torch.amp.GradScaler(
        device.type,
        enabled=torch.cuda.is_available() and model_dtype(policy) == "float16"
    )

    #what must match for a checkpoint to belong to this run
    run_key= {
        "loss_type": loss_type,
        "clip_epsilon": clip_epsilon,
        "kl_beta": kl_beta,
        "updates": total_updates,
        "prompts_per_update": prompts_per_update,
        "policy_epochs": policy_epochs,
        "num_generations": num_generations,
        "seed": seed,
        "base_model": cfg["base_model"],
        "dataset": bundle["prompt_path"],
        "max_completion_length": max_completion_length,
    }

    start_update= 0
    prev_elapsed= 0.0
    prev_peak= 0
    n_retries= 0
    n_no_gradient= 0
    generated_tokens= 0

    if ckpt_dir is not None:
        state= torch.load(ckpt_dir / "trainer_state.pt", map_location=device)

        if state["run_key"] != run_key:
            raise SystemExit(
                f"checkpoint {ckpt_dir} was made with different settings:\n"
                f"  checkpoint: {state['run_key']}\n"
                f"  this run:   {run_key}"
            )

        load_result= set_peft_model_state_dict(
            policy,
            load_file(str(ckpt_dir / "adapter_model.safetensors"))
        )

        if len(load_result.unexpected_keys) > 0:
            raise SystemExit(f"adapter in {ckpt_dir} does not match the model: {load_result.unexpected_keys[:4]}")

        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])

        start_update= state["update"]
        prev_elapsed= state["elapsed"]
        prev_peak= state["peak_vram"]
        n_retries= state["n_retries"]
        n_no_gradient= state["n_no_gradient"]
        generated_tokens= state["generated_tokens"]

        #drop lines written after this checkpoint, those updates are redone
        write_jsonl(log_path, [r for r in read_jsonl(log_path) if r["update"] <= start_update])
        write_jsonl(rollouts_path, [r for r in read_jsonl(rollouts_path) if r["update"] <= start_update])

        print(f"resumed from {ckpt_dir} (update {start_update}/{total_updates})")
    else:
        write_jsonl(log_path, [])
        write_jsonl(rollouts_path, [])

    print(
        f"run={run_name}, "
        f"loss_type={loss_type}, "
        f"updates={total_updates}, "
        f"K={num_generations}, "
        f"dtype={info['dtype']}, "
        f"grad scaler={scaler.is_enabled()}"
    )

    #dropout off for the whole run, so the rollout log-probs and the log-probs in the step are the same numbers
    policy.eval()

    timer= wall_timer()

    for update in range(start_update, total_updates):
        print(f"\n--- Update {update + 1} ---")

        idx= order[update * prompts_per_update:(update + 1) * prompts_per_update]
        prompts= [prompt_messages(prompt_rows[i]) for i in idx]

        #sampling seed depends only on the update number, so a resumed run redraws the same way
        set_seed(seed + update)

        rollout= collect_rollout(
            policy,
            tokenizer,
            reward_model,
            reward_tokenizer,
            prompts,
            cfg
        )

        n_rows= len(rollout["lengths"])
        n_masked= sum(int(rollout["train_mask"][k].sum().item() == 0) for k in range(n_rows))
        no_gradient= n_masked == n_rows

        losses= []
        policy_terms= []
        kl_terms= []
        clip_fractions= []
        grad_norms= []
        update_retries= 0

        if no_gradient:
            #every completion hit the length cap and is masked, so the loss is 0 and there is nothing to step on
            print("all completions are masked, no optimisation step for this update")

            n_no_gradient += 1
            losses.append(0.0)
            policy_terms.append(0.0)
            kl_terms.append(0.0)
            clip_fractions.append(0.0)
            grad_norms.append(0.0)

        for epoch in range(policy_epochs):
            if no_gradient:
                break

            loss, policy_term, kl_term, clip_fraction, grad_norm, max_ratio_dev, retries = grpo_train(
                policy=policy,
                rollout=rollout,
                optimizer=optimizer,
                scaler=scaler,
                clip_epsilon=clip_epsilon,
                kl_beta=kl_beta,
                loss_type=loss_type,
                max_completion_length=max_completion_length,
                max_grad_norm=max_grad_norm
            )

            if epoch == 0 and max_ratio_dev > RATIO_TOL:
                #nothing has been updated yet, so the new log-probs should equal the rollout ones
                print(f"WARNING: first-epoch ratio is off from 1 by {max_ratio_dev:.4f}")

            losses.append(loss)
            policy_terms.append(policy_term)
            kl_terms.append(kl_term)
            clip_fractions.append(clip_fraction)
            grad_norms.append(grad_norm)
            update_retries += retries

        #same tokens scored again after the update, to see how far the step moved the policy on its own rollout
        log_ratio_sum= 0.0
        outside_tokens= 0.0
        max_dev_after= 0.0

        with torch.no_grad():
            for k in range(n_rows):
                after_logp, _ = sequence_logprobs(policy, rollout, k)

                log_ratio= after_logp[0].float() - rollout["old_logp"][k]
                ratio= torch.exp(log_ratio)

                log_ratio_sum += log_ratio.sum().item()
                outside_tokens += ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float().sum().item()
                max_dev_after= max(max_dev_after, (ratio - 1.0).abs().max().item())

        for p in trainable_parameters(policy):
            if not torch.isfinite(p).all():
                raise RuntimeError(f"policy has non-finite weights after update {update + 1}")

        update_tokens= sum(rollout["lengths"])

        n_retries += update_retries
        generated_tokens += update_tokens
        elapsed= prev_elapsed + timer()

        peak_vram= prev_peak

        if torch.cuda.is_available():
            peak_vram= max(prev_peak, torch.cuda.max_memory_allocated())

        group_stds= rollout["group_stds"]

        record= {
            "update": update + 1,
            "prompt_rows": idx,
            "reward": rollout["rewards"].mean().item(),
            "kl": rollout["kl"],
            #the objective's own kl estimate (released per-token estimator), over the tokens that are in the loss
            "kl_objective": sum(kl_terms) / len(kl_terms),
            "reward_std_within_group": sum(group_stds) / len(group_stds),
            "uninformative_group_fraction": sum(s <= INFORMATIVE_STD_TOL for s in group_stds) / len(group_stds),
            "loss": sum(losses) / len(losses),
            "policy_loss": sum(policy_terms) / len(policy_terms),
            "grad_norm": sum(grad_norms) / len(grad_norms),
            "entropy": rollout["entropy_sampled"],
            "entropy_exact": rollout["entropy_exact"],
            #with one epoch per rollout the ratio is 1 inside the step, so this is 0 by construction
            "clip_fraction": clip_fractions[-1],
            "clip_fraction_after_update": outside_tokens / max(update_tokens, 1),
            "max_ratio_deviation_after_update": max_dev_after,
            #mean log ratio new/old on the rollout tokens: how far the update moved the policy
            "delta_kl": log_ratio_sum / max(update_tokens, 1),
            "response_length": update_tokens / n_rows,
            "truncated_fraction": sum(rollout["truncated"]) / n_rows,
            "masked_completions": n_masked,
            "no_gradient": no_gradient,
            "retries": update_retries,
            "scaler_scale": scaler.get_scale(),
            "generated_tokens": generated_tokens,
            "elapsed_sec": elapsed,
        }

        append_jsonl(log_path, record)

        for k in range(n_rows):
            row= prompt_rows[idx[rollout["group_ids"][k]]]
            n= rollout["lengths"][k]
            masked= rollout["train_mask"][k].sum().item() == 0

            #weight each of this completion's tokens gets in the policy term, before the 1/K over completions
            token_weight= 1.0 / max(n, 1)

            if loss_type == "dr_grpo":
                token_weight= 1.0 / max_completion_length

            advantage= rollout["advantages"][k].item()

            append_jsonl(rollouts_path, {
                "update": update + 1,
                "prompt_id": row.get("prompt_id", row.get("source_index", idx[rollout["group_ids"][k]])),
                "generation_index": k % num_generations,
                "prompt": rollout["prompts"][k][-1]["content"],
                "response": rollout["responses"][k],
                "n_tokens": n,
                "reward": rollout["rewards"][k].item(),
                "advantage": advantage,
                "has_eos": rollout["has_eos"][k],
                "truncated": rollout["truncated"][k],
                "masked": masked,
                "token_weight": token_weight,
                "abs_advantage_times_weight": abs(advantage) * token_weight,
            })

        print(
            f"reward={record['reward']:.4f}, "
            f"reward_std={record['reward_std_within_group']:.4f}, "
            f"kl={record['kl']:.6f}, "
            f"loss={record['loss']:.6f}, "
            f"entropy={record['entropy']:.4f}, "
            f"grad_norm={record['grad_norm']:.4f}"
        )
        print(
            f"length={record['response_length']:.1f}, "
            f"truncated={record['truncated_fraction']:.2f}, "
            f"masked={n_masked}/{n_rows}, "
            f"max_ratio_dev_after={record['max_ratio_deviation_after_update']:.4f}, "
            f"retries={update_retries}, "
            f"time={elapsed:.0f}s"
        )

        save_checkpoint(
            policy,
            optimizer,
            scaler,
            ckpt_root / f"step_{update + 1:05d}",
            {
                "update": update + 1,
                "elapsed": elapsed,
                "peak_vram": peak_vram,
                "n_retries": n_retries,
                "n_no_gradient": n_no_gradient,
                "generated_tokens": generated_tokens,
                "run_key": run_key,
            }
        )

    #final adapter sits at the top of the run folder, intermediate ones stay in checkpoints/
    policy.save_pretrained(str(out))

    peak_vram= prev_peak

    if torch.cuda.is_available():
        peak_vram= max(prev_peak, torch.cuda.max_memory_allocated())

    #a step with non-finite gradients is redone at a lower scale, never skipped
    info["nonfinite_gradient_policy"]= "retry"
    info["optimizer_steps"]= (total_updates - n_no_gradient) * policy_epochs
    info["updates_without_gradient"]= n_no_gradient
    info["step_retries_total"]= n_retries
    info["generated_tokens"]= generated_tokens
    info["wall_clock_sec"]= prev_elapsed + timer()
    info["peak_vram_bytes"]= peak_vram
    info["adapter"]= str(out)
    info["note"]= f"each update uses {prompts_per_update} prompt(s) with {num_generations} completions, so per-update values are noisy"

    save_json(metrics_path, info)

    print(f"saved final adapter to {out}")
    print(f"logs in {results_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.smoke, args.resume, args.allow_cpu)


if __name__ == "__main__":
    main()
