from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task1_dpo.train import SMOKE_MODEL, check_setup, git_commit, latest_checkpoint, model_dtype
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss

SMOKE_UPDATES= 2
SMOKE_RESPONSE_TOKENS= 24
SMOKE_OUTPUT= "outputs/smoke/task2_ppo/standard"
SMOKE_RESULTS= "results/smoke/task2_ppo"
MAX_STEP_RETRIES= 10
RATIO_TOL= 1e-2


def adjust_config(cfg, smoke):
    #shared by training and evaluation so both see the same model, folders and dtype
    if smoke:
        #tiny model, short responses and separate folders so a smoke run can never touch a real run
        cfg["base_model"]= SMOKE_MODEL
        cfg["updates"]= SMOKE_UPDATES
        cfg["fork_updates"]= SMOKE_UPDATES
        cfg["max_response_length"]= SMOKE_RESPONSE_TOKENS
        cfg["eval_max_response_length"]= SMOKE_RESPONSE_TOKENS
        cfg["output"]= SMOKE_OUTPUT
        cfg["results_dir"]= SMOKE_RESULTS

    if not torch.cuda.is_available():
        #float16 on cpu is very slow, the gpu runs keep the config dtype
        cfg["dtype"]= "float32"

    return cfg


def prepare_ppo_continuation(config_path: str, smoke: bool = False, allow_cpu: bool = False):
    cfg = adjust_config(load_yaml(config_path), smoke)
    set_seed(int(cfg["seed"]))

    prompt_path= cfg["paths"]["rl_prompt_train"]
    policy_path= cfg["paths"]["ppo_midpoint_policy"]
    value_path= cfg["paths"]["ppo_midpoint_value"]

    if smoke:
        #the midpoint adapter was trained on the 1.5B model, so the tiny smoke policy gets a fresh lora
        policy_path= None

        if not repo_path(prompt_path).exists():
            #no course assets on this machine, the tracked word-limit prompts stand in
            prompt_path= cfg["paths"]["word_limit_prompts"]

        if not repo_path(value_path).exists():
            #same 0.5B base the staff critic started from, with an untrained value head
            value_path= snapshot_download(
                cfg["value_model_init"],
                allow_patterns=["*.json", "*.safetensors", "*.txt"]
            )

    check_setup([prompt_path], smoke, allow_cpu)

    for path in [policy_path, value_path]:
        if path is not None and not repo_path(path).exists():
            raise SystemExit(
                f"Missing checkpoint {path}. "
                "Run the course asset step first (section 5 of colab/run.ipynb, "
                "or python -m scripts.download_assets)."
            )

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=policy_path,
        trainable=True,
        fresh_lora=smoke,
    )
    #the value head is not a lora weight, so it would stay float16 and adamw (eps 1e-8) turns a float16
    #weight into inf when the gradient is small. the 0.5B critic is cheap to keep in float32
    value_cfg= dict(cfg)
    value_cfg["dtype"]= "float32"

    value_model = load_value_model(
        value_cfg,
        value_path,
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(prompt_path)

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "prompt_path": str(prompt_path),
        "policy_path": policy_path,
        "value_path": str(value_path),
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def response_values(value_model, rollout):
    #V(s_t) is read at the position that predicts response token t, the same shift as the logits
    values= token_values(
        value_model,
        rollout["sequences"],
        rollout["attention_mask"]
    )

    values= values[:, rollout["prompt_width"] - 1:-1]

    return values[:, :rollout["response_ids"].shape[1]].float()


def trainable_state(model):
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def collect_rollout(policy, value_model, tokenizer, reward_model, reward_tokenizer, prompts, cfg, kl_beta):
    #one on-policy batch with everything that stays fixed during the ppo epochs
    gen= cfg["generation"]

    out= batch_generate(
        policy,
        tokenizer,
        prompts,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg["max_response_length"]),
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
        "mask": out["response_mask"].clone().float(),
        "responses": out["responses"],
        "lengths": out["response_lengths"],
        "truncated": out["truncated"],
        "has_eos": out["terminated_with_eos"],
    }

    mask= rollout["mask"]
    device= mask.device

    with torch.no_grad():
        old_logp, old_logits = response_token_logprobs(
            policy,
            rollout["sequences"],
            rollout["attention_mask"],
            rollout["prompt_width"],
            rollout["response_ids"]
        )

        #exact token entropy of the rollout policy, next to the released sampled estimate
        token_logp= F.log_softmax(old_logits.float(), dim=-1)
        token_entropy= -(token_logp.exp() * token_logp).sum(-1)
        rollout["entropy_exact"]= masked_mean(token_entropy, mask).item()

        del old_logits, token_logp

        #reference= the same model with the lora adapter switched off
        with reference_mode(policy):
            ref_logp, _ = response_token_logprobs(
                policy,
                rollout["sequences"],
                rollout["attention_mask"],
                rollout["prompt_width"],
                rollout["response_ids"]
            )

        old_values= response_values(value_model, rollout)

    old_logp= old_logp.float()
    ref_logp= ref_logp.float()

    reward= score_reward_pairs(
        reward_model,
        reward_tokenizer,
        prompts,
        rollout["responses"],
        max_length=int(cfg["reward_max_length"])
    ).to(device)

    #a response that never produced eos loses missing_eos_penalty from its terminal reward
    no_eos= torch.tensor([0.0 if e else 1.0 for e in rollout["has_eos"]], device=device)
    task_reward= reward - float(cfg["missing_eos_penalty"]) * no_eos

    rewards= shaped_rewards(
        task_reward,
        old_logp,
        ref_logp,
        mask,
        kl_beta
    )

    advantages, returns = compute_gae(
        rewards,
        old_values,
        mask,
        float(cfg["gamma"]),
        float(cfg["gae_lambda"])
    )

    #explained variance of the critic as it was before this update, over valid response tokens
    valid= mask.bool()
    var_returns= returns[valid].var(unbiased=False).item()
    explained_variance= float("nan")

    if var_returns > 0:
        explained_variance= 1.0 - (returns[valid] - old_values[valid]).var(unbiased=False).item() / var_returns

    rollout["old_logp"]= old_logp
    rollout["ref_logp"]= ref_logp
    rollout["old_values"]= old_values
    rollout["reward"]= reward
    rollout["task_reward"]= task_reward
    rollout["advantages"]= normalize_advantages(advantages, mask)
    rollout["returns"]= returns
    rollout["explained_variance"]= explained_variance
    rollout["kl"]= sampled_kl(old_logp, ref_logp, mask).item()
    rollout["entropy_sampled"]= sample_entropy(old_logp, mask).item()

    return rollout


def ppo_train(policy, value_model, rollout, policy_optimizer, value_optimizer, scaler, clip_epsilon, value_coef, max_grad_norm):
    #one optimisation step on the fixed rollout, redone at a lower scale if float16 gradients overflow
    policy_params= trainable_parameters(policy)
    value_params= trainable_parameters(value_model)

    retries= 0

    while True:
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()

        new_logp, _ = response_token_logprobs(
            policy,
            rollout["sequences"],
            rollout["attention_mask"],
            rollout["prompt_width"],
            rollout["response_ids"]
        )

        policy_loss, ratio, clip_fraction = ppo_policy_loss(
            new_logp.float(),
            rollout["old_logp"],
            rollout["advantages"],
            rollout["mask"],
            clip_epsilon
        )

        value_loss= value_mse_loss(
            response_values(value_model, rollout),
            rollout["returns"],
            rollout["mask"]
        )

        tot_loss= policy_loss + value_coef * value_loss

        if not torch.isfinite(tot_loss):
            raise RuntimeError("non-finite ppo loss")

        scaler.scale(tot_loss).backward()

        #back to the true gradient scale before clipping
        scaler.unscale_(policy_optimizer)
        scaler.unscale_(value_optimizer)

        policy_grad_norm= torch.nn.utils.clip_grad_norm_(policy_params, max_grad_norm).item()
        value_grad_norm= torch.nn.utils.clip_grad_norm_(value_params, max_grad_norm).item()

        if math.isfinite(policy_grad_norm) and math.isfinite(value_grad_norm):
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

    scaler.step(policy_optimizer)
    scaler.step(value_optimizer)
    scaler.update()

    max_ratio_dev= ((ratio - 1.0).abs() * rollout["mask"]).max().item()

    return (
        policy_loss.item(),
        value_loss.item(),
        clip_fraction.item(),
        policy_grad_norm,
        value_grad_norm,
        max_ratio_dev,
        retries
    )


def save_checkpoint(policy, value_model, policy_optimizer, value_optimizer, scaler, ckpt_dir, state):
    os.makedirs(ckpt_dir, exist_ok=True)

    policy.save_pretrained(str(ckpt_dir))

    state= dict(state)
    state["value_state"]= trainable_state(value_model)
    state["policy_optimizer"]= policy_optimizer.state_dict()
    state["value_optimizer"]= value_optimizer.state_dict()
    state["scaler"]= scaler.state_dict()

    #written last and renamed into place, so its presence marks a complete checkpoint
    tmp_path= ckpt_dir / "trainer_state.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, ckpt_dir / "trainer_state.pt")


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard", smoke: bool = False, resume: bool = False, allow_cpu: bool = False):
    if run_name == "standard" and not smoke and (updates is not None or clip_epsilon is not None or kl_beta is not None):
        raise SystemExit(
            "run name 'standard' is the 20-update release-config run reused by Task 4. "
            "Use another --run-name together with --updates, --clip-epsilon or --kl-beta."
        )

    bundle = prepare_ppo_continuation(config_path, smoke, allow_cpu)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)

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
    value_model= bundle["value_model"]
    reward_model= bundle["reward_model"]
    reward_tokenizer= bundle["reward_tokenizer"]
    prompt_rows= bundle["prompt_rows"]
    policy_optimizer= bundle["policy_optimizer"]
    value_optimizer= bundle["value_optimizer"]

    seed= int(cfg["seed"])
    total_updates= int(cfg["updates"])
    prompts_per_update= int(cfg["prompts_per_update"])
    ppo_epochs= int(cfg["ppo_epochs"])
    clip_epsilon= float(cfg["clip_epsilon"])
    kl_beta= float(cfg["kl_beta"])
    value_coef= float(cfg["value_coef"])
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
        "clip_epsilon": clip_epsilon,
        "kl_beta": kl_beta,
        "updates": total_updates,
        "prompts_per_update": prompts_per_update,
        "ppo_epochs": ppo_epochs,
        "base_model": cfg["base_model"],
        "policy_init": bundle["policy_path"],
        "value_init": bundle["value_path"],
        "dataset": bundle["prompt_path"],
        "n_prompts_in_pool": len(prompt_rows),
        "gpu_name": gpu_name,
        "dtype": model_dtype(policy),
        "value_dtype": model_dtype(value_model),
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
        "clip_epsilon": clip_epsilon,
        "kl_beta": kl_beta,
        "updates": total_updates,
        "prompts_per_update": prompts_per_update,
        "ppo_epochs": ppo_epochs,
        "seed": seed,
        "base_model": cfg["base_model"],
        "dataset": bundle["prompt_path"],
        "max_response_length": int(cfg["max_response_length"]),
    }

    start_update= 0
    prev_elapsed= 0.0
    prev_peak= 0
    n_retries= 0
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

        load_result= value_model.load_state_dict(state["value_state"], strict=False)

        if len(load_result.unexpected_keys) > 0:
            raise SystemExit(f"value state in {ckpt_dir} does not match the model: {load_result.unexpected_keys[:4]}")

        policy_optimizer.load_state_dict(state["policy_optimizer"])
        value_optimizer.load_state_dict(state["value_optimizer"])
        scaler.load_state_dict(state["scaler"])

        start_update= state["update"]
        prev_elapsed= state["elapsed"]
        prev_peak= state["peak_vram"]
        n_retries= state["n_retries"]
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
        f"updates={total_updates}, "
        f"clip_epsilon={clip_epsilon}, "
        f"kl_beta={kl_beta}, "
        f"dtype={info['dtype']}, "
        f"grad scaler={scaler.is_enabled()}"
    )

    #dropout off for the whole run, so the rollout log-probs and the first-epoch log-probs are the same numbers
    policy.eval()
    value_model.eval()

    timer= wall_timer()

    for update in range(start_update, total_updates):
        print(f"\n--- Update {update + 1} ---")

        idx= order[update * prompts_per_update:(update + 1) * prompts_per_update]
        prompts= [prompt_messages(prompt_rows[i]) for i in idx]

        #sampling seed depends only on the update number, so a resumed run redraws the same way
        set_seed(seed + update)

        rollout= collect_rollout(
            policy,
            value_model,
            tokenizer,
            reward_model,
            reward_tokenizer,
            prompts,
            cfg,
            kl_beta
        )

        policy_losses= []
        value_losses= []
        clip_fractions= []
        policy_grad_norms= []
        value_grad_norms= []
        update_retries= 0

        for epoch in range(ppo_epochs):
            policy_loss, value_loss, clip_fraction, policy_grad_norm, value_grad_norm, max_ratio_dev, retries = ppo_train(
                policy=policy,
                value_model=value_model,
                rollout=rollout,
                policy_optimizer=policy_optimizer,
                value_optimizer=value_optimizer,
                scaler=scaler,
                clip_epsilon=clip_epsilon,
                value_coef=value_coef,
                max_grad_norm=max_grad_norm
            )

            if epoch == 0 and max_ratio_dev > RATIO_TOL:
                #nothing has been updated yet, so the new log-probs should equal the rollout ones
                print(f"WARNING: first-epoch ratio is off from 1 by {max_ratio_dev:.4f}")

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            clip_fractions.append(clip_fraction)
            policy_grad_norms.append(policy_grad_norm)
            value_grad_norms.append(value_grad_norm)
            update_retries += retries

        for name, model in [("policy", policy), ("value model", value_model)]:
            for p in trainable_parameters(model):
                if not torch.isfinite(p).all():
                    raise RuntimeError(f"{name} has non-finite weights after update {update + 1}")

        n_retries += update_retries
        generated_tokens += sum(rollout["lengths"])
        elapsed= prev_elapsed + timer()

        peak_vram= prev_peak

        if torch.cuda.is_available():
            peak_vram= max(prev_peak, torch.cuda.max_memory_allocated())

        n_rows= len(idx)

        record= {
            "update": update + 1,
            "prompt_rows": idx,
            "reward": rollout["reward"].mean().item(),
            "reward_after_eos_penalty": rollout["task_reward"].mean().item(),
            "kl": rollout["kl"],
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "value_loss": sum(value_losses) / len(value_losses),
            "entropy": rollout["entropy_sampled"],
            "entropy_exact": rollout["entropy_exact"],
            "grad_norm": sum(policy_grad_norms) / len(policy_grad_norms),
            "value_grad_norm": sum(value_grad_norms) / len(value_grad_norms),
            #the first epoch always has ratio 1, so the last epoch is the one where clipping can act
            "clip_fraction": clip_fractions[-1],
            "response_length": sum(rollout["lengths"]) / n_rows,
            "truncated_fraction": sum(rollout["truncated"]) / n_rows,
            "no_eos_fraction": 1.0 - sum(rollout["has_eos"]) / n_rows,
            "explained_variance": rollout["explained_variance"],
            "policy_loss_epochs": policy_losses,
            "value_loss_epochs": value_losses,
            "clip_fraction_epochs": clip_fractions,
            "grad_norm_epochs": policy_grad_norms,
            "value_grad_norm_epochs": value_grad_norms,
            "retries": update_retries,
            "scaler_scale": scaler.get_scale(),
            "generated_tokens": generated_tokens,
            "elapsed_sec": elapsed,
        }

        append_jsonl(log_path, record)

        for k in range(n_rows):
            row= prompt_rows[idx[k]]

            append_jsonl(rollouts_path, {
                "update": update + 1,
                "prompt_id": row.get("prompt_id", row.get("source_index", idx[k])),
                "prompt": prompts[k][-1]["content"],
                "response": rollout["responses"][k],
                "n_tokens": rollout["lengths"][k],
                "reward": rollout["reward"][k].item(),
                "has_eos": rollout["has_eos"][k],
            })

        print(
            f"reward={record['reward']:.4f}, "
            f"kl={record['kl']:.4f}, "
            f"policy_loss={record['policy_loss']:.4f}, "
            f"value_loss={record['value_loss']:.4f}, "
            f"entropy={record['entropy']:.4f}, "
            f"clip_fraction={record['clip_fraction']:.4f}"
        )
        print(
            f"grad_norm={record['grad_norm']:.4f}, "
            f"length={record['response_length']:.1f}, "
            f"no_eos={record['no_eos_fraction']:.2f}, "
            f"explained_variance={record['explained_variance']:.4f}, "
            f"retries={update_retries}, "
            f"time={elapsed:.0f}s"
        )

        save_checkpoint(
            policy,
            value_model,
            policy_optimizer,
            value_optimizer,
            scaler,
            ckpt_root / f"step_{update + 1:05d}",
            {
                "update": update + 1,
                "elapsed": elapsed,
                "peak_vram": peak_vram,
                "n_retries": n_retries,
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
    info["optimizer_steps"]= total_updates * ppo_epochs
    info["step_retries_total"]= n_retries
    info["generated_tokens"]= generated_tokens
    info["wall_clock_sec"]= prev_elapsed + timer()
    info["peak_vram_bytes"]= peak_vram
    info["adapter"]= str(out)
    info["note"]= f"each update uses {prompts_per_update} prompt(s), so per-update values are noisy"

    save_json(metrics_path, info)

    print(f"saved final adapter to {out}")
    print(f"logs in {results_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name, args.smoke, args.resume, args.allow_cpu)


if __name__ == "__main__":
    main()
