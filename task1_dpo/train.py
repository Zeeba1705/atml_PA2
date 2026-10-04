from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, load_json, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss

SMOKE_MODEL= "Qwen/Qwen2.5-0.5B-Instruct"
SMOKE_EXAMPLES= 8
SMOKE_GRAD_ACCUM= 2
SMOKE_ROOT= "outputs/smoke/task1_dpo"
SMOKE_RESULTS= "results/smoke/task1_dpo"


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None, smoke: bool = False):
    cfg = load_yaml(config_path)

    if smoke:
        #tiny model and separate folders so a smoke run can never touch a real run
        cfg["base_model"]= SMOKE_MODEL
        cfg["grad_accum_steps"]= SMOKE_GRAD_ACCUM
        cfg["standard_output"]= SMOKE_ROOT + "/standard"
        cfg["length_output"]= SMOKE_ROOT + "/length_balanced"
        cfg["results_dir"]= SMOKE_RESULTS

        if max_examples is None:
            max_examples= SMOKE_EXAMPLES

    if not torch.cuda.is_available():
        #float16 on cpu is very slow, the gpu runs keep the config dtype
        cfg["dtype"]= "float32"

    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
        "dataset_path": str(path),
    }


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path(".")),
            text=True
        ).strip()
    except Exception:
        return "unknown"


def model_dtype(model):
    return str(next(model.parameters()).dtype).replace("torch.", "")


def run_info(cfg, model, beta, dataset_path, n_examples):
    gpu_name= "cpu"

    if torch.cuda.is_available():
        gpu_name= torch.cuda.get_device_name(0)

    return {
        "git_commit": git_commit(),
        "seed": int(cfg["seed"]),
        "beta": beta,
        "base_model": cfg["base_model"],
        "dataset": dataset_path,
        "n_examples": n_examples,
        "gpu_name": gpu_name,
        "dtype": model_dtype(model),
    }


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def prompt_fits(tokenizer, row, max_length):
    #same condition encode_prompt_response uses before it raises
    prompt_ids= tokenizer.apply_chat_template(
        prompt_messages_from_preference(row),
        tokenize=True,
        add_generation_prompt=True
    )

    return len(prompt_ids) < max_length


def filter_long_prompts(tokenizer, rows, cfg, dataset_path):
    #training and evaluation both go through this, so no run ever sees an unfiltered row
    #it depends only on the tokenizer and max_sequence_length, so it is the same for every run
    max_length= int(cfg["max_sequence_length"])

    indices= []
    dropped= []

    for i in range(len(rows)):
        if prompt_fits(tokenizer, rows[i], max_length):
            indices.append(i)
        else:
            dropped.append({
                "row_index": i,
                "prompt_id": rows[i].get("prompt_id"),
                "source_index": rows[i].get("source_index"),
                "length_stratum": rows[i].get("length_stratum"),
            })

    record= {
        "dataset": dataset_path,
        "rows_loaded": len(rows),
        "max_sequence_length": max_length,
        "tokenizer": cfg["base_model"],
        "n_kept": len(indices),
        "n_dropped": len(dropped),
        "dropped": dropped,
    }

    if len(rows) > 0 and "length_stratum" in rows[0]:
        kept_per_stratum= {}

        for i in indices:
            stratum= rows[i]["length_stratum"]
            kept_per_stratum[stratum]= kept_per_stratum.get(stratum, 0) + 1

        record["kept_per_stratum"]= kept_per_stratum

    #one shared file, one entry per dataset file and number of rows loaded
    path= Path(cfg["results_dir"]) / "filtered_examples.json"
    records= {}

    if repo_path(path).exists():
        records= load_json(path)

    records[f"{dataset_path} (first {len(rows)} rows)"]= record
    save_json(path, records)

    print(
        f"examples: {len(rows)} loaded, "
        f"{len(dropped)} dropped (prompt >= {max_length} tokens), "
        f"{len(indices)} used"
    )

    return indices


def reference_logps(model, tokenizer, rows, indices, cfg, dataset_path, cache_dir):
    #reference= the same model with the lora adapter switched off
    #one cache file per dataset/model/dtype/length, keyed by row number in the dataset file
    max_length= int(cfg["max_sequence_length"])
    batch_size= int(cfg["batch_size"])
    device= next(model.parameters()).device

    cache_name= (
        f"{Path(dataset_path).stem}__"
        f"{cfg['base_model'].split('/')[-1]}__"
        f"{model_dtype(model)}__"
        f"len{max_length}.json"
    )
    cache_path= Path(cache_dir) / cache_name

    cache= {"chosen": {}, "rejected": {}}

    if repo_path(cache_path).exists():
        cache= load_json(cache_path)

    missing= [i for i in indices if str(i) not in cache["chosen"]]

    print(
        f"reference log-probs: {len(indices) - len(missing)} cached, "
        f"{len(missing)} to compute ({cache_path})"
    )

    if len(missing) == 0:
        return cache

    collate= make_collate(tokenizer, max_length)

    with torch.no_grad():
        with reference_mode(model):
            for start in range(0, len(missing), batch_size):
                idx= missing[start:start + batch_size]

                chosen_batch, rejected_batch = collate([rows[i] for i in idx])

                ref_chosen, _, _ = response_sequence_logprobs(model, to_device(chosen_batch, device))
                ref_rejected, _, _ = response_sequence_logprobs(model, to_device(rejected_batch, device))

                for k, i in enumerate(idx):
                    cache["chosen"][str(i)]= ref_chosen[k].item()
                    cache["rejected"][str(i)]= ref_rejected[k].item()

                #save as we go so a dead colab session keeps what was computed
                if (start // batch_size + 1) % 100 == 0:
                    save_json(cache_path, cache)
                    print(f"reference log-probs: {start + len(idx)}/{len(missing)}")

    save_json(cache_path, cache)

    return cache


def latest_checkpoint(ckpt_root):
    #a folder without trainer_state.pt is a save that was cut off, ignore it
    steps= []

    if ckpt_root.exists():
        for d in ckpt_root.iterdir():
            if d.name.startswith("step_") and (d / "trainer_state.pt").exists():
                steps.append(int(d.name.split("_")[1]))

    if len(steps) == 0:
        return None

    return ckpt_root / f"step_{max(steps):05d}"


def save_checkpoint(model, optimizer, scaler, ckpt_dir, state):
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_pretrained(str(ckpt_dir))

    state= dict(state)
    state["optimizer"]= optimizer.state_dict()
    state["scaler"]= scaler.state_dict()

    #written last and renamed into place, so its presence marks a complete checkpoint
    tmp_path= ckpt_dir / "trainer_state.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, ckpt_dir / "trainer_state.pt")


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None, smoke: bool = False, resume: bool = False, save_every: int = 10):
    if run_name == "standard" and max_examples is not None and not smoke:
        raise SystemExit(
            "run name 'standard' is the full one-epoch run reused by Task 4. "
            "Use another --run-name (e.g. quicktest) together with --max-examples."
        )

    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples, smoke)
    cfg = bundle["cfg"]

    if output_path:
        output = repo_path(output_path)
    elif run_name == "standard":
        output = repo_path(cfg["standard_output"])
    elif run_name == "length_balanced":
        output = repo_path(cfg["length_output"])
    else:
        #the released default sent every run to standard_output, which would overwrite it
        output = repo_path(Path(cfg["standard_output"]).parent / run_name)

    output.parent.mkdir(parents=True, exist_ok=True)

    rows= bundle["rows"]
    tokenizer= bundle["tokenizer"]
    model= bundle["model"]
    optimizer= bundle["optimizer"]
    beta= bundle["beta"]
    dataset_path= bundle["dataset_path"]

    seed= int(cfg["seed"])
    batch_size= int(cfg["batch_size"])
    grad_accum= int(cfg["grad_accum_steps"])
    max_length= int(cfg["max_sequence_length"])
    max_grad_norm= float(cfg["max_grad_norm"])

    device= next(model.parameters()).device
    print("Using device:", device)

    results_dir= Path(cfg["results_dir"]) / run_name
    log_path= results_dir / "train_log.jsonl"
    metrics_path= results_dir / "train_metrics.json"
    ckpt_root= output / "checkpoints"
    cache_dir= Path(cfg["standard_output"]).parent / "ref_cache"

    finished= (output / "adapter_config.json").exists() and repo_path(metrics_path).exists()
    ckpt_dir= latest_checkpoint(ckpt_root)

    if finished and resume:
        print(f"run '{run_name}' is already finished, adapter at {output}")
        return

    if (finished or ckpt_dir is not None) and not resume:
        raise SystemExit(
            f"run '{run_name}' already has saved state in {output}. "
            "Pass --resume to continue it, or delete that folder to start again."
        )

    #encode_prompt_response raises when the prompt alone fills max_sequence_length, drop those rows
    #--max-examples slices first (in prepare_dpo_run) and the filter comes after, never topped back up
    indices= filter_long_prompts(
        tokenizer,
        rows,
        cfg,
        dataset_path
    )
    n_skipped= len(rows) - len(indices)

    info= run_info(cfg, model, beta, dataset_path, len(indices))
    info["run_name"]= run_name
    info["n_examples_loaded"]= len(rows)
    info["n_skipped_long_prompt"]= n_skipped
    info["max_examples"]= max_examples
    info["smoke"]= smoke

    save_json(results_dir / "config.json", {"config": cfg, "run": info})

    ref= reference_logps(
        model,
        tokenizer,
        rows,
        indices,
        cfg,
        dataset_path,
        cache_dir
    )

    #fixed shuffle from the seed, so a resumed run sees the same batches in the same order
    g= torch.Generator()
    g.manual_seed(seed)
    perm= torch.randperm(len(indices), generator=g).tolist()

    order= [indices[p] for p in perm]
    batches= [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    total_steps= math.ceil(len(batches) / grad_accum)

    #float16 gradients can underflow to zero, the scaler multiplies the loss up before backward
    use_scaler= torch.cuda.is_available() and model_dtype(model) == "float16"
    scaler= torch.amp.GradScaler("cuda", enabled=use_scaler)

    #what must match for a checkpoint to belong to this run
    run_key= {
        "beta": beta,
        "dataset": dataset_path,
        "n_examples": len(indices),
        "seed": seed,
        "base_model": cfg["base_model"],
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
    }

    start_step= 0
    prev_elapsed= 0.0
    prev_peak= 0
    n_skipped_steps= 0

    if ckpt_dir is not None:
        state= torch.load(ckpt_dir / "trainer_state.pt", map_location=device)

        if state["run_key"] != run_key:
            raise SystemExit(
                f"checkpoint {ckpt_dir} was made with different settings:\n"
                f"  checkpoint: {state['run_key']}\n"
                f"  this run:   {run_key}"
            )

        load_result= set_peft_model_state_dict(
            model,
            load_file(str(ckpt_dir / "adapter_model.safetensors"))
        )

        if len(load_result.unexpected_keys) > 0:
            raise SystemExit(f"adapter in {ckpt_dir} does not match the model: {load_result.unexpected_keys[:4]}")

        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])

        start_step= state["step"]
        prev_elapsed= state["elapsed"]
        prev_peak= state["peak_vram"]
        n_skipped_steps= state["n_skipped_steps"]

        #drop log lines written after this checkpoint, those steps are redone
        old_log= [r for r in read_jsonl(log_path) if r["step"] <= start_step]
        write_jsonl(log_path, old_log)

        print(f"resumed from {ckpt_dir} (step {start_step}/{total_steps})")
    else:
        write_jsonl(log_path, [])

    print(
        f"run={run_name}, "
        f"beta={beta}, "
        f"batches={len(batches)}, "
        f"optimizer steps={total_steps}, "
        f"dtype={info['dtype']}, "
        f"grad scaler={use_scaler}"
    )

    collate= make_collate(tokenizer, max_length)
    params= trainable_parameters(model)
    timer= wall_timer()

    model.train()
    optimizer.zero_grad()

    step_loss= float("nan")

    for step in range(start_step, total_steps):
        #the last group of the epoch can hold fewer than grad_accum batches
        group= batches[step * grad_accum:(step + 1) * grad_accum]

        losses= []
        margins= []
        accs= []

        for idx in group:
            chosen_batch, rejected_batch = collate([rows[i] for i in idx])

            pol_chosen, _, _ = response_sequence_logprobs(model, to_device(chosen_batch, device))
            pol_rejected, _, _ = response_sequence_logprobs(model, to_device(rejected_batch, device))

            ref_chosen= torch.tensor([ref["chosen"][str(i)] for i in idx], device=device)
            ref_rejected= torch.tensor([ref["rejected"][str(i)] for i in idx], device=device)

            loss, diag = dpo_loss(
                pol_chosen,
                pol_rejected,
                ref_chosen,
                ref_rejected,
                beta
            )

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}, rows {idx}")

            scaler.scale(loss / len(group)).backward()

            losses.append(loss.item())
            margins.append(diag["logit_mean"].item() / beta)
            accs.append(diag["preference_accuracy"].item())

        #back to the true gradient scale before clipping
        scaler.unscale_(optimizer)
        grad_norm= torch.nn.utils.clip_grad_norm_(params, max_grad_norm).item()

        step_skipped= not math.isfinite(grad_norm)

        if step_skipped and not use_scaler:
            raise RuntimeError(f"non-finite gradient at step {step + 1}")

        #with the scaler on, a step with inf/nan gradients is skipped and the scale is lowered
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step_skipped:
            n_skipped_steps += 1

        step_loss= sum(losses) / len(losses)
        step_margin= sum(margins) / len(margins)
        step_acc= sum(accs) / len(accs)
        elapsed= prev_elapsed + timer()

        if step == 0 and abs(step_loss - math.log(2.0)) > 0.02:
            #lora starts at zero so policy= reference and the first loss should be log 2
            print(f"WARNING: first step loss {step_loss:.4f} is not close to log 2 (0.6931)")

        append_jsonl(log_path, {
            "step": step + 1,
            "loss": step_loss,
            "margin": step_margin,
            "reward_accuracy": step_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "step_skipped": step_skipped,
            "examples_seen": min((step + 1) * grad_accum * batch_size, len(order)),
            "elapsed_sec": elapsed,
        })

        print(
            f"step {step + 1}/{total_steps}: "
            f"loss={step_loss:.4f}, "
            f"margin={step_margin:.4f}, "
            f"acc={step_acc:.4f}, "
            f"grad_norm={grad_norm:.4f}, "
            f"skipped={step_skipped}, "
            f"time={elapsed:.0f}s"
        )

        if (step + 1) % save_every == 0 and (step + 1) < total_steps:
            peak_vram= prev_peak

            if torch.cuda.is_available():
                peak_vram= max(prev_peak, torch.cuda.max_memory_allocated())

            save_checkpoint(
                model,
                optimizer,
                scaler,
                ckpt_root / f"step_{step + 1:05d}",
                {
                    "step": step + 1,
                    "elapsed": elapsed,
                    "peak_vram": peak_vram,
                    "n_skipped_steps": n_skipped_steps,
                    "run_key": run_key,
                }
            )

            print(f"saved checkpoint at step {step + 1}")

    #final adapter sits at the top of the run folder, intermediate ones stay in checkpoints/
    model.save_pretrained(str(output))

    peak_vram= prev_peak

    if torch.cuda.is_available():
        peak_vram= max(prev_peak, torch.cuda.max_memory_allocated())

    info["optimizer_steps"]= total_steps
    info["skipped_optimizer_steps"]= n_skipped_steps
    info["final_step_train_loss"]= step_loss
    info["wall_clock_sec"]= prev_elapsed + timer()
    info["peak_vram_bytes"]= peak_vram
    info["adapter"]= str(output)

    save_json(metrics_path, info)

    print(f"saved final adapter to {output}")
    print(f"logs in {results_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples, args.smoke, args.resume, args.save_every)


if __name__ == "__main__":
    main()
