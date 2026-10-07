# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

Training commands (run from the repository root; on Colab use `colab/run.ipynb`):

```bash
# unit tests for the objective
python -m pytest tests -q

# smoke: Qwen2.5-0.5B, 8 examples, writes under outputs/smoke/ and results/smoke/
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard --smoke

# quick test of the real model on 32 examples, checkpoint after every optimizer step
python -m task1_dpo.train --config configs/dpo.yaml --run-name quicktest --max-examples 32 --save-every 1 --resume

# standard DPO, one epoch, final adapter at outputs/task1_dpo/standard
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard --resume
```

Evaluation commands:

```bash
# quick test: 8 held-out pairs, one sample per word-limit prompt
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/quicktest --name quicktest --max-examples 8 --word-limit-samples 1

# standard run on the fixed held-out pairs and the word-limit prompts
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard

# untrained reference policy as a baseline (no --adapter)
python -m task1_dpo.evaluate --config configs/dpo.yaml --name reference

# preference accuracy per length stratum on the length-stratified held-out set
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard --eval-set length
```

The two studies (each trains what is missing, then evaluates; finished runs and existing metrics files are skipped):

```bash
# beta study: short forks at the config betas, summary in results/task1_dpo/beta_ablation.json
python -m task1_dpo.ablate_beta --config configs/dpo.yaml

# length study: length-balanced run plus per-stratum accuracy, summary in results/task1_dpo/length_analysis.json
python -m task1_dpo.analyze_length --config configs/dpo.yaml

# smoke versions
python -m task1_dpo.ablate_beta --config configs/dpo.yaml --smoke
python -m task1_dpo.analyze_length --config configs/dpo.yaml --smoke
```

Evaluation writes `eval_metrics.json`, `pairs.jsonl` and `generations.jsonl` (or `stratified_metrics.json` and `stratified_pairs.jsonl` for `--eval-set length`) to `results/task1_dpo/<name>/`.

Full (non-smoke) runs stop early with a message if there is no GPU, a data file is missing, or the adapter is not found; `--allow-cpu` on `train` and `evaluate` overrides the GPU check.

If an optimizer step produces non-finite gradients in float16, the gradient scale is halved and the same batches are redone, so no step is skipped; retries are logged per step in `train_log.jsonl` and totalled in `train_metrics.json`. The `standard` and `length_balanced` runs predate this and record their skipped steps instead; the first beta forks, which also skipped steps, are kept in `results/task1_dpo/_superseded/`.

`--resume` continues from the latest checkpoint in `outputs/task1_dpo/<run_name>/checkpoints/` and starts from scratch when there is none. `--save-every` is in optimizer steps (default 10). Rows whose prompt alone reaches `max_sequence_length` are dropped for every run (after the `--max-examples` slice, never topped back up); the dropped `prompt_id`s and counts are written to `results/task1_dpo/filtered_examples.json`.

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

Training commands (run from the repository root; on Colab use `colab/run.ipynb`):

```bash
# unit tests for the objective
python -m pytest tests/test_ppo.py -q

# smoke: Qwen2.5-0.5B with a fresh LoRA, 2 updates, 24 response tokens, writes under outputs/smoke/ and results/smoke/
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard --smoke

# standard continuation: 20 updates from the supplied midpoint, final adapter in outputs/task2_ppo/standard
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard --resume
```

Evaluation commands (same held-out prompts, seed, batch size and decoding for every policy):

```bash
# smoke: 3 prompts, 24 response tokens, base policy
python -m task2_ppo.evaluate --config configs/ppo.yaml --name sft --smoke

# untouched base policy, supplied midpoint, standard continuation
python -m task2_ppo.evaluate --config configs/ppo.yaml --name sft
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
```

`evaluate` samples one response per prompt in `data/rl_prompt_pool_eval.jsonl` with `eval_max_response_length` and writes `results/task2_ppo/<name>/eval_metrics.json` and `generations.jsonl`. Without `--adapter` it evaluates the base policy. Every headline metric has a 95% bootstrap interval (2000 resamples of prompts, config seed); token-level means (KL, entropy) are resampled as a sum over tokens divided by the token count. `generations.jsonl` keeps the per-prompt sums so paired comparisons between runs need no regeneration.

Clipping study:

```bash
# smoke (needs the course assets: cached rollout and prompt pools)
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --smoke

# cached batch only: clipped surrogate and clip fraction for each epsilon, no training
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --part cached

# matched 8-update forks clip_0.05, clip_0.2, clip_0.5 from the midpoint, each evaluated on the held-out prompts
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --part forks
```

`--part cached` rebuilds the 32 cached rollouts from their text, scores them under the midpoint policy (and under `outputs/task2_ppo/standard` when it exists) and writes `results/task2_ppo/clipping_cached_<name>.json`. Advantages come from the cached values and rewards and are normalised once over the batch, so only epsilon changes. `clip_fraction` is the fraction of valid response tokens whose ratio is outside `[1-eps, 1+eps]` before clipping; `binding_fraction` is the fraction where the objective actually uses the clipped term.

`--part forks` needs the finished standard run. It writes `results/task2_ppo/clipping_forks.json` with the held-out reward, KL and length of each fork and two stability statistics, fixed before any fork was run:

- `delta_kl` of an update is the sampled KL estimate after the update minus before it, on that update's own rollout tokens. The reference log-probs cancel in that difference, so `delta_kl` equals the mean change in log-prob of the sampled tokens (the mean log ratio new/old). It measures how far one update moved the policy on its own rollout; it is not a fresh estimate of the KL from the reference, because the tokens were sampled before the update.
- S1 is the population standard deviation of `delta_kl` over the fork's updates.
- S2 is the number of updates with `|delta_kl|` above a threshold of 3 x the median `|delta_kl|` of the standard 20-update run. The threshold is computed once and stored in `results/task2_ppo/stability_threshold.json`.

Each update uses one prompt, so per-update values are noisy; all forks see the same prompt sequence. Forks that are already trained or evaluated are skipped, so the command is safe to rerun.

KL-pressure study and qualitative candidates:

```bash
# smoke
python -m task2_ppo.ablate_kl --config configs/ppo.yaml --smoke

# matched 8-update forks kl_0.0, kl_0.1, kl_0.2 from the midpoint, each evaluated on the held-out prompts
python -m task2_ppo.ablate_kl --config configs/ppo.yaml

# list reward-vs-quality candidates for every evaluated run against the midpoint's responses
python -m task2_ppo.find_candidates --config configs/ppo.yaml
```

`ablate_kl` writes `results/task2_ppo/kl_ablation.json`: held-out reward, KL, entropy and length for each fork with bootstrap intervals, plus the per-update training trajectories (reward, KL, `delta_kl`, entropy, length). Only `kl_beta` changes between the forks; finished forks are skipped on a rerun.

`find_candidates` needs the midpoint evaluation (`--name midpoint`). For every other evaluated run it compares each response with the midpoint's response to the same prompt and writes `results/task2_ppo/candidates.json`, sorted by reward gain:

- `suspect`: reward is higher than the midpoint's and at least one flag is set: `much_longer` (at least 1.5 x the midpoint's tokens), `no_eos` (hit the generation cap), `repeated_ngram` (a run of 8 whitespace-separated words appears 3 or more times).
- `agree`: reward is higher, the response has more tokens than the midpoint's but under 1.5 x, and no flag is set.

These are candidates from one sampled response per prompt; the examples are chosen and judged by hand.

Each update samples `prompts_per_update` prompts from a seeded order of `data/rl_prompt_pool_train.jsonl` (the same sequence for every fork), runs `ppo_epochs` optimisation steps on that rollout, appends one line to `results/task2_ppo/<run_name>/train_log.jsonl` and the sampled response to `rollouts.jsonl`, and saves a checkpoint to `outputs/task2_ppo/<run_name>/checkpoints/`. `--resume` continues from the latest checkpoint and starts from the midpoint when there is none.

Choices that are not fixed by the handout:

- A rollout that ends without EOS has `missing_eos_penalty` subtracted from its terminal reward. `reward` in the log is the raw reward-model score, `reward_after_eos_penalty` is what PPO trained on.
- The policy and the critic are kept in eval mode (LoRA dropout off) during the optimisation steps, so the ratio is exactly 1 on the first epoch and the clip fraction counts real policy movement only. `clip_fraction` is the last epoch's value; `clip_fraction_epochs` has every epoch.
- Advantages are normalised over the valid response tokens of the rollout with the released `normalize_advantages`; returns use the unnormalised advantages.
- `entropy` is the released sampled estimate (minus the mean log-prob of the sampled tokens), `entropy_exact` is the token entropy from the full distribution.
- Non-finite float16 gradients are handled as in Task 1: the step is redone at a lower gradient scale, never skipped.

In smoke mode on a machine without the course assets, the tracked word-limit prompts stand in for the prompt pool and the critic starts from `value_model_init` with an untrained head.

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

Commands (run from the repository root):

```bash
# unit tests for the objective and the study helpers
python -m pytest tests/test_grpo.py tests/test_grpo_accumulation.py tests/test_grpo_studies.py -q

# smoke: Qwen2.5-0.5B with a fresh LoRA, 2 updates, 16 completion tokens, writes under outputs/smoke/ and results/smoke/
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard --smoke

# standard continuation: 20 updates from the supplied midpoint, final adapter in outputs/task3_grpo/standard
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard --resume

# held-out evaluation, same protocol as Task 2 (no --adapter = base policy)
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.evaluate --config configs/grpo.yaml --name sft

# group-size study on the cached completions (CPU, no training)
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml

# matched 8-update forks norm_grpo and norm_dr_grpo from the midpoint, each evaluated on the held-out prompts
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

Each update samples `num_generations` completions of one prompt from a seeded prompt order (the same sequence for every fork), computes group-relative advantages over all of them, and takes `policy_epochs` optimisation steps. It appends one line to `results/task3_grpo/<run_name>/train_log.jsonl`, one line per completion to `rollouts.jsonl` (reward, advantage, length, whether it was masked, and its token weight), and saves a checkpoint to `outputs/task3_grpo/<run_name>/checkpoints/`.

Choices that are not fixed by the handout:

- A group is informative when its within-group reward standard deviation (population) is above 1e-6, the `eps` default of `group_relative_advantages`. The same tolerance is used in training and in the group-size study.
- Completions that hit `max_completion_length` are masked out of the loss (`mask_truncated_completions`) but still count in their group's mean and standard deviation. When every completion of an update is masked the loss is zero and no optimisation step is taken; the update is logged with `no_gradient: true` and counted in `train_metrics.json`.
- Completions go through the model one at a time and their gradients are accumulated, to fit a T4. The sum is the released batch loss: the policy term weighted by 1/K and the KL term by the completion's share of the valid tokens (`tests/test_grpo_accumulation.py`).
- LoRA dropout is off during the step, as in Task 2. With one epoch per rollout the ratio is exactly 1 inside the step, so `clip_fraction` is 0 by construction; `clip_fraction_after_update` and `max_ratio_deviation_after_update` are measured on the same rollout after the step.
- The evaluation reads `eval_max_response_length` and `reward_max_length` from `configs/ppo.yaml` and calls the Task 2 scoring functions, so Tasks 2 and 3 share one held-out protocol.

`analyze_group_size` splits each prompt's 8 cached completions into 8/K groups of K (200 random splits, config seed) so every K uses the same completions, and writes `results/task3_grpo/group_size_study.json`. It reports the informative-group rate, the mean within-group reward standard deviation, the variance of the group-relative advantage, and the variance of the centred reward (reward minus group mean), overall and for three difficulty bins: prompts ranked by mean reward over their 8 completions, bottom third hard, middle third medium, top third easy. The advantage variance is close to the informative-group rate by construction, because a normalised advantage has variance near 1 inside every informative group; the centred-reward variance is the signal before that normalisation.

`compare_normalization` writes `results/task3_grpo/normalization_study.json`. Its length-conditioned statistic is the per-token weight |A_k| x w_k (w_k = 1/T_k for `grpo`, 1/`max_completion_length` for `dr_grpo`, 0 for a masked completion), averaged within quartiles of the training completion length pooled over both forks, together with each quartile's share of the total per-token weight and of the total per-sequence weight (per-token weight x T_k).

### Task 4 - Safety calibration

The judge loader/parser are supplied. You must implement the requested generation aggregation and evaluation.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.
