# CLAUDE.md — ATML PA2 (LLM post-training), Tasks 1–5

Ask me which task we're on if it isn't clear from my message. Task 1 work lives on branch `task1-dpo`, Task 2 on `task2-ppo`.

Course assignment, individual, due 11 Oct 2026. Public GitHub repo. I am responsible for and must understand every line,
so explain non-trivial changes before making them. Code style: follow CODE_STYLE.md for all new code.

## Hard rules (never break these)
- NEVER write, edit, or draft anything in `report/` or report prose/analysis. The PDF report must be my own words.
  You may produce plots, tables (CSV/markdown of numbers), and scripts that generate them.
- Do not modify fixed course assets: `data/`, `cached/`, `checkpoints/`, the tracked Task 1 word-limit prompt file,
  the Task 5 transfer file, configs' fixed eval settings. Never commit weights, datasets, or `outputs/`.
- Do not change seed, decoding config, max generation length, held-out pairs, or prompt IDs between conditions
  being compared. Read constants from `configs/`; don't hardcode duplicates or silently "improve" hyperparameters.
- Never use Task 4 (XSTest) or Task 5 data to tune anything in Task 1.
- Don't pick a setting after looking at held-out results.
- If you reuse external code (e.g. adapted from TRL), tell me and add it to the README "Attribution" section.
- Don't run `git push --force`, `git reset --hard`, or delete branches without asking.

## Compute setup
- No local GPU. I write/test locally with you, then run on Google Colab via `colab/run.ipynb`.
- Everything must be checkable on CPU first:
  - unit tests in `tests/` on small hand-made tensors (`python -m pytest tests -q`)
  - every train/eval script takes `--smoke` (a few steps, tiny batch, few eval examples; may use
    `Qwen/Qwen2.5-0.5B-Instruct` on CPU) so the pipeline is tested end to end without a GPU.
- Colab sessions die: train scripts save a checkpoint to `outputs/<run_name>/` every N steps and support `--resume`.
- Colab GPU may be a T4 (no bf16): pick dtype from the hardware, support gradient checkpointing and grad accumulation.
- `colab/run.ipynb` is a thin launcher only: clone/pull, pip install, mount Drive, symlink
  `checkpoints/ cached/ outputs/ results/` to Drive, download/validate assets, then call scripts. No experiment logic.

## Before writing code: explore
The starter repo is https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining. Filenames may differ from what's below —
read `task1_dpo/`, `common/`, `configs/`, `scripts/` first and use what's actually there. Reuse the released
helpers (data loading, sequence log-probs, KL estimator, reward model scoring, generation, length/word-limit
metrics) rather than writing new versions. The handout is at `handout/PA2.pdf` (gitignored) — Task 1 is pages 3–5,
metric definitions page 3.

Known starter layout: `common/{data,models,generation,metrics,logging_utils}.py`; `task1_dpo/{dpo,train,evaluate,
ablate_beta,analyze_length}.py` (train/evaluate/ablate/analyze contain `NotImplementedError` TODOs to fill in).
Key config values: max_sequence_length 768, max_generation_tokens 256, batch_size 2, grad_accum_steps 8, lr 2e-5,
max_grad_norm 1.0, `quantize_frozen_models: true`, generation temperature 0.7 / top_p 0.9.
Pinned env: transformers 4.57.1, trl 0.27.2, peft 0.17.1 — don't upgrade them.
`.claude/settings.json` blocks edits to data/, cached/, checkpoints/, manifests/, report/; don't try to work around it.

## Task 1 spec (from the handout)
Policy: `Qwen/Qwen2.5-1.5B-Instruct` + supplied LoRA config, all DPO runs start from this same initialization.
Reference: frozen initial policy (prefer the base model with the LoRA adapter disabled, to save memory).
Reference log-probs never change: compute them once per dataset and cache to `outputs/task1_dpo/ref_cache/`
(the three β forks share the same 600 examples). Config: configs/base.yaml (seed 6304, fp16, LoRA r=8) + configs/dpo.yaml.
Data: course UltraFeedback preference files (fixed subsets), fixed held-out pairs, length-stratified held-out set,
length-balanced training subset, common word-limit prompt set. Don't construct or rebalance data yourself.

Objective:
  L_DPO = -E[ log σ( β [ (log πθ(y+|x) - log πref(y+|x)) - (log πθ(y-|x) - log πref(y-|x)) ] ) ]
Sequence log-prob = SUM of log-probs over RESPONSE tokens only (prompt and padding masked), teacher forcing,
with the usual one-token shift between logits and labels.

### Step 0 — find the deliberate defect (do this first, nothing else until done)
The released DPO objective contains exactly one deliberate algorithmic defect. Procedure:
1. Read the objective code and write out, line by line, what it computes vs the equation above.
   Check: sign of the loss, log-sigmoid vs sigmoid, where β is applied, chosen/rejected ordering, policy vs
   reference ordering, sum vs mean over tokens, response-only mask, logit/label shift, detach/no_grad on reference.
2. Write `tests/test_dpo_loss.py` that computes the correct loss by hand on tiny tensors and compares.
   Include cases: identical policy/ref → loss = log 2; chosen favoured → loss < log 2; masked tokens don't matter.
3. Show me the defect and the test failing. Wait for my OK, then fix, then show the test passing.
4. Commit the fix alone: "Fix DPO objective: <what>".

### Step 1 — standard DPO
Train ONE epoch on the fixed preference set with the release config. Evaluate on the fixed held-out pairs.
Report: held-out DPO loss, preference accuracy (fraction with margin m > 0), KL from reference (released
sampled-response estimator, same averaging convention for every condition), reward-model score on generated
responses, response length (mean + std and IQR, in tokens, same tokenizer and cap).
The final adapter of this run is reused in Task 4 — it MUST end up at `outputs/task1_dpo/standard`
(`standard_output` in configs/dpo.yaml; configs/feedback.yaml loads it from there). Keep intermediate
checkpoints in a subfolder so the final adapter is unambiguous.

### Step 2 — β study
Short-run release config (`short_ablation_examples: 600` in configs/dpo.yaml), from the original init, β ∈ {0.03, 0.10, 0.30}.
Implement the orchestration in `task1_dpo/ablate_beta.py`. Only β changes (same data subset,
number of examples, optimizer, seed, LoRA). Same metrics as Step 1 on the identical held-out prompts/decoding.
Results must mark that these short forks have a different budget than the one-epoch standard run.

### Step 3 — length confounding
Implement in `task1_dpo/analyze_length.py`. Train one DPO model from the original init on the length-balanced subset (standard config unless the release says otherwise).
Evaluate BOTH standard and length-balanced models on the length-stratified held-out set, reporting preference
accuracy separately for: preferred-longer, matched-length, rejected-longer. Also compare generated length and
word-limit compliance on the common word-limit prompt set.

### Qualitative support (for me to pick from)
Save generated responses with prompt IDs so I can find: (i) a case where preference/reward is higher but the
response is worse (correctness, concision, instruction following), (ii) length-bias / word-limit behavior.
Write a helper script that lists candidate examples (e.g. high reward but over the word limit); I choose and interpret them.

## Results and logging
- `results/task1_dpo/<run_name>/` per run (results_dir in configs/dpo.yaml): `config.json` (full resolved config), `train_log.jsonl` (step, loss,
  margin, reward accuracy, lr, grad norm), `eval_metrics.json`, `generations.jsonl` (prompt_id, prompt, response,
  n_tokens, reward, word_limit_ok), and stratified metrics for Step 3.
- Every metrics file also records: git commit hash, seed, β, dataset file + number of examples, GPU name,
  dtype, wall-clock, peak VRAM (`torch.cuda.max_memory_allocated`).
- Run names: `standard`, `beta_0.03`, `beta_0.1`, `beta_0.3`, `length_balanced`. Adapters under `outputs/task1_dpo/<run_name>`
  (length-balanced uses `length_output` from the config).
- `task1_dpo/make_tables.py` collects all runs into `results/task1_dpo/summary.csv` and makes plots in
  `results/task1_dpo/figures/` (I'll copy what I need into the report myself).

## README
Keep a "Task 1" section in README.md with the exact command for each experiment (smoke + full), e.g.
`python -m task1_dpo.train --config configs/dpo.yaml --run-name beta_0.1 --beta 0.1 --max-examples 600`.
Released entry points: `task1_dpo.train`, `task1_dpo.evaluate`, `task1_dpo.ablate_beta`, `task1_dpo.analyze_length`;
the objective is `task1_dpo/dpo.py::dpo_loss`. Keep the released CLI flag names (they use dashes).

## Workflow
- Branch `task1-dpo`. One coherent change per commit (objective fix / training loop / eval / ablation runner / tables).
  End commit messages normally; no giant "everything" commits.
- Order: Step 0 defect → training loop + `--smoke` → eval script → smoke both on CPU → I run Step 1 on Colab →
  β forks → length-balanced run → tables/plots.
- Before any full-run command, show me the command and the expected outputs. After I paste a Colab error,
  fix the root cause and add a smoke check that would have caught it.
- When unsure about a spec detail, ask me instead of guessing; quote the handout line you're unsure about.

## Task 2 spec — PPO (handout pages 5–7)
Continue from the SUPPLIED midpoint, never restart PPO: `checkpoints/ppo_midpoint_policy` (policy LoRA),
`checkpoints/ppo_midpoint_value` (value model: Qwen2.5-0.5B-Instruct init + value LoRA + head), frozen reference,
reward model `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`, prompts from `data/rl_prompt_pool_train.jsonl` / `_eval.jsonl`.
Every fork starts from the IDENTICAL supplied policy/value state. The value model is intentionally imperfect —
treat critic behaviour as something to report, not fix.
Config (configs/ppo.yaml): updates 20, fork_updates 8, prompts_per_update 1, ppo_epochs 2, policy lr 3e-6,
value LoRA lr 1e-4, value head lr 3e-4, clip_epsilon 0.20, kl_beta 0.10, gamma 1.0, gae_lambda 0.95, value_coef 0.5,
missing_eos_penalty 1.0, max_prompt_length 256, max_response_length 512 (training), eval_max_response_length 768,
reward_max_length 1280. Don't change these.

Objectives (handout):
  ρ_t = πθ(a_t|s_t) / πold(a_t|s_t)
  L_clip = E_t[ min( ρ_t A_t , clip(ρ_t, 1-ε, 1+ε) A_t ) ]   (maximise; loss = -L_clip)
  δ_t = r_t + γ V(s_{t+1}) - V(s_t),   A_t = Σ_k (γλ)^k δ_{t+k}
  r_t = r_task·1[t=T] - β_KL (log πθ(a_t|s_t) - log πref(a_t|s_t))
Clip fraction = fraction of valid response tokens with ρ_t outside [1-ε, 1+ε] BEFORE clipping.

### Step 0 — find the deliberate defect in task2_ppo/ppo.py (first, nothing else until done)
Exactly one deliberate defect in the core objective code (`ppo_policy_loss`, `compute_gae`, `shaped_rewards`,
`value_mse_loss`, `normalize_advantages`). Same procedure as Task 1: line-by-line comparison with the equations,
then `tests/test_ppo.py` on tiny hand-made tensors. Must include: clipped surrogate for A>0 and A<0 with ratios
inside and outside [1-ε,1+ε]; GAE vs a hand-computed 3-step example with a padded row; KL-shaped reward puts
r_task only on the last valid token. Show me the defect + failing test, wait for my OK, fix, commit alone:
"Fix PPO objective: <what>". Also sanity-check the other functions even after finding one defect.

### Step 1 — standard continuation (task2_ppo/continue_train.py)
20 updates from the midpoint. Log per update to `results/task2_ppo/standard/train_log.jsonl`: mean learned reward,
KL from reference, policy loss, value loss, entropy, grad norm, clip fraction, response length (+ truncated/no-EOS
fraction). Also log critic explained variance per update: EV = 1 − Var(returns − values)/Var(returns) over valid
response tokens. Record peak VRAM and wall-clock (required in the report). Final adapter → `outputs/task2_ppo/standard`
(configs/feedback.yaml loads it from there for Task 4). Checkpoint after every update + `--resume`.
Same fp16 rule as Task 1: retry a step on overflow so every run applies every update; log retries.

### Step 2 — clipping study (analyze_clipping.py)
(a) On the cached batch `cached/ppo_rollout.pt` (CPU is fine): clipped surrogate + clip/affected-token fraction for
ε ∈ {0.05, 0.20, 0.50} using the same response mask. (b) Short forks (fork_updates=8) from the same midpoint for each ε,
identical reward/KL settings; report held-out reward, KL, length and the two stability statistics below.
Stability statistics (FIXED definitions — implement exactly, don't change after seeing results):
  - S1 = std over updates of ΔKL_t = KL_t − KL_{t−1} (per-update change in KL from reference, t = 1..8)
  - S2 = number of updates with |ΔKL_t| > τ, where τ = 3 × median |ΔKL_t| of the STANDARD 20-update run
    (compute τ once from results/task2_ppo/standard and save it to results/task2_ppo/stability_threshold.json
    before any fork is analysed).
Note in outputs that each update uses 1 prompt (prompts_per_update=1), so per-update KL is noisy; all forks see
the same prompt sequence.

### Step 3 — KL-pressure study (ablate_kl.py)
Short forks (8 updates) from the same midpoint for β_KL ∈ {0, 0.10, 0.20}. Held-out reward, KL, entropy, length,
plus saved generations so I can find cases where reward rises but quality doesn't.

### Task 2 evaluation (task2_ppo/evaluate.py)
Common held-out protocol on `data/rl_prompt_pool_eval.jsonl`, eval_max_response_length 768, release decoding,
same prompts/seed/batch size for every fork. Save generations.jsonl (prompt_id, prompt, response, n_tokens, reward,
has_eos) for qualitative picks.
Reward-vs-quality candidate finder (helper script, I pick the examples): flag held-out prompts where a fork's reward
is higher than the midpoint's response to the same prompt AND the response is ≥1.5× longer, has no EOS, or repeats
an n-gram (n=8) 3+ times. Also list cases where reward and length both rise modestly (reward/quality agree candidates). Compare forks at equal generated-token and update budgets.
Run names: `standard`, `clip_0.05`, `clip_0.2`, `clip_0.5`, `kl_0.0`, `kl_0.1`, `kl_0.2`.

## Cross-task analysis choices (apply to every task)
- Bootstrap 95% CIs on every headline metric: 2000 resamples, seed 6304, resampling prompts/pairs. When two conditions
  share the same prompts, use a PAIRED bootstrap on the per-prompt difference. Save CIs next to the point estimates.
- SFT baseline row: evaluate the untouched base policy with each task's held-out protocol (Tasks 1–3) as `sft`.
- Every analysis definition below is fixed before results are inspected; if something must change, ask me first.

## Task 3 spec — GRPO (handout pages 7–9)
Continue from `checkpoints/grpo_midpoint_policy`, same policy family, reward model, reference and UltraFeedback
prompt pool as PPO. Config (configs/grpo.yaml): updates 20, fork_updates 8, prompts_per_update 1, policy_epochs 1,
lr 5e-6, clip_epsilon 0.20, kl_beta 0.10, num_generations K=4, max_prompt_length 256, max_completion_length 512,
mask_truncated_completions true, max_grad_norm 1.0. Don't change these.
Objective:
  A_k = (r_k − μ_r) / (σ_r + ε),  μ_r = mean of the K rewards for the prompt
  L_GRPO = −E_k[ (1/T_k) Σ_t min(ρ_{k,t} A_k, clip(ρ_{k,t}, 1−ϵ, 1+ϵ) A_k) ] + β D_KL(πθ‖πref)
  Dr. GRPO (`--loss-type dr_grpo`, released in grpo.py): per-sequence token sum divided by the constant
  max_completion_length (512) instead of each response's own length T_k. Use the released form; don't invent one.
Informative group = within-group reward std > the tolerance used by the released helper.

### Step 0 — defect in task3_grpo/grpo.py
One deliberate defect in `group_relative_advantages`, `grpo_policy_loss` or `mask_truncated_sequences`. Same
procedure: `tests/test_grpo.py` on hand-made tensors — advantages vs hand-computed values for 2 groups (one with
zero std), clipped term for A>0/A<0 in/out of range, 1/T_k vs Dr. GRPO weighting on two sequences of different
length, truncated sequences fully masked. Show defect + failing test, wait for my OK, fix, commit alone.

### Step 1 — standard continuation (continue_train.py)
20 updates, K=4. Log per update: reward, KL, mean within-group reward std, uninformative-group fraction, policy loss,
grad norm, entropy, response length, truncated fraction. Peak VRAM + wall-clock. Final adapter →
`outputs/task3_grpo/standard` (Task 4 loads it). Checkpoint every update + `--resume`; fp16 step retry as in Task 1.

### Step 2 — group-size study (analyze_group_size.py, no training, CPU)
Use `cached/grpo_k_cache.jsonl` (8 completions/prompt). Equal total generations: each prompt's 8 completions split
into 4 groups of K=2, 2 of K=4, or 1 of K=8. Repeat the random partition R=200 times (seed 6304) and average.
Report per K: informative-group rate, mean within-group reward std, variance of the group-relative advantage.
Difficulty bins (FIXED): per-prompt mean reward over all 8 cached completions, split into tertiles by rank:
bottom third = hard, middle = medium, top third = easy. Report every metric per bin as well.

### Step 3 — normalisation study (compare_normalization.py)
Matched short forks (8 updates) from the same midpoint: `grpo` vs `dr_grpo`. Same prompts, generation settings,
reward, β, ϵ and generated-token budget. Report held-out reward, KL, response length, and the length-conditioned
statistic (FIXED): effective per-token gradient weight |A_k| × w_k (w_k = 1/T_k for canonical, 1/512 for
Dr. GRPO), aggregated by completion-length bin (quartiles of T_k pooled over both runs), plus the share
of total weight going to the shortest vs longest quartile. Save generations for qualitative examples.
Run names: `standard`, `norm_grpo`, `norm_dr_grpo`. Evaluate with task3_grpo/evaluate.py using the same held-out
protocol as Task 2.

## Task 4 spec — safety calibration (handout pages 9–11)
Only run after Tasks 1–3 standard runs are final. Never use Task 4 results to choose or retune anything earlier.
Policies (fixed, configs/feedback.yaml): sft (no adapter), dpo = outputs/task1_dpo/standard,
ppo = outputs/task2_ppo/standard, grpo = outputs/task3_grpo/standard. No ablation forks.
1. generate_responses.py: one deterministic (greedy, do_sample=False) response per policy per XSTest prompt,
   safety_max_new_tokens 256, fixed prompt order, same batch size for all policies → use the released output names (e.g. results/task4_safety/generated_<policy>.jsonl,
   which make_audit_sheet.py expects).
2. judge_responses.py: apply the supplied `judge_one` (Qwen2.5-3B judge, temperature 0) to every response; cache labels
   (SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS + confidence).
3. evaluate_safety.py: per policy — safe-prompt answer rate, safe over-refusal rate, unsafe-prompt unsafe-compliance
   rate, unsafe justified-refusal rate, ambiguous rate, mean response length; label distribution per XSTest category.
4. Manual audit — HARD RULE: the manual labels are MINE. make_audit_sheet.py builds the fixed 60-example subset
   (released fixed_audit_ids: 30 SAFE + 30 UNSAFE XSTest ids, seed 6304) as a CSV with prompt, response, category and an
   empty `manual_label` column, and the AI label HIDDEN (stored in a separate file). Never fill, suggest, pre-fill or
   guess manual labels, and don't show me the AI labels for audit items until I say I've finished labelling.
5. Agreement (after I finish): raw agreement and Cohen's kappa over the 4 non-ambiguous labels, 5×5 confusion matrix,
   AMBIGUOUS reported separately (judge-ambiguous and manual-ambiguous counts; not counted as agreement), plus a
   list of disagreements for me to classify as policy difference / judge error / both.

## Task 5 spec — RLVR vs RLAIF (handout pages 11–13)
Frozen supplied policies: sft (no adapter), rlvr = checkpoints/rlvr_policy, rlaif = checkpoints/rlaif_policy.
Use the supplied verifier (task5_feedback/rlvr.py) and pairwise judge (task5_feedback/rlaif.py) unchanged.
math_max_new_tokens 512, judge temperature 0. `data/math_transfer_eval.jsonl` is fixed — never regenerate or edit it.
1. evaluate_math.py --dataset gsm: SFT/RLVR/RLAIF on data/gsm8k_eval.jsonl — exact accuracy, format compliance
   (designated final answer parsed), mean/std length, RLAIF pairwise win rate vs SFT (win 1, tie 0.5, loss 0; report
   ties separately).
   Verifier–judge agreement (FIXED): over response pairs (policy vs SFT, same problem) where exactly one is
   verifier-correct, the fraction where the judge prefers the correct one; also report judge ties on those pairs.
2. score_perturbations.py: score the 100-row diagnostic set with both mechanisms. Controlled pairs = clean vs each of the
   4 perturbations per problem. Per category and mechanism: better-response rate, tie rate, wrong-preference rate.
   S_reason = Pr[R(clean) > R(reason-corrupt)] (same correct final), S_outcome = Pr[R(correct final) > R(wrong final)].
   Verifier ties on reasoning-only changes are expected — report them, don't treat as errors.
3. evaluate_math.py --dataset transfer: same three policies on the fixed SVAMP subset — exact accuracy, pairwise win
   rate vs SFT, length, and drop from the in-domain metric.
4. Failure types — HARD RULE: the labels are MINE. Build a CSV of ~30 wrong transfer-set responses per policy
   (fixed seed 6304 sample) with an empty `failure_type` column; my categories: format_failure, arithmetic_slip,
   misread_problem, right_method_wrong_final, other. Never pre-fill or suggest labels. Afterwards, tabulate them.
5. compare_feedback.py: combine in-domain, diagnostic and transfer results into summary tables/figures (numbers only;
   the discussion of coverage/noise/exploitability/cost is mine).
