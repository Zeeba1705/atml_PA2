# CLAUDE.md — ATML PA2 (LLM post-training), current focus: Task 1 (DPO)

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
