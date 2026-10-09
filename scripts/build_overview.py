from __future__ import annotations

import argparse
import html
import subprocess

import pandas as pd

from common.data import repo_path
from common.logging_utils import load_json

OUTPUT= "results/overview.html"

STYLE= """
:root { --bg: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --line: #e1e0d9; --head: #f0efec; }
@media (prefers-color-scheme: dark) { :root { --bg: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --line: #383835; --head: #2c2c2a; } }
body { background: var(--bg); color: var(--ink); font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 14px; line-height: 1.45; margin: 0 auto; max-width: 1180px; padding: 16px; }
h1 { font-size: 22px; } h2 { border-top: 2px solid var(--line); font-size: 19px; margin-top: 40px; padding-top: 18px; } h3 { font-size: 15px; margin-top: 26px; }
p.note { color: var(--ink2); margin: 4px 0 10px 0; }
div.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 12.5px; margin: 6px 0 14px 0; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 9px; text-align: right; white-space: nowrap; }
th { background: var(--head); } th:first-child, td:first-child { text-align: left; }
img { background: #fcfcfb; border: 1px solid var(--line); display: block; margin: 8px 0 16px 0; max-width: 100%; }
nav a { color: var(--ink2); margin-right: 14px; }
code { font-size: 12.5px; }
"""


def fmt(value):
    #plain numbers for a table: small values in scientific notation, the rest to 3-4 significant places
    if isinstance(value, bool):
        return str(value)

    if isinstance(value, float):
        if pd.isna(value):
            return ""

        if value == int(value) and abs(value) < 1e9:
            return str(int(value))

        if abs(value) < 1e-3:
            return f"{value:.2e}"

        if abs(value) >= 100:
            return f"{value:.1f}"

        return f"{value:.3f}"

    return html.escape(str(value))


def ci(df, key):
    #"value [low, high]" as one cell
    return [f"{fmt(float(v))} [{fmt(float(lo))}, {fmt(float(hi))}]" if not pd.isna(v) else "" for v, lo, hi in zip(df[key], df[key + "_ci_low"], df[key + "_ci_high"])]


def table(df, columns):
    #columns= list of (header, column name or list of ready-made cells)
    head= "".join(f"<th>{html.escape(h)}</th>" for h, _ in columns)
    body= []

    for i in range(len(df)):
        cells= []

        for _, source in columns:
            value= source[i] if isinstance(source, list) else df[source].iloc[i]
            cells.append(f"<td>{value if isinstance(source, list) else fmt(value)}</td>")

        body.append("<tr>" + "".join(cells) + "</tr>")

    return f'<div class="scroll"><table><tr>{head}</tr>{"".join(body)}</table></div>'


def figure(path):
    if not repo_path("results/" + path).exists():
        return f'<p class="note">figure not found: {html.escape(path)}</p>'

    return f'<img src="{html.escape(path)}" alt="{html.escape(path)}">'


def read(path):
    p= repo_path("results/" + path)

    if not p.exists():
        return None

    return pd.read_csv(p)


def missing(path):
    return f'<p class="note">not available yet: <code>results/{html.escape(path)}</code></p>'


def task1():
    out= ['<h2 id="task1">Task 1 - DPO</h2>']

    s= read("task1_dpo/summary.csv")

    if s is None:
        return out + [missing("task1_dpo/summary.csv")]

    out.append("<h3>DPO summary: standard run and the three beta forks</h3>")
    out.append('<p class="note">Held-out pairs: 290. The standard run (one epoch) and the beta forks (600 examples each) use different training budgets. The reference row is the untrained base policy: its margin is 0 for every pair, so its preference accuracy (margin &gt; 0) is 0 and its loss is log 2 by definition. KL has no interval because per-response values were not saved for Task 1.</p>')
    out.append(table(s, [
        ("run", "run"), ("budget", "budget"), ("beta", "beta"), ("train examples", "train_examples"), ("optimizer steps", "train_optimizer_steps"),
        ("held-out DPO loss", ci(s, "dpo_loss")), ("preference accuracy", ci(s, "preference_accuracy")), ("mean margin", ci(s, "margin_mean")),
        ("KL per token", "kl_token_mean"), ("KL per response", "kl_sequence_mean"), ("reward", ci(s, "reward_mean")),
        ("length mean", ci(s, "length_mean")), ("length std", "length_std"), ("length IQR", "length_iqr"),
    ]))
    out.append(figure("task1_dpo/figures/beta_study.png"))
    out.append(figure("task1_dpo/figures/training_one_epoch.png"))
    out.append(figure("task1_dpo/figures/training_beta_forks.png"))

    out.append("<h3>Training budget and cost</h3>")
    out.append(table(s, [
        ("run", "run"), ("skipped optimizer steps", "train_skipped_optimizer_steps"), ("step retries", "train_step_retries_total"),
        ("wall-clock (s)", "train_wall_clock_sec"), ("peak VRAM (bytes)", "train_peak_vram_bytes"), ("GPU", "gpu_name"),
    ]))

    st= read("task1_dpo/stratified.csv")

    if st is not None:
        out.append("<h3>Length-confounding diagnostics: preference accuracy by length stratum</h3>")
        out.append('<p class="note">Length-stratified held-out pairs: 237. Preference accuracy counts pairs with margin &gt; 0 (relative to the reference). Raw preference accuracy counts pairs where the policy alone gives the preferred response the higher summed log-probability.</p>')
        out.append(table(st, [
            ("run", "run"), ("stratum", "stratum"), ("pairs", "n_pairs"), ("preference accuracy", ci(st, "preference_accuracy")),
            ("raw preference accuracy", ci(st, "raw_preference_accuracy")), ("mean margin", ci(st, "margin_mean")),
        ]))
        out.append(figure("task1_dpo/figures/length_strata.png"))

    out.append("<h3>Generated length and word-limit compliance</h3>")
    out.append('<p class="note">Word-limit set: 10 prompts, 5 sampled responses each. The interval resamples prompts.</p>')
    sub= s[s["run"].isin(["reference", "standard", "length_balanced"])].reset_index(drop=True)
    out.append(table(sub, [
        ("run", "run"), ("word-limit compliance", ci(sub, "word_limit_compliance")), ("word-limit length (tokens)", "word_limit_length_mean"),
        ("word-limit length (words)", "word_limit_words_mean"), ("held-out length mean", ci(sub, "length_mean")), ("held-out length IQR", "length_iqr"),
        ("truncated fraction", "truncated_fraction"),
    ]))
    out.append(figure("task1_dpo/figures/length_behaviour.png"))

    p= read("task1_dpo/paired_differences.csv")

    if p is not None:
        out.append("<h3>Paired differences on the same held-out items</h3>")
        out.append('<p class="note">Run minus baseline, with a paired bootstrap interval. Rows with a stratum are on the length-stratified pairs.</p>')
        p= p.fillna({"stratum": "all held-out"})
        cells= lambda key: [f"{fmt(float(v))} [{fmt(float(lo))}, {fmt(float(hi))}]" if not pd.isna(v) else "" for v, lo, hi in zip(p[key + "_diff"], p[key + "_diff_ci_low"], p[key + "_diff_ci_high"])]
        out.append(table(p, [
            ("run", "run"), ("baseline", "baseline"), ("set", "stratum"), ("pairs", "n_pairs"),
            ("preference accuracy difference", cells("preference_accuracy")), ("reward difference", cells("reward")), ("length difference (tokens)", cells("n_tokens")),
        ]))

    out.append('<p class="note">Qualitative material: <code>results/task1_dpo/&lt;run&gt;/generations.jsonl</code> (prompt, response, reward, word_limit_ok per response).</p>')

    return out


def heldout_columns(s):
    return [
        ("run", "run"), ("reward", ci(s, "reward_mean")), ("KL per token", ci(s, "kl_token_mean")), ("entropy (sampled)", ci(s, "entropy")),
        ("entropy (exact)", ci(s, "entropy_exact")), ("length mean", ci(s, "length_mean")), ("length std", "length_std"), ("length IQR", "length_iqr"),
        ("no EOS", ci(s, "no_eos_fraction")),
    ]


def paired_cells(p, key):
    return [f"{fmt(float(v))} [{fmt(float(lo))}, {fmt(float(hi))}]" for v, lo, hi in zip(p[key + "_diff"], p[key + "_diff_ci_low"], p[key + "_diff_ci_high"])]


def task2():
    out= ['<h2 id="task2">Task 2 - PPO</h2>']

    s= read("task2_ppo/summary.csv")

    if s is None:
        return out + [missing("task2_ppo/summary.csv")]

    out.append("<h3>Standard PPO continuation diagnostics</h3>")
    out.append('<p class="note">20 updates from the supplied midpoint, one prompt per update, so per-update values are noisy. Per-update numbers: <code>results/task2_ppo/standard_trajectory.csv</code>.</p>')
    out.append(figure("task2_ppo/figures/standard_trajectories.png"))
    train= s[s["train_updates"].notna()].reset_index(drop=True)
    out.append(table(train, [
        ("run", "run"), ("clip epsilon", "train_clip_epsilon"), ("KL beta", "train_kl_beta"), ("updates", "train_updates"), ("training tokens", "train_generated_tokens"),
        ("fp16 step retries", "train_step_retries_total"), ("wall-clock (s)", "train_wall_clock_sec"), ("peak VRAM (bytes)", "train_peak_vram_bytes"), ("GPU", "train_gpu_name"),
    ]))

    out.append("<h3>Held-out evaluation (200 prompts, one sampled response each)</h3>")
    held= s[s["reward_mean"].notna()].reset_index(drop=True)
    not_done= [r for r in s["run"] if r not in set(held["run"])]

    if len(not_done) > 0:
        out.append(f'<p class="note">No held-out evaluation yet for: {", ".join(not_done)}.</p>')

    out.append(table(held, heldout_columns(held)))
    out.append(figure("task2_ppo/figures/heldout_comparison.png"))

    p= read("task2_ppo/paired_differences.csv")

    if p is not None:
        out.append("<h3>Paired differences on the same 200 prompts</h3>")
        out.append('<p class="note">Run minus baseline, paired bootstrap interval. "identical responses" counts prompts where both runs produced exactly the same text.</p>')
        out.append(table(p, [
            ("run", "run"), ("baseline", "baseline"), ("identical responses", "identical_responses"),
            ("reward difference", paired_cells(p, "reward")), ("length difference (tokens)", paired_cells(p, "n_tokens")),
        ]))
        out.append(figure("task2_ppo/figures/paired_differences.png"))

    c= read("task2_ppo/clipping_cached.csv")

    if c is not None:
        out.append("<h3>Clipping study: cached rollout batch</h3>")
        out.append('<p class="note">32 cached rollouts, 8814 response tokens, rebuilt from their text and scored under each policy. Clip fraction: ratio outside [1-eps, 1+eps] before clipping. Binding fraction: the objective uses the clipped term. Per-row counts are in <code>results/task2_ppo/clipping_cached_&lt;policy&gt;.json</code>; row 4 holds all tokens outside the 0.5 range.</p>')
        out.append(table(c, [
            ("policy", "policy"), ("epsilon", "clip_epsilon"), ("clipped surrogate", "clipped_surrogate"), ("unclipped surrogate", "unclipped_surrogate"),
            ("clip fraction", ci(c, "clip_fraction")), ("binding fraction", ci(c, "binding_fraction")),
        ]))
        out.append(figure("task2_ppo/figures/clipping_cached.png"))

    out.append("<h3>Clipping study: matched forks (8 updates each)</h3>")
    threshold= repo_path("results/task2_ppo/stability_threshold.json")

    if threshold.exists():
        t= load_json(threshold)
        out.append(f'<p class="note">S1: population std of delta_kl over the fork\'s updates. S2: number of updates with |delta_kl| above the threshold {t["threshold"]:.3e} (3 x median |delta_kl| of the standard run). delta_kl is the mean log ratio new/old on the update\'s own rollout tokens.</p>')

    clip= s[s["run"].str.startswith("clip_")].reset_index(drop=True)
    out.append(table(clip, [
        ("run", "run"), ("epsilon", "train_clip_epsilon"), ("S1", "s1_std_delta_kl"), ("S2", "s2_updates_above_threshold"),
        ("reward", ci(clip, "reward_mean")), ("KL per token", ci(clip, "kl_token_mean")), ("length mean", ci(clip, "length_mean")),
    ]))
    out.append(figure("task2_ppo/figures/clipping_fork_trajectories.png"))

    out.append("<h3>KL-pressure study: matched forks (8 updates each)</h3>")
    kl= s[s["run"].str.startswith("kl_")].reset_index(drop=True)
    out.append(table(kl, [
        ("run", "run"), ("KL beta", "train_kl_beta"), ("reward", ci(kl, "reward_mean")), ("KL per token", ci(kl, "kl_token_mean")),
        ("entropy (sampled)", ci(kl, "entropy")), ("length mean", ci(kl, "length_mean")), ("no EOS", ci(kl, "no_eos_fraction")),
    ]))
    out.append(figure("task2_ppo/figures/kl_fork_trajectories.png"))

    candidates= repo_path("results/task2_ppo/candidates.json")

    if candidates.exists():
        counts= load_json(candidates)["counts"]
        df= pd.DataFrame([{"run": k, **v} for k, v in counts.items()])

        out.append("<h3>Qualitative candidates (counts only)</h3>")
        out.append('<p class="note">Against the midpoint\'s response to the same prompt. Suspect: reward higher and the response is at least 1.5x longer, has no EOS, or repeats an 8-word run 3+ times. Agree: reward higher, longer but under 1.5x, no flag. The texts are in <code>results/task2_ppo/candidates.json</code>.</p>')
        out.append(table(df, [("run", "run"), ("prompts", "n_prompts"), ("reward higher", "reward_up"), ("suspect", "suspect"), ("agree", "agree")]))

    return out


def task3():
    out= ['<h2 id="task3">Task 3 - GRPO</h2>']

    s= read("task3_grpo/summary.csv")

    if s is None:
        return out + [missing("task3_grpo/summary.csv")]

    out.append("<h3>Standard GRPO continuation diagnostics</h3>")
    out.append('<p class="note">20 updates from the supplied midpoint, one prompt with K = 4 completions per update. Completions that hit the 512-token cap are masked from the loss; an update with all four masked takes no optimizer step. Per-update numbers: <code>results/task3_grpo/standard_trajectory.csv</code>.</p>')
    out.append(figure("task3_grpo/figures/standard_trajectories.png"))
    train= s[s["train_updates"].notna()].reset_index(drop=True)
    out.append(table(train, [
        ("run", "run"), ("loss type", "train_loss_type"), ("updates", "train_updates"), ("optimizer steps", "train_optimizer_steps"), ("updates without gradient", "train_updates_without_gradient"),
        ("mean within-group reward std", "train_reward_std_within_group_mean"), ("uninformative-group fraction", "train_uninformative_group_fraction"),
        ("masked completions", "train_masked_completions_total"), ("training tokens", "train_generated_tokens"),
        ("wall-clock (s)", "train_wall_clock_sec"), ("peak VRAM (bytes)", "train_peak_vram_bytes"), ("GPU", "train_gpu_name"),
    ]))

    g= read("task3_grpo/group_size.csv")

    if g is not None:
        out.append("<h3>Equal-generation group-size study</h3>")
        out.append('<p class="note">24 cached prompts x 8 completions, split into 8/K groups of K (200 random splits). Informative: within-group reward std above 1e-6. Difficulty: prompts ranked by mean reward over their 8 completions, bottom / middle / top third. Intervals resample prompts. The variance of the normalised advantage equals the informative rate by construction, so the centred-reward variance (reward minus group mean) is shown as well.</p>')
        out.append(table(g, [
            ("K", "group_size"), ("difficulty", "difficulty"), ("prompts", "n_prompts"), ("informative-group rate", ci(g, "informative_rate")),
            ("within-group reward std", ci(g, "reward_std")), ("advantage variance", ci(g, "advantage_variance")), ("centred-reward variance", ci(g, "centred_reward_variance")),
        ]))
        out.append(figure("task3_grpo/figures/group_size.png"))

    out.append("<h3>Canonical vs Dr. GRPO: held-out evaluation (200 prompts)</h3>")
    held= s[s["reward_mean"].notna()].reset_index(drop=True)
    not_done= [r for r in s["run"] if r not in set(held["run"])]

    if len(not_done) > 0:
        out.append(f'<p class="note">No held-out evaluation yet for: {", ".join(not_done)}. The sft row is the Task 2 evaluation of the base policy under the identical protocol.</p>')

    out.append(table(held, heldout_columns(held)))
    out.append(figure("task3_grpo/figures/heldout_comparison.png"))

    p= read("task3_grpo/paired_differences.csv")

    if p is not None:
        out.append("<h3>Paired differences on the same 200 prompts</h3>")
        out.append(table(p, [
            ("run", "run"), ("baseline", "baseline"), ("identical responses", "identical_responses"),
            ("reward difference", paired_cells(p, "reward")), ("length difference (tokens)", paired_cells(p, "n_tokens")),
        ]))
        out.append(figure("task3_grpo/figures/paired_differences.png"))

    b= read("task3_grpo/length_bins.csv")

    if b is not None:
        out.append("<h3>Length-conditioned weight in the policy term</h3>")
        out.append('<p class="note">Per-token weight |A_k| x w_k, with w_k = 1/T_k for grpo and 1/512 for dr_grpo, 0 for a masked completion. Quartiles of the training completion length pooled over both forks, 8 completions per quartile per fork. Per-sequence weight = per-token weight x T_k.</p>')
        out.append(table(b, [
            ("run", "run"), ("quartile", "bin"), ("completions", "n_completions"), ("masked", "n_masked"), ("mean length", "length_mean"),
            ("mean per-token weight", "per_token_weight_mean"), ("share of per-token weight", "per_token_weight_share"),
            ("mean per-sequence weight", "per_sequence_weight_mean"), ("share of per-sequence weight", "per_sequence_weight_share"),
        ]))
        out.append(figure("task3_grpo/figures/length_bins.png"))
        out.append(figure("task3_grpo/figures/normalization_fork_trajectories.png"))

    out.append('<p class="note">Qualitative material: <code>results/task3_grpo/&lt;run&gt;/generations.jsonl</code> (held-out) and <code>rollouts.jsonl</code> (training completions with reward, advantage and mask).</p>')

    return out


def task4():
    out= ['<h2 id="task4">Task 4 - safety calibration</h2>']

    s= read("task4_safety/safety_summary.csv")

    if s is None:
        return out + [missing("task4_safety/safety_summary.csv")]

    out.append("<h3>Safety-calibration comparison</h3>")
    out.append('<p class="note">450 XSTest prompts (250 safe, 200 unsafe), one greedy response per policy, labelled by the supplied judge. Each rate is a judge label counted on prompts of one benchmark class. The judge is not told the benchmark class, so the rates of a class need not sum to 1; the next table has all five labels.</p>')
    out.append(table(s, [
        ("policy", "policy"), ("safe-answer rate", ci(s, "safe_answer_rate")), ("safe over-refusal rate", ci(s, "safe_over_refusal_rate")),
        ("unsafe-compliance rate", ci(s, "unsafe_compliance_rate")), ("unsafe justified-refusal rate", ci(s, "unsafe_justified_refusal_rate")),
        ("ambiguous rate", ci(s, "ambiguous_rate")), ("judge parse failures", "judge_parse_failures"),
        ("mean response tokens", ci(s, "response_tokens_mean")), ("response tokens std", "response_tokens_std"),
    ]))
    out.append(figure("task4_safety/figures/safety_rates.png"))

    c= read("task4_safety/class_label_distribution.csv")

    if c is not None:
        out.append("<h3>Judge-label counts by benchmark class</h3>")
        out.append(table(c, [
            ("policy", "policy"), ("benchmark class", "benchmark_class"), ("prompts", "n"), ("SAFE_ANSWER", "SAFE_ANSWER"), ("JUSTIFIED_REFUSAL", "JUSTIFIED_REFUSAL"),
            ("UNSAFE_COMPLIANCE", "UNSAFE_COMPLIANCE"), ("OVER_REFUSAL", "OVER_REFUSAL"), ("AMBIGUOUS", "AMBIGUOUS"),
        ]))

    p= read("task4_safety/paired_vs_sft.csv")

    if p is not None:
        out.append("<h3>Paired differences from the base policy on the same prompts</h3>")
        out.append(table(p, [
            ("policy", "policy"), ("safe-answer", paired_cells(p, "safe_answer_rate")), ("safe over-refusal", paired_cells(p, "safe_over_refusal_rate")),
            ("unsafe-compliance", paired_cells(p, "unsafe_compliance_rate")), ("unsafe justified-refusal", paired_cells(p, "unsafe_justified_refusal_rate")),
            ("response tokens", paired_cells(p, "response_tokens")),
        ]))

    out.append("<h3>Category-level behaviour</h3>")
    out.append('<p class="note">25 prompts per XSTest category. Counts: <code>results/task4_safety/category_label_distribution.csv</code>.</p>')
    out.append(figure("task4_safety/figures/category_labels.png"))

    agreement= repo_path("results/task4_safety/audit_agreement.json")

    if agreement.exists():
        a= load_json(agreement)

        out.append("<h3>Manual-audit agreement</h3>")
        out.append('<p class="note">60 responses (30 safe and 30 unsafe prompt IDs, 15 per policy), labelled by hand without seeing the judge label or the policy. Agreement and kappa are over the examples where neither label is AMBIGUOUS; an AMBIGUOUS label on either side never counts as agreement.</p>')

        df= pd.DataFrame([{
            "examples": a["n_examples"], "manual AMBIGUOUS": a["manual_ambiguous"], "judge AMBIGUOUS": a["judge_ambiguous"],
            "both non-ambiguous": a["n_both_non_ambiguous"], "agreements": a["agreements_non_ambiguous"],
            "raw agreement (non-ambiguous)": a["raw_agreement_non_ambiguous"], "Cohen kappa (non-ambiguous)": a["cohen_kappa_non_ambiguous"],
            "raw agreement (all)": f'{fmt(a["raw_agreement_all"])} [{fmt(a["raw_agreement_all_ci95"][0])}, {fmt(a["raw_agreement_all_ci95"][1])}]',
        }])
        out.append(table(df, [(c, c) for c in df.columns]))

        confusion= read("task4_safety/audit_confusion.csv")
        out.append('<p class="note">Confusion matrix: manual label in rows, judge label in columns.</p>')
        out.append(table(confusion, [(c, c) for c in confusion.columns]))

        per= pd.DataFrame([{"group": k, "examples": v["n_examples"], "agreements": v["agreements"]} for k, v in list(a["per_policy"].items()) + list(a["per_benchmark_class"].items())])
        out.append(table(per, [("policy or class", "group"), ("examples", "examples"), ("agreements", "agreements")]))
    else:
        out.append('<p class="note">Manual-audit agreement not computed yet.</p>')

    out.append('<p class="note">Qualitative material: <code>results/task4_safety/audit_disagreements.csv</code> (the audit disagreements, with a disagreement_type column to fill) and <code>qualitative_candidates.csv</code> (responses picked by rule from the judge labels: an unsafe prompt that was answered, a safe prompt that was refused, or policies that got different labels).</p>')

    return out


def main(args):
    try:
        commit= subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_path(".")), text=True).strip()
    except Exception:
        commit= "unknown"

    parts= [
        "<!doctype html><html><head><meta charset='utf-8'><title>PA2 results overview</title>",
        f"<style>{STYLE}</style></head><body>",
        "<h1>ATML PA2 - results overview, Tasks 1-4</h1>",
        f'<p class="note">Numbers and figures only, generated from the saved result files by <code>python -m scripts.build_overview</code> at commit {commit}. Brackets are 95% bootstrap intervals (2000 resamples, seed 6304). Every value traces to a CSV or JSON under <code>results/</code>.</p>',
        '<nav><a href="#task1">Task 1 - DPO</a><a href="#task2">Task 2 - PPO</a><a href="#task3">Task 3 - GRPO</a><a href="#task4">Task 4 - safety</a></nav>',
    ]

    parts += task1() + task2() + task3() + task4()
    parts.append("</body></html>")

    path= repo_path(args.output)
    path.write_text("\n".join(parts), encoding="utf-8")

    print("saved", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT
    )

    args = parser.parse_args()

    main(args)
