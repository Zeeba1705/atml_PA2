from __future__ import annotations

import re
import numpy as np
import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def sampled_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor):
    return masked_mean(policy_logp - ref_logp, mask)


def sample_entropy(sampled_logp: torch.Tensor, mask: torch.Tensor):
    return -masked_mean(sampled_logp, mask)


def mean_response_length(mask: torch.Tensor):
    return float(mask.sum(-1).float().mean().item())


def preference_accuracy(chosen_logp, rejected_logp):
    return float((chosen_logp > rejected_logp).float().mean().item())


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def parse_word_limit(prompt: str):
    patterns = [
        r"(?:at most|no more than|under|within)\s+(\d+)\s+words?",
        r"(?:in|use)\s+(\d+)\s+words?\s+(?:or fewer|max(?:imum)?)",
        r"(?:maximum|max)\s+(?:of\s+)?(\d+)\s+words?",
    ]
    lower = str(prompt).lower()
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            return int(m.group(1))
    return None


def word_limit_compliance(prompt: str, response: str):
    limit = parse_word_limit(prompt)
    if limit is None:
        return None
    return float(word_count(response) <= limit)


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


BOOTSTRAP_RESAMPLES= 2000


def bootstrap_ci(values, weights=None, seed=6304, n_resamples=BOOTSTRAP_RESAMPLES):
    #95% interval from resampling prompts with replacement
    #without weights the statistic is the mean over prompts
    #with weights it is sum(values) / sum(weights), for means taken over tokens (values= per-prompt sums, weights= token counts)
    values= np.asarray(values, dtype=float)
    rng= np.random.default_rng(seed)

    idx= rng.integers(0, len(values), size=(n_resamples, len(values)))

    if weights is None:
        stats= values[idx].mean(axis=1)
    else:
        weights= np.asarray(weights, dtype=float)
        stats= values[idx].sum(axis=1) / np.maximum(weights[idx].sum(axis=1), 1.0)

    low, high = np.percentile(stats, [2.5, 97.5])

    return [float(low), float(high)]


def paired_bootstrap_ci(values_a, values_b, seed=6304, n_resamples=BOOTSTRAP_RESAMPLES):
    #two conditions on the same prompts: resample the per-prompt difference a - b
    diff= np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)

    return bootstrap_ci(diff, None, seed, n_resamples)
