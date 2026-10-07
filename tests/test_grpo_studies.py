import math

from task3_grpo.analyze_group_size import difficulty_bins
from task3_grpo.analyze_group_size import partition_metrics
from task3_grpo.analyze_group_size import regroup_equal_generation_budget
from task3_grpo.compare_normalization import length_bin
from task3_grpo.compare_normalization import length_conditioned_stats

TOL= 1e-5


def make_prompt(rewards):
    return [{"reward": r, "generation_index": i} for i, r in enumerate(rewards)]


def test_regrouping_keeps_every_completion_for_every_group_size():
    by_prompt= {
        "a": make_prompt([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        "b": make_prompt([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
    }

    for k, n_groups in [(2, 8), (4, 4), (8, 2)]:
        groups= regroup_equal_generation_budget(by_prompt, k)

        assert len(groups) == n_groups
        assert all(len(g[1]) == k for g in groups)

        #same 16 completions whatever the group size
        assert sorted(r for g in groups for r in g[1]) == sorted([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

    #without shuffling the cached order is kept
    assert regroup_equal_generation_budget(by_prompt, 4)[0] == ("a", [1.0, 2.0, 3.0, 4.0])


def test_partition_metrics_on_one_informative_and_one_flat_group():
    #group 1= 1, 3, 5, 5: mean 3.5, variance 2.75. group 2= 2, 2, 2, 2: no spread
    groups= [("p", [1.0, 3.0, 5.0, 5.0]), ("p", [2.0, 2.0, 2.0, 2.0])]

    m= partition_metrics(groups)["p"]

    assert abs(m["informative_rate"] - 0.5) < TOL
    assert abs(m["reward_std"] - math.sqrt(2.75) / 2.0) < TOL

    #advantages have variance 1 in the informative group and 0 in the flat one: 4 * 1 / 8
    assert abs(m["advantage_variance"] - 0.5) < TOL

    #squared distances from the group mean: 6.25 + 0.25 + 2.25 + 2.25= 11, over 8 completions
    assert abs(m["centred_reward_variance"] - 11.0 / 8.0) < TOL


def test_partition_metrics_are_kept_per_prompt():
    groups= [("easy", [5.0, 5.0]), ("hard", [0.0, 2.0])]

    m= partition_metrics(groups)

    assert m["easy"]["informative_rate"] == 0.0
    assert m["hard"]["informative_rate"] == 1.0
    assert abs(m["hard"]["reward_std"] - 1.0) < TOL


def test_difficulty_bins_are_thirds_by_mean_reward():
    by_prompt= {}

    #6 prompts whose mean reward is 0, 1, 2, 3, 4, 5, given out of order
    for name, mean in [("c", 2.0), ("a", 0.0), ("f", 5.0), ("b", 1.0), ("e", 4.0), ("d", 3.0)]:
        by_prompt[name]= make_prompt([mean] * 8)

    bins= difficulty_bins(by_prompt)

    assert bins == {"a": "hard", "b": "hard", "c": "medium", "d": "medium", "e": "easy", "f": "easy"}


def test_length_bin_puts_an_edge_value_in_the_lower_bin():
    edges= [10.0, 20.0, 30.0]

    assert length_bin(3, edges) == 0
    assert length_bin(10, edges) == 0
    assert length_bin(11, edges) == 1
    assert length_bin(30, edges) == 2
    assert length_bin(31, edges) == 3


def test_length_conditioned_weights_and_shares():
    edges= [10.0, 20.0, 30.0]

    #canonical grpo weights: |A| / T
    rows= [
        {"n_tokens": 5, "masked": False, "abs_advantage_times_weight": 1.0 / 5},
        {"n_tokens": 10, "masked": False, "abs_advantage_times_weight": 2.0 / 10},
        {"n_tokens": 40, "masked": False, "abs_advantage_times_weight": 1.0 / 40},
        {"n_tokens": 50, "masked": True, "abs_advantage_times_weight": 3.0 / 50},
    ]

    stats= length_conditioned_stats(rows, edges)

    shortest= stats[0]
    longest= stats[3]

    assert shortest["n_completions"] == 2
    assert longest["n_completions"] == 2
    assert longest["n_masked"] == 1

    #per-token weights: 0.2 and 0.2 in the shortest bin, 0.025 and 0 (masked) in the longest
    assert abs(shortest["per_token_weight_mean"] - 0.2) < TOL
    assert abs(longest["per_token_weight_mean"] - 0.0125) < TOL
    assert abs(shortest["per_token_weight_share"] - 0.4 / 0.425) < TOL

    #per-sequence weights are |A|: 1 and 2 in the shortest bin, 1 and 0 in the longest
    assert abs(shortest["per_sequence_weight_share"] - 3.0 / 4.0) < TOL
    assert abs(longest["per_sequence_weight_share"] - 1.0 / 4.0) < TOL

    #the two middle bins are empty
    assert stats[1]["n_completions"] == 0
    assert stats[1]["per_token_weight_mean"] is None
