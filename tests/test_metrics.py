import numpy as np

from common.metrics import bootstrap_ci
from common.metrics import paired_bootstrap_ci

TOL= 1e-9


def test_bootstrap_of_constant_values_is_that_value():
    low, high = bootstrap_ci([3.0, 3.0, 3.0, 3.0])

    assert abs(low - 3.0) < TOL
    assert abs(high - 3.0) < TOL


def test_bootstrap_is_the_same_for_the_same_seed():
    values= [0.1, 0.9, 0.4, 0.7, 0.2, 0.5]

    assert bootstrap_ci(values, None, 6304) == bootstrap_ci(values, None, 6304)
    assert bootstrap_ci(values, None, 6304) != bootstrap_ci(values, None, 1)


def test_bootstrap_interval_contains_the_mean():
    values= np.arange(50) / 10.0

    low, high = bootstrap_ci(values)

    assert low < values.mean() < high


def test_weighted_bootstrap_is_a_ratio_of_sums():
    #every prompt has the same per-token value 0.5, whatever its length, so the token mean is always 0.5
    n_tokens= [10.0, 200.0, 35.0, 4.0]
    sums= [0.5 * n for n in n_tokens]

    low, high = bootstrap_ci(sums, n_tokens)

    assert abs(low - 0.5) < TOL
    assert abs(high - 0.5) < TOL


def test_paired_bootstrap_uses_the_per_prompt_difference():
    #b is always exactly 1 below a, so the difference has no spread even though a and b do
    values_a= [0.3, 2.0, -1.0, 5.5]
    values_b= [v - 1.0 for v in values_a]

    low, high = paired_bootstrap_ci(values_a, values_b)

    assert abs(low - 1.0) < TOL
    assert abs(high - 1.0) < TOL
