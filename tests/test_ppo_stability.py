import math

from task2_ppo.analyze_clipping import delta_kls
from task2_ppo.analyze_clipping import run_name_for
from task2_ppo.analyze_clipping import stability_stats

TOL= 1e-9


def test_delta_kls_reads_the_logged_change_per_update():
    log_rows= [
        {"update": 1, "kl": 0.10, "kl_after_update": 0.13, "delta_kl": 0.03},
        {"update": 2, "kl": 0.20, "kl_after_update": 0.19, "delta_kl": -0.01},
    ]

    assert delta_kls(log_rows) == [0.03, -0.01]


def test_s1_is_the_population_std_of_delta_kl():
    #mean= 0.02, squared deviations= 0.0001, 0.0009, 0.0004, 0.0000 -> variance 0.00035
    deltas= [0.03, -0.01, 0.04, 0.02]

    stats= stability_stats(deltas, 1.0)

    assert abs(stats["s1_std_delta_kl"] - math.sqrt(0.00035)) < TOL


def test_s2_counts_updates_above_the_threshold_in_absolute_value():
    deltas= [0.03, -0.05, 0.04, 0.02, -0.01]

    #|delta| above 0.035: -0.05 and 0.04
    stats= stability_stats(deltas, 0.035)

    assert stats["s2_updates_above_threshold"] == 2


def test_s2_does_not_count_a_change_exactly_at_the_threshold():
    stats= stability_stats([0.5, -0.5, 0.25], 0.5)

    assert stats["s2_updates_above_threshold"] == 0


def test_fork_run_names():
    assert run_name_for(0.05) == "clip_0.05"
    assert run_name_for(0.20) == "clip_0.2"
    assert run_name_for(0.50) == "clip_0.5"
