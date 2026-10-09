from task5_feedback.evaluate_math import judge_pairs
from task5_feedback.evaluate_math import pairwise_summary
from task5_feedback.evaluate_math import policy_metrics
from task5_feedback.evaluate_math import verifier_judge_agreement
from task5_feedback.rlvr import exact_reward
from task5_feedback.score_perturbations import judge_outcome
from task5_feedback.score_perturbations import rates
from task5_feedback.score_perturbations import summarize
from task5_feedback.score_perturbations import verifier_outcome

TOL= 1e-9


def make_pair(preference, a_correct, b_correct):
    return {"preference": preference, "a_correct": a_correct, "b_correct": b_correct, "identical_responses": False}


def test_verifier_reads_only_the_designated_final_answer():
    assert exact_reward("2 + 2 = 4\n#### 4", "4") == 1.0
    assert exact_reward("the answer is 4", "4") == 0.0

    #the gold number in the reasoning does not count when the designated final is wrong
    assert exact_reward("maybe 4, but\n#### 5", "4") == 0.0

    #the last designated final is the one that counts
    assert exact_reward("#### 5\nno wait\n#### 1,200", "1200") == 1.0


def test_win_rate_counts_a_tie_as_half():
    pairs= [make_pair("A", True, False), make_pair("A", True, True), make_pair("TIE", False, False), make_pair("B", False, True)]

    out= pairwise_summary(pairs, 6304)

    assert (out["wins"], out["ties"], out["losses"]) == (2, 1, 1)
    assert abs(out["win_rate"] - 2.5 / 4.0) < TOL
    assert abs(out["tie_rate"] - 0.25) < TOL


class NeverCalledJudge:
    def compare(self, problem, a, b):
        raise AssertionError("the judge must not be asked to rank two identical responses")


class AlwaysAJudge:
    def compare(self, problem, a, b):
        return "A"


def test_identical_responses_are_a_tie_without_calling_the_judge():
    a= [{"problem_id": "1", "question": "q", "response": "same text", "correct": True}]
    b= [{"problem_id": "1", "question": "q", "response": "same text", "correct": True}]

    rows, _ = judge_pairs(NeverCalledJudge(), a, b, "rlvr", "sft")

    assert rows[0]["preference"] == "TIE"
    assert rows[0]["identical_responses"] is True

    b[0]["response"]= "other text"

    rows, _ = judge_pairs(AlwaysAJudge(), a, b, "rlvr", "sft")

    assert rows[0]["preference"] == "A"
    assert rows[0]["identical_responses"] is False


def test_agreement_uses_only_pairs_where_exactly_one_response_is_correct():
    pairs= [
        make_pair("A", True, False),    #judge prefers the correct one
        make_pair("B", True, False),    #judge prefers the wrong one
        make_pair("B", False, True),    #judge prefers the correct one
        make_pair("TIE", False, True),  #judge tie
        make_pair("A", True, True),     #both correct, not counted
        make_pair("B", False, False),   #both wrong, not counted
    ]

    out= verifier_judge_agreement(pairs)

    assert out["n_pairs_exactly_one_correct"] == 4
    assert out["judge_prefers_correct"] == 2
    assert out["judge_ties"] == 1
    assert out["judge_prefers_wrong"] == 1
    assert abs(out["agreement"] - 0.5) < TOL


def test_policy_metrics_on_a_hand_example():
    records= [
        {"correct": True, "format_ok": True, "response_tokens": 100, "truncated": False},
        {"correct": False, "format_ok": True, "response_tokens": 200, "truncated": False},
        {"correct": False, "format_ok": False, "response_tokens": 512, "truncated": True},
        {"correct": True, "format_ok": True, "response_tokens": 188, "truncated": False},
    ]

    out= policy_metrics(records, 6304)

    assert abs(out["exact_accuracy"] - 0.5) < TOL
    assert abs(out["format_compliance"] - 0.75) < TOL
    assert abs(out["response_tokens_mean"] - 250.0) < TOL
    assert abs(out["truncated_fraction"] - 0.25) < TOL
    assert abs(out["accuracy_given_format_ok"] - 2.0 / 3.0) < TOL


def test_verifier_ties_when_both_rewards_are_equal():
    assert verifier_outcome(1.0, 0.0) == "better"
    assert verifier_outcome(1.0, 1.0) == "tie"
    assert verifier_outcome(0.0, 1.0) == "wrong"


def test_judge_outcome_is_from_the_clean_responses_side():
    assert judge_outcome("A") == "better"
    assert judge_outcome("TIE") == "tie"
    assert judge_outcome("B") == "wrong"


def make_diag_pairs():
    #2 problems x 4 perturbations. the verifier is ideal: ties on correct-final variants, better on wrong-final ones
    pairs= []

    judge= {
        "corrupt_reasoning_correct_final": ["better", "tie"],
        "good_reasoning_wrong_final": ["better", "better"],
        "persuasive_filler_correct": ["wrong", "tie"],
        "gold_distractor_wrong_final": ["better", "wrong"],
    }

    for variant in judge:
        for k in range(2):
            wrong_final= variant.endswith("wrong_final")

            pairs.append({
                "problem_id": str(k),
                "perturbation": variant,
                "verifier_outcome": "better" if wrong_final else "tie",
                "judge_outcome": judge[variant][k],
            })

    return pairs


def test_rates_per_perturbation():
    pairs= make_diag_pairs()

    out= rates(pairs, "judge", ["persuasive_filler_correct"], 6304)

    assert out["n_pairs"] == 2
    assert abs(out["better_rate"] - 0.0) < TOL
    assert abs(out["tie_rate"] - 0.5) < TOL
    assert abs(out["wrong_rate"] - 0.5) < TOL


def test_s_reason_and_s_outcome():
    summary= summarize(make_diag_pairs(), 6304)

    verifier= summary["sensitivity"]["verifier"]
    judge= summary["sensitivity"]["judge"]

    #a binary verifier cannot see a reasoning-only change, and always sees a changed final answer
    assert abs(verifier["s_reason"] - 0.0) < TOL
    assert abs(verifier["s_reason_tie_rate"] - 1.0) < TOL
    assert abs(verifier["s_outcome"] - 1.0) < TOL

    assert abs(judge["s_reason"] - 0.5) < TOL
    assert abs(judge["s_outcome"] - 1.0) < TOL

    #with the gold-distractor pairs added: 3 of 4 better
    assert abs(judge["s_outcome_all_wrong_final"] - 0.75) < TOL
