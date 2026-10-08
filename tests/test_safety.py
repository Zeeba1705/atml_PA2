from task4_safety.evaluate_safety import agreement_stats
from task4_safety.evaluate_safety import cohen_kappa
from task4_safety.evaluate_safety import policy_summary
from task4_safety.make_audit_sheet import assign_policies
from task4_safety.make_audit_sheet import build_audit

TOL= 1e-9


def make_rows(policy):
    #100 prompts, ids 0-49 SAFE and 50-99 UNSAFE
    rows= []

    for i in range(100):
        rows.append({
            "xstest_id": i,
            "policy": policy,
            "prompt": f"prompt {i}",
            "benchmark_class": "SAFE" if i < 50 else "UNSAFE",
            "type": "homonyms" if i < 50 else "contrast_homonyms",
            "response": f"{policy} answer {i}",
            "response_tokens": 10,
        })

    return rows


def test_kappa_on_a_hand_example():
    #observed agreement 3/4, by chance 0.5*0.25 + 0.5*0.75= 0.5, kappa= 0.25 / 0.5
    manual= ["A", "A", "B", "B"]
    judge= ["A", "B", "B", "B"]

    assert abs(cohen_kappa(manual, judge, ["A", "B"]) - 0.5) < TOL


def test_kappa_is_one_for_identical_labels_and_zero_for_chance():
    assert abs(cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"], ["A", "B"]) - 1.0) < TOL

    #judge says A and B equally often whatever the manual label is
    assert abs(cohen_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"], ["A", "B"]) - 0.0) < TOL


def test_ambiguous_is_reported_separately_and_never_counts_as_agreement():
    manual= ["SAFE_ANSWER", "SAFE_ANSWER", "AMBIGUOUS", "OVER_REFUSAL", "AMBIGUOUS"]
    judge= ["SAFE_ANSWER", "OVER_REFUSAL", "AMBIGUOUS", "AMBIGUOUS", "SAFE_ANSWER"]

    out= agreement_stats(manual, judge)

    assert out["n_examples"] == 5
    assert out["manual_ambiguous"] == 2
    assert out["judge_ambiguous"] == 2
    assert out["n_both_non_ambiguous"] == 2

    #of the two examples with clear labels on both sides one agrees
    assert abs(out["raw_agreement_non_ambiguous"] - 0.5) < TOL

    #over all five, the pair where both said AMBIGUOUS is not an agreement
    assert abs(out["raw_agreement_all"] - 0.2) < TOL

    assert out["confusion_rows_manual_columns_judge"]["AMBIGUOUS"]["AMBIGUOUS"] == 1
    assert out["confusion_rows_manual_columns_judge"]["SAFE_ANSWER"]["OVER_REFUSAL"] == 1


def test_rates_are_conditioned_on_the_benchmark_class():
    rows= []

    #4 safe prompts: 3 answered, 1 refused. 2 unsafe prompts: 1 complied, 1 ambiguous
    for label in ["SAFE_ANSWER", "SAFE_ANSWER", "SAFE_ANSWER", "OVER_REFUSAL"]:
        rows.append({"benchmark_class": "SAFE", "label": label, "rationale_tag": "", "response_tokens": 10})

    rows.append({"benchmark_class": "UNSAFE", "label": "UNSAFE_COMPLIANCE", "rationale_tag": "", "response_tokens": 40})
    rows.append({"benchmark_class": "UNSAFE", "label": "AMBIGUOUS", "rationale_tag": "parse_failure", "response_tokens": 40})

    out= policy_summary(rows, 6304)

    assert abs(out["safe_answer_rate"] - 0.75) < TOL
    assert abs(out["safe_over_refusal_rate"] - 0.25) < TOL
    assert abs(out["unsafe_compliance_rate"] - 0.5) < TOL
    assert abs(out["unsafe_justified_refusal_rate"] - 0.0) < TOL
    assert abs(out["ambiguous_rate"] - 1.0 / 6.0) < TOL
    assert out["judge_parse_failures"] == 1
    assert abs(out["response_tokens_mean"] - 20.0) < TOL


def test_audit_gives_every_policy_the_same_number_of_examples():
    ids= list(range(60))
    classes= {i: ("SAFE" if i < 30 else "UNSAFE") for i in ids}

    assignment= assign_policies(ids, classes, 6304)

    counts= {}

    for i in ids:
        counts[assignment[i]]= counts.get(assignment[i], 0) + 1

    assert counts == {"sft": 15, "dpo": 15, "ppo": 15, "grpo": 15}


def test_audit_sheet_is_balanced_and_hides_the_policy():
    generated= {name: make_rows(name) for name in ["sft", "dpo", "ppo", "grpo"]}

    ids, sheet, key = build_audit(generated, 30, 6304)

    assert len(ids) == 60
    assert len(sheet) == 60
    assert sum(k["benchmark_class"] == "SAFE" for k in key) == 30
    assert sum(k["benchmark_class"] == "UNSAFE" for k in key) == 30

    #the labeller sees these columns only
    assert sorted(sheet[0].keys()) == ["audit_id", "category", "manual_label", "prompt", "response"]
    assert all(row["manual_label"] == "" for row in sheet)

    #the response in each sheet row is the one of the policy named in the hidden key
    for row, k in zip(sheet, key):
        assert row["response"] == f"{k['policy']} answer {k['xstest_id']}"

    #same seed, same sheet
    assert build_audit(generated, 30, 6304)[1] == sheet
