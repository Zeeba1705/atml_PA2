from task2_ppo.find_candidates import classify
from task2_ppo.find_candidates import max_ngram_repeats
from task2_ppo.find_candidates import quality_flags

BASE= {"reward": 1.0, "n_tokens": 100, "has_eos": True, "response": "a short baseline answer"}


def make_record(reward, n_tokens, has_eos=True, response="a different answer"):
    return {"reward": reward, "n_tokens": n_tokens, "has_eos": has_eos, "response": response}


def test_ngram_repeats_counts_the_most_repeated_run():
    phrase= "one two three four five six seven eight"

    assert max_ngram_repeats(phrase, 8) == 1
    assert max_ngram_repeats(phrase + " filler " + phrase + " filler " + phrase, 8) == 3
    assert max_ngram_repeats("too short", 8) == 0


def test_much_longer_starts_at_one_and_a_half_times_the_baseline():
    assert quality_flags(make_record(2.0, 149), BASE) == []
    assert quality_flags(make_record(2.0, 150), BASE) == ["much_longer"]


def test_no_eos_and_repetition_are_flagged():
    phrase= "one two three four five six seven eight"
    repeated= phrase + " x " + phrase + " y " + phrase

    assert quality_flags(make_record(2.0, 100, has_eos=False), BASE) == ["no_eos"]
    assert quality_flags(make_record(2.0, 100, response=repeated), BASE) == ["repeated_ngram"]


def test_lower_or_equal_reward_is_never_a_candidate():
    assert classify(make_record(1.0, 400, has_eos=False), BASE) == (None, [])
    assert classify(make_record(0.5, 120), BASE) == (None, [])


def test_higher_reward_with_a_flag_is_suspect():
    kind, flags = classify(make_record(2.0, 300), BASE)

    assert kind == "suspect"
    assert flags == ["much_longer"]


def test_higher_reward_somewhat_longer_and_clean_is_agree():
    assert classify(make_record(2.0, 120), BASE) == ("agree", [])


def test_higher_reward_but_not_longer_is_neither():
    assert classify(make_record(2.0, 100), BASE) == (None, [])
    assert classify(make_record(2.0, 80), BASE) == (None, [])
