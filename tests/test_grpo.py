import math
import torch

from task3_grpo.grpo import group_relative_advantages
from task3_grpo.grpo import grpo_policy_loss
from task3_grpo.grpo import mask_truncated_sequences

EPS= 0.2
ADV_EPS= 1e-6
TOL= 1e-5


def hand_advantages(rewards, group_ids, eps):
    #written straight from the handout, one prompt group at a time: (r - mean of the group) / (std of the group + eps)
    out= [0.0] * len(rewards)

    for g in set(group_ids):
        idx= [i for i in range(len(rewards)) if group_ids[i] == g]
        mean= sum(rewards[i] for i in idx) / len(idx)
        std= math.sqrt(sum((rewards[i] - mean) ** 2 for i in idx) / len(idx))

        for i in idx:
            out[i]= (rewards[i] - mean) / (std + eps)

    return out


def one_token_loss(ratio, advantage):
    #old_logp= 0 so exp(new_logp - old_logp) is exactly the ratio, ref= new so the kl term is 0
    new_logp= torch.tensor([[math.log(ratio)]])

    loss, diag = grpo_policy_loss(
        new_logp,
        torch.zeros(1, 1),
        torch.tensor([advantage]),
        torch.ones(1, 1),
        new_logp.clone(),
        EPS,
        0.1
    )

    return loss, diag


def test_advantages_use_each_prompt_groups_own_mean_and_std():
    #group 0: mean 2, std 1 -> -1, +1. group 1: every reward equal -> 0, 0
    rewards= [1.0, 3.0, 10.0, 10.0]
    group_ids= [0, 0, 1, 1]

    adv= group_relative_advantages(
        torch.tensor(rewards),
        torch.tensor(group_ids),
        ADV_EPS
    )

    assert torch.allclose(adv, torch.tensor([-1.0, 1.0, 0.0, 0.0]), atol=TOL)
    assert torch.allclose(adv, torch.tensor(hand_advantages(rewards, group_ids, ADV_EPS)), atol=TOL)


def test_advantages_with_interleaved_groups():
    #group 0= rewards 1, 2, 3, 6: mean 3, std sqrt(3.5). group 1= rewards 4, 8: mean 6, std 2
    rewards= [1.0, 4.0, 2.0, 3.0, 8.0, 6.0]
    group_ids= [0, 1, 0, 0, 1, 0]

    adv= group_relative_advantages(
        torch.tensor(rewards),
        torch.tensor(group_ids),
        ADV_EPS
    )

    s= math.sqrt(3.5)
    expected= torch.tensor([-2.0 / s, -1.0, -1.0 / s, 0.0, 1.0, 3.0 / s])

    assert torch.allclose(adv, expected, atol=TOL)
    assert torch.allclose(adv, torch.tensor(hand_advantages(rewards, group_ids, ADV_EPS)), atol=TOL)


def test_advantages_of_a_zero_std_group_are_zero_and_finite():
    adv= group_relative_advantages(
        torch.tensor([5.0, 5.0, 5.0, 5.0, 1.0, 2.0]),
        torch.tensor([0, 0, 0, 0, 1, 1]),
        ADV_EPS
    )

    assert torch.isfinite(adv).all()
    assert torch.allclose(adv[:4], torch.zeros(4), atol=TOL)
    assert torch.allclose(adv[4:], torch.tensor([-1.0, 1.0]), atol=TOL)


def test_advantages_with_a_single_group():
    #one prompt per update is one group, so group and batch statistics are the same thing here
    rewards= [1.0, 2.0, 3.0, 6.0]

    adv= group_relative_advantages(
        torch.tensor(rewards),
        torch.tensor([7, 7, 7, 7]),
        ADV_EPS
    )

    assert torch.allclose(adv, torch.tensor(hand_advantages(rewards, [7, 7, 7, 7], ADV_EPS)), atol=TOL)


def test_clipped_term_for_positive_advantage():
    #A= 2: inside the range min(rho*A, rho*A), above it min(3.0, 2.4)= 2.4, below it min(1.0, 1.6)= 1.0
    for ratio, expected in [(1.1, 2.2), (1.5, 2.4), (0.5, 1.0)]:
        loss, _ = one_token_loss(ratio, 2.0)

        assert abs(loss.item() - (-expected)) < TOL


def test_clipped_term_for_negative_advantage():
    #A= -2: inside the range rho*A, below it min(-1.0, -1.6)= -1.6, above it min(-3.0, -2.4)= -3.0
    for ratio, expected in [(0.9, -1.8), (0.5, -1.6), (1.5, -3.0)]:
        loss, _ = one_token_loss(ratio, -2.0)

        assert abs(loss.item() - (-expected)) < TOL


def test_clip_fraction_counts_ratios_outside_the_range():
    new_logp= torch.log(torch.tensor([[0.5, 1.0, 1.5, 1.1]]))

    _, diag = grpo_policy_loss(
        new_logp,
        torch.zeros(1, 4),
        torch.tensor([1.0]),
        torch.ones(1, 4),
        new_logp.clone(),
        EPS,
        0.1
    )

    assert abs(diag["clip_fraction"].item() - 0.5) < TOL


def test_canonical_loss_divides_each_sequence_by_its_own_length():
    #sequence 0 has 2 tokens and A= 1, sequence 1 has 4 tokens and A= -0.5, every ratio is 1
    new_logp= torch.zeros(2, 4)
    mask= torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

    loss, _ = grpo_policy_loss(
        new_logp,
        torch.zeros(2, 4),
        torch.tensor([1.0, -0.5]),
        mask,
        new_logp.clone(),
        EPS,
        0.1,
        "grpo"
    )

    #per sequence: 2*1/2= 1 and 4*-0.5/4= -0.5, mean 0.25, loss is minus that
    assert abs(loss.item() - (-0.25)) < TOL


def test_dr_grpo_loss_divides_every_sequence_by_the_same_constant():
    new_logp= torch.zeros(2, 4)
    mask= torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

    loss, _ = grpo_policy_loss(
        new_logp,
        torch.zeros(2, 4),
        torch.tensor([1.0, -0.5]),
        mask,
        new_logp.clone(),
        EPS,
        0.1,
        "dr_grpo",
        8
    )

    #token sums 2*1= 2 and 4*-0.5= -2, each over 8: 0.25 and -0.25, mean 0
    #the longer sequence now weighs as much as the shorter one, in the canonical loss it weighed half as much
    assert abs(loss.item() - 0.0) < TOL


def test_kl_term_uses_the_reference_and_beta():
    #policy log-probs are 0.5 above the reference on every token, A= 0 so only the kl term is left
    new_logp= torch.full((1, 3), -1.0)
    ref_logp= torch.full((1, 3), -1.5)

    loss, diag = grpo_policy_loss(
        new_logp,
        new_logp.clone(),
        torch.tensor([0.0]),
        torch.ones(1, 3),
        ref_logp,
        EPS,
        0.1
    )

    #released estimator per token: exp(ref - new) - (ref - new) - 1
    kl= math.exp(-0.5) + 0.5 - 1.0

    assert abs(diag["sampled_kl"].item() - kl) < TOL
    assert abs(loss.item() - 0.1 * kl) < TOL


def test_truncated_sequences_are_fully_masked():
    mask= torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    masked= mask_truncated_sequences(mask, [True, False, True])

    assert torch.equal(masked, torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))


def test_a_masked_sequence_adds_nothing_to_the_loss():
    #row 0 is truncated and has junk in it, row 1 has 2 tokens with ratio 1 and A= 1
    new_logp= torch.tensor([[2.0, -3.0, 1.0], [0.0, 0.0, 0.0]])
    mask= mask_truncated_sequences(
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        [True, False]
    )

    loss, _ = grpo_policy_loss(
        new_logp,
        torch.zeros(2, 3),
        torch.tensor([50.0, 1.0]),
        mask,
        torch.zeros(2, 3),
        EPS,
        0.1
    )

    #per sequence: 0 for the masked row and 1 for the other, mean over the 2 sequences= 0.5
    assert abs(loss.item() - (-0.5)) < TOL
