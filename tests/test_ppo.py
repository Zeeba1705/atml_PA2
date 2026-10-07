import math
import torch

from task2_ppo.ppo import compute_gae
from task2_ppo.ppo import normalize_advantages
from task2_ppo.ppo import ppo_policy_loss
from task2_ppo.ppo import shaped_rewards
from task2_ppo.ppo import value_mse_loss

EPS= 0.2
TOL= 1e-5


def hand_clip_objective(ratio, advantage, eps):
    #written straight from the handout, one token at a time: min(rho*A, clip(rho, 1-eps, 1+eps)*A)
    clipped= min(max(ratio, 1.0 - eps), 1.0 + eps)

    return min(ratio * advantage, clipped * advantage)


def hand_gae(rewards, values, n_valid, gamma, lam):
    #forward sum from the handout, A_t = sum_k (gamma*lam)^k delta_{t+k}, V after the last valid token is 0
    deltas= []

    for t in range(n_valid):
        next_value= values[t + 1] if t + 1 < n_valid else 0.0

        deltas.append(rewards[t] + gamma * next_value - values[t])

    advantages= []

    for t in range(n_valid):
        adv= 0.0

        for k in range(n_valid - t):
            adv += (gamma * lam) ** k * deltas[t + k]

        advantages.append(adv)

    return advantages


def one_token_loss(ratio, advantage):
    #old_logp= 0 so exp(new_logp - old_logp) is exactly the ratio we want
    new_logp= torch.tensor([[math.log(ratio)]], requires_grad=True)
    old_logp= torch.zeros(1, 1)

    loss, _, clip_fraction = ppo_policy_loss(
        new_logp,
        old_logp,
        torch.tensor([[advantage]]),
        torch.ones(1, 1),
        EPS
    )

    return loss, new_logp, clip_fraction


def test_ratio_inside_range_is_unclipped():
    for advantage in [2.0, -2.0]:
        for ratio in [0.9, 1.0, 1.1]:
            loss, _, clip_fraction = one_token_loss(ratio, advantage)

            assert abs(loss.item() - (-ratio * advantage)) < TOL
            assert clip_fraction.item() == 0.0


def test_positive_advantage_ratio_above_range_is_clipped():
    #A= 2, rho= 1.5: min(3.0, 1.2*2= 2.4)= 2.4
    loss, _, clip_fraction = one_token_loss(1.5, 2.0)

    assert abs(loss.item() - (-2.4)) < TOL
    assert clip_fraction.item() == 1.0


def test_positive_advantage_ratio_below_range_is_not_clipped():
    #A= 2, rho= 0.5: min(1.0, 0.8*2= 1.6)= 1.0
    loss, _, clip_fraction = one_token_loss(0.5, 2.0)

    assert abs(loss.item() - (-1.0)) < TOL
    assert clip_fraction.item() == 1.0


def test_negative_advantage_ratio_below_range_is_clipped():
    #A= -2, rho= 0.5: min(-1.0, 0.8*-2= -1.6)= -1.6
    loss, _, clip_fraction = one_token_loss(0.5, -2.0)

    assert abs(loss.item() - 1.6) < TOL
    assert clip_fraction.item() == 1.0


def test_negative_advantage_ratio_above_range_is_not_clipped():
    #A= -2, rho= 1.5: min(-3.0, 1.2*-2= -2.4)= -3.0
    loss, _, clip_fraction = one_token_loss(1.5, -2.0)

    assert abs(loss.item() - 3.0) < TOL
    assert clip_fraction.item() == 1.0


def test_no_gradient_once_the_ratio_moved_past_the_clip_in_the_good_direction():
    #the point of clipping: no more reward for pushing a good token above 1+eps or a bad token below 1-eps
    for ratio, advantage in [(1.5, 2.0), (0.5, -2.0)]:
        loss, new_logp, _ = one_token_loss(ratio, advantage)

        loss.backward()

        assert abs(new_logp.grad.item()) < TOL


def test_gradient_kept_when_the_ratio_moved_in_the_bad_direction():
    #d(-rho*A)/d(new_logp)= -rho*A, the update must still be able to undo a bad move
    for ratio, advantage in [(0.5, 2.0), (1.5, -2.0)]:
        loss, new_logp, _ = one_token_loss(ratio, advantage)

        loss.backward()

        assert abs(new_logp.grad.item() - (-ratio * advantage)) < TOL


def test_policy_loss_matches_hand_computed_batch():
    ratios= [[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]]
    advantages= [[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]]

    loss, ratio, clip_fraction = ppo_policy_loss(
        torch.log(torch.tensor(ratios)),
        torch.zeros(2, 3),
        torch.tensor(advantages),
        torch.ones(2, 3),
        EPS
    )

    total= 0.0

    for b in range(2):
        for t in range(3):
            total += hand_clip_objective(ratios[b][t], advantages[b][t], EPS)

    #(1.0 + 2.0 + 2.4 - 1.6 - 2.0 - 3.0) / 6= -0.2, loss is minus that
    assert abs(total / 6 - (-0.2)) < TOL
    assert abs(loss.item() - 0.2) < TOL

    assert torch.allclose(ratio, torch.tensor(ratios), atol=TOL)

    #4 of the 6 ratios are outside [0.8, 1.2] before clipping
    assert abs(clip_fraction.item() - 4.0 / 6.0) < TOL


def test_policy_loss_ignores_masked_tokens():
    #row 1 has one padded position with junk values in it
    new_logp= torch.log(torch.tensor([[0.5, 1.0, 1.5], [1.5, 0.9, 5.0]]))
    advantage= torch.tensor([[2.0, 2.0, 2.0], [-2.0, -2.0, 100.0]])
    mask= torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    loss, _, clip_fraction = ppo_policy_loss(
        new_logp,
        torch.zeros(2, 3),
        advantage,
        mask,
        EPS
    )

    #valid tokens: 1.0, 2.0, 2.4, -3.0, -1.8 -> mean -0.12 over 5 tokens
    assert abs(loss.item() - (-(1.0 + 2.0 + 2.4 - 3.0 - 1.8) / 5.0)) < TOL

    #outside the range among valid tokens: 0.5, 1.5, 1.5 -> 3 of 5
    assert abs(clip_fraction.item() - 3.0 / 5.0) < TOL


def test_gae_three_steps_with_a_padded_row():
    gamma= 0.9
    lam= 0.5

    #row 0 has 3 valid tokens, row 1 has 2 and junk in the padded position
    rewards= torch.tensor([[1.0, 0.0, 2.0], [0.5, 1.0, 9.0]])
    values= torch.tensor([[0.5, 1.0, -0.5], [0.2, 0.4, 7.0]])
    mask= torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    advantages, returns = compute_gae(
        rewards,
        values,
        mask,
        gamma,
        lam
    )

    #row 0: delta= [1.4, -1.45, 2.5]
    #A2= 2.5, A1= -1.45 + 0.45*2.5= -0.325, A0= 1.4 + 0.45*-0.325= 1.25375
    #row 1: delta= [0.66, 0.6], A1= 0.6, A0= 0.66 + 0.45*0.6= 0.93, padded position= 0
    expected= torch.tensor([[1.25375, -0.325, 2.5], [0.93, 0.6, 0.0]])

    assert torch.allclose(advantages, expected, atol=TOL)

    #same numbers from the forward sum in the handout
    row0= hand_gae([1.0, 0.0, 2.0], [0.5, 1.0, -0.5], 3, gamma, lam)
    row1= hand_gae([0.5, 1.0], [0.2, 0.4], 2, gamma, lam)

    assert torch.allclose(advantages[0], torch.tensor(row0), atol=TOL)
    assert torch.allclose(advantages[1, :2], torch.tensor(row1), atol=TOL)

    #returns= advantages + values on the valid tokens
    assert torch.allclose(returns[0], torch.tensor([1.75375, 0.675, 2.0]), atol=TOL)
    assert torch.allclose(returns[1, :2], torch.tensor([1.13, 1.0]), atol=TOL)


def test_gae_padded_values_dont_leak_into_valid_tokens():
    rewards= torch.tensor([[0.5, 1.0, 9.0]])
    values= torch.tensor([[0.2, 0.4, 7.0]])
    mask= torch.tensor([[1.0, 1.0, 0.0]])

    advantages, _ = compute_gae(rewards, values, mask, 0.9, 0.5)

    changed_rewards= torch.tensor([[0.5, 1.0, -3.0]])
    changed_values= torch.tensor([[0.2, 0.4, 55.0]])

    changed_advantages, _ = compute_gae(changed_rewards, changed_values, mask, 0.9, 0.5)

    assert torch.allclose(advantages, changed_advantages, atol=TOL)


def test_shaped_reward_puts_task_reward_only_on_last_valid_token():
    beta_kl= 0.1

    task_reward= torch.tensor([3.0, -1.0])
    policy_logp= torch.tensor([[-1.0, -2.0, -0.5], [-1.5, -0.2, -4.0]])
    ref_logp= torch.tensor([[-1.5, -1.0, -0.5], [-1.0, -1.2, -9.0]])
    mask= torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    rewards= shaped_rewards(
        task_reward,
        policy_logp,
        ref_logp,
        mask,
        beta_kl
    )

    #kl part= -0.1 * (policy - ref): row 0 [-0.05, 0.1, 0.0], row 1 [0.05, -0.1, masked]
    #task reward goes on index 2 for row 0 and index 1 for row 1
    expected= torch.tensor([[-0.05, 0.1, 3.0], [0.05, -1.1, 0.0]])

    assert torch.allclose(rewards, expected, atol=TOL)


def test_shaped_reward_with_zero_beta_is_only_the_terminal_reward():
    task_reward= torch.tensor([3.0, -1.0])
    policy_logp= torch.tensor([[-1.0, -2.0, -0.5], [-1.5, -0.2, -4.0]])
    ref_logp= torch.tensor([[-1.5, -1.0, -0.5], [-1.0, -1.2, -9.0]])
    mask= torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    rewards= shaped_rewards(
        task_reward,
        policy_logp,
        ref_logp,
        mask,
        0.0
    )

    expected= torch.tensor([[0.0, 0.0, 3.0], [0.0, -1.0, 0.0]])

    assert torch.allclose(rewards, expected, atol=TOL)


def test_value_loss_is_masked_mean_squared_error():
    predicted= torch.tensor([[1.0, 2.0, 50.0]])
    returns= torch.tensor([[0.0, 4.0, 0.0]])
    mask= torch.tensor([[1.0, 1.0, 0.0]])

    loss= value_mse_loss(predicted, returns, mask)

    #(1 + 4) / 2 valid tokens
    assert abs(loss.item() - 2.5) < TOL


def test_normalized_advantages_use_valid_tokens_only():
    advantages= torch.tensor([[1.0, 3.0, 100.0]])
    mask= torch.tensor([[1.0, 1.0, 0.0]])

    normed= normalize_advantages(advantages, mask)

    #valid mean= 2, population std= 1
    assert torch.allclose(normed, torch.tensor([[-1.0, 1.0, 0.0]]), atol=TOL)
