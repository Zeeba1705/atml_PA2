import torch

from task3_grpo.continue_train import sequence_loss
from task3_grpo.grpo import grpo_policy_loss

EPS= 0.2
BETA= 0.1
MAX_LEN= 8
TOL= 1e-6


def make_batch():
    torch.manual_seed(0)

    #3 completions of 5, 3 and 4 tokens, right padded to 5. the last one is masked as if truncated
    old_logp= -torch.rand(3, 5)
    new_logp= (old_logp + 0.4 * torch.randn(3, 5)).requires_grad_(True)
    ref_logp= old_logp + 0.1 * torch.randn(3, 5)
    adv= torch.tensor([1.2, -0.7, 0.3])
    mask= torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    return new_logp, old_logp, ref_logp, adv, mask


def accumulated_loss(new_logp, old_logp, ref_logp, adv, mask, loss_type):
    #the way the training loop does it: one completion at a time, masked ones skipped
    total= 0.0

    for k in range(new_logp.shape[0]):
        if mask[k].sum().item() == 0:
            continue

        loss, _, _, _ = sequence_loss(
            new_logp[k:k + 1],
            old_logp[k:k + 1],
            adv[k:k + 1],
            mask[k:k + 1],
            ref_logp[k:k + 1],
            EPS,
            BETA,
            loss_type,
            MAX_LEN,
            new_logp.shape[0],
            mask.sum().clamp_min(1.0)
        )

        total= total + loss

    return total


def test_one_completion_at_a_time_gives_the_released_batch_loss():
    for loss_type in ["grpo", "dr_grpo"]:
        new_logp, old_logp, ref_logp, adv, mask = make_batch()

        batch_loss, _ = grpo_policy_loss(
            new_logp,
            old_logp,
            adv,
            mask,
            ref_logp,
            EPS,
            BETA,
            loss_type,
            MAX_LEN
        )

        total= accumulated_loss(new_logp, old_logp, ref_logp, adv, mask, loss_type)

        assert abs(total.item() - batch_loss.item()) < TOL


def test_one_completion_at_a_time_gives_the_released_batch_gradient():
    for loss_type in ["grpo", "dr_grpo"]:
        new_logp, old_logp, ref_logp, adv, mask = make_batch()

        batch_loss, _ = grpo_policy_loss(
            new_logp,
            old_logp,
            adv,
            mask,
            ref_logp,
            EPS,
            BETA,
            loss_type,
            MAX_LEN
        )

        batch_grad= torch.autograd.grad(batch_loss, new_logp)[0]

        total= accumulated_loss(new_logp, old_logp, ref_logp, adv, mask, loss_type)

        total_grad= torch.autograd.grad(total, new_logp)[0]

        assert torch.allclose(total_grad, batch_grad, atol=TOL)

        #the masked completion gets no gradient at all
        assert torch.equal(total_grad[2], torch.zeros(5))
