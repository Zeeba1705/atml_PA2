import math
import torch
import torch.nn.functional as F

from types import SimpleNamespace

from common.generation import response_sequence_logprobs
from task1_dpo.dpo import dpo_loss

LOG2= math.log(2.0)
TOL= 1e-5


def hand_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta):
    #written straight from the handout equation, one pair at a time
    losses= []

    for i in range(len(pol_chosen)):
        m= (pol_chosen[i] - ref_chosen[i]) - (pol_rejected[i] - ref_rejected[i])

        losses.append(-torch.log(torch.sigmoid(beta * m)))

    return sum(losses) / len(losses)


def fixed_logits_model(logits):
    #stands in for the LM, returns the same logits whatever the input ids are
    def model(input_ids, attention_mask, use_cache, return_dict):
        return SimpleNamespace(logits=logits)

    return model


def test_identical_policy_and_ref_gives_log2():
    chosen= torch.tensor([-10.0, -20.0])
    rejected= torch.tensor([-14.0, -19.0])

    loss, _ = dpo_loss(
        chosen,
        rejected,
        chosen.clone(),
        rejected.clone(),
        0.1
    )

    assert abs(loss.item() - LOG2) < TOL


def test_matches_hand_computed_loss():
    pol_chosen= torch.tensor([-10.0, -20.0])
    pol_rejected= torch.tensor([-14.0, -19.0])
    ref_chosen= torch.tensor([-12.0, -21.0])
    ref_rejected= torch.tensor([-13.0, -18.0])

    loss, _ = dpo_loss(
        pol_chosen,
        pol_rejected,
        ref_chosen,
        ref_rejected,
        0.1
    )

    expected= hand_loss(
        pol_chosen,
        pol_rejected,
        ref_chosen,
        ref_rejected,
        0.1
    )

    assert abs(loss.item() - expected.item()) < TOL


def test_chosen_favoured_gives_loss_below_log2():
    #policy raised chosen by 2 nats over the reference, so m= 2
    loss, _ = dpo_loss(
        torch.tensor([-18.0]),
        torch.tensor([-10.0]),
        torch.tensor([-20.0]),
        torch.tensor([-10.0]),
        0.1
    )

    assert loss.item() < LOG2
    assert abs(loss.item() - (-math.log(1.0 / (1.0 + math.exp(-0.2))))) < TOL


def test_rejected_favoured_gives_loss_above_log2():
    #policy raised rejected by 2 nats over the reference, so m= -2
    loss, _ = dpo_loss(
        torch.tensor([-10.0]),
        torch.tensor([-18.0]),
        torch.tensor([-10.0]),
        torch.tensor([-20.0]),
        0.1
    )

    assert loss.item() > LOG2
    assert abs(loss.item() - (-math.log(1.0 / (1.0 + math.exp(0.2))))) < TOL


def test_beta_scales_the_margin_inside_the_sigmoid():
    pol_chosen= torch.tensor([-10.0, -20.0])
    pol_rejected= torch.tensor([-14.0, -19.0])
    ref_chosen= torch.tensor([-12.0, -21.0])
    ref_rejected= torch.tensor([-13.0, -18.0])

    for beta in [0.03, 0.1, 0.3]:
        loss, _ = dpo_loss(
            pol_chosen,
            pol_rejected,
            ref_chosen,
            ref_rejected,
            beta
        )

        expected= hand_loss(
            pol_chosen,
            pol_rejected,
            ref_chosen,
            ref_rejected,
            beta
        )

        assert abs(loss.item() - expected.item()) < TOL


def test_masked_tokens_dont_matter():
    torch.manual_seed(0)

    #batch 2, length 6, vocab 5
    logits= torch.randn(2, 6, 5)
    model= fixed_logits_model(logits)

    #row 0 is left padded by one token, last two tokens are the response
    #row 1 has no padding, last four tokens are the response
    input_ids= torch.tensor([
        [0, 1, 2, 3, 4, 1],
        [2, 3, 1, 0, 4, 2]
    ])
    attention_mask= torch.tensor([
        [0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]
    ])
    response_mask= torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    ])

    batch= {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask
    }

    seq_logp, _, _ = response_sequence_logprobs(model, batch)

    #token t is predicted by the logits at t-1, summed over response tokens only
    logp= F.log_softmax(logits, dim=-1)
    expected= torch.zeros(2)

    for b in range(2):
        for t in range(1, 6):
            if response_mask[b, t] == 1.0:
                expected[b] += logp[b, t - 1, input_ids[b, t]]

    assert torch.allclose(seq_logp, expected, atol=TOL)

    #change every prompt and padding token id, the response sum must not move
    changed_ids= input_ids.clone()
    changed_ids[response_mask == 0.0] = (changed_ids[response_mask == 0.0] + 1) % 5

    changed_batch= {
        "input_ids": changed_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask
    }

    changed_logp, _, _ = response_sequence_logprobs(model, changed_batch)

    assert torch.allclose(changed_logp, seq_logp, atol=TOL)
