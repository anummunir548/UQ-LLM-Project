"""
mc_dropout.py
--------------
Theory
------
Standard inference gives you ONE prediction:

    P(y | x)

MC Dropout asks: what if dropout (randomly zeroing neurons), which is
normally OFF at inference time, stayed ON? Each forward pass now
"routes" through a slightly different random sub-network, so you get
a slightly different prediction each time. Do this T times:

    Run 1 -> P_1(y | x)
    Run 2 -> P_2(y | x)
    ...
    Run T -> P_T(y | x)

Two numbers fall out of this:

    mean       = (1/T) * sum_t P_t(y | x)        -> the actual prediction
    variance   = (1/T) * sum_t (P_t - mean)^2     -> epistemic uncertainty

Why "epistemic"? Because this variance captures uncertainty due to the
MODEL not being sure which sub-network is "correct" -- i.e. uncertainty
from limited training data / model knowledge. This is DIFFERENT from
"aleatoric" uncertainty (irreducible noise in the data itself, e.g. a
genuinely ambiguous sentence). MC Dropout only estimates the former.

We also compute predictive entropy, a second common uncertainty
measure that works directly off the MEAN probability distribution
(not the per-sample variance):

    H[P(y|x)] = - sum_k P(y=k|x) * log P(y=k|x)

Entropy is high when the mean probability is close to uniform (e.g.
[0.5, 0.5] for binary classification -> maximally uncertain) and low
when it's close to one-hot (e.g. [0.99, 0.01] -> confident). Unlike
variance, entropy is well-defined even for a SINGLE deterministic
forward pass -- but combining it with MC Dropout's mean gives you a
version of entropy that already reflects the model's epistemic doubt.
"""

import torch
import torch.nn.functional as F

import config


def enable_dropout(model):
    """
    Force every Dropout submodule into train() mode while leaving
    everything else (BatchNorm, etc.) in eval() mode. This is the key
    trick that makes MC Dropout work: model.eval() would normally
    freeze dropout, defeating the whole method.
    """
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model, input_ids, attention_mask, num_samples=30):
    """
    Run T stochastic forward passes over an already-tokenized batch
    and return the mean probability, standard deviation, predictive
    entropy, and the raw per-sample probabilities.

    Parameters
    ----------
    model : a HuggingFace sequence classification model (already on device)
    input_ids, attention_mask : tensors, already on device, shape [batch, seq_len]
    num_samples : int, number of stochastic forward passes (T)

    Returns
    -------
    dict with keys: mean, std, entropy, all_probs
        mean         : [batch, num_classes]  -- the actual prediction to report
        std          : [batch, num_classes]  -- per-class epistemic uncertainty
        entropy      : [batch]               -- scalar uncertainty per example
        all_probs    : [num_samples, batch, num_classes] -- raw samples, for plotting
    """
    model.eval()          # freeze BatchNorm etc.
    enable_dropout(model)  # ...but force dropout back on

    all_probs = []

    for _ in range(num_samples):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        all_probs.append(probs)

    all_probs = torch.stack(all_probs, dim=0)  # [T, batch, num_classes]

    mean = all_probs.mean(dim=0)
    std = all_probs.std(dim=0)

    # Predictive entropy computed from the MEAN distribution.
    # Add a tiny epsilon to avoid log(0) for near-one-hot predictions.
    eps = 1e-12
    entropy = -(mean * torch.log(mean + eps)).sum(dim=1)

    return {
        "mean": mean,
        "std": std,
        "entropy": entropy,
        "all_probs": all_probs,
    }


"""
Exercise (Phase 2)
-------------------
1. Run mc_dropout_predict with num_samples=5 vs num_samples=100 on the
   same input. How much does `std` change? At what point does adding
   more samples stop meaningfully changing the estimate? (This is the
   "T" you'd justify in a paper's experimental setup section.)
2. Find one test example where `entropy` is high AND the model's
   prediction is WRONG, and one where entropy is low and the model is
   correct. This pairing -- "uncertain AND wrong" vs "confident AND
   right" -- is the qualitative story every UQ paper tells before the
   quantitative one (Phase 5/6).

Expected output: this file has no __main__ block -- it's imported by
experiment_mc_dropout.py, which loads the trained checkpoint and runs
this over the real test set.
"""