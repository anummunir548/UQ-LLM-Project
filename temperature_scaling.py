"""
temperature_scaling.py
------------------------
Theory
------
Temperature scaling is the simplest post-hoc calibration fix that
exists: divide every logit by a single learned scalar T ("temperature")
before applying softmax:

    calibrated_probs = softmax(logits / T)

Why does dividing by a number fix calibration? Softmax squashes logits
into probabilities, and its "sharpness" depends on the SCALE of the
logits. If T > 1, dividing shrinks the logits toward zero, which
flattens the softmax output -- probabilities move closer to uniform,
making the model LESS confident across the board. If T < 1, the
opposite happens -- probabilities get pushed toward 0/1, MORE
confident. T = 1 is a no-op (unchanged).

Crucially, temperature scaling changes ONLY confidence, never the
model's actual prediction (argmax(logits) == argmax(logits / T) for
any T > 0) -- so accuracy is completely unaffected. This is a purely
"how confident should I sound" adjustment, not a "am I right" fix.

T is found by minimizing negative log-likelihood (equivalently,
cross-entropy) on a held-out validation set:

    T* = argmin_T  NLL(softmax(logits / T), labels)

This file implements that optimization via a simple 1D search rather
than requiring a full autograd-based optimizer, since T is a single
scalar and NLL as a function of T is smooth and unimodal in practice
-- easy to search directly.
"""

import torch
import torch.nn.functional as F


def fit_temperature(logits, labels, t_min=0.1, t_max=5.0, n_steps=200):
    """
    Grid search over candidate temperatures to find the one that
    minimizes negative log-likelihood on the given (logits, labels).

    IMPORTANT: in a proper setup, `logits`/`labels` here should come
    from a held-out VALIDATION set, separate from both the training
    set (used to fit the model) and the test set (used for final
    reporting) -- fitting T on the test set and then evaluating
    calibration on the same test set is a form of data leakage. Given
    our current small dataset, we don't have a separate validation
    split yet -- see the exercise below.

    Returns
    -------
    best_T : float
    """
    candidate_Ts = torch.linspace(t_min, t_max, n_steps)
    best_T = 1.0
    best_nll = float("inf")

    for T in candidate_Ts:
        scaled_logits = logits / T
        nll = F.cross_entropy(scaled_logits, labels).item()
        if nll < best_nll:
            best_nll = nll
            best_T = T.item()

    return best_T


def apply_temperature(logits, T):
    """Apply a fitted temperature to logits and return calibrated probabilities."""
    return F.softmax(logits / T, dim=1)


"""
Exercise (Phase 5)
-------------------
1. Modify dataset.py to carve out a small validation split (e.g. 10%
   of the training set) that's used ONLY for fitting T, never for
   training or final test reporting. Refit T on this proper validation
   split instead of the test set, then re-measure ECE on the test set
   with the properly-fit T. Does ECE change much? This exercise is
   specifically about understanding data leakage in calibration
   pipelines -- a common mistake in published UQ papers.
2. Temperature scaling only has ONE parameter (T). Compare this to
   "vector scaling" or "matrix scaling" (per-class temperature or a
   full learned linear transform of the logits) -- these can fix
   MORE calibration problems but risk overfitting with limited data.
   Write one sentence on when you'd choose one over the other.

Expected output: no __main__ block -- calibration.py drives this by
loading test logits, fitting T, and reporting before/after ECE.
"""
