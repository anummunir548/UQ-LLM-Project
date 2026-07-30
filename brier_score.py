"""
brier_score.py
---------------
Theory
------
The Brier score is a second, complementary way to measure whether
predicted probabilities are trustworthy -- while ECE bins predictions
and compares group-level accuracy vs confidence, Brier score is a
per-example squared error between the predicted probability vector
and the true outcome, averaged over the dataset:

    Brier = (1/N) * sum_i sum_k (p_i,k - y_i,k)^2

where p_i,k is the predicted probability of class k for example i, and
y_i,k is 1 if k is the true class for example i, else 0 (a one-hot
encoding of the label).

Intuition: if the model predicts [0.9, 0.1] and the true class is 0
(one-hot [1, 0]), the squared error is (0.9-1)^2 + (0.1-0)^2 = 0.02 --
small, because it was confident AND correct. If it predicts [0.9, 0.1]
but the true class is actually 1 ([0, 1]), the error is
(0.9-0)^2 + (0.1-1)^2 = 1.62 -- large, because it was confident AND
WRONG. This is why Brier score specifically PUNISHES confident wrong
answers much more than ECE does -- ECE only cares about the gap
between confidence and accuracy on average within a bin, while Brier
score penalizes every individual overconfident mistake directly.

Range: 0 (perfect) to 2 (worst possible, for this two-term
formulation) for binary classification; lower is always better.
"""

import torch
import torch.nn.functional as F


def brier_score(probs, labels, num_classes=None):
    """
    Parameters
    ----------
    probs  : [n_examples, num_classes] tensor of predicted probabilities
    labels : [n_examples] tensor of true class labels (int)
    num_classes : inferred from probs.shape[1] if not given

    Returns
    -------
    float : the mean Brier score over the dataset
    """
    if num_classes is None:
        num_classes = probs.shape[1]

    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    squared_error = (probs - one_hot) ** 2
    per_example_score = squared_error.sum(dim=1)  # sum over classes
    return per_example_score.mean().item()


"""
Exercise (Phase 5)
-------------------
1. Find the single test example with the WORST (highest) individual
   Brier contribution across all three UQ methods. Is it the same
   example for each method, or different? A consistently-bad example
   across all methods might indicate a genuinely mislabeled or
   ambiguous piece of text worth inspecting manually.
2. Brier score decomposes mathematically into (reliability - resolution
   + uncertainty) terms -- look this decomposition up as background for
   your paper's related-work section; you don't need to implement it.

Expected output: no __main__ block -- calibration.py imports and uses
this for every method's results, alongside ece.py.
"""
