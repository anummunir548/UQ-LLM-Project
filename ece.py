"""
ece.py
------
Theory
------
Accuracy answers "how often is the model right?" Calibration answers a
DIFFERENT question: "when the model SAYS it's 80% confident, is it
actually right about 80% of the time?" A model can be 90% accurate and
still be badly calibrated -- e.g. if it says "99% confident" on almost
everything, including things it gets wrong.

Expected Calibration Error (ECE) measures this gap directly:

    1. Take the model's confidence for each prediction
       (confidence = the probability of whichever class it predicted,
       i.e. max(probs), NOT the probability of the true class)
    2. Sort predictions into B bins by confidence (e.g. bin 1 = confidence
       in [0.0, 0.1), bin 2 = [0.1, 0.2), ..., bin 10 = [0.9, 1.0])
    3. For each bin, compute:
           accuracy(bin)   = fraction of predictions in this bin that were correct
           confidence(bin) = average confidence of predictions in this bin
    4. ECE = weighted average of |accuracy(bin) - confidence(bin)|,
       weighted by how many predictions fall in each bin:

           ECE = sum_b (n_b / N) * |accuracy(b) - confidence(b)|

A PERFECTLY calibrated model has accuracy(bin) == confidence(bin) for
every bin -> ECE = 0. Lower ECE is better. There's no universal
"good" threshold, but ECE < 0.05 is often considered well-calibrated
for standard classification benchmarks; > 0.15 signals a real problem.

Note: ECE only needs mean_prob (the final probability output), so it
applies identically to plain softmax predictions AND to the mean
predictions from MC Dropout / Deep Ensembles / Bayesian Last Layer --
this is one of the metrics you'll compute for all four (raw + 3 UQ
methods) in Phase 7's comparison table.
"""

import torch


def expected_calibration_error(probs, labels, n_bins=10):
    """
    Parameters
    ----------
    probs  : [n_examples, num_classes] tensor of predicted probabilities
    labels : [n_examples] tensor of true class labels
    n_bins : number of confidence bins (10 is standard in the literature)

    Returns
    -------
    ece : float
    bin_data : list of dicts (one per bin) with keys:
        bin_range, count, accuracy, confidence -- useful for plotting
        a reliability diagram in Phase 6.
    """
    confidences, predictions = probs.max(dim=1)
    correct = (predictions == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    n_total = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        # Include the right edge only in the last bin, so confidence == 1.0
        # lands somewhere instead of being dropped.
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)

        count = in_bin.sum().item()
        if count == 0:
            bin_data.append({"bin_range": (lo, hi), "count": 0, "accuracy": None, "confidence": None})
            continue

        bin_accuracy = correct[in_bin].mean().item()
        bin_confidence = confidences[in_bin].mean().item()

        ece += (count / n_total) * abs(bin_accuracy - bin_confidence)

        bin_data.append({
            "bin_range": (lo, hi),
            "count": count,
            "accuracy": bin_accuracy,
            "confidence": bin_confidence,
        })

    return ece, bin_data


"""
Exercise (Phase 5)
-------------------
1. Compute ECE for your MC Dropout, Deep Ensemble, and Bayesian Last
   Layer results (using their saved `mean_prob` values). Which method
   has the lowest ECE? Does that match which method had the best
   correct-vs-incorrect entropy gap from Phase 2-4? (It doesn't
   have to -- entropy gap and ECE measure different things.)
2. Try n_bins=5 vs n_bins=20. ECE estimates get noisier with more bins
   and fewer examples per bin -- with only 50 test examples, how
   trustworthy is a 20-bin ECE number?

Expected output: no __main__ block -- calibration.py imports and
uses this for every method's results.
"""
