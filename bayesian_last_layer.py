"""
bayesian_last_layer.py
------------------------
Theory
------
MC Dropout (Phase 2) needed 30 forward passes. Deep Ensembles (Phase 3)
needed training 5 whole models. Both are expensive. Bayesian Last
Layer asks: what if we ONLY put a Bayesian treatment on the cheapest,
smallest part of the network -- the final linear classification
layer -- and leave the giant pretrained DistilBERT backbone as a
single fixed (frozen) feature extractor?

Concretely, your classifier currently computes:

    features = ReLU(pre_classifier(pooled_bert_output))   -- frozen, deterministic
    logits   = classifier(features) = W @ features + b     -- this is what we go Bayesian on
    probs    = softmax(logits)

Standard training gives you a single point estimate for W and b (the
"MAP" -- Maximum A Posteriori -- estimate, found by your existing
train.py run). A Laplace Approximation says: instead of treating W as
one fixed value, treat it as a random variable with a GAUSSIAN
distribution centered at that MAP estimate:

    W ~ N(W_MAP, Sigma)

Where does Sigma (the uncertainty around each weight) come from? From
the curvature of the loss function around the MAP solution -- if the
loss function is very "sharply curved" around a weight (small nudges
hurt a lot), we should be CONFIDENT in that weight (small variance).
If the loss is nearly FLAT around a weight (nudging it barely changes
anything), we should be UNCERTAIN about it (large variance). That
curvature is exactly the Hessian matrix of the loss.

Computing a full Hessian is expensive, so we use a standard
simplification: a DIAGONAL Gauss-Newton approximation, which for a
softmax classifier works out to, for each weight w_j feeding into
class k:

    H_jk = prior_precision + sum_i p_i,k * (1 - p_i,k) * phi_i,j^2

    (prior_precision is a small constant from assuming a Gaussian
    prior N(0, 1/prior_precision) on the weights -- this is also what
    makes the Hessian invertible even with limited data)

Then the posterior variance for that weight is just 1/H_jk, and at
prediction time, instead of ONE forward pass, we:

    1. Sample many different W's from N(W_MAP, Sigma)
    2. Run each sampled W through the (frozen) features
    3. Average the resulting probabilities -- same mean/std/entropy
       recipe as MC Dropout and Deep Ensembles

This gives you uncertainty that's much cheaper than Deep Ensembles (no
retraining) and doesn't require T forward passes through the WHOLE
network like MC Dropout (only through the tiny final linear layer,
after the backbone's frozen features are computed once).
"""

import torch
import torch.nn.functional as F

import config
from baseline_model import build_model


PRIOR_PRECISION = 1.0  # inverse of the assumed Gaussian prior variance on weights.
                         # Larger -> stronger regularization -> more confident (smaller) posterior.
                         # Smaller -> weaker prior -> posterior variance driven more by data curvature.


def load_trained_model():
    """Load the Phase 1 checkpoint -- this is our frozen backbone + MAP classifier head."""
    model = build_model()
    state_dict = torch.load(config.BASELINE_CHECKPOINT, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, loader):
    """
    Run the frozen DistilBERT backbone + pre_classifier over a
    DataLoader and return the pre-final-layer features (what actually
    feeds into `classifier`), along with labels. We do this ONCE for
    the whole dataset -- this is the expensive part (full transformer
    forward pass), but unlike MC Dropout we only pay this cost once,
    not 30 times.

    DistilBERT-for-sequence-classification's forward pipeline is:
        distilbert(...) -> hidden_state
        pooled = hidden_state[:, 0]           (the [CLS] token)
        pooled = ReLU(pre_classifier(pooled))
        pooled = dropout(pooled)               (skipped at eval time)
        logits = classifier(pooled)
    We stop right before the final `classifier` call.
    """
    all_features = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)

        distilbert_output = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output[0]           # [batch, seq_len, hidden_size]
        pooled = hidden_state[:, 0]                    # [batch, hidden_size]  -- the [CLS] token
        pooled = model.pre_classifier(pooled)
        pooled = torch.relu(pooled)
        # (dropout skipped -- model is in eval mode)

        all_features.append(pooled.cpu())
        all_labels.append(batch["labels"])

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def fit_laplace_posterior(model, features, labels, prior_precision=PRIOR_PRECISION):
    """
    Compute the diagonal Laplace approximation's posterior variance
    for the final classifier layer's weights, using the MAP weights
    already sitting in model.classifier (from training in Phase 1).

    Parameters
    ----------
    features : [n_examples, hidden_size] -- output of extract_features()
    labels   : [n_examples] -- true class labels (int)

    Returns
    -------
    W_map, b_map : the existing trained classifier weight/bias (unchanged)
    W_var, b_var : posterior variance for each weight/bias, same shape as W_map/b_map
    """
    classifier = model.classifier
    W_map = classifier.weight.data.clone()   # [num_classes, hidden_size]
    b_map = classifier.bias.data.clone()     # [num_classes]

    with torch.no_grad():
        logits = features @ W_map.T + b_map    # [n_examples, num_classes]
        probs = F.softmax(logits, dim=1)        # [n_examples, num_classes]

    num_classes = W_map.shape[0]
    hidden_size = W_map.shape[1]

    # Diagonal Gauss-Newton Hessian, computed per class.
    # p_k * (1 - p_k) is the softmax curvature term (same role as the
    # logistic sigmoid variance p*(1-p) in binary logistic regression,
    # applied per-class here as the standard diagonal approximation).
    W_hessian = torch.zeros(num_classes, hidden_size)
    b_hessian = torch.zeros(num_classes)

    for k in range(num_classes):
        pk = probs[:, k]                       # [n_examples]
        curvature = pk * (1 - pk)               # [n_examples]
        # sum_i curvature_i * phi_i^2, elementwise over hidden_size
        W_hessian[k] = prior_precision + (curvature.unsqueeze(1) * (features ** 2)).sum(dim=0)
        b_hessian[k] = prior_precision + curvature.sum()

    W_var = 1.0 / W_hessian
    b_var = 1.0 / b_hessian

    return W_map, b_map, W_var, b_var


@torch.no_grad()
def bayesian_last_layer_predict(W_map, b_map, W_var, b_var, features, num_samples=30):
    """
    Monte Carlo integration over the Laplace-approximated posterior:
    sample many (W, b) pairs from N(W_map, W_var) / N(b_map, b_var),
    run the (already-computed, frozen) features through each sampled
    linear layer, and average -- same output contract (mean, std,
    entropy) as mc_dropout_predict and ensemble_predict, so all three
    methods can be compared apples-to-apples in Phase 7.
    """
    W_std = W_var.sqrt()
    b_std = b_var.sqrt()

    all_probs = []
    for _ in range(num_samples):
        W_sample = W_map + torch.randn_like(W_map) * W_std
        b_sample = b_map + torch.randn_like(b_map) * b_std

        logits = features @ W_sample.T + b_sample
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)

    all_probs = torch.stack(all_probs, dim=0)  # [num_samples, n_examples, num_classes]

    mean = all_probs.mean(dim=0)
    std = all_probs.std(dim=0)

    eps = 1e-12
    entropy = -(mean * torch.log(mean + eps)).sum(dim=1)

    return {
        "mean": mean,
        "std": std,
        "entropy": entropy,
        "all_probs": all_probs,
    }


"""
Exercise (Phase 4)
-------------------
1. Try PRIOR_PRECISION = 0.1 vs PRIOR_PRECISION = 10.0. How does the
   average entropy change? A stronger prior (higher precision) should
   shrink posterior variance and lower average entropy -- confirm this
   empirically.
2. This implementation uses a DIAGONAL approximation to the Hessian
   (ignoring correlations between different weights). Look up
   "Kronecker-Factored Approximate Curvature" (K-FAC) as the standard
   next step up in accuracy over diagonal Laplace -- you don't need to
   implement it, just be able to explain in one sentence why it's more
   accurate but more expensive.

Expected output: no __main__ block here -- experiment_bayesian_last_layer.py
drives this by loading the checkpoint, extracting features once, fitting
the Laplace posterior, and evaluating over the test set.
"""
