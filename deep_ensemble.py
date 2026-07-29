"""
deep_ensemble.py
-----------------
Theory
------
MC Dropout (Phase 2) simulates "many models" using ONE set of trained
weights and random dropout masks. Deep Ensembles take a more literal
(and often more reliable) approach:

    Train N completely separate models, from N different random
    initializations (different seeds), on the SAME data.

Each model converges to a different point in parameter space, because
neural network loss landscapes have many different local minima that
all fit the training data reasonably well but generalize slightly
differently. When you feed the same input to all N models:

    Model 1 -> P_1(y | x)
    Model 2 -> P_2(y | x)
    ...
    Model N -> P_N(y | x)

Just like MC Dropout, you get:

    mean     = (1/N) * sum_n P_n(y | x)         -> the ensemble's prediction
    variance = (1/N) * sum_n (P_n - mean)^2      -> epistemic uncertainty

The key conceptual difference from MC Dropout: disagreement here comes
from genuinely different trained models, not from randomly zeroing
neurons in the SAME model. Empirically, Deep Ensembles are widely
considered a stronger (if more expensive -- N times the training cost)
uncertainty estimate than MC Dropout, which is exactly why comparing
them is worth a paper section.

This file only defines HOW to train one ensemble member and how to
combine predictions from a list of trained models. The actual "train N
of them and save N checkpoints" loop lives in train_ensemble.py, kept
separate so this file can be unit-tested/reasoned about on its own.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW

import config
from baseline_model import build_model


def train_single_member(train_loader, test_loader, seed, epochs=config.EPOCHS):
    """
    Train ONE ensemble member from a fresh random initialization.

    Setting torch.manual_seed(seed) before build_model() ensures the
    classifier head's random init (and dropout patterns during
    training) differ across members -- this is the actual source of
    "ensemble diversity". The pretrained DistilBERT backbone weights
    are the same across members (that part isn't randomly
    initialized), but the new classification head, training-time
    dropout, and data shuffling order all vary with the seed.
    """
    torch.manual_seed(seed)

    model = build_model()
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        print(f"    [seed {seed}] epoch {epoch}/{epochs} - train loss: {total_loss / len(train_loader):.4f}")

    return model


@torch.no_grad()
def ensemble_predict(models, input_ids, attention_mask):
    """
    Run every model in the ensemble over the same batch and combine
    their predictions -- the Deep Ensemble analog of mc_dropout_predict
    in Phase 2. Same output shape/keys on purpose, so experiment and
    comparison code in later phases can treat both methods uniformly.

    Parameters
    ----------
    models : list of trained nn.Module, all in eval() mode, same device
    input_ids, attention_mask : tensors already on device

    Returns
    -------
    dict with keys: mean, std, entropy, all_probs
    """
    all_probs = []

    for model in models:
        model.eval()  # deterministic forward pass per model -- no dropout tricks here
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        all_probs.append(probs)

    all_probs = torch.stack(all_probs, dim=0)  # [N_models, batch, num_classes]

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
Exercise (Phase 3)
-------------------
1. Compare the `std` values from Deep Ensembles (N=5 models) against
   MC Dropout (T=30 samples) on the SAME test examples. Which method
   gives higher uncertainty on the examples both methods get wrong?
2. Deep Ensembles cost N times the training compute of one model.
   MC Dropout costs T times the INFERENCE compute of one model, but
   only 1x training. Write one sentence for your paper's "Limitations"
   section about this tradeoff.

Expected output: no __main__ block here -- train_ensemble.py drives
this by calling train_single_member() N times and saving N checkpoints.
"""
