"""
evaluate.py
-----------
Theory
------
Phase 1 evaluation is just standard classification accuracy:

    accuracy = (# correct predictions) / (# total predictions)

Note what this DOESN'T tell you: whether the model's confidence is
trustworthy. A model can be 90% accurate while being wildly
overconfident on the 10% it gets wrong. That gap between "accuracy"
and "trustworthy confidence" is exactly what Phase 5 (calibration:
ECE, Brier score) will measure. Keep this file simple for now --
Phase 6/7 will import and extend it, not replace it.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Run the model in eval mode (deterministic, no dropout) over a
    DataLoader and compute accuracy.

    Returns a dict so later phases can add more keys (ece, brier,
    ood_score, ...) without changing this function's signature.
    """
    model.eval()

    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        probs = F.softmax(outputs.logits, dim=1)
        preds = probs.argmax(dim=1)

        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

        all_probs.append(probs.cpu())
        all_labels.append(batch["labels"].cpu())

    return {
        "accuracy": correct / total,
        "probs": torch.cat(all_probs, dim=0),      # kept for Phase 5/6 reuse
        "labels": torch.cat(all_labels, dim=0),
    }


"""
Exercise (Phase 1)
-------------------
1. Add precision/recall/F1 (sklearn.metrics.classification_report) to
   the returned dict. For imbalanced datasets these matter more than
   raw accuracy.
2. Save a confusion matrix plot to outputs/. You'll want this again
   in Phase 6 when comparing OOD vs ID behavior.

Expected output: this file has no __main__ block by itself --
it's imported by train.py and later by mc_dropout / ensemble / etc.
evaluation scripts, all of which reuse the same `evaluate()` contract:
model + loader + device in, dict of results out.
"""
