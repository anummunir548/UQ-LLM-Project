"""
experiment_mc_dropout.py
-------------------------
Theory
------
This is the "experiment script" pattern you'll repeat in every later
phase: load the ALREADY-TRAINED checkpoint (never retrain here),
run the UQ method over the test set, save results to outputs/ so
Phase 7's compare_all.py can read them back in without recomputing
anything.

We deliberately do NOT retrain the model in this file. Training
happens once, in train.py. Every UQ method in this project is a
different way of asking questions of that SAME trained model.

Run with:
    python experiment_mc_dropout.py
"""

import os
import json
import torch

import config
from dataset import load_train_test
from preprocessing import get_tokenizer, make_dataloader
from baseline_model import build_model
from mc_dropout import mc_dropout_predict


def load_trained_model():
    """
    Rebuild the model architecture, then load the fine-tuned weights
    saved by train.py. This is the "load a checkpoint" pattern every
    later phase (ensembles, Bayesian last layer) will also use.
    """
    model = build_model()
    state_dict = torch.load(config.BASELINE_CHECKPOINT, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    return model


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"Loading trained checkpoint from {config.BASELINE_CHECKPOINT} ...")
    model = load_trained_model()

    # Reuse the tokenizer that was saved alongside the checkpoint, so
    # we're guaranteed to match exactly what the model was trained on.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{config.CHECKPOINT_DIR}/tokenizer")

    _, _, test_texts, test_labels = load_train_test()
    test_loader = make_dataloader(test_texts, test_labels, tokenizer, shuffle=False)

    results = []

    print(f"Running MC Dropout (num_samples=30) over {len(test_texts)} test examples ...")
    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        labels = batch["labels"]

        out = mc_dropout_predict(model, input_ids, attention_mask, num_samples=30)

        preds = out["mean"].argmax(dim=1).cpu()

        for i in range(len(labels)):
            results.append({
                "true_label": int(labels[i]),
                "predicted_label": int(preds[i]),
                "correct": bool(preds[i] == labels[i]),
                "mean_prob": out["mean"][i].cpu().tolist(),
                "std": out["std"][i].cpu().tolist(),
                "entropy": float(out["entropy"][i].cpu()),
            })

        if (batch_idx + 1) % 5 == 0:
            print(f"  processed batch {batch_idx + 1}/{len(test_loader)}")

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_entropy = sum(r["entropy"] for r in results) / len(results)
    avg_entropy_correct = sum(r["entropy"] for r in results if r["correct"]) / max(sum(r["correct"] for r in results), 1)
    incorrect = [r for r in results if not r["correct"]]
    avg_entropy_incorrect = sum(r["entropy"] for r in incorrect) / len(incorrect) if incorrect else float("nan")

    print("\n--- MC Dropout Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average predictive entropy (all):       {avg_entropy:.4f}")
    print(f"Average predictive entropy (correct):    {avg_entropy_correct:.4f}")
    print(f"Average predictive entropy (incorrect):  {avg_entropy_incorrect:.4f}")
    print("\n(A good uncertainty method should show HIGHER entropy on incorrect")
    print(" predictions than correct ones -- that gap is the whole point of UQ.)")

    out_path = f"{config.OUTPUT_DIR}/mc_dropout_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "avg_entropy_all": avg_entropy,
            "avg_entropy_correct": avg_entropy_correct,
            "avg_entropy_incorrect": avg_entropy_incorrect,
            "per_example": results,
        }, f, indent=2)
    print(f"\nSaved results to {out_path}")


"""
Exercise (Phase 2)
-------------------
1. Sort `results` by entropy descending and print the top 5 most
   uncertain test examples' text. Do they look genuinely ambiguous to
   YOU as a human reader? That qualitative check matters more than
   people think.
2. Plot a histogram of entropy for correct vs incorrect predictions
   (two overlapping histograms). This is a preview of what Phase 6's
   reliability_diagram.py will formalize.

Expected output when you run this file directly (numbers will vary,
especially with your small smoke-test checkpoint):
    Loading trained checkpoint from checkpoints/baseline_distilbert.pt ...
    Running MC Dropout (num_samples=30) over 50 test examples ...
      processed batch 5/4
    --- MC Dropout Results ---
    Accuracy: 0.90
    Average predictive entropy (all):       0.31
    Average predictive entropy (correct):   0.22
    Average predictive entropy (incorrect): 0.58
    Saved results to outputs/mc_dropout_results.json
"""

if __name__ == "__main__":
    main()
