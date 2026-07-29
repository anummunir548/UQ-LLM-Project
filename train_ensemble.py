"""
train_ensemble.py
-------------------
Theory
------
This is the "experiment driver" for Deep Ensembles: train N members
(different seeds), save each one to its own checkpoint file, then run
ensemble_predict() over the test set and report the same
correct-vs-incorrect entropy comparison you got from MC Dropout in
Phase 2 -- so the two methods are directly comparable in Phase 7.

N=5 is a common choice in the Deep Ensembles literature (Lakshminarayanan
et al. 2017 found diminishing returns past ~5). We keep it configurable
here since your dataset is currently small (fast to retrain) -- feel
free to bump it up once you're running the full 4000-example dataset,
if your compute budget allows it.

Run with:
    python train_ensemble.py
"""

import os
import json
import torch

import config
from dataset import load_train_test
from preprocessing import get_tokenizer, make_dataloader
from deep_ensemble import train_single_member, ensemble_predict

N_ENSEMBLE_MEMBERS = 5
SEEDS = [0, 1, 2, 3, 4]
ENSEMBLE_CHECKPOINT_DIR = f"{config.CHECKPOINT_DIR}/ensemble"


def main():
    os.makedirs(ENSEMBLE_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_texts, train_labels, test_texts, test_labels = load_train_test()
    tokenizer = get_tokenizer()
    train_loader = make_dataloader(train_texts, train_labels, tokenizer, shuffle=True)
    test_loader = make_dataloader(test_texts, test_labels, tokenizer, shuffle=False)

    models = []
    for i, seed in enumerate(SEEDS[:N_ENSEMBLE_MEMBERS], start=1):
        print(f"\n=== Training ensemble member {i}/{N_ENSEMBLE_MEMBERS} (seed={seed}) ===")
        model = train_single_member(train_loader, test_loader, seed=seed)

        ckpt_path = f"{ENSEMBLE_CHECKPOINT_DIR}/member_{i}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved {ckpt_path}")

        models.append(model)

    # --- Evaluate the ensemble as a whole over the test set ---
    print(f"\nEvaluating {N_ENSEMBLE_MEMBERS}-model ensemble over {len(test_texts)} test examples ...")

    results = []
    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        labels = batch["labels"]

        out = ensemble_predict(models, input_ids, attention_mask)
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

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_entropy = sum(r["entropy"] for r in results) / len(results)
    correct_r = [r for r in results if r["correct"]]
    incorrect_r = [r for r in results if not r["correct"]]
    avg_entropy_correct = sum(r["entropy"] for r in correct_r) / len(correct_r) if correct_r else float("nan")
    avg_entropy_incorrect = sum(r["entropy"] for r in incorrect_r) / len(incorrect_r) if incorrect_r else float("nan")

    print("\n--- Deep Ensemble Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average predictive entropy (all):       {avg_entropy:.4f}")
    print(f"Average predictive entropy (correct):    {avg_entropy_correct:.4f}")
    print(f"Average predictive entropy (incorrect):  {avg_entropy_incorrect:.4f}")

    out_path = f"{config.OUTPUT_DIR}/ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_members": N_ENSEMBLE_MEMBERS,
            "accuracy": accuracy,
            "avg_entropy_all": avg_entropy,
            "avg_entropy_correct": avg_entropy_correct,
            "avg_entropy_incorrect": avg_entropy_incorrect,
            "per_example": results,
        }, f, indent=2)
    print(f"\nSaved results to {out_path}")


"""
Exercise (Phase 3)
-------------------
1. Reduce N_ENSEMBLE_MEMBERS to 2 and rerun. How much does the
   correct-vs-incorrect entropy gap shrink? This tells you how much
   "ensemble size" matters for uncertainty quality vs just accuracy.
2. Open outputs/mc_dropout_results.json and outputs/ensemble_results.json
   side by side. For examples both methods get wrong, does one method
   assign consistently higher entropy than the other?

Expected output (numbers will vary):
    === Training ensemble member 1/5 (seed=0) ===
        [seed 0] epoch 1/3 - train loss: 0.68
        ...
    --- Deep Ensemble Results ---
    Accuracy: 0.86
    Average predictive entropy (all):       0.40
    Average predictive entropy (correct):   0.30
    Average predictive entropy (incorrect): 0.65
    Saved results to outputs/ensemble_results.json
"""

if __name__ == "__main__":
    main()
