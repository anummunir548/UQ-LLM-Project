"""
evaluate_ensemble_existing.py
-------------------------------
Evaluates whatever ensemble member checkpoints already exist in
checkpoints/ensemble/, without training anything. Use this when you
already have some members trained (e.g. member_1.pt, member_2.pt,
member_3.pt) and just want results from those, instead of retraining
all 5 from scratch.

Run with:
    python evaluate_ensemble_existing.py
"""

import os
import json
import glob

import config
from dataset import load_train_test
from preprocessing import get_tokenizer, make_dataloader
from deep_ensemble import evaluate_ensemble_streaming

ENSEMBLE_CHECKPOINT_DIR = f"{config.CHECKPOINT_DIR}/ensemble"


def main():
    checkpoint_paths = sorted(glob.glob(f"{ENSEMBLE_CHECKPOINT_DIR}/member_*.pt"))
    if not checkpoint_paths:
        print(f"No checkpoints found in {ENSEMBLE_CHECKPOINT_DIR}/. Run train_ensemble.py first.")
        return

    print(f"Found {len(checkpoint_paths)} existing member checkpoint(s):")
    for p in checkpoint_paths:
        print(f"  {p}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    _, _, test_texts, test_labels = load_train_test()
    tokenizer = get_tokenizer()
    test_loader = make_dataloader(test_texts, test_labels, tokenizer, shuffle=False)

    print(f"\nEvaluating {len(checkpoint_paths)}-model ensemble over {len(test_texts)} test examples ...")
    out = evaluate_ensemble_streaming(checkpoint_paths, test_loader, config.DEVICE)
    preds = out["mean"].argmax(dim=1)
    labels = out["labels"]

    results = []
    for i in range(len(labels)):
        results.append({
            "true_label": int(labels[i]),
            "predicted_label": int(preds[i]),
            "correct": bool(preds[i] == labels[i]),
            "mean_prob": out["mean"][i].tolist(),
            "std": out["std"][i].tolist(),
            "entropy": float(out["entropy"][i]),
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_entropy = sum(r["entropy"] for r in results) / len(results)
    correct_r = [r for r in results if r["correct"]]
    incorrect_r = [r for r in results if not r["correct"]]
    avg_entropy_correct = sum(r["entropy"] for r in correct_r) / len(correct_r) if correct_r else float("nan")
    avg_entropy_incorrect = sum(r["entropy"] for r in incorrect_r) / len(incorrect_r) if incorrect_r else float("nan")

    print("\n--- Deep Ensemble Results ---")
    print(f"N members: {len(checkpoint_paths)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average predictive entropy (all):       {avg_entropy:.4f}")
    print(f"Average predictive entropy (correct):    {avg_entropy_correct:.4f}")
    print(f"Average predictive entropy (incorrect):  {avg_entropy_incorrect:.4f}")

    out_path = f"{config.OUTPUT_DIR}/ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_members": len(checkpoint_paths),
            "accuracy": accuracy,
            "avg_entropy_all": avg_entropy,
            "avg_entropy_correct": avg_entropy_correct,
            "avg_entropy_incorrect": avg_entropy_incorrect,
            "per_example": results,
        }, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
