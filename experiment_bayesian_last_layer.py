"""
experiment_bayesian_last_layer.py
-----------------------------------
Theory
------
Same "experiment driver" pattern as Phase 2 and Phase 3: load the
already-trained checkpoint (no retraining), apply the UQ method, save
results in the same format so Phase 7's compare_all.py can read all
three methods side by side.

The Bayesian Last Layer pipeline has one extra step compared to MC
Dropout/Ensembles: we fit the Laplace posterior using the TRAINING set
(that's what tells us the curvature of the loss around the MAP
weights), then evaluate uncertainty on the TEST set using that fitted
posterior.

Run with:
    python experiment_bayesian_last_layer.py
"""

import os
import json

import config
from dataset import load_train_test
from preprocessing import get_tokenizer, make_dataloader
from bayesian_last_layer import (
    load_trained_model,
    extract_features,
    fit_laplace_posterior,
    bayesian_last_layer_predict,
)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"Loading trained checkpoint from {config.BASELINE_CHECKPOINT} ...")
    model = load_trained_model()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{config.CHECKPOINT_DIR}/tokenizer")

    train_texts, train_labels, test_texts, test_labels = load_train_test()
    train_loader = make_dataloader(train_texts, train_labels, tokenizer, shuffle=False)
    test_loader = make_dataloader(test_texts, test_labels, tokenizer, shuffle=False)

    print("Extracting frozen backbone features for the training set (used to fit the posterior) ...")
    train_features, train_labels_t = extract_features(model, train_loader)

    print("Fitting Laplace posterior over the final classification layer ...")
    W_map, b_map, W_var, b_var = fit_laplace_posterior(model, train_features, train_labels_t)
    print(f"  Average posterior std (weights): {W_var.sqrt().mean().item():.6f}")

    print("Extracting frozen backbone features for the test set ...")
    test_features, test_labels_t = extract_features(model, test_loader)

    print("Running Bayesian Last Layer prediction (num_samples=30) over the test set ...")
    out = bayesian_last_layer_predict(W_map, b_map, W_var, b_var, test_features, num_samples=30)
    preds = out["mean"].argmax(dim=1)

    results = []
    for i in range(len(test_labels_t)):
        results.append({
            "true_label": int(test_labels_t[i]),
            "predicted_label": int(preds[i]),
            "correct": bool(preds[i] == test_labels_t[i]),
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

    print("\n--- Bayesian Last Layer Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average predictive entropy (all):       {avg_entropy:.4f}")
    print(f"Average predictive entropy (correct):    {avg_entropy_correct:.4f}")
    print(f"Average predictive entropy (incorrect):  {avg_entropy_incorrect:.4f}")

    out_path = f"{config.OUTPUT_DIR}/bayesian_last_layer_results.json"
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
Exercise (Phase 4)
-------------------
1. Compare this method's runtime against experiment_mc_dropout.py
   and train_ensemble.py (add simple time.time() calls around the
   main computation). Bayesian Last Layer should be dramatically
   faster than Deep Ensembles and somewhat faster than MC Dropout,
   since the expensive backbone forward pass only runs ONCE per
   example instead of T times.
2. Compare the three methods' correct-vs-incorrect entropy gaps side
   by side (you now have all three JSON files in outputs/). Which
   method's uncertainty estimate is most useful on your data?

Expected output (numbers will vary):
    Loading trained checkpoint from checkpoints/baseline_distilbert.pt ...
    Extracting frozen backbone features for the training set ...
    Fitting Laplace posterior over the final classification layer ...
      Average posterior std (weights): 0.0XX
    Extracting frozen backbone features for the test set ...
    Running Bayesian Last Layer prediction (num_samples=30) over the test set ...
    --- Bayesian Last Layer Results ---
    Accuracy: 0.84
    Average predictive entropy (all):       0.4X
    Average predictive entropy (correct):   0.3X
    Average predictive entropy (incorrect): 0.6X
    Saved results to outputs/bayesian_last_layer_results.json
"""

if __name__ == "__main__":
    main()
