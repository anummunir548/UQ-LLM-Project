"""
experiment_evaluation.py
------------------------

Runs the Phase 6 evaluation:
1. Selective Prediction
2. OOD Detection

Author: UQ-LLM Project
"""

import os
import json
import numpy as np

from selective_prediction import (
    selective_prediction,
    plot_selective_prediction,
)

from ood_detection import (
    summarize_uncertainty,
    plot_ood_histogram,
)


def main():

    os.makedirs("outputs", exist_ok=True)

    np.random.seed(42)

    # -----------------------------
    # Demo predictions
    # Replace these with your model outputs later.
    # -----------------------------
    n = 1000

    y_true = np.random.randint(0, 2, n)

    # Simulate ~85% accuracy
    y_pred = y_true.copy()
    flip = np.random.choice(n, int(0.15 * n), replace=False)
    y_pred[flip] = 1 - y_pred[flip]

    uncertainty = np.random.rand(n)

    # -----------------------------
    # Selective Prediction
    # -----------------------------
    sp_results = selective_prediction(
        y_true,
        y_pred,
        uncertainty
    )

    plot_selective_prediction(
        sp_results,
        "outputs/selective_prediction.png"
    )

    # -----------------------------
    # OOD Demo
    # -----------------------------
    id_uncertainty = np.random.normal(0.30, 0.08, 500)
    ood_uncertainty = np.random.normal(0.70, 0.10, 500)

    id_uncertainty = np.clip(id_uncertainty, 0, 1)
    ood_uncertainty = np.clip(ood_uncertainty, 0, 1)

    ood_results = summarize_uncertainty(
        id_uncertainty,
        ood_uncertainty
    )

    plot_ood_histogram(
        id_uncertainty,
        ood_uncertainty,
        "outputs/ood_uncertainty_histogram.png"
    )

    results = {
        "selective_prediction": sp_results,
        "ood_detection": ood_results
    }

    with open("outputs/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nEvaluation complete.")
    print("Saved:")
    print("  outputs/selective_prediction.png")
    print("  outputs/ood_uncertainty_histogram.png")
    print("  outputs/evaluation_results.json")


if __name__ == "__main__":
    main()
