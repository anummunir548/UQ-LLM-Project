"""
ood_detection.py
----------------

Simple Out-of-Distribution (OOD) uncertainty analysis.

This module compares uncertainty values for:
1. In-Distribution (ID) data
2. Out-of-Distribution (OOD) data

Higher uncertainty on OOD data indicates better uncertainty estimation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt


def summarize_uncertainty(id_uncertainty, ood_uncertainty):
    id_uncertainty = np.asarray(id_uncertainty)
    ood_uncertainty = np.asarray(ood_uncertainty)

    results = {
        "id_mean": float(np.mean(id_uncertainty)),
        "ood_mean": float(np.mean(ood_uncertainty)),
        "id_std": float(np.std(id_uncertainty)),
        "ood_std": float(np.std(ood_uncertainty)),
    }

    print("\nOOD Detection Summary")
    print("-" * 40)
    print(f"ID Mean Uncertainty : {results['id_mean']:.4f}")
    print(f"OOD Mean Uncertainty: {results['ood_mean']:.4f}")
    print(f"ID Std              : {results['id_std']:.4f}")
    print(f"OOD Std             : {results['ood_std']:.4f}")

    return results


def plot_ood_histogram(id_uncertainty,
                       ood_uncertainty,
                       save_path="outputs/ood_uncertainty_histogram.png"):

    plt.figure(figsize=(7,5))

    plt.hist(id_uncertainty,
             bins=20,
             alpha=0.6,
             label="ID")

    plt.hist(ood_uncertainty,
             bins=20,
             alpha=0.6,
             label="OOD")

    plt.xlabel("Uncertainty")
    plt.ylabel("Number of Samples")
    plt.title("OOD Uncertainty Distribution")
    plt.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved histogram to: {save_path}")


def save_results(results,
                 save_path="outputs/ood_results.json"):

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to: {save_path}")


if __name__ == "__main__":

    np.random.seed(42)

    # Demo data
    id_uncertainty = np.random.normal(0.30, 0.08, 500)
    ood_uncertainty = np.random.normal(0.70, 0.10, 500)

    id_uncertainty = np.clip(id_uncertainty, 0, 1)
    ood_uncertainty = np.clip(ood_uncertainty, 0, 1)

    results = summarize_uncertainty(
        id_uncertainty,
        ood_uncertainty
    )

    plot_ood_histogram(
        id_uncertainty,
        ood_uncertainty
    )

    save_results(results)
