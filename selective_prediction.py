"""
selective_prediction.py
-----------------------

Selective Prediction (Selective Classification)

Reject the most uncertain predictions and measure how
accuracy improves as coverage decreases.
"""

import numpy as np
import matplotlib.pyplot as plt


def selective_prediction(y_true, y_pred, uncertainty, rejection_rates=None):
    if rejection_rates is None:
        rejection_rates = np.arange(0.0, 0.55, 0.05)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    uncertainty = np.asarray(uncertainty)

    order = np.argsort(uncertainty)
    y_true = y_true[order]
    y_pred = y_pred[order]

    results = {
        "coverage": [],
        "accuracy": [],
    }

    n = len(y_true)

    print("\nSelective Prediction Results")
    print("-" * 45)

    for reject in rejection_rates:
        keep = max(1, int(n * (1 - reject)))

        kept_true = y_true[:keep]
        kept_pred = y_pred[:keep]

        acc = np.mean(kept_true == kept_pred)

        results["coverage"].append(1 - reject)
        results["accuracy"].append(float(acc))

        print(
            f"Reject {reject*100:>4.0f}% | "
            f"Coverage {(1-reject)*100:>4.0f}% | "
            f"Accuracy {acc:.4f}"
        )

    return results


def plot_selective_prediction(results,
                              save_path="outputs/selective_prediction.png"):

    plt.figure(figsize=(7, 5))
    plt.plot(
        results["coverage"],
        results["accuracy"],
        marker="o"
    )

    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.title("Selective Prediction Curve")
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figure to: {save_path}")


if __name__ == "__main__":

    np.random.seed(42)

    n = 100

    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    uncertainty = np.random.rand(n)

    results = selective_prediction(
        y_true,
        y_pred,
        uncertainty
    )

    plot_selective_prediction(results)
