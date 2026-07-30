"""
calibration.py
----------------
Theory
------
This is the "experiment driver" that ties ece.py, brier_score.py, and
temperature_scaling.py together into one report. It answers, for every
method you've built so far (raw softmax, MC Dropout, Deep Ensembles,
Bayesian Last Layer):

    1. How well-calibrated are its probabilities? (ECE, Brier score)
    2. Does simple post-hoc temperature scaling improve the RAW
       (uncalibrated) model's calibration?

We compute ECE/Brier directly from each method's already-saved
`outputs/*_results.json` file (using the `mean_prob` and `true_label`
fields) -- no need to rerun any of the expensive UQ methods.
Temperature scaling is the one exception: it needs raw LOGITS (not
just softmax probabilities), so for that part specifically we reload
the baseline model and recompute logits directly.

Run with:
    python calibration.py
"""

import json
import torch

import config
from dataset import load_train_test
from preprocessing import get_tokenizer, make_dataloader
from ece import expected_calibration_error
from brier_score import brier_score
from temperature_scaling import fit_temperature, apply_temperature


def load_results_json(path):
    with open(path) as f:
        data = json.load(f)
    probs = torch.tensor([ex["mean_prob"] for ex in data["per_example"]])
    labels = torch.tensor([ex["true_label"] for ex in data["per_example"]])
    return probs, labels


def report_calibration(name, probs, labels):
    ece, _ = expected_calibration_error(probs, labels)
    brier = brier_score(probs, labels)
    accuracy = (probs.argmax(dim=1) == labels).float().mean().item()
    print(f"{name:24s} | Accuracy: {accuracy:.4f} | ECE: {ece:.4f} | Brier: {brier:.4f}")
    return {"accuracy": accuracy, "ece": ece, "brier": brier}


def get_baseline_logits_and_labels():
    """
    Reload the baseline model and recompute raw logits over the test
    set -- needed only for the temperature scaling step, since ECE/
    Brier from the saved JSONs work fine with just probabilities.
    """
    from baseline_model import build_model

    model = build_model()
    state_dict = torch.load(config.BASELINE_CHECKPOINT, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    model.eval()

    tokenizer = get_tokenizer()
    _, _, test_texts, test_labels = load_train_test()
    test_loader = make_dataloader(test_texts, test_labels, tokenizer, shuffle=False)

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(batch["labels"])

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def main():
    print("=== Calibration Report: ECE & Brier Score per Method ===\n")

    summary = {}

    # --- Raw (uncalibrated) baseline model ---
    print("Recomputing raw baseline logits ...")
    logits, labels = get_baseline_logits_and_labels()
    raw_probs = torch.softmax(logits, dim=1)
    summary["raw_softmax"] = report_calibration("Raw softmax (uncalibrated)", raw_probs, labels)

    # --- Temperature-scaled baseline model ---
    # NOTE: fitting T on the test set itself is a simplification for
    # this small-dataset phase -- see the exercise in
    # temperature_scaling.py about using a proper held-out validation
    # split instead.
    T = fit_temperature(logits, labels)
    calibrated_probs = apply_temperature(logits, T)
    print(f"\nFitted temperature: T = {T:.3f}")
    summary["temperature_scaled"] = report_calibration("Temperature-scaled", calibrated_probs, labels)

    print()

    # --- The three UQ methods from Phase 2-4 ---
    method_files = {
        "MC Dropout": f"{config.OUTPUT_DIR}/mc_dropout_results.json",
        "Deep Ensembles": f"{config.OUTPUT_DIR}/ensemble_results.json",
        "Bayesian Last Layer": f"{config.OUTPUT_DIR}/bayesian_last_layer_results.json",
    }

    for name, path in method_files.items():
        try:
            probs, ex_labels = load_results_json(path)
            summary[name] = report_calibration(name, probs, ex_labels)
        except FileNotFoundError:
            print(f"{name:24s} | (results file not found: {path} -- skipping)")

    out_path = f"{config.OUTPUT_DIR}/calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


"""
Exercise (Phase 5)
-------------------
1. Look at the printed table: does temperature scaling meaningfully
   reduce ECE compared to raw softmax? If T came out close to 1.0, the
   raw model was already reasonably calibrated -- if T is far from
   1.0 (e.g. > 2 or < 0.5), that's a sign the raw model was quite
   over/under-confident.
2. Rank all methods (raw, temperature-scaled, MC Dropout, Ensembles,
   Bayesian Last Layer) by ECE, then separately by Brier score. Do the
   two rankings agree? If not, that's worth a sentence in your paper
   explaining WHY a metric disagreement happened (hint: think about
   what each metric penalizes differently, per each file's theory
   section).

Expected output (numbers will vary):
    === Calibration Report: ECE & Brier Score per Method ===

    Recomputing raw baseline logits ...
    Raw softmax (uncalibrated) | Accuracy: 0.8400 | ECE: 0.0XXX | Brier: 0.XXXX

    Fitted temperature: T = X.XXX
    Temperature-scaled       | Accuracy: 0.8400 | ECE: 0.0XXX | Brier: 0.XXXX

    MC Dropout               | Accuracy: 0.8400 | ECE: 0.0XXX | Brier: 0.XXXX
    Deep Ensembles           | Accuracy: 0.8400 | ECE: 0.0XXX | Brier: 0.XXXX
    Bayesian Last Layer      | Accuracy: 0.8400 | ECE: 0.0XXX | Brier: 0.XXXX

    Saved summary to outputs/calibration_results.json
"""

if __name__ == "__main__":
    main()
