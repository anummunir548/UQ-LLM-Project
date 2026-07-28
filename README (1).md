# Uncertainty Quantification for LLM Text Classification

A from-scratch, paper-shaped research project comparing uncertainty
quantification (UQ) methods on a fine-tuned DistilBERT text
classifier. Built in phases, each one a self-contained, documented
module with theory, code, comments, and exercises.

## Roadmap

- **Phase 1 (this drop): Foundation** — `dataset.py`, `preprocessing.py`,
  `baseline_model.py`, `train.py`, `evaluate.py`. A working, checkpointed
  DistilBERT classifier. Everything else builds on this.
- **Phase 2: MC Dropout** — `mc_dropout.py`, `experiment_mc_dropout.py`
- **Phase 3: Deep Ensembles** — `deep_ensemble.py`, `train_ensemble.py`
- **Phase 4: Bayesian Last Layer** — `bayesian_last_layer.py` (Laplace approximation)
- **Phase 5: Calibration** — `temperature_scaling.py`, `calibration.py`, `ece.py`, `brier_score.py`
- **Phase 6: Evaluation** — `reliability_diagram.py`, `ood_detection.py`, `selective_prediction.py`
- **Phase 7: Final comparison** — `compare_all.py`: one table (Accuracy / ECE / Brier / OOD / Selective
  Prediction) and plots across every method, ready for a workshop paper.

## Setup

```bash
pip install -r requirements.txt
```

## Run Phase 1

```bash
python train.py        # trains + saves checkpoints/baseline_distilbert.pt
python evaluate.py      # (imported by train.py; run standalone once you extend it in exercises)
```

Each file's docstring contains the theory behind that step, a walkthrough
of the code, and an exercise to check your understanding before moving
to the next phase.

## Before Phase 2

Swap `config.DATASET_NAME` for your real dataset (e.g. an AI-vs-human
text detection dataset) and confirm `python train.py` runs end-to-end
and prints a reasonable test accuracy. That checkpoint is what every
later phase (MC Dropout, Ensembles, Bayesian Last Layer) will load and
build on.
