# Comparing Uncertainty Quantification Methods for LLM-Based Text Classification

## Project Overview

This project investigates **Uncertainty Quantification (UQ)** techniques for transformer-based Large Language Models (LLMs) on a binary text classification task.

Rather than evaluating a single uncertainty estimation method, this project provides a systematic comparison of several Bayesian and calibration approaches using the same fine-tuned DistilBERT model.

The goal is to determine which methods produce the most reliable confidence estimates while maintaining strong classification performance.

---

## Research Motivation

Deep neural networks often produce highly confident predictions, even when they are incorrect.

In safety-critical applications such as:

- Healthcare
- Finance
- Legal document analysis
- Misinformation detection
- AI-generated text detection

knowing **how uncertain** a prediction is can be just as important as the prediction itself.

This project explores multiple uncertainty estimation methods and compares their effectiveness under a unified experimental framework.

---

# Objectives

The objectives of this project are to:

- Fine-tune a DistilBERT classifier using PyTorch
- Compare multiple uncertainty quantification methods
- Evaluate model calibration
- Analyze prediction confidence
- Study Bayesian approaches for transformer models
- Produce reproducible research experiments

---

# Methods Compared

The project compares the following uncertainty estimation techniques:

### 1. Baseline Softmax Confidence

Uses the raw softmax probability as a confidence estimate.

---

### 2. Monte Carlo Dropout

Runs multiple stochastic forward passes with dropout enabled during inference.

Measures:

- Mean prediction
- Predictive entropy
- Prediction variance

---

### 3. Deep Ensembles

Trains multiple independently initialized DistilBERT models.

Prediction disagreement across models is used as an uncertainty estimate.

---

### 4. Bayesian Last Layer (Laplace Approximation)

Treats only the final classification layer as Bayesian while freezing the transformer backbone.

Provides Bayesian uncertainty at significantly lower computational cost.

---

### 5. Temperature Scaling

Post-processing calibration technique.

Improves probability calibration without affecting classification accuracy.

---

## Evaluation Metrics

The following metrics are used throughout the project:

- Accuracy
- Expected Calibration Error (ECE)
- Brier Score
- Predictive Entropy
- Reliability Diagrams
- Selective Prediction
- Out-of-Distribution (OOD) Detection

---

# Project Structure

```
UQ-LLM-Project/

│
├── config.py
├── dataset.py
├── train.py
├── evaluate.py
│
├── mc_dropout.py
├── experiment_mc_dropout.py
│
├── deep_ensemble.py
├── experiment_deep_ensemble.py
│
├── bayesian_last_layer.py
├── experiment_bayesian_last_layer.py
│
├── calibration.py
├── experiment_calibration.py
│
├── selective_prediction.py
├── ood_detection.py
├── experiment_evaluation.py
│
├── outputs/
├── checkpoints/
└── README.md
```

---

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy
- Matplotlib
- Scikit-learn

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your_username/UQ-LLM-Project.git
cd UQ-LLM-Project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Train the baseline model

```bash
python train.py
```

## Monte Carlo Dropout

```bash
python experiment_mc_dropout.py
```

## Deep Ensembles

```bash
python experiment_deep_ensemble.py
```

## Bayesian Last Layer

```bash
python experiment_bayesian_last_layer.py
```

## Calibration

```bash
python experiment_calibration.py
```

## Evaluation

```bash
python experiment_evaluation.py
```

---

# Current Status

✅ Baseline DistilBERT Training

✅ Monte Carlo Dropout

✅ Deep Ensembles

✅ Bayesian Last Layer

✅ Temperature Scaling

✅ Expected Calibration Error (ECE)

✅ Brier Score

✅ Reliability Diagrams

🚧 Selective Prediction (In Progress)

🚧 Out-of-Distribution Detection (In Progress)

---

# Future Work

Planned extensions include:

- Evaluation on larger language models (Llama, Mistral)
- Additional Bayesian inference techniques
- Active Learning using uncertainty estimates
- Conformal Prediction
- More challenging out-of-distribution benchmarks
- Multi-class text classification
- AI-generated text detection

---

# Research Contribution

This project provides a reproducible framework for comparing uncertainty quantification methods on transformer-based language models.

The implementation demonstrates practical applications of:

- Bayesian Deep Learning
- Neural Network Calibration
- Transformer Fine-tuning
- Uncertainty Quantification
- Experimental Machine Learning

The framework can be extended to misinformation detection, toxicity classification, AI-generated text detection, fake news detection, and other NLP tasks.

---

# License

This project is released for educational and research purposes.

---

# Author

**Anum Munir**



Research Interests:

- Bayesian Deep Learning
- Uncertainty Quantification
- Explainable AI
- Natural Language Processing
- Large Language Models
