"""
config.py
---------
Central place for every hyperparameter, path, and constant used across
the project. Real research repos almost always have a file like this so
that no "magic numbers" are scattered across train.py / evaluate.py /
mc_dropout.py etc. When you write your paper's "Experimental Setup"
section, this file IS that section.
"""

import torch

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
# We use a HuggingFace `datasets` source. Swap this for whatever dataset
# you want the UQ project to run on (AI-text-detection, sentiment,
# toxicity, misinformation, etc). Binary classification keeps every
# downstream UQ method (entropy, ECE, Brier score) simple to compute
# and simple to explain in a paper.
DATASET_NAME = "imdb"          # placeholder: swap for your real dataset
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
NUM_CLASSES = 2

N_TRAIN_SAMPLES = 4000          # keep small while iterating on code
N_TEST_SAMPLES = 1000
TEST_SIZE = 0.2

# ---------------------------------------------------------------------
# Tokenizer / model
# ---------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
BASELINE_CHECKPOINT = f"{CHECKPOINT_DIR}/baseline_distilbert.pt"
