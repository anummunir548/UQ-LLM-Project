"""
dataset.py
----------
Theory
------
Every UQ experiment needs three data splits with DIFFERENT roles:

    train  -> fit model parameters
    test   -> in-distribution (ID) evaluation: "how good/calibrated is
              the model on data that looks like what it was trained on?"
    ood    -> out-of-distribution evaluation, used in Phase 6. Good
              uncertainty methods should output HIGH uncertainty here,
              even if the model is confidently right or wrong.

Phase 1 only needs train/test. We add an `ood` hook now so later
phases don't require touching this file again.

This file is intentionally the ONLY place that talks to the raw
HuggingFace `datasets` library. Everything downstream (train.py,
evaluate.py, mc_dropout.py, ...) just consumes plain Python lists of
(text, label) pairs. That separation is what lets you swap in your own
CSV, a different HF dataset, or a scraped dataset later without
touching any modeling code.
"""

import random
from datasets import load_dataset

import config


def set_seed(seed: int = config.SEED):
    """Make sampling and shuffling reproducible across runs."""
    random.seed(seed)


def load_train_test():
    """
    Load a balanced random subsample of the configured dataset and
    split it into train/test text + label lists.

    Returns
    -------
    train_texts, train_labels, test_texts, test_labels : list, list, list, list
    """
    set_seed()

    print(f"Loading dataset '{config.DATASET_NAME}' ...")
    raw = load_dataset(config.DATASET_NAME)

    # Most HF text-classification datasets expose a "train" and a
    # "test" split already. If your dataset only has "train", split it
    # yourself using sklearn's train_test_split instead (see Phase 1
    # exercise below).
    train_split = raw["train"].shuffle(seed=config.SEED)
    test_split = raw["test"].shuffle(seed=config.SEED)

    n_train = min(config.N_TRAIN_SAMPLES, len(train_split))
    n_test = min(config.N_TEST_SAMPLES, len(test_split))

    train_subset = train_split.select(range(n_train))
    test_subset = test_split.select(range(n_test))

    train_texts = list(train_subset[config.TEXT_COLUMN])
    train_labels = list(train_subset[config.LABEL_COLUMN])
    test_texts = list(test_subset[config.TEXT_COLUMN])
    test_labels = list(test_subset[config.LABEL_COLUMN])

    print(f"Train: {len(train_texts)} examples | Test: {len(test_texts)} examples")
    return train_texts, train_labels, test_texts, test_labels


def load_ood_texts(dataset_name: str, text_column: str, n_samples: int = 500):
    """
    Load a sample of text from a DIFFERENT dataset/domain, used in
    Phase 6 for out-of-distribution (OOD) uncertainty evaluation.

    Example: if you trained on movie reviews (IMDB), a good OOD set
    might be news headlines, tweets, or Wikipedia sentences -- text
    the model has never seen the "style" of.
    """
    raw = load_dataset(dataset_name)
    split = raw["train"].shuffle(seed=config.SEED)
    n = min(n_samples, len(split))
    return list(split.select(range(n))[text_column])


"""
Exercise (Phase 1)
-------------------
1. Swap DATASET_NAME in config.py for a dataset with only a "train"
   split (many do). Modify load_train_test() to use
   sklearn.model_selection.train_test_split instead of relying on a
   pre-made "test" split.
2. Print the class balance (how many 0s vs 1s) in train_labels. Is it
   balanced? If not, why might that matter for calibration later
   (Phase 5)?

Expected output when you run this file directly:
    Loading dataset 'imdb' ...
    Train: 4000 examples | Test: 1000 examples
"""

if __name__ == "__main__":
    tr_x, tr_y, te_x, te_y = load_train_test()
    print("Sample text:", tr_x[0][:120], "...")
    print("Sample label:", tr_y[0])
