"""
preprocessing.py
-----------------
Theory
------
A transformer like DistilBERT never sees raw text. It sees integers.
The tokenizer does two jobs:

1. Text -> subword tokens -> integer IDs (a fixed vocabulary lookup).
2. Padding/truncation so every sequence in a batch has the SAME length
   (required because tensors are rectangular).

We also build a torch.utils.data.Dataset wrapper, which is the
standard PyTorch interface for "give me example i as tensors".
DataLoader then wraps that to produce shuffled mini-batches.

Key equation-free but important detail: `attention_mask` tells the
model which tokens are real text (1) vs padding (0), so padding
doesn't influence the model's attention computation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import config


def get_tokenizer():
    """Load the pretrained DistilBERT tokenizer (subword vocabulary)."""
    return AutoTokenizer.from_pretrained(config.MODEL_NAME)


class TextClassificationDataset(Dataset):
    """
    Wraps (texts, labels) as a PyTorch Dataset.

    __getitem__ tokenizes ONE example on the fly. For small/medium
    datasets this is fine; for huge datasets you'd pre-tokenize once
    and cache the result to avoid repeating tokenization every epoch.
    """

    def __init__(self, texts, labels, tokenizer, max_length=config.MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # encoding tensors have a leading batch dim of 1 (from a single
        # string) -- squeeze it out since DataLoader adds its own
        # batch dimension when it collates multiple items together.
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_dataloader(texts, labels, tokenizer, batch_size=config.BATCH_SIZE, shuffle=True):
    """Convenience wrapper: texts+labels -> ready-to-iterate DataLoader."""
    dataset = TextClassificationDataset(texts, labels, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


"""
Exercise (Phase 1)
-------------------
1. Print the shape of `input_ids` and `attention_mask` for a single
   batch. Confirm both are [batch_size, MAX_LENGTH].
2. Try MAX_LENGTH=64 vs MAX_LENGTH=256 in config.py. How does batch
   tokenization time change? (This matters a lot once you're doing 30
   MC Dropout forward passes per example in Phase 2 -- shorter
   sequences = much faster experiments.)

Expected output when you run this file directly:
    input_ids shape: torch.Size([16, 256])
    attention_mask shape: torch.Size([16, 256])
    labels shape: torch.Size([16])
"""

if __name__ == "__main__":
    from dataset import load_train_test

    tr_x, tr_y, te_x, te_y = load_train_test()
    tok = get_tokenizer()
    loader = make_dataloader(tr_x, tr_y, tok)
    batch = next(iter(loader))
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("labels shape:", batch["labels"].shape)
