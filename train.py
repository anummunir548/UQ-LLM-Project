"""
train.py
--------
Theory
------
Standard supervised fine-tuning loop. For each batch:

    1. Forward pass:  logits = model(input_ids, attention_mask)
    2. Loss:          L = CrossEntropy(logits, labels)
                       = -log P(y_true | x)     (negative log-likelihood)
    3. Backward pass: compute dL/dtheta for every parameter theta
    4. Optimizer step: theta <- theta - lr * dL/dtheta   (Adam variant)

Minimizing cross-entropy is equivalent to maximum likelihood
estimation of the model's parameters -- this is the "MLE" baseline
that every later Bayesian/ensemble method in this project will be
compared against. Phase 1's whole job is to produce ONE well-trained
deterministic model and a checkpoint file. Every later phase reuses
this training loop's structure (Deep Ensembles = run this loop N
times with different seeds; Bayesian Last Layer = run this loop then
post-hoc fit a Bayesian layer on top).

Run with:
    python train.py
"""

import os
import torch
from torch.optim import AdamW

import config
from dataset import load_train_test
from preprocessing import get_tokenizer, make_dataloader
from baseline_model import build_model
from evaluate import evaluate


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = {k: v.to(config.DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss  # HF computes cross-entropy internally when labels are passed

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_texts, train_labels, test_texts, test_labels = load_train_test()

    tokenizer = get_tokenizer()
    train_loader = make_dataloader(train_texts, train_labels, tokenizer, shuffle=True)
    test_loader = make_dataloader(test_texts, test_labels, tokenizer, shuffle=False)

    model = build_model()
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    for epoch in range(1, config.EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch}/{config.EPOCHS} - train loss: {avg_loss:.4f}")

        metrics = evaluate(model, test_loader, config.DEVICE)
        print(f"  test accuracy: {metrics['accuracy']:.4f}")

    torch.save(model.state_dict(), config.BASELINE_CHECKPOINT)
    print(f"\nSaved checkpoint to {config.BASELINE_CHECKPOINT}")

    # Also save the tokenizer alongside the checkpoint so evaluate.py /
    # mc_dropout.py never have to guess which tokenizer config was used.
    tokenizer.save_pretrained(f"{config.CHECKPOINT_DIR}/tokenizer")


"""
Exercise (Phase 1)
-------------------
1. Add a learning-rate scheduler (e.g. get_linear_schedule_with_warmup
   from transformers). Does test accuracy improve or become more
   stable across epochs?
2. Log train loss AND test accuracy per epoch to a CSV in outputs/.
   You will reuse this pattern for every later phase's experiment
   logs (this is what compare_all.py in Phase 7 will read from).

Expected output when you run this file directly (numbers will vary):
    Epoch 1/3 - train loss: 0.42
      test accuracy: 0.86
    Epoch 2/3 - train loss: 0.21
      test accuracy: 0.89
    Epoch 3/3 - train loss: 0.11
      test accuracy: 0.90
    Saved checkpoint to checkpoints/baseline_distilbert.pt
"""

if __name__ == "__main__":
    main()
