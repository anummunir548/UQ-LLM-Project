"""
baseline_model.py
------------------
Theory
------
We fine-tune DistilBERT for sequence classification. Concretely:

    text -> DistilBERT encoder -> [CLS]-like pooled representation
         -> linear classification head -> logits (raw scores)
         -> softmax -> P(y | x)

For a binary problem, logits has shape [batch_size, 2]. Softmax turns
those two raw numbers into a valid probability distribution:

    P(y=k | x) = exp(logit_k) / sum_j exp(logit_j)

Crucially: this softmax output is what most people INCORRECTLY treat
as "the model's confidence." A model can output P = [0.99, 0.01] and
still be wrong, and worse, can be wrong in a way that no amount of
looking at that single softmax output reveals. That's the whole
motivation for Phases 2-4 (MC Dropout, Ensembles, Bayesian layers):
they give you a DISTRIBUTION over predictions instead of one number,
which is what lets you distinguish "confidently right", "confidently
wrong", and "actually unsure".

HuggingFace's AutoModelForSequenceClassification already implements
the encoder + pooling + linear head + cross-entropy loss for us. We
wrap it in a tiny helper so the rest of the codebase doesn't need to
know HuggingFace-specific details.
"""

from transformers import AutoModelForSequenceClassification

import config


def build_model():
    """
    Construct a fresh (pretrained-backbone, randomly-initialized head)
    DistilBERT classifier, moved to config.DEVICE.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_CLASSES,
    )
    model.to(config.DEVICE)
    return model


"""
Exercise (Phase 1)
-------------------
1. Print model.config.hidden_size and model.config.num_hidden_layers.
   These are DistilBERT's architecture constants -- knowing them
   matters when you later add a custom Bayesian head in Phase 4 (you
   need to know the exact size of the pooled representation feeding
   into it).
2. Count trainable parameters with:
       sum(p.numel() for p in model.parameters() if p.requires_grad)
   Compare that number to a full BERT-base model. This is why we
   picked DistilBERT for a project you're building solo, on limited
   compute, in a few weeks.

Expected output when you run this file directly:
    hidden_size: 768
    num_hidden_layers: 6
    trainable parameters: ~66,000,000
"""

if __name__ == "__main__":
    model = build_model()
    print("hidden_size:", model.config.hidden_size)
    print("num_hidden_layers:", model.config.num_hidden_layers)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}")
