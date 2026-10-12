"""Mechanistic probes: refusal-direction geometry and output entropy.

This file answers the question the behavioural results cannot: if small models
really do recover more of their safety tax, what is physically different about
them?

The hypothesis we test is ENTANGLEMENT. Safety alignment installs a refusal
feature. In a model with enough capacity, that feature can be conditional:
"refuse this kind of request, answer that one." In a model without the
capacity, the cheapest solution found by gradient descent is a blunt,
surface-triggered feature that also fires on benign prompts that merely look
risky. If that is right, then in small models the refusal direction should sit
closer to the subspace the model uses for ordinary task processing, and
on-policy self-distillation should pull the two apart.

Every measurement here is correlational. We observe geometry; we do not
intervene on it. Causal confirmation means ablating the direction and checking
whether that alone reproduces the capability recovery, which is the natural
follow-up and is deliberately out of scope for the first paper.
"""

from __future__ import annotations

import numpy as np
import torch

from .models import apply_chat_template


@torch.no_grad()
def collect_activations(model, tokenizer, prompts: list[str], batch_size: int = 8):
    """Residual-stream activations at the final prompt token, every layer.

    The last prompt position is the standard readout point: it is where the
    model has seen the whole request and is about to decide how to begin its
    answer, which is exactly where a refusal decision shows up.

    Returns a (n_layers, n_prompts, hidden) float32 array on CPU.
    """
    device = next(model.parameters()).device
    model.eval()
    per_batch = []

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        rendered = [apply_chat_template(tokenizer, p) for p in chunk]
        batch = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch, output_hidden_states=True)
        # hidden_states is a tuple of (B, L, H), one per layer plus embeddings.
        # Left padding means the final prompt token is always at index -1.
        stacked = torch.stack([h[:, -1, :] for h in outputs.hidden_states])
        per_batch.append(stacked.float().cpu())

    return torch.cat(per_batch, dim=1).numpy()


def refusal_direction(harmful_acts: np.ndarray, harmless_acts: np.ndarray) -> np.ndarray:
    """Difference-in-means refusal direction, one unit vector per layer.

    This is the standard construction: the direction separating harmful from
    harmless prompts in activation space. Returns (n_layers, hidden).
    """
    direction = harmful_acts.mean(axis=1) - harmless_acts.mean(axis=1)
    norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    return direction / np.clip(norms, 1e-8, None)


def task_subspace(task_acts: np.ndarray, n_components: int = 8) -> np.ndarray:
    """Top principal components of activations on benign task prompts.

    This is our stand-in for "the directions the model uses to do ordinary
    work". Returns (n_layers, n_components, hidden) with orthonormal rows.
    """
    n_layers, _, hidden = task_acts.shape
    basis = np.zeros((n_layers, n_components, hidden), dtype=np.float32)

    for layer in range(n_layers):
        centered = task_acts[layer] - task_acts[layer].mean(axis=0, keepdims=True)
        # Right singular vectors are the principal directions in feature space.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis[layer] = vt[:n_components]

    return basis


def entanglement(direction: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """How much of the refusal direction lies inside the benign task subspace.

    For each layer, project the unit refusal direction onto the orthonormal
    task basis and return the squared norm of the projection, which lands in
    [0, 1]: 0 means refusal is geometrically orthogonal to ordinary task
    processing, 1 means it is entirely inside it.

    THE PREDICTION: entanglement is higher in smaller safety-aligned models,
    and on-policy self-distillation reduces it in proportion to the NTR it
    delivers. If entanglement is flat across scale while NTR is peaked, this
    account is wrong and should be reported as wrong.
    """
    # (n_layers, n_components) coefficients, then sum of squares per layer.
    coefficients = np.einsum("lch,lh->lc", basis, direction)
    return (coefficients ** 2).sum(axis=-1)


@torch.no_grad()
def output_entropy(model, tokenizer, prompts: list[str], batch_size: int = 8) -> float:
    """Mean next-token entropy (nats) at the first generated position.

    The competing "sharpening" story says self-distillation helps simply by
    reducing output entropy, and that small models have more entropy to
    remove. The entropy-sharpening control arm in distill.py plus this
    measurement are what separate that story from the teacher mattering.
    """
    device = next(model.parameters()).device
    model.eval()
    totals, count = 0.0, 0

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        rendered = [apply_chat_template(tokenizer, p) for p in chunk]
        batch = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(**batch).logits[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        entropy = -(logp.exp() * logp).sum(dim=-1)

        totals += entropy.sum().item()
        count += entropy.shape[0]

    return totals / max(1, count)


def run_probes(model, tokenizer, harmful: list[str], harmless: list[str],
               task_prompts: list[str], n_components: int = 8) -> dict:
    """Run the full probe suite against one checkpoint."""
    harmful_acts = collect_activations(model, tokenizer, harmful)
    harmless_acts = collect_activations(model, tokenizer, harmless)
    task_acts = collect_activations(model, tokenizer, task_prompts)

    direction = refusal_direction(harmful_acts, harmless_acts)
    basis = task_subspace(task_acts, n_components=n_components)
    per_layer = entanglement(direction, basis)

    return {
        "entanglement_per_layer": per_layer.tolist(),
        "entanglement_mean": float(per_layer.mean()),
        "entanglement_peak_layer": int(per_layer.argmax()),
        "output_entropy": output_entropy(model, tokenizer, task_prompts),
    }
