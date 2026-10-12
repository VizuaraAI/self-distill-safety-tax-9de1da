"""The on-policy self-distillation training loop.

The whole idea of the method lives in two steps that alternate:

    1. ROLLOUT. The student generates K completions for each prompt, with
       sampling, under no_grad. These are sequences the student can actually
       produce, which is what "on-policy" means.
    2. CORRECT. The frozen teacher scores those same sequences token by token,
       and we take a gradient step on the KL between the two distributions,
       restricted to the completion positions.

The single most common way to get this wrong is to misalign the logits with
the tokens they predict. Logits at position t predict the token at position
t + 1, so the distribution that produced completion token j lives at index
j - 1. `_completion_slice` is the one place that offset is handled; if your
loss looks suspiciously flat or suspiciously huge, check there first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .models import apply_chat_template


@dataclass
class Rollouts:
    """A batch of student-generated sequences, ready for teacher scoring.

    sequences    (B, L) prompt tokens followed by completion tokens
    attention    (B, L) 1 for real tokens, 0 for padding
    prompt_len   number of prompt tokens; identical across the batch because
                 the tokenizer pads on the left
    """

    sequences: torch.Tensor
    attention: torch.Tensor
    prompt_len: int


@torch.no_grad()
def sample_rollouts(pair, prompts: list[str], cfg) -> Rollouts:
    """Generate K completions per prompt from the model that will be trained.

    Note that we sample from `pair.student`. Sampling from the teacher instead
    is the off-policy control arm, and the only difference between the two arms
    is which model is passed here.
    """
    tokenizer = pair.tokenizer
    rendered = [apply_chat_template(tokenizer, p) for p in prompts]

    batch = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.models.max_seq_len - cfg.distill.max_new_tokens,
    ).to(pair.device)

    prompt_len = batch["input_ids"].shape[1]

    model = pair.student
    was_training = model.training
    model.eval()
    sequences = model.generate(
        **batch,
        do_sample=True,
        temperature=float(cfg.distill.temperature),
        top_p=float(cfg.distill.top_p),
        max_new_tokens=int(cfg.distill.max_new_tokens),
        num_return_sequences=int(cfg.distill.rollouts_per_prompt),
        pad_token_id=tokenizer.pad_token_id,
    )
    if was_training:
        model.train()

    attention = (sequences != tokenizer.pad_token_id).long()
    # The prompt region is real input even where it was left-padded, and the
    # left padding itself is already masked by the comparison above.
    return Rollouts(sequences=sequences, attention=attention, prompt_len=prompt_len)


def _completion_slice(logits: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Keep only the logits that predict completion tokens.

    Logits at index t predict the token at t + 1, so the predictions for
    completion tokens [prompt_len, L) live at indices [prompt_len - 1, L - 1).
    """
    return logits[:, prompt_len - 1 : -1, :]


def _completion_mask(rollouts: Rollouts) -> torch.Tensor:
    """(B, L - prompt_len) mask over completion token positions."""
    return rollouts.attention[:, rollouts.prompt_len :].float()


def kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    direction: str = "forward",
    topk: int = 0,
) -> torch.Tensor:
    """Token-level divergence between teacher and student, averaged over
    non-padding completion tokens.

    forward  KL(teacher || student), mean-seeking: the student must cover
             everything the teacher puts mass on.
    reverse  KL(student || teacher), mode-seeking: the choice made in MiniLLM.
    jsd      the symmetric midpoint, useful when the two disagree.

    `topk > 0` truncates both distributions to the teacher's top-k tokens
    before renormalizing. With a 150k vocabulary the full-vocabulary tensors
    dominate memory, so this is the knob to reach for on a small GPU. It does
    change the objective slightly, so hold it fixed across the whole ladder.
    """
    if topk and topk > 0:
        _, index = teacher_logits.topk(topk, dim=-1)
        student_logits = student_logits.gather(-1, index)
        teacher_logits = teacher_logits.gather(-1, index)

    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)

    if direction == "forward":
        per_token = F.kl_div(student_logp, teacher_logp, reduction="none", log_target=True)
    elif direction == "reverse":
        per_token = F.kl_div(teacher_logp, student_logp, reduction="none", log_target=True)
    elif direction == "jsd":
        mixture_logp = torch.logsumexp(
            torch.stack([student_logp, teacher_logp]) + math.log(0.5), dim=0
        )
        per_token = 0.5 * F.kl_div(
            mixture_logp, teacher_logp, reduction="none", log_target=True
        ) + 0.5 * F.kl_div(
            mixture_logp, student_logp, reduction="none", log_target=True
        )
    else:
        raise ValueError(f"unknown kl_direction: {direction}")

    per_token = per_token.sum(dim=-1)                 # (B, T)
    denominator = mask.sum().clamp(min=1.0)
    return (per_token * mask).sum() / denominator


def distill_step(pair, rollouts: Rollouts, cfg) -> torch.Tensor:
    """One forward pass pair plus loss. Backward is the caller's job, so that
    gradient accumulation stays visible in the training loop."""
    inputs = {
        "input_ids": rollouts.sequences,
        "attention_mask": rollouts.attention,
    }

    with torch.no_grad():
        teacher_logits = _completion_slice(
            pair.teacher(**inputs).logits, rollouts.prompt_len
        )

    student_logits = _completion_slice(
        pair.student(**inputs).logits, rollouts.prompt_len
    )

    return kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        mask=_completion_mask(rollouts),
        direction=cfg.distill.kl_direction,
        topk=int(cfg.distill.kl_topk),
    )


def train(pair, prompts: list[str], cfg, logger=None) -> None:
    """The training loop. Deliberately plain: an explicit optimizer, explicit
    accumulation, explicit clipping, so that every number in the log can be
    traced to a line here."""
    distill = cfg.distill
    params = [p for p in pair.student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(distill.lr), weight_decay=0.0)

    total_steps = int(distill.steps)
    warmup = max(1, int(total_steps * float(distill.warmup_ratio)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            step / warmup
            if step < warmup
            else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        ),
    )

    batch_size = int(distill.batch_size)
    accum = int(distill.grad_accum)
    cursor = 0
    pair.student.train()

    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        for _ in range(accum):
            if cursor + batch_size > len(prompts):
                cursor = 0
            chunk = prompts[cursor : cursor + batch_size]
            cursor += batch_size

            rollouts = sample_rollouts(pair, chunk, cfg)
            loss = distill_step(pair, rollouts, cfg) / accum
            loss.backward()
            running += loss.item()

        torch.nn.utils.clip_grad_norm_(params, float(distill.max_grad_norm))
        optimizer.step()
        scheduler.step()

        if logger is not None and step % int(distill.log_every) == 0:
            logger(
                {
                    "step": step,
                    "kl": running,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        if (step + 1) % int(distill.save_every) == 0:
            pair.student.save_pretrained(f"{cfg.output_dir}/step-{step + 1}")

    pair.student.save_pretrained(f"{cfg.output_dir}/final")
    pair.tokenizer.save_pretrained(f"{cfg.output_dir}/final")
