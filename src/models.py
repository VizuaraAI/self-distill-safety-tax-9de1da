"""Checkpoint loading, LoRA wiring, and the teacher/student pairing.

One invariant matters more than anything else in this file: the teacher and
the student must share a tokenizer. Token-level KL between two different
vocabularies is meaningless, and because both checkpoints come from the same
family that is easy to guarantee, so we assert it rather than hope.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class Pair:
    """A trainable student, a frozen teacher, and the shared tokenizer."""

    student: torch.nn.Module
    teacher: torch.nn.Module
    tokenizer: object

    @property
    def device(self) -> torch.device:
        return next(self.student.parameters()).device


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding so that generated continuations start at a common offset,
    # which keeps the rollout slicing in distill.py simple.
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_id: str, cfg, trainable: bool = False):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=DTYPES[cfg.dtype],
        attn_implementation=cfg.attn_implementation,
    )
    model.to(resolve_device())

    if trainable:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.train()
    else:
        model.eval()
        model.requires_grad_(False)

    return model


def attach_lora(model, distill_cfg):
    """Wrap the student in LoRA adapters.

    The rank sweep here is the capacity ablation: it varies how much the model
    is *allowed* to change while holding the number of forward-pass parameters
    fixed, which is how we tell "small models benefit more" apart from
    "models with less adaptation capacity benefit more".
    """
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=int(distill_cfg.lora_rank),
        lora_alpha=int(distill_cfg.lora_alpha),
        lora_dropout=float(distill_cfg.lora_dropout),
        target_modules=list(distill_cfg.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def build_pair(cfg, teacher_id: str) -> Pair:
    """Load the student (trainable) and the teacher (frozen)."""
    tokenizer = load_tokenizer(cfg.models.base_id)
    teacher_tokenizer = load_tokenizer(teacher_id)

    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            "teacher and student tokenizers differ, so token-level KL is not "
            f"defined between them: {cfg.models.base_id} vs {teacher_id}. Pick "
            "a teacher from the same model family."
        )

    student = load_model(cfg.models.student_id, cfg.models, trainable=True)
    if cfg.distill.use_lora:
        student = attach_lora(student, cfg.distill)

    teacher = load_model(teacher_id, cfg.models, trainable=False)
    return Pair(student=student, teacher=teacher, tokenizer=tokenizer)


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Render one user turn. Falls back to the raw prompt for base models,
    which have no chat template."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt
