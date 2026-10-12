"""Data loading: benign prompt pools, safety-alignment data, evaluation suites.

Nothing here ships with the repo. Every loader pulls a public dataset from the
Hugging Face Hub on first use and caches it. If a dataset id has moved (the
safety sets get re-hosted fairly often), change it in the config rather than
here, and note the change in your experiment log.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from datasets import load_dataset


@dataclass
class Example:
    """One evaluation or training item.

    prompt    the user turn, before chat templating
    answer    gold answer string, if the suite has one
    choices   answer options for multiple-choice suites
    meta      suite-specific extras (e.g. the XSTest prompt type)
    """

    prompt: str
    answer: str | None = None
    choices: list[str] | None = None
    meta: dict = field(default_factory=dict)


def _subsample(rows, n: int, seed: int = 0):
    """Deterministic subsample. Fixing this seed is what makes the cross-scale
    numbers comparable, so do not vary it between rungs of the ladder."""
    rows = list(rows)
    if n <= 0 or n >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


# --------------------------------------------------------------------------
# Benign prompt pool for the self-distillation rollouts
# --------------------------------------------------------------------------

def load_prompt_pool(sources: list[dict], seed: int = 0) -> list[str]:
    """Build the benign prompt pool the student samples its own rollouts on.

    These prompts must be benign. The whole claim is that on-policy
    self-distillation restores capability on harmless requests without
    touching behaviour on harmful ones, so mixing harmful prompts in here
    would confound the result.
    """
    prompts: list[str] = []
    for source in sources:
        dataset = load_dataset(
            source["id"],
            source.get("config"),
            split=source.get("split", "train"),
        )
        field_name = source.get("field", "question")
        rows = _subsample(dataset, int(source.get("n", 0)), seed=seed)
        for row in rows:
            text = (row.get(field_name) or "").strip()
            if text:
                prompts.append(text)

    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


# --------------------------------------------------------------------------
# Safety-alignment data (stage 1: this is what creates the tax)
# --------------------------------------------------------------------------

def load_safety_align_data(cfg, seed: int = 0) -> list[dict]:
    """Preference pairs for the safety-alignment stage.

    PKU-SafeRLHF gives two responses per prompt plus safety labels. We build
    (chosen, rejected) from the safer/less safe pair, which works for both the
    SFT arm (train on `chosen` only) and the DPO arm.
    """
    dataset = load_dataset(cfg.dataset_id, split=cfg.split)
    rows = _subsample(dataset, int(cfg.n_examples), seed=seed)

    pairs: list[dict] = []
    for row in rows:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        safer = row.get("safer_response_id")
        if safer is None:
            continue
        chosen = row.get(f"response_{int(safer)}")
        rejected = row.get(f"response_{1 - int(safer)}")
        if not chosen or not rejected:
            continue
        pairs.append(
            {"prompt": prompt, "chosen": chosen.strip(), "rejected": rejected.strip()}
        )
    return pairs


# --------------------------------------------------------------------------
# Evaluation suites
# --------------------------------------------------------------------------

def load_eval_suite(spec: dict, seed: int = 0) -> list[Example]:
    """Load one evaluation suite described by a config entry.

    `kind` selects the parser and, downstream in evaluate.py, the scorer:
        exact_match          free-form generation vs a gold string (GSM8K)
        multiple_choice      pick a letter (MMLU, ARC)
        over_refusal         safe prompts that look unsafe (XSTest)
        harmful_compliance   genuinely harmful requests (AdvBench)
    """
    dataset = load_dataset(spec["id"], spec.get("config"), split=spec.get("split", "test"))
    rows = _subsample(dataset, int(spec.get("n", 0)), seed=seed)
    kind = spec["kind"]

    parsers = {
        "exact_match": _parse_gsm8k,
        "multiple_choice": _parse_multiple_choice,
        "over_refusal": _parse_over_refusal,
        "harmful_compliance": _parse_harmful,
    }
    if kind not in parsers:
        raise ValueError(f"unknown eval kind: {kind}")

    parse = parsers[kind]
    examples: list[Example] = []
    for row in rows:
        example = parse(row)
        if example is not None:
            example.meta["kind"] = kind
            example.meta["suite"] = spec["id"]
            examples.append(example)
    return examples


def _parse_gsm8k(row) -> Example | None:
    question = (row.get("question") or "").strip()
    raw_answer = (row.get("answer") or "").strip()
    if not question or "####" not in raw_answer:
        return None
    gold = raw_answer.split("####")[-1].strip().replace(",", "")
    return Example(prompt=question, answer=gold)


def _parse_multiple_choice(row) -> Example | None:
    question = (row.get("question") or "").strip()
    if not question:
        return None

    # MMLU stores `choices` + integer `answer`; ARC nests text/label lists.
    if isinstance(row.get("choices"), list):
        choices = [str(c) for c in row["choices"]]
        index = row.get("answer")
        gold = chr(ord("A") + int(index)) if isinstance(index, int) else str(index).strip()
    elif isinstance(row.get("choices"), dict):
        choices = [str(c) for c in row["choices"].get("text", [])]
        labels = [str(l) for l in row["choices"].get("label", [])]
        key = str(row.get("answerKey", "")).strip()
        gold = chr(ord("A") + labels.index(key)) if key in labels else key
    else:
        return None

    if not choices or not gold:
        return None
    return Example(prompt=question, answer=gold, choices=choices)


def _parse_over_refusal(row) -> Example | None:
    # XSTest: safe prompts phrased so they superficially resemble unsafe ones,
    # plus a matched unsafe contrast set. We only score the safe half, and
    # a refusal there counts against the model.
    prompt = (row.get("prompt") or row.get("question") or "").strip()
    if not prompt:
        return None
    prompt_type = str(row.get("type") or row.get("label") or "safe")
    is_safe = not prompt_type.lower().startswith("contrast")
    return Example(prompt=prompt, meta={"is_safe": is_safe, "type": prompt_type})


def _parse_harmful(row) -> Example | None:
    prompt = (row.get("prompt") or row.get("goal") or "").strip()
    if not prompt:
        return None
    return Example(prompt=prompt, meta={"is_safe": False})


# --------------------------------------------------------------------------
# Probe data: matched harmful / harmless prompt pairs
# --------------------------------------------------------------------------

def load_probe_pairs(n_pairs: int = 256, seed: int = 0) -> tuple[list[str], list[str]]:
    """Harmful and harmless prompt sets for the refusal-direction probe.

    The difference-in-means direction is only meaningful if the two sets are
    matched in surface form and length, so we draw the harmless half from
    XSTest's safe split (which is deliberately written to *look* unsafe)
    rather than from generic instructions.
    """
    harmful_rows = load_dataset("walledai/AdvBench", split="train")
    harmful = [
        (row.get("prompt") or row.get("goal") or "").strip()
        for row in _subsample(harmful_rows, n_pairs, seed=seed)
    ]

    xstest = load_dataset("walledai/XSTest", split="train")
    harmless_all = [
        example.prompt
        for example in (_parse_over_refusal(row) for row in xstest)
        if example is not None and example.meta.get("is_safe")
    ]
    harmless = _subsample(harmless_all, n_pairs, seed=seed)

    size = min(len(harmful), len(harmless))
    return harmful[:size], harmless[:size]
