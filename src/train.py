"""Entry points. Four subcommands, one per stage of the study.

    python -m src.train distill  --config configs/default.yaml
    python -m src.train eval     --config configs/default.yaml --checkpoint PATH
    python -m src.train probe    --config configs/default.yaml --checkpoint PATH
    python -m src.train ladder   --config configs/default.yaml --results DIR

`ladder` does no GPU work. It reads the JSON written by `eval` for the three
checkpoints at each size and prints the paper's main results table, including
the NTR column. Run it early and often, even with one rung filled in: seeing
the table take shape is the fastest way to notice that a cell is missing or
that a number is implausible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config, resolve_teacher_id
from .data import load_probe_pairs, load_prompt_pool
from .distill import train as run_distill
from .evaluate import evaluate_checkpoint, save_results, summarize_ladder
from .models import build_pair, load_model, load_tokenizer
from .probes import run_probes


def _log(record: dict) -> None:
    print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in record.items()), flush=True)


def cmd_distill(args) -> None:
    cfg = load_config(args.config, _overrides(args))
    torch.manual_seed(int(cfg.seed))

    teacher_id = resolve_teacher_id(cfg)
    print(f"student: {cfg.models.student_id}\nteacher: {teacher_id}")

    pair = build_pair(cfg, teacher_id)
    prompts = load_prompt_pool(cfg.distill.prompt_sources, seed=int(cfg.seed))
    print(f"prompt pool: {len(prompts)} benign prompts")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    run_distill(pair, prompts, cfg, logger=_log)
    print(f"saved to {cfg.output_dir}/final")


def cmd_eval(args) -> None:
    cfg = load_config(args.config, _overrides(args))
    checkpoint = args.checkpoint or cfg.models.student_id

    tokenizer = load_tokenizer(cfg.models.base_id)
    model = load_model(checkpoint, cfg.models, trainable=False)

    results = evaluate_checkpoint(model, tokenizer, cfg, seed=int(cfg.seed))
    results["checkpoint"] = checkpoint

    out = args.out or f"{cfg.output_dir}/eval.json"
    save_results(results, out)

    for suite in results["suites"]:
        print(f"{suite['suite']:<28} {suite['kind']:<20} {suite['score']:6.2f}  "
              f"(n={suite['n']})")
    print(f"\ncapability mean: {results['capability_mean']:.2f}")
    print(f"written to {out}")


def cmd_probe(args) -> None:
    cfg = load_config(args.config, _overrides(args))
    checkpoint = args.checkpoint or cfg.models.student_id

    tokenizer = load_tokenizer(cfg.models.base_id)
    model = load_model(checkpoint, cfg.models, trainable=False)

    harmful, harmless = load_probe_pairs(
        n_pairs=int(cfg.probes.n_prompt_pairs), seed=int(cfg.seed)
    )
    task_prompts = load_prompt_pool(cfg.distill.prompt_sources, seed=int(cfg.seed))
    task_prompts = task_prompts[: int(cfg.probes.n_prompt_pairs)]

    results = run_probes(
        model, tokenizer, harmful, harmless, task_prompts,
        n_components=int(cfg.probes.task_subspace_components),
    )
    results["checkpoint"] = checkpoint

    out = args.out or f"{cfg.output_dir}/probes.json"
    save_results(results, out)

    print(f"mean refusal/task entanglement: {results['entanglement_mean']:.4f}")
    print(f"peak at layer:                  {results['entanglement_peak_layer']}")
    print(f"mean output entropy (nats):     {results['output_entropy']:.4f}")
    print(f"written to {out}")


def cmd_ladder(args) -> None:
    """Assemble the paper's main table from per-checkpoint eval JSON.

    Expects files named <size>_{base,safe,sd}.json inside --results.
    """
    directory = Path(args.results)
    rows = []

    for size in args.sizes.split(","):
        size = size.strip()
        try:
            scores = {
                tag: json.loads((directory / f"{size}_{tag}.json").read_text())[
                    "capability_mean"
                ]
                for tag in ("base", "safe", "sd")
            }
        except FileNotFoundError as missing:
            print(f"skipping {size}: {Path(missing.filename).name} not found")
            continue
        rows.append(
            {"model": size, "base": scores["base"], "safe": scores["safe"],
             "distilled": scores["sd"]}
        )

    if not rows:
        print("no complete rungs found; run `eval` for each checkpoint first")
        return

    table = summarize_ladder(rows)
    header = f"{'Model':<14}{'base':>8}{'safe':>8}{'sd':>8}{'tax':>8}{'delta':>8}{'NTR':>8}"
    print(header)
    print("-" * len(header))
    for row in table:
        ntr = "n/a" if row["ntr"] is None else f"{row['ntr']:.2f}"
        print(f"{row['model']:<14}{row['s_base']:>8}{row['s_safe']:>8}"
              f"{row['s_sd']:>8}{row['safety_tax']:>8}{row['delta_raw']:>8}{ntr:>8}")

    print("\nNTR is 'n/a' where safety alignment cost that model nothing, so "
          "there was no tax to recover. Report those rungs as excluded rather "
          "than filling them in.")
    save_results({"table": table}, directory / "ladder.json")


def _overrides(args) -> dict:
    overrides = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        overrides[key] = parsed
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler in (
        ("distill", cmd_distill),
        ("eval", cmd_eval),
        ("probe", cmd_probe),
        ("ladder", cmd_ladder),
    ):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/default.yaml")
        p.add_argument("--set", action="append",
                       help="dotted override, e.g. --set distill.lora_rank=64")
        p.set_defaults(func=handler)

        if name in {"eval", "probe"}:
            p.add_argument("--checkpoint", default=None)
            p.add_argument("--out", default=None)
        if name == "ladder":
            p.add_argument("--results", default="outputs")
            p.add_argument("--sizes", default="qwen3-0.6b,qwen3-1.7b,qwen3-4b,qwen3-8b")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
