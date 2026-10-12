# Experiment log

One row per run. The table below is deliberately the same shape as Table 1 in
`paper/main.tex`, so filling this in fills in the paper. When a rung is
complete, copy its numbers straight across.

Keep the discipline of writing the row **before** you read the result. Decide
what would count as success, write it in "prediction", then record what
happened. That is what stops you from fitting a story to whatever the number
turned out to be.

---

## Main table (maps directly to Table 1)

| Date | Run ID | Model | Arm | s(base) | s(safe) | s(sd) | Safety tax | Δ raw | **NTR** | Seeds | Notes |
|------|--------|-------|-----|---------|---------|-------|-----------|-------|---------|-------|-------|
| | | Qwen3-0.6B | on-policy SD | | | | | | | | |
| | | Qwen3-1.7B | on-policy SD | | | | | | | | |
| | | Qwen3-4B | on-policy SD | | | | | | | | |
| | | Qwen3-8B | on-policy SD | | | | | | | | |
| | | Qwen3-0.6B | off-policy | | | | | | | | |
| | | Qwen3-1.7B | off-policy | | | | | | | | |
| | | Qwen3-0.6B | SFT on teacher | | | | | | | | |
| | | Qwen3-1.7B | SFT on teacher | | | | | | | | |
| | | Qwen3-0.6B | weight interpolation | | | | | | | | |
| | | Llama-3.2-1B | on-policy SD | | | | | | | | |
| | | Llama-3.2-3B | on-policy SD | | | | | | | | |

NTR = (s(sd) - s(safe)) / (s(base) - s(safe)). Leave it blank, never zero, if
the tax was not meaningfully positive: a rung where alignment cost nothing is
an excluded rung, not a rung that scored zero.

## Safety audit (maps to Figure 2c)

Run this for every row above. A capability gain with a harm-compliance rise is
not a result, it is de-alignment.

| Run ID | Model | XSTest over-refusal (safe answered, %) | AdvBench refusal (%) | Verdict |
|--------|-------|----------------------------------------|----------------------|---------|
| | | | | |

## Probes (maps to Section 5, Result 5)

| Run ID | Model | Checkpoint | Mean entanglement | Peak layer | Output entropy (nats) |
|--------|-------|------------|-------------------|------------|-----------------------|
| | | safe | | | |
| | | sd | | | |

Prediction to beat: entanglement is higher at smaller scale in the `safe`
checkpoints, and the drop from `safe` to `sd` correlates with that rung's NTR.

## Ablations (maps to Section 5)

| Run ID | Ablation | Setting | Model | Δ raw | NTR | What it rules in or out |
|--------|----------|---------|-------|-------|-----|-------------------------|
| | LoRA rank | r=4 | | | | size vs adaptation capacity |
| | LoRA rank | r=16 | | | | |
| | LoRA rank | r=64 | | | | |
| | LoRA rank | r=256 | | | | |
| | Teacher gap | teacher=0.6B | student 0.6B | | | self vs closeness |
| | Teacher gap | teacher=1.7B | student 0.6B | | | |
| | Teacher gap | teacher=8B | student 0.6B | | | |
| | On-policy λ | 0.0 / 0.5 / 1.0 | | | | distribution mismatch |
| | KL direction | forward / reverse / jsd | | | | sharpening vs coverage |
| | Rollouts K | 1 / 2 / 4 / 8 | | | | sample coverage artifact |
| | Parser | strict vs permissive | | | | format artifact bound |
| | Matched accuracy | large model degraded | | | | headroom artifact |

---

## Run notes

Copy this block per run.

### Run `YYYY-MM-DD-nn`

- **Command:** `...`
- **Config diff from default:** `...`
- **Commit:** `...`
- **Prediction (written before looking):** what I expect, and what result would
  change my mind.
- **Result:** the numbers.
- **Surprise:** anything that did not match the prediction. This is the most
  valuable line in the log; most real findings start as an entry here.
- **Next action:** one concrete thing.
