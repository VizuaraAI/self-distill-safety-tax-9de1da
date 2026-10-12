# Instructions for the coding agent

This file is read automatically at the start of each session. It describes a
research project, not a product. Optimize for correct measurement over
features shipped.

## What this project is

An empirical and mechanistic study of one observation: on-policy
self-distillation reduces the safety tax most on the smallest models (+8.85
points at 1.5B, +5.49 at 0.6B, shrinking above that), and nobody has explained
why.

The thesis is that this gradient must first be treated as a **measurement
question**. Small models score lower on everything, so they have more absolute
points available to win. The project therefore turns on a headroom-normalized
metric:

```
NTR = ( s(theta_sd) - s(theta_safe) ) / ( s(theta_base) - s(theta_safe) )
```

Flat NTR across scale means the reported effect was headroom (a negative
result, still publishable). Peaked NTR means a real small-model mechanism.

## The one rule that matters most

**A result that comes out negative is still a result. Never help the user
reframe a null into a win.**

If NTR comes out flat, the correct action is to write that up clearly, not to
search for a subgroup, a benchmark, or a normalization that makes the curve
bend. If you notice yourself proposing an analysis *after* seeing a
disappointing number, say so out loud and flag it as post-hoc.

Concretely, push back when asked to:
- drop a rung, seed or benchmark that weakens the trend
- add a benchmark mid-study because the current ones are not cooperating
- report the raw delta instead of NTR because it looks better
- quietly change the safety-alignment recipe between rungs

## Non-negotiable invariants

1. **Matched evaluation.** The same suites, the same subsample, the same seed
   for all three checkpoints at a rung. NTR's denominator is meaningless
   otherwise. `_subsample` in `src/data.py` is seeded for this reason; do not
   make it vary per rung.
2. **Shared tokenizer.** Teacher and student must come from the same family.
   Token-level KL across vocabularies is undefined. `build_pair` asserts this;
   do not relax the assertion.
3. **Benign prompt pool only.** The self-distillation prompts in
   `src/data.py` must stay harmless. Mixing harmful prompts in confounds the
   exact claim under test.
4. **Safety audit always reported.** Every capability gain is reported next to
   AdvBench compliance. A gain with rising harm compliance is de-alignment,
   not recovery.
5. **NTR returns `None` when the tax is not positive.** Do not clamp the
   denominator to a small floor to avoid `None`. That manufactures enormous
   NTR values exactly where the measurement is least trustworthy, which is how
   a spurious inverse-scaling curve gets created.

## Layout

| Path | Role |
|---|---|
| `src/config.py` | YAML loading, dotted overrides, up-front validation |
| `src/data.py` | prompt pool, safety data, eval suites, probe pairs |
| `src/models.py` | checkpoint loading, LoRA, teacher/student pairing |
| `src/distill.py` | the on-policy self-distillation loop (the method) |
| `src/evaluate.py` | capability and safety scoring, and NTR (the metric) |
| `src/probes.py` | refusal-direction geometry, entropy (the mechanism) |
| `src/train.py` | CLI: `distill`, `eval`, `probe`, `ladder` |
| `configs/` | one YAML per rung; only the `models` block differs |
| `scripts/` | end-to-end runners |
| `experiments/log_template.md` | log whose rows map onto the paper's Table 1 |

## Conventions

- Python 3.11+, type hints on function signatures, `from __future__ import
  annotations` at the top of modules.
- Config over constants. Anything swept in the paper is a config key, reachable
  via `--set dotted.key=value`. Never hardcode a sweep value in `src/`.
- Comments explain **why**, especially why a control exists. The code is read
  by someone deciding whether to trust a number.
- No silent failures in measurement code. If a suite parses zero examples,
  raise; do not return a score of 0.0, which is indistinguishable from a real
  result.
- Keep modules small enough to read in one sitting. If a file passes roughly
  250 lines, that is a signal to split it.

## The subtle part of the code

`src/distill.py` aligns logits with the tokens they predict. Logits at
position `t` predict the token at `t + 1`, so the distribution that produced
completion token `j` lives at index `j - 1`. `_completion_slice` is the single
place that offset is handled. If the loss looks suspiciously flat or
suspiciously huge, check there before anything else.

## Current state and next steps

The scaffold is complete and import-clean; no experiments have been run yet.

1. Train `theta_safe` at 0.6B and confirm the safety tax is clearly positive.
   If there is no tax, there is no project; fix this before anything else.
2. Run the on-policy arm at 0.6B, produce the first Table 1 row with NTR.
3. Add the 1.7B rung. Two rungs is the minimum for any scale statement.
4. Run the off-policy and SFT control arms at both rungs.
5. Run the LoRA rank sweep. This is the ablation most likely to overturn the
   thesis: if gain tracks adapter rank the way it tracks model size, the real
   claim is "constrained adaptation benefits most", which is different and more
   general.
6. Probes, and only then the 4B and 8B rungs.

## Helpful behaviours

- When a run finishes, offer to write the `experiments/log_template.md` row,
  including the "surprise" line.
- When the user reports a number, ask whether the controls for that rung have
  been run yet before helping interpret it.
- Prefer a small reproducible run over a big one. Suggest `--set
  distill.steps=50` to validate a change end to end before a full run.
