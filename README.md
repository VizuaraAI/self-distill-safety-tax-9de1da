# Why does on-policy self-distillation help small models most?

Starter repository for Giribalaji, Vizuara SLM Bootcamp.

## The question

On-policy self-distillation recovers much of the capability that safety
alignment removes, and the reported gains are largest on the smallest models:
**+8.85 points at 1.5B and +5.49 at 0.6B**, shrinking above that.

Nobody has explained why. This repository is built to find out, and it is
built around a specific suspicion: **a benefit that grows as models shrink is
usually either a mechanism with no room to operate at scale, or an artifact of
how the benefit was measured.** Small models score lower on every benchmark,
so they have more absolute points available to win. A raw point gain cannot
tell those two stories apart.

So the central quantity here is not the raw gain. It is **normalized tax
recovery**:

```
NTR = ( s(theta_sd) - s(theta_safe) ) / ( s(theta_base) - s(theta_safe) )
```

the fraction of a model's *own* safety tax that self-distillation gives back.
Each model is scored against its own ceiling, which divides headroom out.

- If **NTR is flat** across scale, the headline gradient was bookkeeping. That
  is a negative result, and a genuinely useful one.
- If **NTR is peaked**, there is a real small-model effect, and the peak
  location is a number worth explaining.

Both outcomes are publishable. Commit to the metric before you see the curve.

## The three checkpoints

Everything rests on training three models at every rung of a size ladder:

| Checkpoint | What it is | Role |
|---|---|---|
| `theta_base` | released instruct model | capability ceiling, and the teacher |
| `theta_safe` | **you train this**, SFT + DPO on safety data | the model that paid the tax |
| `theta_sd` | **you train this**, on-policy self-distillation | the model that recovered some of it |

You must train `theta_safe` yourself. Using a vendor-aligned model would leave
the tax confounded with undocumented vendor post-training, and the denominator
of NTR would mean nothing.

## Setup

```bash
git clone <your-fork> && cd self-distill-safety-tax
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

huggingface-cli login     # for gated model weights
```

A single 24GB GPU is enough for the 0.6B and 1.7B rungs with LoRA, which is
where the interesting part of the curve lives. Start there. The 4B and 8B
rungs mostly serve to show the effect shrinking.

## The first three things to run

**1. Confirm the plumbing works, on CPU, in seconds.**

```bash
python -c "
from src.config import load_config
from src.evaluate import normalized_tax_recovery as ntr
cfg = load_config('configs/default.yaml')
print('config ok:', cfg.run_name)
print('NTR(60,50,57) =', ntr(60,50,57))   # 0.7: recovered 70% of a 10-point tax
print('NTR(50,50,55) =', ntr(50,50,55))   # None: no tax, so nothing to recover
"
```

**2. Create a tax, and measure it.** This is the make-or-break step: if your
safety recipe does not cost capability, there is nothing for stage 2 to
recover and the project has no subject.

```bash
scripts/run_safety_align.sh Qwen/Qwen3-0.6B outputs/qwen3-0.6b-safety-aligned

python -m src.train eval --checkpoint Qwen/Qwen3-0.6B \
  --out results/qwen3-0.6b_base.json
python -m src.train eval --checkpoint outputs/qwen3-0.6b-safety-aligned \
  --out results/qwen3-0.6b_safe.json
```

Compare the two `capability_mean` values. You want a clearly positive gap. If
it is near zero, make the safety stage more aggressive (more epochs, more
data, stronger DPO beta) before going further.

**3. Recover it, and get your first table row.**

```bash
python -m src.train distill --config configs/default.yaml
scripts/run_eval.sh qwen3-0.6b Qwen/Qwen3-0.6B \
  outputs/qwen3-0.6b-safety-aligned \
  outputs/qwen3-0.6b-onpolicy-sd/final
```

That prints one row of the paper's Table 1, NTR column included. One row, on
one rung, is a real result and a good first week.

## Layout

```
configs/      one YAML per rung; change only the `models` block between them
src/
  config.py     config loading, validated up front
  data.py       benign prompt pool, safety data, eval suites (all downloaded)
  models.py     checkpoint loading, LoRA, teacher/student pairing
  distill.py    the on-policy self-distillation loop  <- the method
  evaluate.py   capability + safety scoring, and NTR  <- the metric
  probes.py     refusal-direction geometry, entropy   <- the mechanism
  train.py      CLI: distill | eval | probe | ladder
scripts/      end-to-end runners, including the full ladder
experiments/  the log template; its rows map onto Table 1
```

## Running the whole study

```bash
SIZES="qwen3-0.6b qwen3-1.7b" scripts/run_ladder.sh
```

Validate on one rung first. The full four-rung sweep is many hours of compute.

## Two rules worth keeping

**Run the controls.** The on-policy number alone proves nothing. The claim is
that on-policy beats off-policy and plain SFT, and that the margin widens as
models shrink. Without the other arms there is no margin to report.

**Always report the safety audit.** A capability gain paired with a rise in
harmful-request compliance is not recovery, it is de-alignment. `run_eval.sh`
scores AdvBench alongside everything else for exactly this reason.

## Where the paper lives

`paper/main.tex` is a placeholder draft: the structure, the argument and the
table are real, and the numbers are marked `XX.X` until you replace them.
Fill in `experiments/log_template.md` as you go and the table fills itself in.
