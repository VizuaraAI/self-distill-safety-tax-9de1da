#!/usr/bin/env bash
# Stage 2: on-policy self-distillation, plus its control arms.
#
# Run the controls. The on-policy number alone proves nothing: the paper's
# claim is that on-policy beats off-policy and plain SFT, and that the margin
# widens as models shrink. Without the other arms there is no margin to report.
#
# Usage: scripts/run_distill.sh configs/qwen3_0.6b.yaml

set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
echo "=== Stage 2: on-policy self-distillation ==="
echo "config: ${CONFIG}"

# Main arm: fully on-policy, student-sampled rollouts.
python -m src.train distill --config "${CONFIG}" \
  --set distill.on_policy_fraction=1.0

# Control 1: off-policy, teacher-sampled rollouts. Isolates the contribution
# of sampling from the student rather than the teacher.
python -m src.train distill --config "${CONFIG}" \
  --set distill.on_policy_fraction=0.0 \
  --set run_name=offpolicy \
  --set output_dir=outputs/offpolicy

# Control 2: reverse KL (the MiniLLM choice). If the small-model effect shows
# up under only one divergence direction, that localizes the mechanism.
python -m src.train distill --config "${CONFIG}" \
  --set distill.kl_direction=reverse \
  --set run_name=reverse-kl \
  --set output_dir=outputs/reverse-kl

echo
echo "Done. Evaluate every arm before drawing any conclusion."
