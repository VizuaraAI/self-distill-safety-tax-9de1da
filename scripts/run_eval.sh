#!/usr/bin/env bash
# Stage 3: evaluate all three checkpoints at one rung, then print the row.
#
# The three evaluations must use the same suites, the same subsample and the
# same seed. That is the only reason the NTR denominator means anything.
#
# Usage: scripts/run_eval.sh qwen3-0.6b Qwen/Qwen3-0.6B outputs/qwen3-0.6b-safety-aligned outputs/qwen3-0.6b-onpolicy-sd/final

set -euo pipefail

SIZE="${1:?usage: run_eval.sh <size_tag> <base_ckpt> <safe_ckpt> <sd_ckpt>}"
BASE_CKPT="${2:?missing base checkpoint}"
SAFE_CKPT="${3:?missing safety-aligned checkpoint}"
SD_CKPT="${4:?missing self-distilled checkpoint}"
RESULTS="${RESULTS:-results}"

mkdir -p "${RESULTS}"

echo "=== Stage 3: evaluating rung ${SIZE} ==="
for pair in "base:${BASE_CKPT}" "safe:${SAFE_CKPT}" "sd:${SD_CKPT}"; do
  tag="${pair%%:*}"
  ckpt="${pair#*:}"
  echo "--- ${tag} (${ckpt})"
  python -m src.train eval \
    --checkpoint "${ckpt}" \
    --out "${RESULTS}/${SIZE}_${tag}.json"
done

echo
echo "=== Current state of the main results table ==="
python -m src.train ladder --results "${RESULTS}" --sizes "${SIZE}"
