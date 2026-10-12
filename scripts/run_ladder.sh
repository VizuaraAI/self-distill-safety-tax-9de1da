#!/usr/bin/env bash
# The full study: every rung, end to end, then the paper's main table.
#
# This is the script that produces Table 1 and Figure 2. Expect it to run for
# many hours; start with SIZES="qwen3-0.6b" to validate the pipeline on one
# rung before committing a weekend to all four.
#
# Usage: SIZES="qwen3-0.6b qwen3-1.7b" scripts/run_ladder.sh

set -euo pipefail

SIZES="${SIZES:-qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b}"
RESULTS="${RESULTS:-results}"
mkdir -p "${RESULTS}"

declare -A BASE_IDS=(
  [qwen3-0.6b]="Qwen/Qwen3-0.6B"
  [qwen3-1.7b]="Qwen/Qwen3-1.7B"
  [qwen3-4b]="Qwen/Qwen3-4B"
  [qwen3-8b]="Qwen/Qwen3-8B"
)

for size in ${SIZES}; do
  base="${BASE_IDS[$size]}"
  safe="outputs/${size}-safety-aligned"
  sd="outputs/${size}-onpolicy-sd/final"
  config="configs/${size}.yaml"
  [ -f "${config}" ] || config="configs/default.yaml"

  echo "############ ${size} ############"
  scripts/run_safety_align.sh "${base}" "${safe}"
  python -m src.train distill --config "${config}" \
    --set "models.base_id=${base}" \
    --set "models.student_id=${safe}" \
    --set "output_dir=outputs/${size}-onpolicy-sd"
  scripts/run_eval.sh "${size}" "${base}" "${safe}" "${sd}"

  # Probes on the aligned and self-distilled checkpoints: the entanglement
  # comparison is between these two, at every rung.
  python -m src.train probe --checkpoint "${safe}" --out "${RESULTS}/${size}_safe_probes.json"
  python -m src.train probe --checkpoint "${sd}"   --out "${RESULTS}/${size}_sd_probes.json"
done

echo
echo "=== Table 1 ==="
python -m src.train ladder --results "${RESULTS}" \
  --sizes "$(echo ${SIZES} | tr ' ' ',')"

echo
echo "Read the NTR column, not the delta column. If NTR is flat across rungs,"
echo "you have a negative result worth publishing: the inverse-scale gain was"
echo "headroom. Write that up honestly rather than hunting for a framing that"
echo "makes it look positive."
