#!/usr/bin/env bash
# Stage 1: create the safety tax.
#
# Trains theta_safe from theta_base with SFT then DPO on safety preference
# data. You must run this before anything else, because the whole study
# measures a quantity (the tax) that this script creates.
#
# Use TRL's CLI rather than hand-rolling a trainer: this stage is standard and
# not where the novelty is, so the less of it you own the better.
#
# Usage: scripts/run_safety_align.sh Qwen/Qwen3-0.6B outputs/qwen3-0.6b-safety-aligned

set -euo pipefail

BASE_MODEL="${1:?usage: run_safety_align.sh <base_model_id> <output_dir>}"
OUT_DIR="${2:?usage: run_safety_align.sh <base_model_id> <output_dir>}"

echo "=== Stage 1: safety alignment ==="
echo "base:   ${BASE_MODEL}"
echo "output: ${OUT_DIR}"

# SFT on the safer response of each preference pair.
python -m trl.scripts.sft \
  --model_name_or_path "${BASE_MODEL}" \
  --dataset_name PKU-Alignment/PKU-SafeRLHF \
  --output_dir "${OUT_DIR}-sft" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_length 1024 \
  --bf16 \
  --logging_steps 25

# DPO on the full preference pairs, starting from the SFT checkpoint.
python -m trl.scripts.dpo \
  --model_name_or_path "${OUT_DIR}-sft" \
  --dataset_name PKU-Alignment/PKU-SafeRLHF \
  --output_dir "${OUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-7 \
  --beta 0.1 \
  --max_length 1024 \
  --bf16 \
  --logging_steps 25

echo
echo "Done. Now measure the tax you just created:"
echo "  python -m src.train eval --checkpoint ${BASE_MODEL} --out results/base.json"
echo "  python -m src.train eval --checkpoint ${OUT_DIR}   --out results/safe.json"
echo "If the gap between those two capability means is near zero, this recipe"
echo "did not produce a tax, and there is nothing for stage 2 to recover."
