#!/usr/bin/env bash
# Phase 4 precision check: shared-vs-fresh argmax agreement in bf16 and fp32.
set -u
cd "$(dirname "$0")/.."
FIX=reference/openjev/benchmarks/data/authored144.jsonl
MODEL=${1:-google/gemma-3-1b-pt}
TAG=$(basename "$MODEL")
for dt in bfloat16 float32; do
  out="results/local/shared-check-$TAG-$dt.json"
  [ -e "$out" ] && continue
  echo "=== $TAG $dt $(date +%T)"
  uv run python bench/shared_check.py --model "$MODEL" --dtype $dt --input "$FIX" --output "$out" 2>&1 | grep -v -i "warn\|Loading weights"
done
