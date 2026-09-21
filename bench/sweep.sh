#!/usr/bin/env bash
# Phase 5 sweep: full authored144 fixture, one create-only result per (model, style).
# Usage: bench/sweep.sh   (results land in results/local/, which is gitignored)
set -u
cd "$(dirname "$0")/.."
FIX=reference/openjev/benchmarks/data/authored144.jsonl
run() { # model style tag
  out="results/local/phase0-$3.json"
  [ -e "$out" ] && { echo "skip $3 (exists)"; return; }
  echo "=== $3 ($1, $2) $(date +%T)"
  uv run python bench/phase0.py --model "$1" --style "$2" --input "$FIX" --output "$out" 2>&1 \
    | grep -v -i "warn\|Loading weights"
}
run google/gemma-3-270m     completion gemma3-270m-completion
run google/gemma-3-1b-pt    completion gemma3-1b-pt-completion
run google/gemma-3-1b-it    completion gemma3-1b-it-completion
run google/gemma-3-1b-it    chat       gemma3-1b-it-chat
run google/gemma-3-4b-pt    completion gemma3-4b-pt-completion
run google/gemma-3-4b-it    chat       gemma3-4b-it-chat
run google/gemma-4-E2B      completion gemma4-e2b-completion
run google/gemma-4-E2B-it   chat       gemma4-e2b-it-chat
