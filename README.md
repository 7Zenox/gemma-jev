# gemma-jev

Generation-free typed decisions on Gemma models, on a Mac.

Most decisions software asks a model to make are small: *route this ticket*, *is this
claim supported*, *does this need approval*. A chat model can answer, but it spends its
time generating text that the caller immediately parses back into an `if`.

This project reads the answer straight out of the logits instead. Options are presented
as lettered slots, and the probability distribution is taken over the token IDs of those
letters. Nothing is sampled, so nothing can be malformed, and one state prefill can be
shared across every question asked about that state.

It is a port of [SAGAR-TAMANG/sarvam-jev](https://github.com/SAGAR-TAMANG/sarvam-jev)
(itself built on TheoLeeCJ's [SemIf](https://github.com/TheoLeeCJ/SemIf), formerly
OpenJev) from Sarvam-1 to Gemma 3 and Gemma 4, run on Apple Silicon (MPS, bf16) instead
of CUDA. The interface pattern is TypeSafe's Jev, whose model and training are not
public; this reproduces the pattern with open models, not Jev itself.

## Status

| Phase | Goal | State |
| --- | --- | --- |
| 1 | Scaffold: load Gemma, get logits and a reusable KV cache | **done** (MPS/CPU/CUDA, `--device`, `--dtype`) |
| 2 | Letter-slot probing on Gemma's tokenizer | **done**, pinned in `tests/test_prompts.py` |
| 3 | Prompting: base-model completion and instruct-model chat | **done**, both measured |
| 4 | Shared-state batching | code ported and branch-vs-fresh checked on Gemma 3 270M; **bf16 vs fp32 argmax-agreement check not run** |
| 5 | Benchmark against the published ladder | **done** for Gemma 3 (270M, 1B, 4B), Gemma 4 E2B, and Qwen3.5-4B |
| 6 | Demos (server split-screen, browser) | carried over from upstream, **not verified on Gemma**; browser demo not copied |

## Phase 5 results

144 authored decisions, 3 options each (`authored144.jsonl` from SemIf), Apple M3 Pro
18GB, MPS, bf16, transformers 5.17.0. The metric is mean family balanced accuracy.
Evidence is in [`results/`](results/): row-level predictions in `results/raw/`,
checksummed in `results/raw/SHA256SUMS`, headline numbers in `results/phase0-summary.json`.

| System | Prompt | Score |
| --- | --- | ---: |
| chance | | 0.333 |
| Gemma 3 270M (base) | 3-shot completion | 0.293 |
| Gemma 3 1B (base) | 3-shot completion | 0.331 |
| Gemma 3 1B-it | 3-shot completion | 0.355 |
| Qwen3-0.6B *(SemIf, published)* | JSON chat | 0.440 |
| Gemma 4 E2B (base) | 3-shot completion | 0.508 |
| sarvam-1 2B base *(sarvam-jev, published)* | 3-shot completion | 0.516 |
| Gemma 3 4B (base) | 3-shot completion | 0.566 |
| MiniCPM5-2B *(SemIf, published)* | JSON chat | 0.686 |
| Gemma 4 E2B-it | plain-text chat | 0.778 |
| **Gemma 4 E2B-it** | **JSON chat** | **0.807** |
| **Qwen3.5-4B** *(this repo, reproduces SemIf)* | JSON chat | **0.813** |
| Qwen3.5-4B *(SemIf, published)* | JSON chat | 0.813 |

Per family (balanced accuracy) for the two top rows:

| | candidate selection | evidence interpretation | rule application |
| --- | ---: | ---: | ---: |
| Gemma 4 E2B-it, JSON chat | 0.925 | 0.848 | 0.648 |
| Qwen3.5-4B, JSON chat | 0.690 | 0.871 | 0.879 |

### What the numbers say

- **Instruction tuning is what matters, not the model family.** Base Gemma models score
  0.29 to 0.57 with a few-shot prompt; the instruct Gemma 4 E2B-it scores 0.81 with a
  chat prompt. Moving from base E2B to E2B-it, 0.508 to 0.807, is the largest effect
  measured here. The upstream finding that "the mechanism works and the model is the
  limit" holds for Gemma too.
- **The port is faithful.** Qwen3.5-4B through this code scores 0.8132 against SemIf's
  published 0.813, on different hardware (MPS, not an RTX 3090).
- **Gemma 4 E2B-it matches Qwen3.5-4B overall** (0.807 vs 0.813, within noise) at a similar
  size on disk (9.6GB vs 8.7GB). They fail in different places: Gemma is far better at
  candidate selection, Qwen at rule application.
- **The prompt layout is a smaller effect.** The JSON payload beat plain text by 0.03
  on E2B-it, which is inside the noise.
- **Gemma 3 at 1B and below is at chance** on this task, with or without instruction
  tuning. Gemma 3 4B base reaches 0.566, in the same tier as sarvam-1 (0.516).

### Caveats

- 144 rows, 48 per family: differences under about 0.04 are within noise.
- The published rows come from other runs on other hardware, so cross-row comparisons
  with them are not controlled. Only the Qwen3.5-4B row was re-measured here.
- Probabilities are **conditional on the supplied options and uncalibrated**; do not
  threshold on them.
- Gold labels in the fixture are model-reviewed, not human-adjudicated.
- Not run: bf16 vs fp32 agreement (the shared-state precision check), Gemma 4 E4B,
  any Gemma above 4B, and quantized (GGUF) builds. `gemma4:12b` in Ollama is a Q4_K_M
  instruct build and cannot supply the letter-logit readout, since Ollama does not
  expose restricted logits or a branchable KV cache.
- The plain-text chat run of E2B-it predates the `enable_thinking=False` flag (see
  below). The JSON-chat run was repeated with it and scored identically (0.8067), so
  Gemma's template is unaffected, but the plain-text run was not repeated.
- Result files record `revision: unpinned-local-development`. The Qwen3.5-4B weights
  were downloaded at SemIf's pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

## Gemma-specific details

Three things differ from the Sarvam original, and each one broke a run before it was
fixed:

1. **`<bos>` is required.** Gemma degrades badly without a leading BOS, and the upstream
   code never adds one. `core.encode_text` adds exactly one to every encoding.
2. **Chat prompts read bare letters, completion prompts read `▁A`.** After `Answer:` the
   natural token is `▁A`; after a chat turn's `model\n` it is a bare `A`. Probing only for
   boundary stability picked `▁A` in chat and read about 0% of the probability mass.
   `core.slot_separators(style)` orders the probe by style.
3. **Thinking is switched off** (`enable_thinking=False`, as SemIf does). Qwen3.5's
   template ends the prompt on `<think>`, where the model wants to reason, not answer:
   the first Qwen run scored 0.49 with about 0% mass on the letters, and 0.813 once fixed.

Gemma 3 and Gemma 4 share one SentencePiece vocabulary, and letters resolve to single
tokens `▁A`..`▁D` (or bare `A`..`D` in chat). The KV cache is a `DynamicCache` with
`reorder_cache`, and `logits_to_keep` accepts a tensor of positions, so the shared-state
path needed no rewrite.

## Quick start

```bash
uv sync          # creates .venv from pyproject.toml + uv.lock
uv run pytest tests/ -q
```

`reference/` is gitignored; fetch the fixture once:

```bash
mkdir -p reference/openjev/benchmarks/data
gh api repos/TheoLeeCJ/SemIf/contents/benchmarks/data/authored144.jsonl \
  -H "Accept: application/vnd.github.raw" > reference/openjev/benchmarks/data/authored144.jsonl
```

Score the fixture. Instruct models use a chat style; base models use completion:

```bash
uv run python bench/phase0.py --model google/gemma-4-E2B-it --style chat-json \
  --input reference/openjev/benchmarks/data/authored144.jsonl \
  --output results/local/phase0-gemma4-e2b-it-chat-json.json

uv run python bench/phase0.py --model google/gemma-4-E2B --style completion \
  --input reference/openjev/benchmarks/data/authored144.jsonl \
  --output results/local/phase0-gemma4-e2b-completion.json
```

`--model` also takes a local directory, so a model can be downloaded once
(`hf download <id> --local-dir ~/models/<name>`) and reused. Add `--device cpu|mps|cuda`
and `--dtype float32|bfloat16` as needed. Benchmark outputs are create-only.

Score JSONL directly, or check shared-state scoring against fresh scoring:

```bash
uv run python -m gemma_jev.cli --mode shared --style chat-json \
  --model google/gemma-4-E2B-it --input decisions.jsonl --output out.jsonl
uv run python bench/shared_check.py --model google/gemma-3-1b-pt --dtype float32 \
  --input reference/openjev/benchmarks/data/authored144.jsonl \
  --output results/local/shared-check.json
```

## How it works

```
                        ┌── "which team?"     ──→ logits[A,B,C,D] ──→ softmax
state (prefilled once)  ├── "how urgent?"     ──→ logits[A,B,C,D] ──→ softmax
   KV cache             ├── "is it angry?"    ──→ logits[A,B]     ──→ softmax
                        └── "needs approval?" ──→ logits[A,B]     ──→ softmax
                            (one batched forward pass)
```

1. **Prefill** the shared head (instructions, demonstrations, the state) into a KV cache.
2. **Branch** the cache across criteria, one per batch row, with `position_ids`
   continuing from the end of the prefix. Branches cannot see each other.
3. **Read** the logits at each branch's final position, restricted to that criterion's
   answer-letter token IDs, and softmax over just those.

Three prompt styles: `completion` (few-shot, for base models), `chat` (plain text in the
model's chat template), and `chat-json` (SemIf's zero-shot prompt: a system instruction
plus a JSON `{evidence, criterion, options}` user message).

## Honest limits

- Branch isolation comes from the batch dimension, so the state KV is held once per
  branch. This is the same shortcut both reference implementations take.
- BF16 batching is not bit-exact. Upstream measured shared and fresh scoring disagreeing
  on 8 of 72 argmaxes for a near-indifferent base model and 5 to 6 of 777 for a confident
  4B model. That check has not been run on Gemma here.
- The server demo (`server/`, `gemma_jev.generate`) was carried over from upstream with
  its environment variables renamed to `GEMMA_JEV_*`; it has not been run on Gemma.

## Credits

The engine structure, the letter-slot readout, the prefix-verification discipline and the
evaluation metric follow [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) (MIT), via
the Sarvam adaptation [SAGAR-TAMANG/sarvam-jev](https://github.com/SAGAR-TAMANG/sarvam-jev).
This project is not affiliated with either, or with TypeSafe.
