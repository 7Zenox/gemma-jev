# gemma-jev: Porting sarvam-jev to Gemma — Implementation Plan

Source project: [SAGAR-TAMANG/sarvam-jev](https://github.com/SAGAR-TAMANG/sarvam-jev)
Goal: reproduce the generation-free, letter-logit-readout inference engine on Gemma models, and benchmark it against the existing sarvam-1 / Qwen / MiniCPM results.

---

## 0. Background (why this is expected to work)

The engine does three things, none of which are Sarvam-specific:

1. **Prefill once** — the shared state (instructions, few-shot demos, ticket/context) goes through the model a single time, producing a KV cache.
2. **Branch per question** — each typed criterion is appended as its own branch along the batch dimension, `position_ids` continuing from the prefix end. Branches can't see each other.
3. **Read letter logits** — options are rendered as `A.`, `B.`, `C.`… and instead of sampling, the code reads logits at each branch's final position restricted to that criterion's letter-token IDs, then softmaxes over just those.

Everything here is standard `transformers` KV-cache + logits access. The two genuinely model-specific risk points are the **tokenizer's letter-slot behavior** and **base-vs-instruction-tuned prompting**, both of which the original repo already isolates into their own modules — so this is a targeted port, not a rewrite.

---

## Phase 1 — Fork & scaffold

- [ ] Fork `sarvam-jev` → `gemma-jev`
- [ ] Update `pyproject.toml` dependency pins if needed (Gemma requires a recent `transformers`; check min version for Gemma 2/3 support)
- [ ] Add config plumbing for `--model google/gemma-2b`, `google/gemma-2-9b`, `google/gemma-2-9b-it`, etc. (start with the smallest base model to mirror sarvam-1's 2B-base comparison)
- [ ] Confirm the model loads and produces logits + a reusable KV cache locally before touching any Jev-specific logic

**Deliverable:** a branch that loads a Gemma checkpoint through the existing `sarvam_jev` scaffolding, unmodified, just to confirm plumbing works.

---

## Phase 2 — Port `core.answer_slots` (tokenizer probing)

This is the highest-risk, highest-priority piece. Sarvam's SentencePiece vocab has both an ordinary `▁A` token and a rare byte-level `A`; the original code probes both and keeps whichever continues the prompt as exactly one token without disturbing the prompt's own tokenization.

- [ ] Write a standalone probing script against Gemma's tokenizer: for each letter A–Z (or however many options are needed, typically A–D), test leading-space and no-space variants
- [ ] Verify each candidate encodes to exactly one token
- [ ] Verify appending it does **not** retroactively change the tokenization of the preceding prompt text (this is the invariant `tests/test_prompts.py` pins in the original repo — port that test file first, against Gemma)
- [ ] Document findings: does Gemma have the same dual-token ambiguity, a different one, or none at all? This determines how much of `answer_slots` needs rewriting vs. reusing as-is
- [ ] Handle Gemma's own special tokens (`<bos>`, `<start_of_turn>`, etc.) — confirm they don't interfere with letter-slot placement, especially if testing the `-it` chat template later

**Deliverable:** `answer_slots` ported and passing a Gemma-specific version of `test_prompts.py`.

---

## Phase 3 — Prompting strategy

Two tracks, both worth running since the original repo found chat-template prompting *underperformed* few-shot completion prompting on their base model (0.404 vs. 0.516 balanced accuracy) — an unintuitive result that may or may not hold for Gemma.

**Track A — Gemma base model, completion style**
- [ ] Port the fixed few-shot demonstration set, with correct answers deliberately spread across letter positions (avoids teaching a letter-position prior instead of the task)
- [ ] Tune demo count (repo tested 0/1/3-shot on sarvam-1)

**Track B — Gemma-it, chat style**
- [ ] Adapt to Gemma's official chat template (`<start_of_turn>user ... <end_of_turn>`)
- [ ] Test whether instruction-tuning changes the completion-vs-chat performance ordering found on sarvam-1

- [ ] Run both tracks on the same fixture and compare — don't assume one wins without testing

**Deliverable:** a working prompt template module for at least the base-model track, with the chat-model track as a comparison point.

---

## Phase 4 — Shared-state batching (mostly framework code)

This layer is the least Gemma-specific and should port with minimal edits.

- [ ] Swap `AutoModelForCausalLM.from_pretrained(...)` / tokenizer calls to Gemma checkpoints
- [ ] Confirm KV-cache prefill + batch-dim branching works with Gemma's attention implementation (check if Gemma's grouped-query attention or sliding-window attention in Gemma 2/3 changes how `position_ids` need to be handled — this is the one place Gemma's architecture *could* diverge from a plain decoder)
- [ ] Re-run the shared-vs-fresh scoring consistency check the original repo did (they found BF16 batching caused a ~11% argmax disagreement rate on sarvam-1 because it sat near-indifferent between options, and fp32 collapsed that drift to ~0). Repeat this precision check on Gemma rather than assuming it transfers.

**Deliverable:** `--mode shared` scoring working end-to-end on a Gemma checkpoint, plus a precision-sensitivity note (BF16 vs fp32 argmax agreement rate).

---

## Phase 5 — Benchmark

- [ ] Run `bench/phase0.py` (or its Gemma-adapted equivalent) against `authored144.jsonl` with each Gemma variant tested
- [ ] Record mean family balanced accuracy for at least:
  - Gemma base (smallest size, e.g. 2B) — direct comparison point to sarvam-1's 0.516
  - Gemma-it (same size) — direct comparison point to the chat-template question from Phase 3
  - One larger Gemma variant if compute allows, as a comparison to the Qwen3.5-4B (0.813) and MiniCPM5-2B (0.686) reference points
- [ ] Reuse the existing table format so results slot directly next to the published sarvam-1/Qwen/MiniCPM numbers

**Deliverable:** a results table comparable to the original repo's Phase 0 table, committed under `results/`.

---

## Phase 6 (optional / stretch) — Demo parity

- [ ] Port the server split-screen demo (typed readout vs. token-by-token JSON generation) to a Gemma backend
- [ ] Port the browser (wllama/WebGPU) demo if a GGUF-quantized Gemma build is available and licensing permits redistribution
- [ ] Add Gemma presets alongside the existing Hindi/Tamil/Bengali/English fixture presets if there's an Indic-language angle worth testing (Gemma's tokenizer efficiency on Devanagari etc. is untested and was the original motivation for choosing Sarvam — worth a side-by-side token-count comparison)

---

## Open risks to revisit as you go

| Risk | Why it matters | How to resolve |
|---|---|---|
| Letter-slot tokenization differs from Sarvam's | Core mechanism depends on exactly-one-token, non-disturbing letter appends | Phase 2 probing script, test against real Gemma tokenizer, don't assume |
| Base model may lack "decision semantics" like sarvam-1 did | Repo's own finding: mechanism worked, model quality was the bottleneck (0.516 vs 0.813 for a bigger/better model) | Test multiple Gemma sizes/variants, don't over-index on the smallest one |
| BF16 argmax instability | Could be worse or better than Sarvam's ~11% disagreement rate depending on how confident/indifferent Gemma's softmax outputs are | Run the fp32 vs BF16 consistency check in Phase 4 before trusting shared-state results |
| Gemma 2/3 attention differences (sliding window, GQA) | Could affect how KV-cache branching and `position_ids` continuation behave | Explicit test in Phase 4, not just an assumption of drop-in compatibility |

---

## Suggested order of work

1. Phase 1 (scaffold) → 2 (answer slots) → 4 (batching, since it's low-risk) can happen roughly in parallel once Phase 1 is done.
2. Phase 3 (prompting) is the most experimental and should follow once Phase 2 confirms letter slots work.
3. Phase 5 (benchmark) is the checkpoint that tells you whether the port was worth it — treat it as the go/no-go gate before investing in Phase 6.