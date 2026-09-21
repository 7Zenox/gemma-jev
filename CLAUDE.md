# Repository instructions

## Environment

- Apple Silicon Mac (M3 Pro, 18GB unified memory). Models run on MPS in bf16 by default;
  use `--dtype float32` for the precision reference. A 4B-class model in bf16 fits; a 12B does not.
- Use `uv run <command>`: it resolves against `pyproject.toml` and `uv.lock` (Python 3.12
  `.venv`). Do not `uv pip install` into `.venv`.
- `reference/` (the benchmark fixture) and `results/local/` are gitignored scratch space.

## Commands

```bash
uv run pytest tests/ -q                       # tokenizer-level; needs cached Gemma tokenizers
uv run python bench/phase0.py --model <id-or-local-dir> --style chat-json|chat|completion \
  --input reference/openjev/benchmarks/data/authored144.jsonl --output results/local/<new-name>.json
```

Benchmark outputs are create-only: `bench/` scripts refuse to overwrite an existing output
path. Use a new filename rather than deleting the old one.

## Model downloads

Bandwidth is limited (about 5MB/s), so do not download models speculatively or twice.
Before fetching, check what is already on disk: `du -shL` on `~/.cache/huggingface/hub/<model>`
(plain `du` does not follow the symlinks into `blobs/` and reads as almost empty) and `~/models`.
Fetch once with `hf download <id> --local-dir ~/models/<name>` and load from that path.
Never `rm -rf` a completed model cache to save space.

## Conventions

- Probabilities from this engine are **conditional on the supplied options and uncalibrated**.
  Every surface that shows them says so. Do not remove or soften that wording.
- Headline numbers in `README.md` and `results/phase0-summary.json` must be backed by a
  committed row-level file under `results/raw/` (checksummed in `SHA256SUMS`). If a number
  changes, regenerate the evidence.
- Every prompt goes through `core.encode_text` (adds the single leading `<bos>`), chat prompts
  pass `enable_thinking=False`, and answer slots come from `core.answer_slots` with
  `core.slot_separators(style)`. A run with allowed-token mass near 0 is a broken readout, not
  a bad model: check the slots and the prompt tail first.
- Do not hand an autoregressive baseline a worse prompt than the readout gets.
