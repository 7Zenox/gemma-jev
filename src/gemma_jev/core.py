"""Input validation, prompt construction, answer-slot discovery, and model loading.

Adapted from TheoLeeCJ/openjev (MIT) via SAGAR-TAMANG/sarvam-jev. The substantive
changes are model-family specific:

* Base (pretrained) models are driven by a few-shot completion prompt; instruction-
  tuned models by their chat template. Both styles are kept side by side.
* SentencePiece tokenizers never emit a bare ``A`` after ``Answer:`` -- the natural
  continuation is ``_A``. Answer slots are therefore discovered by probing which
  continuation keeps the prompt boundary stable, rather than being assumed. Gemma 3
  and Gemma 4 share one vocabulary and resolve to ``_A``.._D`` with a space separator.
* Gemma is trained with a leading ``<bos>`` and degrades badly without one, so every
  encoding goes through :func:`encode_text`, which adds it unless the text has one.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"

INSTRUCTION = (
    "Apply the criterion to the evidence and choose exactly one listed option. "
    "Answer with only the uppercase letter of that option."
)

# Fixed, project-authored demonstrations for base (non-instruction-tuned) models.
# Deliberately not drawn from any evaluation fixture, and the correct letters are
# spread across positions so the format is taught without teaching a letter prior.
FEWSHOT = [
    {
        "state": "The parcel left the Chennai sorting facility on Tuesday. "
                 "The recipient reports it has not arrived.",
        "question": "Assess the claim: the parcel has been delivered.",
        "options": [
            "The evidence establishes the claim",
            "The evidence does not establish either",
            "The evidence establishes the opposite",
        ],
        "answer": 2,
    },
    {
        "state": "A customer writes that their card was charged twice for one order "
                 "and asks for one of the charges to be reversed.",
        "question": "Which queue should handle this request?",
        "options": [
            "Account access and authentication support",
            "Billing and payment support",
            "Sales and product evaluation",
        ],
        "answer": 1,
    },
    {
        "state": "Policy: refunds above 5000 rupees need manager approval. "
                 "Request: refund 900 rupees for a damaged item.",
        "question": "Does this request need manager approval under the stated policy?",
        "options": [
            "Manager approval is required",
            "Manager approval is not required",
        ],
        "answer": 0,
    },
]


def validate_row(row: dict) -> None:
    """Reject anything the scorers cannot score exactly as written."""
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError(f"options must contain 2-{len(LETTERS)} entries")
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str) \
                or not isinstance(option.get("description"), str):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")


def render_state(state) -> str:
    """Render a state as prompt text, preserving structure for JSON states."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def _block(state, question: str, descriptions: list[str]) -> tuple[str, str]:
    """Return (state_part, tail_part) for one decision.

    The split point matters: everything in ``state_part`` is identical across all
    decisions that share a state, which is what makes prefix reuse possible.
    """
    state_part = f"Evidence:\n{render_state(state)}\n"
    lines = [f"\nCriterion: {question}", "Options:"]
    for letter, description in zip(LETTERS, descriptions):
        lines.append(f"{letter}. {description}")
    lines.append("Answer:")
    return state_part, "\n".join(lines)


def _fewshot_text(shots: int) -> str:
    parts = [INSTRUCTION, ""]
    for example in FEWSHOT[:shots]:
        state_part, tail = _block(example["state"], example["question"], example["options"])
        parts.append(state_part + tail + " " + LETTERS[example["answer"]] + "\n")
    return "\n".join(parts)


def completion_prompt(row: dict, shots: int = len(FEWSHOT)) -> tuple[str, str]:
    """Build a base-model completion prompt, split at the state boundary.

    Returns ``(head, tail)`` where ``head + tail`` is the full prompt and ``head``
    ends immediately after the state. ``head`` is shared by every decision over the
    same state.
    """
    validate_row(row)
    preamble = _fewshot_text(shots) if shots else INSTRUCTION + "\n\n"
    state_part, tail = _block(
        row["state"], row["question"], [option["description"] for option in row["options"]]
    )
    return preamble + state_part, tail


def chat_prompt(tokenizer, row: dict) -> tuple[str, str]:
    """Build an instruction-model prompt via the tokenizer's chat template.

    Thinking is switched off (``enable_thinking=False``, as SemIf does): a thinking
    template ends on ``<think>``, where the model wants to reason, not answer, and the
    letter slots carry ~0 probability mass. Templates without the variable ignore it.

    Split at the state boundary the same way, by locating the rendered state inside
    the templated text.
    """
    validate_row(row)
    state_part, tail = _block(
        row["state"], row["question"], [option["description"] for option in row["options"]]
    )
    content = state_part + tail
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if prompt.count(content) != 1:
        raise ValueError("Chat template altered or duplicated the decision payload")
    head = prompt[: prompt.index(content)] + state_part
    return head, prompt[len(head):]


# Both chat builders pass enable_thinking=False, as SemIf does: a thinking template ends
# on <think>, where the model wants to reason rather than answer, so the letter slots carry
# ~0 probability mass. Templates without that variable ignore it.
#
# SemIf's zero-shot chat instruction (TheoLeeCJ/SemIf, src/semif_phase1/core.py); the
# 0.813 Qwen3.5-4B reference was measured with this system message.
JSON_INSTRUCTION = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def chat_json_prompt(tokenizer, row: dict) -> tuple[str, str]:
    """SemIf-style chat prompt: system instruction plus one JSON user payload.

    ``{"evidence": ..., "criterion": ..., "options": [{"letter", "description"}]}``,
    zero-shot. Evidence comes first in the payload so everything up to the end of the
    evidence is a shared head across criteria over one state.
    """
    validate_row(row)
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [{"letter": LETTERS[i], "description": option["description"]}
                    for i, option in enumerate(row["options"])],
    }
    content = json.dumps(payload, ensure_ascii=False)
    head_content = '{"evidence": ' + json.dumps(row["state"], ensure_ascii=False)
    if not content.startswith(head_content):
        raise ValueError("JSON payload does not start with its evidence")
    messages = [
        {"role": "system", "content": JSON_INSTRUCTION},
        {"role": "user", "content": content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if prompt.count(content) != 1:
        raise ValueError("Chat template altered or duplicated the decision payload")
    head = prompt[: prompt.index(content)] + head_content
    return head, prompt[len(head):]


def slot_separators(style: str) -> tuple[str, ...]:
    return ("", " ") if style.startswith("chat") else (" ", "")


def build_prompt(tokenizer, row: dict, style: str, shots: int = len(FEWSHOT)):
    if style == "completion":
        return completion_prompt(row, shots)
    if style == "chat":
        return chat_prompt(tokenizer, row)
    if style == "chat-json":
        return chat_json_prompt(tokenizer, row)
    raise ValueError(f"Unknown prompt style {style!r}")


def answer_slots(tokenizer, prompt: str, count: int,
                 separators: tuple[str, ...] = (" ", "")) -> tuple[list[int], str]:
    """Find token IDs for the first ``count`` answer letters at this exact boundary.

    A SentencePiece vocabulary typically holds ``_A`` as the ordinary token and a
    separate rare ``A``; which one continues the prompt depends on the preceding
    character. Probe both and keep whichever appends exactly one token per letter
    without disturbing the prompt's own tokenization.

    ``separators`` is tried in order. A chat prompt ends on a fresh line inside the
    model turn, where the model emits a bare ``A``, not ``_A``, so chat passes
    ``("", " ")``. Probing only for stability picked ``_A`` there and read ~0 of the
    probability mass.
    """
    base = tokenizer.encode(prompt, add_special_tokens=False)
    if not base:
        raise ValueError("Prompt encoded to zero tokens")
    for separator in separators:
        slots = []
        for letter in LETTERS[:count]:
            encoded = tokenizer.encode(prompt + separator + letter, add_special_tokens=False)
            if len(encoded) != len(base) + 1 or encoded[: len(base)] != base:
                slots = []
                break
            slots.append(encoded[-1])
        if slots and len(slots) == len(set(slots)):
            return slots, separator
    raise ValueError(
        "No answer-letter continuation keeps the prompt boundary stable; "
        "the prompt tail needs adjusting for this tokenizer"
    )


def softmax(values: list[float]) -> list[float]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Need at least two finite scores")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def encode_text(tokenizer, text: str) -> list[int]:
    """Token IDs for ``text`` with exactly one leading BOS, when the tokenizer has one.

    Chat templates already render ``<bos>`` into the text, so it is only added when
    absent. ``answer_slots`` deliberately compares prompts without it: the BOS is a
    constant prefix and cannot change how the tail tokenizes.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos = tokenizer.bos_token_id
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids
    return ids


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def resolve_device(torch, device: str = "auto"):
    """Pick the compute device: CUDA, then Apple MPS, then CPU."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one CUDA GPU, e.g. with CUDA_VISIBLE_DEVICES")
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(torch, device) -> None:
    """Block until queued work on ``device`` finishes, so timings mean something."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_causal_model(source: str, revision: str = "", trust_remote_code: bool = False,
                      device: str = "auto", dtype: str = "auto"):
    """Load one pinned causal model onto a single device.

    ``dtype="auto"`` is bfloat16 on CUDA/MPS and float32 on CPU. Pass ``float32`` to
    get the reference side of the precision check; avoid float16, which overflows
    Gemma's activations.
    """
    import torch
    import transformers

    local = Path(source).exists()
    if not local and revision and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A supplied revision must be a 40-character commit hash")
    common = {
        "revision": None if local or not revision else revision,
        "local_files_only": local,
        "trust_remote_code": trust_remote_code,
    }
    target = resolve_device(torch, device)
    if dtype == "auto":
        resolved = torch.float32 if target.type == "cpu" else torch.bfloat16
    else:
        resolved = getattr(torch, dtype)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unknown dtype {dtype!r}")
    device_map = {"": str(target)}

    config = transformers.AutoConfig.from_pretrained(source, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    causal_class = transformers.AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        # The checkpoint is multimodal; score it with the text-only decoder, as SemIf does.
        causal_class = transformers.Qwen3_5ForCausalLM
        config = config.get_text_config()
    model, loading = causal_class.from_pretrained(
        source,
        config=config,
        dtype=resolved,
        device_map=device_map,
        low_cpu_mem_usage=True,
        output_loading_info=True,
        **common,
    )
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load completely: {loading}")
    model.eval()
    metadata = {
        "source": source,
        "revision": revision or "unpinned-local-development",
        "dtype": str(resolved).replace("torch.", ""),
        "device": str(next(model.parameters()).device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata
