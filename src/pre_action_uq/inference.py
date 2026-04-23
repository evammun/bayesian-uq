"""
LLM inference layer using llama-cpp-python for in-process GPU inference.

Key design choices:
  - Model loaded once in-process (no HTTP overhead, no server management)
  - Low-level logit extraction bypasses logits_all=True requirement
  - Qwen3 think mode via manual chat template construction
  - logits_all=False — only the last position's logits are needed

The key insight: llama.cpp always computes logits for the last evaluated
position (it has to, for sampling). We only need logprobs at that one
position (the A/B/C/D answer token). By going below create_completion()
and using eval() + llama_get_logits() directly, we avoid the massive
logits buffer that logits_all=True allocates (n_ctx × vocab_size × 4 bytes).
On Qwen3 with 150K vocab and 8192 context, that saves ~4.5 GB of VRAM.

Logprob format (normalised, used throughout the project):
  [{"token": str, "logprob": float,
    "top_logprobs": [{"token": str, "logprob": float}, ...]}, ...]
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from llama_cpp import Llama
import llama_cpp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Answer letters for MCQA extraction (matches v2 convention)
ANSWER_LETTERS = ["A", "B", "C", "D"]

# Qwen3 chat template tokens
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"

# Token IDs for answer letters — populated on first model load.
# Maps letter -> list of token IDs (bare "A" and space-prefixed " A").
_ANSWER_TOKEN_IDS: dict[str, list[int]] = {}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LlamaCppClient:
    """In-process LLM client using llama-cpp-python.

    Wraps a llama_cpp.Llama instance for GPU-accelerated inference with
    logprob extraction. Uses low-level logit access to avoid the VRAM
    overhead of logits_all=True.

    Usage:
        client = LlamaCppClient("path/to/model.gguf")
        result = client.generate_with_logprobs("...prompt...", think=False)
        print(result["logprobs"])
    """

    def __init__(
        self,
        model_path: str | Path,
        n_gpu_layers: int = -1,
        n_ctx: int = 12288,
        n_batch: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        """Load a GGUF model for inference.

        Args:
            model_path: Path to the GGUF model file.
            n_gpu_layers: Layers to offload to GPU (-1 = all).
            n_ctx: Context window size in tokens.
            n_batch: Max tokens per eval batch. Defaults to n_ctx so the
                full prompt is always processed in one pass. Setting this
                lower than the longest prompt causes chunked eval, which
                triggered batch-size errors during testing.
            seed: Random seed for reproducibility.
            verbose: Print llama.cpp loading diagnostics.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if n_batch is None:
            n_batch = n_ctx

        t0 = time.time()
        self.model = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            seed=seed,
            logits_all=False,  # only need logits at last position
            verbose=verbose,
        )
        self._load_time = time.time() - t0
        self._query_count = 0
        self._n_vocab = self.model.n_vocab()
        self._n_ctx = n_ctx
        # Serialize GPU access — eval + logit extraction is not thread-safe.
        # Kept as a safety net even though the pipeline runs sequentially.
        self._lock = threading.Lock()

        # Discover token IDs for A/B/C/D (bare and space-prefixed)
        self._init_answer_token_ids()

    def _init_answer_token_ids(self) -> None:
        """Find token IDs for 'A', 'B', 'C', 'D' and ' A', ' B', ' C', ' D'."""
        global _ANSWER_TOKEN_IDS
        if _ANSWER_TOKEN_IDS:
            return  # already initialised

        for letter in ANSWER_LETTERS:
            ids = []
            # Bare letter: "A"
            bare = self.model.tokenize(letter.encode(), add_bos=False)
            if len(bare) == 1:
                ids.append(bare[0])
            # Space-prefixed: " A"
            spaced = self.model.tokenize(f" {letter}".encode(), add_bos=False)
            if len(spaced) == 1:
                ids.append(spaced[0])
            _ANSWER_TOKEN_IDS[letter] = ids

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def _full_reset(self) -> None:
        """Clear the KV cache and reset token counter.

        Llama.reset() only sets n_tokens=0 without clearing the KV
        cache, which causes decode failures on subsequent eval() calls.

        The API for clearing the cache has changed across llama-cpp-python
        versions, so we try multiple approaches:
          - 0.3.12+: llama_memory_seq_rm (via llama_get_memory)
          - older: memory_clear on the context object
          - fallback: kv_cache_clear on the context
        """
        ctx = self.model._ctx

        # Try the modern API first (0.3.12+)
        try:
            mem = llama_cpp.llama_get_memory(ctx.ctx)
            llama_cpp.llama_memory_seq_rm(mem, -1, -1, -1)
        except (AttributeError, Exception):
            # Try older API
            try:
                ctx.memory_clear(True)
            except AttributeError:
                try:
                    ctx.kv_cache_clear()
                except AttributeError:
                    # Last resort — Llama.reset() at least zeros n_tokens
                    self.model.reset()

        self.model.n_tokens = 0

    # ------------------------------------------------------------------
    # Low-level logit extraction
    # ------------------------------------------------------------------

    def _extract_answer_letter_logprobs(self) -> list[dict]:
        """Extract logprobs from the last evaluated position.

        Returns top-20 from full vocab plus all A/B/C/D answer tokens
        (even if they're outside the top-20). This ensures
        extract_answer_logprobs() always finds the answer letters.

        Returns:
            Single-element list with one logprob entry dict, compatible
            with the rest of the pipeline.
        """
        ctx = self.model._ctx.ctx
        logits_ptr = llama_cpp.llama_get_logits(ctx)

        if not logits_ptr:
            raise RuntimeError("llama_get_logits returned NULL")

        logits = np.ctypeslib.as_array(logits_ptr, shape=(self._n_vocab,)).copy()

        # Log-softmax (numerically stable)
        max_logit = float(logits.max())
        shifted = logits - max_logit
        log_sum_exp = np.log(np.exp(shifted).sum())
        log_probs = shifted - log_sum_exp

        # Top-20 from full vocab
        top_k = 20
        top_indices = np.argpartition(log_probs, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(log_probs[top_indices])[::-1]]
        top_set = set(top_indices.tolist())

        # Ensure all answer letter token IDs are included
        for letter, ids in _ANSWER_TOKEN_IDS.items():
            for tid in ids:
                top_set.add(tid)

        # Sort all by logprob descending
        all_indices = sorted(top_set, key=lambda i: log_probs[i], reverse=True)

        # Build top_logprobs list
        top_logprobs_list: list[dict[str, Any]] = []
        for idx in all_indices:
            token_text = self._token_id_to_text(idx)
            top_logprobs_list.append({
                "token": token_text,
                "logprob": float(log_probs[idx]),
            })

        # Top-1 token
        top_token_text = top_logprobs_list[0]["token"] if top_logprobs_list else ""
        top_token_lp = top_logprobs_list[0]["logprob"] if top_logprobs_list else 0.0

        return [{
            "token": top_token_text,
            "logprob": top_token_lp,
            "top_logprobs": top_logprobs_list,
        }]

    def _token_id_to_text(self, token_id: int) -> str:
        """Convert a token ID to its text representation."""
        try:
            piece_bytes = llama_cpp.llama_token_get_text(
                self.model._model.vocab, token_id
            )
            if piece_bytes:
                text = piece_bytes.decode("utf-8", errors="replace")
                # llama.cpp uses Ġ (0xC4 0xA0) for space prefix — convert to actual space
                return text.replace("\u0120", " ")
            return f"<{token_id}>"
        except Exception:
            return f"<{token_id}>"

    # ------------------------------------------------------------------
    # Core generation (text only, no logprobs)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        think: bool = False,
    ) -> dict[str, Any]:
        """Generate a text completion (no logprobs).

        For logprob extraction, use generate_with_logprobs() instead.
        This method uses create_completion() which works fine without
        logits_all=True when logprobs are not requested.

        Args:
            prompt: Input text. For think=True, wrapped in chat template.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            stop: Optional stop sequences.
            think: Enable Qwen3 /think reasoning mode.

        Returns:
            Dict with keys: response_text, logprobs (always []), thinking_trace.
        """
        full_prompt = (
            self._build_chat_prompt(prompt, think=True) if think else prompt
        )

        with self._lock:
            completion = self.model.create_completion(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=None,
                stop=stop,
                echo=False,
            )
            self._query_count += 1

        raw_text = completion["choices"][0]["text"]

        if think:
            response_text, thinking_trace = _split_think_response(raw_text)
        else:
            response_text = raw_text
            thinking_trace = ""

        return {
            "response_text": response_text,
            "logprobs": [],
            "thinking_trace": thinking_trace,
        }

    # ------------------------------------------------------------------
    # MCQ logprob extraction (low-level)
    # ------------------------------------------------------------------

    def generate_with_logprobs(
        self,
        prompt: str,
        think: bool = False,
    ) -> dict[str, Any]:
        """Extract logprobs at the answer position via low-level logit access.

        Wraps the prompt in the Qwen3 chat template, then uses
        tokenize → eval → llama_get_logits → log-softmax to extract
        logprobs without requiring logits_all=True.

        The prompt from build_prompt() ends with "\\n\\nAnswer:". We strip
        this from the user message and place it as assistant prefill so
        the model's next token (the one we extract logprobs for) is the
        answer letter. Without this, the model sees a fresh assistant
        turn and wants to output <think> or reasoning tokens instead.

        Args:
            prompt: Full MCQ prompt ending with "Answer:" (from build_prompt).
            think: Enable Qwen3 /think mode. When True, the model reasons
                internally before the answer token.

        Returns:
            Dict with response_text (top-1 token), logprobs, thinking_trace.
        """
        # Split "Answer:" from the user message and use as assistant prefill.
        # The user message contains the passage + question + options.
        # The assistant turn starts with "Answer:" so the next token is A/B/C/D.
        if prompt.endswith("Answer:"):
            user_message = prompt[:-len("Answer:")].rstrip()
            assistant_prefill = "Answer:"
        else:
            user_message = prompt
            assistant_prefill = ""

        chat_prompt = self._build_chat_prompt(user_message, think=think) + assistant_prefill

        with self._lock:
            tokens = self.model.tokenize(chat_prompt.encode(), add_bos=True)

            if len(tokens) > self._n_ctx:
                raise ValueError(
                    f"Prompt ({len(tokens)} tokens) exceeds n_ctx ({self._n_ctx}). "
                    f"Truncate the article or increase n_ctx."
                )

            self._full_reset()
            # Time the eval + logit extraction — this is the full inference cost
            t0 = time.time()
            self.model.eval(tokens)
            logprobs = self._extract_answer_letter_logprobs()
            elapsed = time.time() - t0
            self._query_count += 1

        response_text = logprobs[0]["token"] if logprobs else ""

        return {
            "response_text": response_text,
            "logprobs": logprobs,
            "thinking_trace": "",
            # Timing and token counts for per-query analysis
            "inference_time_s": elapsed,
            "prompt_tokens": len(tokens),
            "output_tokens": 1,  # single answer token extracted, no generation
        }

    # ------------------------------------------------------------------
    # Two-pass CoT
    # ------------------------------------------------------------------

    def _generate_and_extract(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        think: bool = False,
    ) -> dict[str, Any]:
        """Generate reasoning + answer, then extract logprobs at the answer position.

        Single logical pass (two mechanical steps due to llama-cpp-python):
          1. Generate: model reasons (visibly or in <think> block) then answers
          2. Extract: eval the full output up to "Answer:", get logprobs

        Logprobs are conditioned on the model's FULL reasoning — no stripping,
        no anti-leak constraints. This is the natural post-reasoning probability
        distribution at the answer position.

        Args:
            prompt: Question text (becomes user message in chat template).
            max_tokens: Max tokens for generation.
            temperature: Sampling temperature.
            think: Enable Qwen3 /think reasoning.

        Returns:
            Dict with response_text, logprobs, thinking_trace, pass1_answer.
        """
        import re

        # --- Step 1: generate reasoning + answer freely ---
        # Time the entire two-pass process as one block (both passes together)
        t0 = time.time()

        chat_prompt = self._build_chat_prompt(prompt, think=think)

        gen = self.generate(
            prompt=chat_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=None,
            think=False,  # template already applied
        )

        raw_output = gen["response_text"]
        visible_output, thinking_trace = _split_think_response(raw_output)

        # Parse the model's freely chosen answer from visible output
        pass1_answer = ""
        matches = list(re.finditer(r'[Aa]nswer:\s*([A-Da-d])', visible_output))
        if matches:
            pass1_answer = matches[-1].group(1).upper()

        # --- Step 2: eval full output up to "Answer:", extract logprobs ---
        # Keep everything (including <think> block), strip only the answer letter.
        # Find the last "Answer:" in the RAW output (before think/visible split).
        last_answer_pos = -1
        for m in re.finditer(r'[Aa]nswer:', raw_output):
            last_answer_pos = m.start()

        if last_answer_pos > 0:
            output_prefix = raw_output[:last_answer_pos].rstrip()
        else:
            output_prefix = raw_output.rstrip()

        eval_prompt = chat_prompt + output_prefix + "\n\nAnswer:"

        with self._lock:
            tokens = self.model.tokenize(eval_prompt.encode(), add_bos=True)

            if len(tokens) > self._n_ctx:
                raise ValueError(
                    f"Reasoning eval ({len(tokens)} tokens) exceeds n_ctx ({self._n_ctx})."
                )

            self._full_reset()
            self.model.eval(tokens)
            logprobs = self._extract_answer_letter_logprobs()
            self._query_count += 1

        elapsed = time.time() - t0

        # --- Assemble result ---
        response_text = (
            output_prefix + "\nAnswer:" +
            (logprobs[0]["token"] if logprobs else "")
        )

        # Store full reasoning in thinking_trace
        if thinking_trace:
            thinking_trace = thinking_trace + "\n---\n" + visible_output
        else:
            thinking_trace = visible_output

        # Token counts: Pass 1 prompt tokens come from the chat_prompt tokenisation,
        # Pass 2 prompt tokens come from the longer eval_prompt (which includes the
        # generated reasoning). Sum both; output_tokens counts the raw generation
        # length plus the single extracted answer token from Pass 2.
        pass1_prompt_tokens = len(
            self.model.tokenize(chat_prompt.encode(), add_bos=True)
        )
        pass1_output_tokens = len(
            self.model.tokenize(raw_output.encode(), add_bos=False)
        )

        return {
            "response_text": response_text,
            "logprobs": logprobs,
            "thinking_trace": thinking_trace,
            "pass1_answer": pass1_answer,
            # Timing and token counts for per-query analysis
            "inference_time_s": elapsed,
            "prompt_tokens": pass1_prompt_tokens + len(tokens),  # Pass 1 + Pass 2
            "output_tokens": pass1_output_tokens + 1,  # Pass 1 generation + Pass 2 answer token
        }

    def generate_cot(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Natural Chain-of-Thought: model reasons visibly, logprobs conditioned on full reasoning."""
        return self._generate_and_extract(
            prompt, max_tokens=max_tokens, temperature=temperature, think=False,
        )

    def generate_think(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Qwen3 think mode: model reasons in <think> block, logprobs conditioned on full output."""
        return self._generate_and_extract(
            prompt, max_tokens=max_tokens, temperature=temperature, think=True,
        )

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def _build_chat_prompt(
        self,
        user_message: str,
        think: bool = True,
        system_message: str = "You are a helpful assistant.",
    ) -> str:
        """Construct a Qwen3-format chat prompt.

        Args:
            user_message: User's message content.
            think: Prepend /think (True) or /no_think (False).
            system_message: System prompt text.

        Returns:
            Formatted prompt with Qwen3 <|im_start|>/<|im_end|> tokens.
        """
        think_tag = "/think" if think else "/no_think"
        return (
            f"{_IM_START}system\n"
            f"{think_tag}\n{system_message}{_IM_END}\n"
            f"{_IM_START}user\n"
            f"{user_message}{_IM_END}\n"
            f"{_IM_START}assistant\n"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def load_time(self) -> float:
        """Seconds taken to load the model."""
        return self._load_time

    @property
    def query_count(self) -> int:
        """Total inference calls made."""
        return self._query_count


# ---------------------------------------------------------------------------
# Think-mode response splitting
# ---------------------------------------------------------------------------

def _split_think_response(text: str) -> tuple[str, str]:
    """Separate Qwen3 <think>...</think> from visible output.

    Returns:
        (visible_response, thinking_trace).  If no think tags are found
        the full text is returned as the visible response.
    """
    tag_open = "<think>"
    tag_close = "</think>"

    if tag_open not in text:
        return text, ""

    start = text.index(tag_open) + len(tag_open)
    if tag_close in text:
        end = text.index(tag_close)
        thinking = text[start:end].strip()
        visible = text[end + len(tag_close):].strip()
    else:
        thinking = text[start:].strip()
        visible = ""

    return visible, thinking


# ---------------------------------------------------------------------------
# MCQA answer extraction (operates on our standard logprob format)
# ---------------------------------------------------------------------------

def _token_to_letter(token: str) -> str | None:
    """Map a token like 'B' or ' B' to the canonical letter 'B'."""
    stripped = token.strip()
    return stripped if stripped in ANSWER_LETTERS else None


def extract_answer_logprobs(
    logprobs: list[dict],
    answer_permutation: list[int],
    mode: str = "direct",
) -> tuple[dict[str, float], dict[int, float], str, int, int]:
    """Extract per-letter logprobs from model output for MCQA.

    Direct mode: inspects the first token position.
    CoT mode: single entry (two-pass) → treat as direct;
              multi-entry → scan backwards for last A/B/C/D.

    Args:
        logprobs: Normalised logprobs list.
        answer_permutation: Display-position → canonical-index mapping.
        mode: "direct" or "cot".

    Returns:
        Tuple of (display_letter_logprobs, canonical_logprobs,
                  display_answer, canonical_answer, answer_token_idx).

    Raises:
        ValueError: If no answer letters found in logprobs.
    """
    if not logprobs:
        raise ValueError("No logprobs data provided")

    # Locate the answer token position
    if mode == "direct":
        target_entry = logprobs[0]
        answer_token_idx = 0
    else:
        if len(logprobs) == 1:
            target_entry = logprobs[0]
            answer_token_idx = 0
        else:
            found = _find_last_answer_token(logprobs)
            if found is None:
                raise ValueError(
                    "No standalone A/B/C/D token in CoT logprobs"
                )
            target_entry, answer_token_idx = found

    # Extract per-letter logprobs from top_logprobs
    display_letter_logprobs: dict[str, float] = {}
    top_lp_list = target_entry.get("top_logprobs", [])

    for entry in top_lp_list:
        token = entry.get("token", "")
        lp = entry.get("logprob")
        letter = _token_to_letter(token)
        if letter is not None and lp is not None:
            if letter not in display_letter_logprobs or lp > display_letter_logprobs[letter]:
                display_letter_logprobs[letter] = lp

    if not display_letter_logprobs:
        available = [e.get("token", "?") for e in top_lp_list[:10]]
        raise ValueError(
            f"No answer letters in top_logprobs. "
            f"Tokens present: {available}"
        )

    # Map display letters → canonical indices
    canonical_logprobs: dict[int, float] = {}
    for display_pos, letter in enumerate(ANSWER_LETTERS):
        if letter in display_letter_logprobs:
            canonical_logprobs[answer_permutation[display_pos]] = (
                display_letter_logprobs[letter]
            )

    # Model's top answer
    display_answer = max(
        display_letter_logprobs, key=display_letter_logprobs.get  # type: ignore[arg-type]
    )
    display_pos = ANSWER_LETTERS.index(display_answer)
    canonical_answer = answer_permutation[display_pos]

    return (
        display_letter_logprobs,
        canonical_logprobs,
        display_answer,
        canonical_answer,
        answer_token_idx,
    )


def _find_last_answer_token(
    logprobs: list[dict],
) -> tuple[dict, int] | None:
    """Find the last position whose top token is a standalone A/B/C/D."""
    for i in range(len(logprobs) - 1, -1, -1):
        entry = logprobs[i]
        top_lp = entry.get("top_logprobs", [])
        if not top_lp:
            continue
        if _token_to_letter(top_lp[0].get("token", "")) is not None:
            return entry, i
    return None


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

# Default GGUF filename for our primary model
_DEFAULT_GGUF = "qwen3-8b-q4_k_m.gguf"


def find_model_path(model_name: str = "qwen3:8b-q4_K_M") -> Path | None:
    """Find a GGUF model file on disk.

    Search order:
      1. UQ_MODEL_PATH env var (explicit override — use on any machine)
      2. /workspace/models/<gguf>  (vast.ai convention)
      3. <project_root>/models/<gguf>  (local convenience — symlink or copy)
      4. ~/models/<gguf>  (user home)
      5. Largest GGUF (>1 GB) in ~/.ollama/models/blobs/  (legacy — the
         GGUF was originally downloaded via Ollama and still lives there
         on the Windows laptop)

    Args:
        model_name: Ignored for now (single-model project). Kept for
            config compatibility so callers don't need to change.

    Returns:
        Path to the GGUF file, or None if not found anywhere.
    """
    import os

    # --- 1. Explicit env var ---
    env_path = os.environ.get("UQ_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        print(f"  [WARN] UQ_MODEL_PATH={env_path} but file does not exist")

    # --- 2. vast.ai convention ---
    vastai = Path("/workspace/models") / _DEFAULT_GGUF
    if vastai.is_file():
        return vastai

    # --- 3. Project-local models/ directory ---
    # Walk up from this file to find the project root (contains pyproject.toml)
    project_root = _find_project_root()
    if project_root:
        local = project_root / "models" / _DEFAULT_GGUF
        if local.is_file():
            return local

    # --- 4. ~/models/ ---
    home_models = Path.home() / "models" / _DEFAULT_GGUF
    if home_models.is_file():
        return home_models

    # --- 5. Legacy: largest file in Ollama blob store ---
    # The GGUF was originally downloaded by Ollama and lives as a
    # hash-named blob. We just find the biggest file (>1 GB).
    blobs_dir = Path.home() / ".ollama" / "models" / "blobs"
    if blobs_dir.is_dir():
        largest: Path | None = None
        largest_size = 0
        for f in blobs_dir.iterdir():
            if f.is_file():
                sz = f.stat().st_size
                if sz > largest_size:
                    largest = f
                    largest_size = sz
        if largest and largest_size > 1_000_000_000:
            print(f"  [INFO] Using GGUF from Ollama blob store: {largest.name[:40]}...")
            return largest

    return None


def _find_project_root() -> Path | None:
    """Walk up from this file to find the directory containing pyproject.toml."""
    current = Path(__file__).resolve().parent
    for _ in range(5):  # don't walk too far
        if (current / "pyproject.toml").is_file():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    DIVIDER = "=" * 60
    SUBDIV = "-" * 60

    print(DIVIDER)
    print("  llama-cpp-python Inference Smoke Test")
    print(f"  (low-level logit extraction, logits_all=False)")
    print(DIVIDER)

    # --- Find model ---
    model_path = find_model_path()
    if model_path is None:
        print("\nERROR: No GGUF model found.")
        print("Searched: UQ_MODEL_PATH env var, /workspace/models/,")
        print("  <project>/models/, ~/models/, ~/.ollama/models/blobs/")
        print("Set UQ_MODEL_PATH or copy the GGUF to one of these locations.")
        sys.exit(1)

    print(f"\nModel: {model_path.name[:40]}...")
    print(f"Size:  {model_path.stat().st_size / 1e9:.1f} GB")

    client = LlamaCppClient(model_path, n_ctx=8192, verbose=False)
    print(f"Loaded in {client.load_time:.1f}s")
    print(f"n_vocab={client._n_vocab}, logits_all=False, n_batch=n_ctx=8192")

    # --- Test 1: MCQA logprob extraction with chat template ---
    print(f"\n{SUBDIV}")
    print("Test 1: MCQA + chat template + logprob extraction")
    print(SUBDIV)

    mcqa_prompt = (
        "What is the capital of France?\n\n"
        "A) London\n"
        "B) Paris\n"
        "C) Berlin\n"
        "D) Madrid\n\n"
        "Answer:"
    )

    result = client.generate_with_logprobs(mcqa_prompt, think=False)
    print(f"Top token: {result['response_text']!r}")

    if result["logprobs"]:
        first = result["logprobs"][0]
        print(f"Top 10 logprobs:")
        for e in first["top_logprobs"][:10]:
            p = math.exp(e["logprob"])
            print(f"  {e['token']!r:>10s}: {e['logprob']:.4f} (p={p:.4f})")

        try:
            disp_lp, canon_lp, disp_ans, canon_ans, _ = extract_answer_logprobs(
                result["logprobs"], [0, 1, 2, 3], mode="direct",
            )
            print(f"\nExtracted answer: {disp_ans} (canonical {canon_ans})")
            total_p = 0
            for letter in ANSWER_LETTERS:
                if letter in disp_lp:
                    p = math.exp(disp_lp[letter])
                    total_p += p
                    print(f"  {letter}: logprob={disp_lp[letter]:.4f}  p={p:.4f}")
            print(f"  Sum of answer probs: {total_p:.4f}")
        except ValueError as e:
            print(f"Extraction FAILED: {e}")

    # --- Test 2: n_ctx overflow assertion ---
    print(f"\n{SUBDIV}")
    print("Test 2: n_ctx overflow assertion")
    print(SUBDIV)

    huge_prompt = "word " * 20000 + "\n\nAnswer:"
    try:
        client.generate_with_logprobs(huge_prompt, think=False)
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"Correctly caught: {e}")

    # --- Test 3: Two-pass CoT ---
    print(f"\n{SUBDIV}")
    print("Test 3: Two-pass CoT (reasoning + answer logprobs)")
    print(SUBDIV)

    cot_prompt = (
        "What is the capital of France?\n\n"
        "A) London\n"
        "B) Paris\n"
        "C) Berlin\n"
        "D) Madrid\n\n"
        "Reason briefly, then end with:\nAnswer: X"
    )

    result = client.generate_cot(
        prompt=cot_prompt,
        max_tokens=256,
        temperature=0.0,
        top_logprobs=20,
    )

    reasoning = result["thinking_trace"][:200]
    print(f"Reasoning (first 200): {reasoning}")
    print(f"Response tail: ...{result['response_text'][-80:]}")

    if result["logprobs"]:
        first = result["logprobs"][0]
        print(f"\nAnswer position logprobs:")
        for e in first["top_logprobs"][:5]:
            p = math.exp(e["logprob"])
            print(f"  {e['token']!r:>10s}: {e['logprob']:.4f} (p={p:.4f})")

    # --- Summary ---
    print(f"\n{DIVIDER}")
    print(f"All tests complete. Total queries: {client.query_count}")
    print(DIVIDER)
