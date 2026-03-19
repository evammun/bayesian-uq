"""
Llama-cpp-python client for querying local LLMs with logprob extraction.

This is an isolated prototype that mirrors OllamaClient's interface but uses
llama-cpp-python directly instead of the Ollama HTTP API. Designed to test
performance benefits of direct GPU access vs. Ollama's HTTP overhead.

Key differences from OllamaClient:
  - Uses llama_cpp.Llama directly instead of HTTP requests
  - Logprobs returned in a different format (dict per token position)
  - No streaming (llama-cpp-python doesn't support it well)
  - No think mode (not needed for initial prototype)
  - Supports three modes: direct, cot (simple), cot_structured

v2 design:
  - Extract logprobs for answer letters (A/B/C/D) from each query
  - Convert to normalised probability distributions
  - Aggregate across queries: mean of per-query vectors
  - No Dirichlet, no Monte Carlo, no adaptive stopping
"""

import json
import math
import random
import time
from typing import Optional

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError(
        "llama-cpp-python not installed. "
        "Install with: pip install llama-cpp-python"
    )


# Constants matching the project
ANSWER_LETTERS = ["A", "B", "C", "D"]
NUM_CHOICES = 4

# Answer tokens may have a leading space. We match both "A" and " A".
ANSWER_TOKENS = set(ANSWER_LETTERS) | {f" {l}" for l in ANSWER_LETTERS}


def _token_to_letter(token: str) -> Optional[str]:
    """Map a token like 'B' or ' B' to the canonical letter 'B'.

    Returns None if not a match.
    """
    stripped = token.strip()
    if stripped in ANSWER_LETTERS:
        return stripped
    return None


class LlamaCppClient:
    """Client for querying models via llama-cpp-python with logprob extraction.

    This is a direct-inference alternative to the Ollama HTTP API, designed to
    reduce overhead and test performance on a local NVIDIA GPU.

    Usage:
        client = LlamaCppClient(
            model_path="/path/to/qwen3-8b-q4_K_M.gguf",
            prompt_mode="direct",
            n_ctx=2048,
            n_gpu_layers=-1,  # use all GPU layers
        )

        response, logprobs, thinking = client.send_query(
            question_text="What is 2+2?",
            choices=["3", "4", "5", "6"],
            answer_permutation=[2, 0, 3, 1],
        )
    """

    def __init__(
        self,
        model_path: str,
        prompt_mode: str = "direct",
        temperature: float = 0.7,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        """
        Args:
            model_path: Full path to the GGUF model file.
            prompt_mode: One of "direct", "cot", "cot_structured".
            temperature: Sampling temperature.
            n_ctx: Context window size in tokens.
            n_gpu_layers: Number of model layers to offload to GPU.
                         -1 = all layers (requires sufficient VRAM).
            verbose: Whether to print llama-cpp-python debug output.
        """
        self.model_path = model_path
        self.prompt_mode = prompt_mode
        self.temperature = temperature
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._query_count = 0
        self._logged_first_query = False

        # Initialize the model. This may take a while (model loading + compilation).
        print(f"[INFO] Loading model from {model_path}...")
        start = time.monotonic()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            # logits_all=True is NOT needed — we only use top_logprobs
        )
        elapsed = time.monotonic() - start
        print(f"[INFO] Model loaded in {elapsed:.2f}s")

    def send_query(
        self,
        question_text: str,
        choices: list[str],
        answer_permutation: list[int],
    ) -> tuple[str, list[dict], str, Optional[dict]]:
        """
        Send a multiple-choice question to the model and extract logprobs.

        For direct mode: single token completion with logprobs.
        For CoT modes: reasoning then single-token answer completion.

        Args:
            question_text: The question stem (original or paraphrased).
            choices: Answer texts in canonical order (index 0=A, 1=B, 2=C, 3=D).
            answer_permutation: Mapping from display position to canonical index.

        Returns:
            Tuple of (raw_response_text, logprobs_list, thinking_trace, committed).
            - raw_response_text: the full text the model generated
            - logprobs_list: logprobs in Ollama-compatible format (list of dicts)
            - thinking_trace: empty string (no hidden reasoning in direct mode)
            - committed: None (not implemented in this prototype)
        """
        num_choices = len(choices)

        # Build display choices using the permutation
        display_choices = {}
        for display_pos, canonical_idx in enumerate(answer_permutation):
            letter = ANSWER_LETTERS[display_pos]
            display_choices[letter] = choices[canonical_idx]

        # Format choice lines
        choice_lines = "\n".join(
            f"  {letter}) {text}" for letter, text in display_choices.items()
        )

        # Log prompt setup once
        if not self._logged_first_query:
            print(
                f"\n[INFO] Prompt mode: {self.prompt_mode} | "
                f"n_ctx: {self.n_ctx} | top_logprobs: 20 | "
                f"temperature: {self.temperature}",
                flush=True,
            )
            self._logged_first_query = True

        # Build prompt based on mode
        if self.prompt_mode == "direct":
            prompt_text = (
                f"{question_text}\n\n"
                f"{choice_lines}\n\n"
                "Answer:"
            )
            # Single-token completion with logprobs
            response_text, all_logprobs = self._direct_completion(prompt_text)
        elif self.prompt_mode == "cot":
            # Reasoning phase
            reasoning_prompt = (
                f"{question_text}\n\n"
                f"{choice_lines}\n\n"
                "BE CONCISE. 3-4 bullet points of reasoning only "
                "— do NOT name the answer letter in your reasoning.\n\n"
                "End with: Answer: X"
            )
            reasoning, _ = self._cot_reasoning(reasoning_prompt)

            # Extract the reasoning (up to "Answer:")
            reasoning_cleaned = reasoning
            last_answer = reasoning_cleaned.rfind("\nAnswer:")
            if last_answer != -1:
                reasoning_cleaned = reasoning_cleaned[:last_answer]

            # Answer token completion (Pass 2)
            answer_prompt = (
                f"{question_text}\n\n"
                f"{choice_lines}\n\n"
                "Answer concisely, then state your answer."
            )
            # Build a two-message format (simulating /api/chat assistant prefill)
            # For llama-cpp-python, we concatenate directly
            full_prompt = (
                f"{answer_prompt}\n\n"
                f"[Assistant reasoning]\n{reasoning_cleaned}\n\nAnswer:"
            )
            response_token, all_logprobs = self._direct_completion(full_prompt)
            response_text = reasoning_cleaned + f"\nAnswer:{response_token}"

        elif self.prompt_mode == "cot_structured":
            # Structured CoT reasoning phase
            reasoning_prompt = (
                f"{question_text}\n\n"
                f"{choice_lines}\n\n"
                "Which option is correct? Use this exact format:\n\n"
                "A) ✓/✗ reason\n"
                "B) ✓/✗ reason\n"
                "C) ✓/✗ reason\n"
                "D) ✓/✗ reason\n\n"
                "Answer: X\n\n"
                "Example:\n"
                "Q: Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\n"
                "  A) 0  B) 2  C) 1  D) 3\n"
                "A) ✗ x²+0 = x² is reducible\n"
                "B) ✓ x²+2 has no roots in Z_3\n"
                "C) ✗ x²+1 = (x+1)(x+2) in Z_3\n"
                "D) ✗ 3 ∉ Z_3\n"
                "Answer: B"
            )
            reasoning, _ = self._cot_reasoning(reasoning_prompt)

            # Extract reasoning (up to "Answer:")
            reasoning_cleaned = reasoning
            last_answer = reasoning_cleaned.rfind("\nAnswer:")
            if last_answer != -1:
                reasoning_cleaned = reasoning_cleaned[:last_answer]

            # Answer token completion
            answer_prompt = (
                f"{question_text}\n\n"
                f"{choice_lines}\n\n"
                "Answer concisely, then state your answer."
            )
            full_prompt = (
                f"{answer_prompt}\n\n"
                f"[Assistant structured response]\n{reasoning_cleaned}\n\nAnswer:"
            )
            response_token, all_logprobs = self._direct_completion(full_prompt)
            response_text = reasoning_cleaned + f"\nAnswer:{response_token}"
        else:
            raise ValueError(f"Unknown prompt_mode: {self.prompt_mode}")

        self._query_count += 1

        # Log diagnostics for first few queries
        if self._query_count <= 3 and all_logprobs:
            self._log_first_token_diagnostics(all_logprobs)

        return response_text, all_logprobs, "", None

    def _direct_completion(self, prompt: str) -> tuple[str, list[dict]]:
        """Single-token completion with logprobs extraction.

        Returns:
            Tuple of (response_text, logprobs_list in Ollama format).
        """
        # llama-cpp-python's create_completion returns logprobs in a different format
        # than Ollama. We need to convert it.
        output = self.llm.create_completion(
            prompt=prompt,
            max_tokens=1,
            temperature=self.temperature,
            top_logprobs=20,
        )

        response_text = output.get("choices", [{}])[0].get("text", "")

        # Extract logprobs from llama-cpp-python format
        all_logprobs = self._convert_logprobs(output)

        return response_text, all_logprobs

    def _cot_reasoning(self, prompt: str) -> tuple[str, list[dict]]:
        """Generate reasoning with optional stop sequence.

        For CoT modes, we generate up to 300 tokens and stop at "\nAnswer:".
        llama-cpp-python doesn't support streaming, so we generate all at once.

        Returns:
            Tuple of (reasoning_text, logprobs_list).
        """
        output = self.llm.create_completion(
            prompt=prompt,
            max_tokens=300,
            temperature=self.temperature,
            stop=["\nAnswer:"],
            top_logprobs=20,
        )

        reasoning_text = output.get("choices", [{}])[0].get("text", "")

        # Extract logprobs
        all_logprobs = self._convert_logprobs(output)

        return reasoning_text, all_logprobs

    def _convert_logprobs(self, output: dict) -> list[dict]:
        """Convert llama-cpp-python logprobs format to Ollama-compatible format.

        llama-cpp-python returns:
            {
                "choices": [{
                    "logprobs": {
                        "top_logprobs": [
                            {" B": -0.12, " A": -3.21, ...},  # position 0
                            {" C": -0.05, " B": -2.1, ...},   # position 1
                        ]
                    }
                }]
            }

        Ollama format (what we use):
            [
                {"top_logprobs": [{token: logprob}, ...]},  # position 0
                {"top_logprobs": [{token: logprob}, ...]},  # position 1
            ]

        Returns:
            List of dicts, one per token position, in Ollama-compatible format.
        """
        try:
            choices = output.get("choices", [])
            if not choices:
                return []

            choice = choices[0]
            logprobs_data = choice.get("logprobs", {})
            top_logprobs_list = logprobs_data.get("top_logprobs", [])

            # Convert each position's logprobs dict to Ollama format
            result = []
            for top_lp_dict in top_logprobs_list:
                # top_lp_dict is like {" B": -0.12, " A": -3.21, ...}
                # Convert to list of {"token": ..., "logprob": ...} dicts,
                # sorted by logprob descending
                entries = [
                    {"token": tok, "logprob": lp}
                    for tok, lp in top_lp_dict.items()
                ]
                entries.sort(key=lambda x: x["logprob"], reverse=True)
                result.append({"top_logprobs": entries})

            return result
        except (KeyError, TypeError, AttributeError):
            # If conversion fails, return empty list so extraction code
            # can handle it gracefully
            return []

    def _log_first_token_diagnostics(self, all_logprobs: list[dict]) -> None:
        """Log the first token and its top 5 logprobs for debugging."""
        if not all_logprobs:
            return

        top_logprobs_list = all_logprobs[0].get("top_logprobs", [])
        if not top_logprobs_list:
            print(f"    [DIAG] No top_logprobs in first token", flush=True)
            return

        first_token = top_logprobs_list[0].get("token", "?")
        top5 = top_logprobs_list[:5]
        top5_str = " | ".join(
            f"{e.get('token', '?')!r}: {e.get('logprob', 0):.3f}"
            for e in top5
        )
        is_answer = _token_to_letter(first_token) is not None
        marker = "" if is_answer else " [NOT A/B/C/D]"
        print(
            f"    [DIAG] First token: {first_token!r}{marker} | "
            f"Top 5: [{top5_str}]",
            flush=True,
        )


def extract_answer_logprobs(
    all_logprobs: list[dict],
    answer_permutation: list[int],
    prompt_mode: str = "direct",
) -> tuple[dict[str, float], dict[int, float], str, int, int]:
    """Extract per-letter logprobs from logprobs data.

    For direct mode: uses the first token position. Matches both "A" and " A".
    For CoT modes: finds the LAST token that is a single-character A/B/C/D.

    Returns raw logprobs without normalisation. If a letter doesn't appear
    in the top_logprobs, it is simply absent from the returned dict.

    Args:
        all_logprobs: Complete logprobs list (one entry per token position).
        answer_permutation: Mapping from display position to canonical index.
        prompt_mode: "direct", "cot", or "cot_structured".

    Returns:
        Tuple of:
        - display_letter_logprobs: {"A": -0.12, "B": -3.21, ...}
        - canonical_logprobs: {0: -3.21, 1: -0.12, ...}
        - display_answer: the letter with highest logprob (e.g. "B")
        - canonical_answer: canonical index of that letter
        - answer_token_idx: index into all_logprobs of the answer token
    """
    if prompt_mode == "direct":
        # Direct mode: use the first token position
        if not all_logprobs:
            raise ValueError("No logprobs data returned for direct mode query")
        target_logprobs = all_logprobs[0]
        answer_token_idx = 0
    else:
        # CoT modes: find the last single-character A/B/C/D token
        result = _find_last_answer_token_logprobs(all_logprobs)
        if result is None:
            raise ValueError(
                "No standalone A/B/C/D token found in CoT response logprobs"
            )
        target_logprobs, answer_token_idx = result

    # Extract logprobs for each answer letter
    display_letter_logprobs: dict[str, float] = {}
    top_logprobs_list = target_logprobs.get("top_logprobs", [])

    for entry in top_logprobs_list:
        token = entry.get("token", "")
        logprob = entry.get("logprob", None)
        letter = _token_to_letter(token)
        if letter is not None and logprob is not None:
            # Keep the one with higher logprob if we see both "A" and " A"
            if letter not in display_letter_logprobs or logprob > display_letter_logprobs[letter]:
                display_letter_logprobs[letter] = logprob

    # Map display letters to canonical indices via the permutation
    canonical_logprobs: dict[int, float] = {}
    for display_pos, letter in enumerate(ANSWER_LETTERS):
        if letter in display_letter_logprobs:
            canonical_idx = answer_permutation[display_pos]
            canonical_logprobs[canonical_idx] = display_letter_logprobs[letter]

    # Determine the model's answer: the letter with the highest logprob
    if display_letter_logprobs:
        display_answer = max(
            display_letter_logprobs, key=display_letter_logprobs.get
        )
        display_pos = ANSWER_LETTERS.index(display_answer)
        canonical_answer = answer_permutation[display_pos]
    else:
        available = [e.get("token", "?") for e in top_logprobs_list]
        raise ValueError(
            f"No answer letters found in top_logprobs. "
            f"Available tokens: {available}"
        )

    return (
        display_letter_logprobs,
        canonical_logprobs,
        display_answer,
        canonical_answer,
        answer_token_idx,
    )


def _find_last_answer_token_logprobs(
    all_logprobs: list[dict],
) -> Optional[tuple[dict, int]]:
    """Find the logprobs entry for the last standalone A/B/C/D token in a CoT response.

    Scans from the end, looking for a token position where the top token
    is exactly one of A, B, C, D (single character, with or without space).

    Returns (logprobs_entry, index) or None if not found.
    """
    for i in range(len(all_logprobs) - 1, -1, -1):
        entry = all_logprobs[i]
        top_logprobs_list = entry.get("top_logprobs", [])
        if not top_logprobs_list:
            continue

        # Check the top token at this position
        top_token = top_logprobs_list[0].get("token", "")
        if _token_to_letter(top_token) is not None:
            return entry, i

    return None


def generate_permutation(
    num_choices: int = NUM_CHOICES,
    rng: Optional[random.Random] = None,
    shuffle: bool = True,
) -> list[int]:
    """Generate a permutation for answer ordering.

    When shuffle=True, returns a random permutation to integrate out position bias.
    When shuffle=False, returns the identity [0, 1, 2, 3].

    Returns a list where permutation[display_pos] = canonical_index.

    Args:
        num_choices: Number of answer options (default 4).
        rng: Python random.Random instance for reproducibility.
        shuffle: Whether to randomise order.

    Returns:
        List of canonical indices in display order.
    """
    perm = list(range(num_choices))
    if shuffle:
        if rng is None:
            rng = random.Random()
        rng.shuffle(perm)
    return perm
