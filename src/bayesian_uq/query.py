"""
Ollama client wrapper for querying local LLMs with logprob extraction.

v2 changes from v1:
  - No JSON schema enforcement (format parameter removed)
  - Uses /api/generate with raw: True for direct mode (completion-style)
  - Uses /api/chat for CoT modes (needs chat template for reasoning)
  - Extracts answer probability distribution from logprobs
  - top_logprobs: 20 to maximise chance of capturing all four answer letters
  - Streaming for CoT modes to avoid timeouts on long reasoning

Handles:
  - Sending multiple-choice questions to Ollama's API
  - Extracting raw logprobs for answer letters (A/B/C/D) from the response
  - Randomising answer order to control for position bias
  - Mapping logprobs back to canonical answer indices (without normalisation)
"""

import json
import math
import random
import time

import requests

from .config import ANSWER_LETTERS, NUM_CHOICES


# Default Ollama endpoint — use 127.0.0.1 instead of localhost to avoid
# Windows IPv6 DNS resolution delay (~2s per request on some systems)
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# Answer tokens may have a leading space in the generate endpoint's tokenisation.
# We match both "A" and " A" (and similarly for B, C, D).
ANSWER_TOKENS = set(ANSWER_LETTERS) | {f" {l}" for l in ANSWER_LETTERS}

# Context sizes for CoT modes. MCQ prompts are ~150 tokens and CoT responses
# ~300 tokens, so 2048 gives a comfortable 4x margin. Think mode generates
# long hidden reasoning chains that need significantly more headroom.
COT_CONTEXT_SIZE = 2048
THINK_CONTEXT_SIZE = 8192


def _token_to_letter(token: str) -> str | None:
    """Map a token like 'B' or ' B' to the canonical letter 'B'. Returns None if not a match."""
    stripped = token.strip()
    if stripped in ANSWER_LETTERS:
        return stripped
    return None


class OllamaClient:
    """Client for querying models via the Ollama API with logprob extraction.

    Usage:
        client = OllamaClient(model="qwen3:8b-q4_K_M")
        if not client.check_connection():
            raise ConnectionError("Ollama not reachable")

        response, logprobs, thinking = client.send_query(
            question_text="What is 2+2?",
            choices=["3", "4", "5", "6"],
            answer_permutation=[2, 0, 3, 1],
        )
    """

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_OLLAMA_URL,
        think: bool = False,
        prompt_mode: str = "direct",
        temperature: float = 0.7,
        timeout: int = 120,
    ):
        """
        Args:
            model: Ollama model name (e.g. "qwen3:8b-q4_K_M").
            base_url: Ollama server URL (default http://localhost:11434).
            think: Whether to enable the model's thinking/reasoning mode.
            prompt_mode: One of "direct", "cot", "cot_structured".
            temperature: Sampling temperature.
            timeout: HTTP request timeout in seconds (per-chunk for streaming).
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.think = think
        self.prompt_mode = prompt_mode
        self.temperature = temperature
        self.timeout = timeout
        self._logged_first_query = False
        self._query_count = 0  # tracks total queries for diagnostic logging
        # Fewer retries in think mode — runaway think chains will likely repeat
        self.max_retries = 3
        # Total wall-clock limit per generation attempt
        if think:
            self.max_stream_time = 300
        elif prompt_mode in ("cot", "cot_structured"):
            self.max_stream_time = 180
        else:
            self.max_stream_time = 30

    def check_connection(self) -> bool:
        """Check that Ollama is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def send_query(
        self,
        question_text: str,
        choices: list[str],
        answer_permutation: list[int],
    ) -> tuple[str, list[dict], str]:
        """
        Send a multiple-choice question to the model and extract logprobs.

        For direct mode: uses /api/generate with raw=True (completion-style,
        no chat template) and num_predict=1. The prompt ends with "Answer:"
        and the model completes with a single letter token.

        For CoT modes: uses /api/chat so the chat template is applied. The
        model reasons freely and we find the answer token in the output.

        Args:
            question_text: The question stem (original or paraphrased).
            choices: Answer texts in canonical order (index 0=A, 1=B, 2=C, 3=D).
            answer_permutation: Mapping from display position to canonical index.

        Returns:
            Tuple of (raw_response_text, logprobs_list, thinking_trace).
            - raw_response_text: the full text the model generated
            - logprobs_list: the complete logprobs array from Ollama (all token positions)
            - thinking_trace: the model's reasoning tokens (empty for direct mode)

        Raises:
            requests.RequestException: If the Ollama API call fails after retries.
            ValueError: If the model response can't be parsed after retries.
        """
        num_choices = len(choices)

        # Build the display choices using the permutation
        display_choices = {}
        for display_pos, canonical_idx in enumerate(answer_permutation):
            letter = ANSWER_LETTERS[display_pos]
            display_choices[letter] = choices[canonical_idx]

        # Format the choice lines (shared across all modes)
        choice_lines = "\n".join(
            f"  {letter}) {text}" for letter, text in display_choices.items()
        )

        # Build the prompt/message based on prompt_mode
        if self.prompt_mode == "direct":
            # Direct mode: completion-style via /api/generate with raw=True.
            # Prompt ends with "Answer:" — model completes with " B" etc.
            prompt_text = (
                f"{question_text}\n\n"
                f"{choice_lines}\n\n"
                "Answer:"
            )
            payload = {
                "model": self.model,
                "prompt": prompt_text,
                "raw": True,  # skip chat template — pure text completion
                "stream": False,  # single token, no need to stream
                "logprobs": True,
                "top_logprobs": 20,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 1,
                },
            }
        elif self.prompt_mode == "cot":
            user_message = (
                f"{question_text}\n\n"
                f"{choice_lines}\n\n"
                "Consider each option, then state your answer as a single letter."
            )
            payload = self._build_chat_payload(user_message)
        elif self.prompt_mode == "cot_structured":
            user_message = (
                f"{question_text}\n\n"
                f"{choice_lines}\n\n"
                "Which option is correct? Use this format:\n\n"
                "A) \u2713/\u2717 reason\n"
                "B) \u2713/\u2717 reason\n"
                "C) \u2713/\u2717 reason\n"
                "D) \u2713/\u2717 reason\n\n"
                "Answer: X\n\n"
                "Example:\n"
                "A) \u2717 confuses mass with weight\n"
                "B) \u2713 matches Newton's second law\n"
                "C) \u2717 ignores friction\n"
                "D) \u2717 wrong units\n\n"
                "Answer: B"
            )
            payload = self._build_chat_payload(user_message)
        else:
            raise ValueError(f"Unknown prompt_mode: {self.prompt_mode}")

        # Log prompt setup once per experiment
        if not self._logged_first_query:
            api = "generate (raw)" if self.prompt_mode == "direct" else "chat"
            num_pred = payload.get("options", {}).get("num_predict", "default")
            print(
                f"\n  [INFO] Prompt mode: {self.prompt_mode} | API: {api} | "
                f"num_predict: {num_pred} | top_logprobs: 20 | think: {self.think}",
                flush=True,
            )
            self._logged_first_query = True

        # Retry loop for network errors and empty responses
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self.prompt_mode == "direct":
                    raw_content, raw_thinking, all_logprobs = self._send_generate(payload)
                else:
                    raw_content, raw_thinking, all_logprobs = self._stream_chat(payload)
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as e:
                last_error = e
                wait = 10 * (attempt + 1)
                print(
                    f"\n    [retry {attempt + 1}/{self.max_retries}] "
                    f"connection error, waiting {wait}s...",
                    end="", flush=True,
                )
                time.sleep(wait)
                continue

            if raw_content.strip() or all_logprobs:
                self._query_count += 1
                # Diagnostic logging for first 3 queries only
                if self._query_count <= 3 and all_logprobs:
                    self._log_first_token_diagnostics(all_logprobs)
                return raw_content, all_logprobs, raw_thinking

            # Empty response — retry
            last_error = ValueError("Empty response from model")
            wait = 5 * (attempt + 1)
            print(
                f"\n    [retry {attempt + 1}/{self.max_retries}] "
                f"empty response, waiting {wait}s...",
                end="", flush=True,
            )
            time.sleep(wait)

        # All retries exhausted
        raise last_error  # type: ignore[misc]

    def _build_chat_payload(self, user_message: str) -> dict:
        """Build a /api/chat payload for CoT modes."""
        ctx_size = THINK_CONTEXT_SIZE if self.think else COT_CONTEXT_SIZE
        return {
            "model": self.model,
            "messages": [
                {"role": "user", "content": user_message},
            ],
            "think": self.think,
            "stream": True,
            "logprobs": True,
            "top_logprobs": 20,
            "options": {
                "temperature": self.temperature,
                "num_predict": 500,
                "num_ctx": ctx_size,
            },
        }

    def _send_generate(self, payload: dict) -> tuple[str, str, list[dict]]:
        """Send a non-streaming /api/generate request and extract logprobs.

        For direct mode: single token, non-streaming. The logprobs are
        returned as a top-level field in the response JSON.

        Returns:
            Tuple of (content_text, thinking_text, logprobs_list).
        """
        resp = requests.post(
            self.generate_url,
            json=payload,
            timeout=(30, self.timeout),
        )
        resp.raise_for_status()
        data = resp.json()

        raw_content = data.get("response", "")
        # /api/generate doesn't have a separate thinking field
        raw_thinking = ""
        # logprobs is a top-level list in the generate response
        all_logprobs = data.get("logprobs", [])

        return raw_content, raw_thinking, all_logprobs

    def _stream_chat(self, payload: dict) -> tuple[str, str, list[dict]]:
        """Open a streaming /api/chat connection and accumulate content, thinking, and logprobs.

        For CoT modes: streaming to avoid timeouts on long reasoning.
        logprobs and top_logprobs are top-level params in the payload.

        Returns:
            Tuple of (content_text, thinking_text, logprobs_list).
        """
        resp = requests.post(
            self.chat_url,
            json=payload,
            timeout=(30, self.timeout),
            stream=True,
        )
        resp.raise_for_status()

        raw_content = ""
        raw_thinking = ""
        all_logprobs: list[dict] = []
        stream_start = time.monotonic()

        try:
            for line in resp.iter_lines(decode_unicode=True):
                # Total elapsed check — kills runaway generations
                elapsed = time.monotonic() - stream_start
                if elapsed > self.max_stream_time:
                    raise requests.exceptions.ReadTimeout(
                        f"Total stream time exceeded {self.max_stream_time}s"
                    )
                if not line:
                    continue

                chunk = json.loads(line)
                msg = chunk.get("message", {})

                # Accumulate content and thinking tokens separately
                content_token = msg.get("content", "")
                raw_content += content_token
                raw_thinking += msg.get("thinking", "")

                # Collect logprobs for content tokens.
                # In the chat streaming API, logprobs may be in the chunk
                # at the top level or inside the message.
                if content_token:
                    if "logprobs" in chunk:
                        all_logprobs.append(chunk["logprobs"])
                    elif "logprobs" in msg:
                        all_logprobs.append(msg["logprobs"])

                if chunk.get("done", False):
                    break
        finally:
            resp.close()

        return raw_content, raw_thinking, all_logprobs

    def _log_first_token_diagnostics(self, all_logprobs: list[dict]) -> None:
        """Log the first content token and its top 5 logprobs for debugging.

        Called for the first ~33 queries (first 3 questions) so we can
        verify what the model is actually outputting.
        """
        if not all_logprobs:
            return
        top = _get_top_logprobs(all_logprobs[0])
        if not top:
            print(f"\n      [DIAG] No top_logprobs in first token", flush=True)
            return
        first_token = top[0].get("token", "?")
        top5 = top[:5]
        top5_str = " | ".join(
            f"{e.get('token', '?')!r}: {e.get('logprob', 0):.3f}"
            for e in top5
        )
        is_answer = _token_to_letter(first_token) is not None
        marker = "" if is_answer else " [NOT A/B/C/D]"
        print(
            f"\n      [DIAG] First token: {first_token!r}{marker} | "
            f"Top 5: [{top5_str}]",
            flush=True,
        )


def extract_answer_logprobs(
    all_logprobs: list[dict],
    answer_permutation: list[int],
    prompt_mode: str = "direct",
) -> tuple[dict[str, float], dict[int, float], str, int, int]:
    """Extract per-letter logprobs from the Ollama logprobs data.

    For direct mode: uses the first token position. Matches both "A" and " A"
    style tokens (the generate endpoint may add a leading space).

    For CoT modes: finds the LAST token that is exactly one of A/B/C/D
    (single-character tokens only — "Based" doesn't count as "B").

    Returns raw logprobs without normalisation. If a letter doesn't appear
    in the top_logprobs for the selected token, it is simply absent from
    the returned dict.

    Args:
        all_logprobs: Complete logprobs list from Ollama (one entry per token).
        answer_permutation: Mapping from display position to canonical index.
        prompt_mode: "direct", "cot", or "cot_structured".

    Returns:
        Tuple of:
        - display_letter_logprobs: {"A": -0.12, "B": -3.21, ...} raw logprobs
          (always keyed by canonical letter, not the raw token)
        - canonical_logprobs: {0: -3.21, 1: -0.12, ...} mapped via permutation
        - display_answer: the letter with the highest logprob (e.g. "B")
        - canonical_answer: the canonical index of that letter
        - answer_token_idx: index into all_logprobs of the answer token
    """
    # Find the right token position to inspect
    if prompt_mode == "direct":
        # Direct mode: use the first content token position.
        # Even if the top token isn't A/B/C/D, the answer letters may still
        # appear in top_logprobs with meaningful probabilities.
        if not all_logprobs:
            raise ValueError("No logprobs data returned for direct mode query")
        target_logprobs = all_logprobs[0]
        answer_token_idx = 0
    else:
        # CoT modes: find the LAST single-character A/B/C/D token
        result = _find_last_answer_token_logprobs(all_logprobs)
        if result is None:
            raise ValueError(
                "No standalone A/B/C/D token found in CoT response logprobs"
            )
        target_logprobs, answer_token_idx = result

    # Extract logprobs for each answer letter from the target token's top_logprobs.
    # Tokens may be "A", " A", "B", " B" etc. — we normalise to just the letter.
    display_letter_logprobs: dict[str, float] = {}
    top_logprobs_list = _get_top_logprobs(target_logprobs)

    for entry in top_logprobs_list:
        token = entry.get("token", "")
        logprob = entry.get("logprob", None)
        letter = _token_to_letter(token)
        if letter is not None and logprob is not None:
            # If we see both "A" and " A", keep the one with the higher logprob
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
        display_answer = max(display_letter_logprobs, key=display_letter_logprobs.get)  # type: ignore[arg-type]
        display_pos = ANSWER_LETTERS.index(display_answer)
        canonical_answer = answer_permutation[display_pos]
    else:
        raise ValueError(
            f"No answer letters found in top_logprobs. "
            f"Available tokens: {[e.get('token', '?') for e in top_logprobs_list]}"
        )

    return display_letter_logprobs, canonical_logprobs, display_answer, canonical_answer, answer_token_idx


def _find_last_answer_token_logprobs(
    all_logprobs: list[dict],
) -> tuple[dict, int] | None:
    """Find the logprobs entry for the last standalone A/B/C/D token in a CoT response.

    Scans the logprobs list from the end, looking for a token position where
    the top token (highest probability) is exactly one of A, B, C, D as a
    single character (with or without leading space).

    Returns (logprobs_entry, index) for that token position, or None if not found.
    """
    for i in range(len(all_logprobs) - 1, -1, -1):
        entry = all_logprobs[i]
        top_logprobs = _get_top_logprobs(entry)
        if not top_logprobs:
            continue
        # Check the top token at this position
        top_token = top_logprobs[0].get("token", "")
        if _token_to_letter(top_token) is not None:
            return entry, i
    return None


def _get_top_logprobs(logprobs_entry: dict) -> list[dict]:
    """Extract the top_logprobs list from a logprobs entry.

    Handles different possible Ollama response formats:
    - {"top_logprobs": [...]}  (standard format)
    - {"token": ..., "top_logprobs": [...]}  (generate endpoint)
    - Direct list (if the entry itself is the list)
    """
    if isinstance(logprobs_entry, list):
        return logprobs_entry
    if "top_logprobs" in logprobs_entry:
        return logprobs_entry["top_logprobs"]
    return []


def generate_permutation(
    num_choices: int = NUM_CHOICES,
    rng: random.Random | None = None,
    shuffle: bool = True,
) -> list[int]:
    """
    Generate a permutation for answer ordering.

    When shuffle=True (default), returns a random permutation to integrate
    out position bias. When shuffle=False, returns the identity [0, 1, 2, 3]
    so answers stay in canonical order.

    Returns a list where permutation[display_pos] = canonical_index.
    For example, [2, 0, 3, 1] means:
      - Display position A shows canonical choice 2
      - Display position B shows canonical choice 0
      - Display position C shows canonical choice 3
      - Display position D shows canonical choice 1

    Args:
        num_choices: Number of answer options (default 4).
        rng: Python random.Random instance for reproducibility.
        shuffle: Whether to randomise order (False = identity permutation).

    Returns:
        List of canonical indices in display order.
    """
    perm = list(range(num_choices))
    if shuffle:
        if rng is None:
            rng = random.Random()
        rng.shuffle(perm)
    return perm
