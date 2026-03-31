"""
Ollama client wrapper for querying local LLMs with structured output.

Handles:
  - Sending multiple-choice questions to Ollama's chat API
  - Enforcing structured JSON output (answer must be one of A/B/C/D)
  - Randomising answer order to control for position bias
  - Mapping responses back to canonical answer indices
"""

import json
import random
import time

import requests

from .config import ANSWER_LETTERS, NUM_CHOICES

# Default Ollama endpoint (local)
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaClient:
    """Client for querying models via the Ollama API with structured output.

    Usage:
        client = OllamaClient(model="qwen3:8b-q4_K_M")
        if not client.check_connection():
            raise ConnectionError("Ollama not reachable")

        raw_json, canonical_idx = client.send_query(
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
                "direct" — answer only (current baseline).
                "cot" — reasoning field before answer in JSON schema.
                "cot_structured" — per-option evaluation fields before answer.
            temperature: Sampling temperature (ignored when think=True).
            timeout: HTTP request timeout in seconds.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/api/chat"
        self.think = think
        self.prompt_mode = prompt_mode
        self.temperature = temperature
        self.timeout = timeout
        self._logged_first_query = False
        # Fewer retries in think mode — if a question triggers a runaway
        # think chain once, retrying identical input will likely do the same
        self.max_retries = 3 if think else 5
        # Total wall-clock limit per generation attempt — catches runaway
        # think chains that keep producing tokens but never finish.
        # Per-chunk timeout (self.timeout) only fires when NO data flows,
        # so it can't catch a model that thinks forever.
        # Think mode needs much more time — the model can reason for
        # minutes on harder questions. 300s (5 min) vs 90s for nothink.
        # CoT modes also generate more tokens (reasoning text), so give
        # them more time too.
        if think:
            self.max_stream_time = 300
        elif prompt_mode in ("cot", "cot_structured"):
            self.max_stream_time = 180
        else:
            self.max_stream_time = 90

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
    ) -> tuple[str, int, str]:
        """
        Send a multiple-choice question to the model with shuffled answer order.

        Uses streaming so the connection stays alive during long think-mode
        generations. The timeout applies per-chunk (not total), so even a
        10-minute reasoning chain won't timeout as long as tokens keep flowing.

        The answer_permutation controls which canonical choice appears in each
        display slot. For example, permutation [2, 0, 3, 1] means:
          - Display slot A shows canonical choice 2
          - Display slot B shows canonical choice 0
          - Display slot C shows canonical choice 3
          - Display slot D shows canonical choice 1

        The model sees the shuffled version and picks a letter (A/B/C/D).
        We then map that letter back through the permutation to get the
        canonical answer index.

        Args:
            question_text: The question stem (original or paraphrased).
            choices: Answer texts in canonical order (index 0=A, 1=B, 2=C, 3=D).
            answer_permutation: Mapping from display position to canonical index.

        Returns:
            Tuple of (raw_response_json_string, canonical_answer_index,
            thinking_trace). The thinking_trace is the model's reasoning
            tokens (empty string when think=False).

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

        # Format the prompt — choice lines are shared across all modes
        choice_lines = "\n".join(
            f"  {letter}) {text}" for letter, text in display_choices.items()
        )

        # Build the user message and JSON schema based on prompt_mode.
        # The order of fields in the schema matters for autoregressive models:
        # reasoning/evaluation fields MUST come before the answer field so
        # the model generates its thinking before committing to an answer.
        answer_enum = ANSWER_LETTERS[:num_choices]

        if self.prompt_mode == "cot":
            # CoT: free-form reasoning field before answer
            user_message = (
                f"{question_text}\n\n"
                f"{choice_lines}\n\n"
                "Consider each option, then give your answer."
            )
            response_schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "answer": {"type": "string", "enum": answer_enum},
                },
                "required": ["reasoning", "answer"],
            }

        elif self.prompt_mode == "cot_structured":
            # Structured CoT: per-option evaluation fields before answer.
            # Fields use display letters (A/B/C/D) — these correspond to
            # the shuffled positions, not canonical indices.
            user_message = (
                f"{question_text}\n\n"
                f"{choice_lines}\n\n"
                "Evaluate each option, then give your answer."
            )
            option_props = {
                f"option_{letter.lower()}": {"type": "string"}
                for letter in answer_enum
            }
            option_keys = list(option_props.keys())
            response_schema = {
                "type": "object",
                "properties": {
                    **option_props,
                    "answer": {"type": "string", "enum": answer_enum},
                },
                "required": option_keys + ["answer"],
            }

        else:
            # Direct mode: answer only (baseline)
            user_message = (
                f"{question_text}\n\n"
                f"{choice_lines}\n\n"
                "Respond with the letter of the correct answer."
            )
            response_schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "enum": answer_enum},
                },
                "required": ["answer"],
            }

        # No system message — follows the standard MMLU evaluation protocol
        # (EleutherAI lm-eval-harness uses no system prompt)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": user_message},
            ],
            "format": response_schema,
            "think": self.think,
            "stream": True,  # Stream to avoid timeout on long think chains
        }

        # Set model options: temperature always set; think and CoT modes
        # additionally get a larger context window for reasoning traces.
        options = {"temperature": self.temperature}
        if self.think or self.prompt_mode in ("cot", "cot_structured"):
            options["num_ctx"] = 8192
        payload["options"] = options

        # Log schema fields once per experiment to confirm correct setup
        if not self._logged_first_query:
            schema_keys = list(response_schema.get("properties", {}).keys())
            print(f"\n  [INFO] Prompt mode: {self.prompt_mode} | Schema fields: {schema_keys}", flush=True)
            self._logged_first_query = True

        # Retry loop handles network errors and empty/unparseable responses
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            # --- Step 1: Open the streaming connection ---
            try:
                resp = requests.post(
                    self.chat_url,
                    json=payload,
                    # (connect_timeout, read_timeout_per_chunk)
                    # read timeout is per-chunk, not total — so long generations
                    # won't timeout as long as tokens keep flowing
                    timeout=(30, self.timeout),
                    stream=True,
                )
                resp.raise_for_status()
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError) as e:
                last_error = e
                wait = 10 * (attempt + 1)
                print(
                    f"\n    [retry {attempt + 1}/{self.max_retries}] "
                    f"connection error, waiting {wait}s...",
                    end="", flush=True,
                )
                time.sleep(wait)
                continue

            # --- Step 2: Accumulate streamed chunks ---
            raw_content = ""
            raw_thinking = ""
            stream_start = time.monotonic()
            try:
                for line in resp.iter_lines(decode_unicode=True):
                    # Total elapsed check — kills runaway think chains
                    elapsed = time.monotonic() - stream_start
                    if elapsed > self.max_stream_time:
                        raise requests.exceptions.ReadTimeout(
                            f"Total stream time exceeded "
                            f"{self.max_stream_time}s"
                        )
                    if not line:
                        continue
                    chunk = json.loads(line)
                    # Accumulate answer content and thinking tokens separately
                    raw_content += chunk.get("message", {}).get("content", "")
                    raw_thinking += chunk.get("message", {}).get("thinking", "")
                    if chunk.get("done", False):
                        break
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as e:
                last_error = e
                wait = 10 * (attempt + 1)
                print(
                    f"\n    [retry {attempt + 1}/{self.max_retries}] "
                    f"stream interrupted, waiting {wait}s...",
                    end="", flush=True,
                )
                time.sleep(wait)
                continue
            finally:
                resp.close()

            # --- Step 3: Parse the answer ---
            display_letter = None
            try:
                parsed = json.loads(raw_content)
                display_letter = parsed["answer"]
                # Warn if CoT schema enforcement silently failed
                if self.prompt_mode == "cot" and "reasoning" not in parsed:
                    print(f"\n    [WARN] CoT mode but no 'reasoning' field in response", end="", flush=True)
                elif self.prompt_mode == "cot_structured" and "option_a" not in parsed:
                    print(f"\n    [WARN] CoT-structured mode but no 'option_a' field in response", end="", flush=True)
            except (json.JSONDecodeError, KeyError):
                # Fallback: look for a single letter A-D in the raw text
                for letter in ANSWER_LETTERS[:num_choices]:
                    if letter in raw_content:
                        display_letter = letter
                        break

            if display_letter is not None:
                break  # Success — got a valid answer

            # Empty or unparseable response — retry
            last_error = ValueError(
                f"Could not parse model response: {raw_content!r}"
            )
            # Debug: show what the model actually returned so we can
            # diagnose whether it's malformed JSON, empty, or leaked
            # thinking tokens in the content field
            print(
                f"\n    [debug] raw_content: {raw_content[:200]!r}",
                end="", flush=True,
            )
            wait = 5 * (attempt + 1)
            print(
                f"\n    [retry {attempt + 1}/{self.max_retries}] "
                f"empty/bad response, waiting {wait}s...",
                end="", flush=True,
            )
            time.sleep(wait)
        else:
            # All retries exhausted
            raise last_error  # type: ignore[misc]

        # Map the display letter back to a canonical answer index
        display_pos = ANSWER_LETTERS.index(display_letter)
        canonical_idx = answer_permutation[display_pos]

        return raw_content, canonical_idx, raw_thinking


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
