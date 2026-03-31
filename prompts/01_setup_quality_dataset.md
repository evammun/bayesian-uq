# Claude Code Prompt: Remove RGB, Download QuALITY, Port Pipeline, Explore Dataset

Project: `C:\Users\evama\Dropbox\Family Room\Projects\bayesian-uq`
Venv: `.venv\Scripts\python`

## IMPORTANT CONTEXT

Read these files first — they contain the full project context, prior pipeline code, and technical learnings:
- `CLAUDE.md` (project overview, coding style, constraints)
- `paper/paper_running_notes.md` (decision log, what changed and why)
- `paper/brainstorm.md` (research design, context conditions)
- `paper/technical_learnings_v2.md` (inference backend notes, logprob extraction gotchas)

The old v2 experiment code is archived in `v2_mmlu_archive/`. The key files to understand the pipeline:
- `v2_mmlu_archive/src/bayesian_uq/query.py` — Ollama inference layer (logprob extraction, two-pass CoT, think mode)
- `v2_mmlu_archive/src/bayesian_uq/pipeline.py` — experiment orchestration (_IncrementalWriter, run_single_question, parallel execution)
- `v2_mmlu_archive/src/bayesian_uq/config.py` — Pydantic data models (ExperimentConfig, QueryResult, QuestionResult, ExperimentResult)
- `v2_mmlu_archive/src/bayesian_uq/analysis.py` — logprob → probability conversion
- `v2_mmlu_archive/analysis/compute_signals.py` — ALL uncertainty signals (Tier I, II, III), CSV output
- `v2_mmlu_archive/dashboard/app.py` — Streamlit dashboard

---

## Step 1: Clean out RGB

Delete everything in `data/` (rgb_en_refine.json, rgb_en_int.json, rgb_en_fact.json). We are completely done with RGB.

---

## Step 2: Download QuALITY dataset

Download from the GitHub repo: https://github.com/nyu-mll/quality

We need the **htmlstripped** versions (plain text, not HTML) from `data/v1.0.1/`:
- `QuALITY.v1.0.1.htmlstripped.train`
- `QuALITY.v1.0.1.htmlstripped.dev`
- `QuALITY.v1.0.1.htmlstripped.test` (if it exists and has gold labels)

These are JSONL files (one JSON object per line). Download all available splits to `data/`.

**We are NOT using the train/dev/test split** — we're not training anything. We're using QuALITY as a stimulus set for our uncertainty experiments. Merge all splits into a single file: `data/quality_all.jsonl`. If the test set has no gold labels (common for leaderboard benchmarks), skip it and merge train + dev only. Keep the original split files too for reference.

**QuALITY format** (each line = one article's question set):
```json
{
  "article_id": "12345",
  "article": "full plain text of the article (~5000 tokens)",
  "questions": [
    {
      "question": "What was the main reason...",
      "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
      "gold_label": 2,  // 1-indexed! Option 2 is correct
      "difficult": 1,   // 1 = hard (speed-readers got wrong), 0 = easy
      "question_unique_id": "..."
    },
    // ... more questions for this article
  ],
  "set_unique_id": "...",
  "writer_id": "...",
  "source": "Gutenberg" or "slate_com" etc,
  "title": "Article title",
  "topic": "Science" or "Fiction" etc,
  "url": "..."
}
```

IMPORTANT: `gold_label` is **1-indexed** (1-4), not 0-indexed. Account for this everywhere.

After downloading, validate:
- Count articles and total questions in each split
- Confirm the JSONL parses correctly
- Print stats: articles per source/topic, questions per article (min/max/mean), hard vs easy split

---

## Step 3: Port the inference layer (Ollama → llama-cpp-python)

Create `src/pre_action_uq/inference.py` — rewrite the inference layer using llama-cpp-python (already installed in the venv with CUDA support, version 0.3.33).

**Class: `LlamaCppClient`**
- Constructor: `model_path` (GGUF file), `n_gpu_layers=-1`, `n_ctx=8192`, `seed=42`, `verbose=False`
- Load model once via `llama_cpp.Llama(...)`, reuse across calls.

**Method: `generate(prompt, max_tokens, temperature, logprobs, top_logprobs, stop, think)`**
- Use `self.model.create_completion(...)` for raw completion (NOT chat mode)
- `think` param: for Qwen 3, this toggles thinking mode. When think=True, add `/think` at the start of the prompt. When think=False, add `/no_think`. (Check Qwen 3 docs if unsure of exact format.)
- Return dict: `{"response_text": str, "logprobs": list[dict], "thinking_trace": str}`
- Logprobs format must match the old Ollama format exactly: `[{"token": str, "logprob": float, "top_logprobs": [{"token": str, "logprob": float}, ...]}]`
- Extract logprobs from llama-cpp-python's response format and convert to this structure.

**Method: `generate_with_logprobs(prompt, max_tokens=1, temperature=0.0, top_logprobs=20)`**
- Convenience method for the common case: generate exactly 1 token, return its logprobs.
- This is the MCQ logprob extraction call (equivalent to `num_predict=1` in the old Ollama pipeline).

**Two-pass CoT method: `generate_cot(prompt, max_tokens, temperature, top_logprobs, think)`**
- Pass 1: generate reasoning (stop at answer token or stop sequence)
- Pass 2: feed prompt + reasoning back, extract logprobs at the answer position
- Port logic from old `query.py` but adapted for llama-cpp-python's API

**Answer extraction: `extract_answer_logprobs(logprobs_list, mode="direct")`**
- Port from old `query.py` — scan for A/B/C/D tokens
- Direct mode: first token's top_logprobs
- CoT mode: scan backwards for last answer token
- Return: `{"display_letter_logprobs": dict, "canonical_logprobs": dict, "canonical_probs": list, "display_answer": str, "canonical_answer": int}`

**Finding the model GGUF:**
The user has Qwen 3 8B Q4 via Ollama. The GGUF is stored in Ollama's blob store at `C:\Users\evama\.ollama\models\blobs\`. Find the largest file there — it's the model weights (~5GB). If the path doesn't work, print clear instructions.

**Smoke test (if __name__ == "__main__"):**
- Load model, run one direct-mode completion with logprobs, print result
- Run one completion with think=True, print reasoning + logprobs
- Verify logprobs format matches expected structure

---

## Step 4: Port the experiment pipeline for QuALITY

Create `src/pre_action_uq/config.py` — Pydantic data models:
- **ExperimentConfig**: run_name, model_path, think (bool), prompt_mode ("direct"/"cot"), dataset_file, context_condition ("sufficient"/"insufficient"), max_questions, seed, temperature, num_paraphrases, n_ctx, num_workers
- **QueryResult**: query_number, paraphrase_index, query_text, answer_permutation, raw_response, raw_logprobs, display_letter_logprobs, canonical_logprobs, canonical_probs, display_answer, canonical_answer, thinking_trace. Same structure as old format.
- **QuestionResult**: question_id, question_unique_id, article_id, question_text, options, correct_answer (0-indexed internally), difficult (bool), context_condition, query_log (list[QueryResult]), num_queries, correct (bool), mean_probs, final_answer, answer_counts
- **ExperimentResult**: run_name, config, timestamp, question_results

Create `src/pre_action_uq/pipeline.py`:
- Port `_IncrementalWriter` exactly from old pipeline.py (it was well-optimised, O(1) per save)
- **`load_quality_dataset(path)`**: Load JSONL, flatten into list of individual questions. Each question gets the article text, options, gold_label (convert to 0-indexed), difficult flag, article metadata (source, topic, title).
- **`build_prompt(question_text, options, article_text, context_condition, answer_permutation)`**:
  - Construct the MCQ prompt. Format:
    ```
    Read the following passage and answer the question.

    Passage:
    {article_text}

    Question: {question_text}

    A) {options[permutation[0]]}
    B) {options[permutation[1]]}
    C) {options[permutation[2]]}
    D) {options[permutation[3]]}

    Answer:
    ```
  - `context_condition="sufficient"`: use the correct article
  - `context_condition="insufficient"`: use a DIFFERENT article (details below)
  - `answer_permutation`: shuffle which option maps to which letter (same as v2)
- **`run_single_question(client, question, config, rng)`**: Same structure as old — build query schedule (original + N paraphrases), each with a random answer permutation. Execute all queries, aggregate probabilities.
  - For now, "paraphrases" = same question text, different answer permutation (shuffle-only mode, like v2's nopara+shuffle condition). Real paraphrasing comes later.
  - Use ThreadPoolExecutor for parallel inference in direct mode (same as old pipeline)
- **`run_experiment(config)`**: Same loop — iterate questions, incremental saves every 20 questions, progress bar, resume support.
- **Insufficient context construction**: When context_condition="insufficient", for each question, pick a random article from the same dataset that is NOT the correct article. Prefer same topic/source if available, otherwise random. Store the swapped article_id in the result for traceability.

Create `experiments/run_experiment.py` — CLI entry point:
- `--config` (YAML path), `--resume` (optional JSON path)
- Load QuALITY dataset, run experiment, save results

Create experiment configs in `experiments/configs/`:
- `quality_pilot_direct_nothink_sufficient.yaml`: 50 questions from dev set, direct mode, think=false, context_condition=sufficient, num_paraphrases=5, temperature=0.7
- `quality_pilot_direct_nothink_insufficient.yaml`: same but context_condition=insufficient
- `quality_pilot_direct_think_sufficient.yaml`: same but think=true
- `quality_pilot_direct_think_insufficient.yaml`: same but think=true + insufficient

---

## Step 5: Port signal computation

Copy `v2_mmlu_archive/analysis/compute_signals.py` to `analysis/compute_signals.py` and adapt:
- Remove MMLU-specific SUBJECT_TO_PARENT mapping
- Add QuALITY metadata: article_id, source, topic, difficult (easy/hard)
- Keep ALL Tier I and Tier II uncertainty signals — unchanged
- Keep Tier III position signals (we're still shuffling answer positions)
- Add columns: context_condition, difficult, topic, source
- Same CSV output format, same column naming

---

## Step 6: Dataset exploration script

Write `experiments/quality_exploration.py` — thorough dataset profile:

**Basic stats (for the merged quality_all.jsonl, and note how many came from each split):**
- Article count, question count
- Questions per article: min/max/mean/median
- Hard vs easy question split (count and %)
- Articles by source (Gutenberg, Slate, etc.) and topic

**Article analysis:**
- Article length in tokens (approximate with word count / 0.75): min/max/mean/median
- Will articles fit in n_ctx=8192 with prompt overhead? Flag any that won't.

**Question analysis:**
- Question length stats (words)
- Option length stats (words)
- Are options roughly equal length? (Important — length can be a spurious signal)
- Gold label distribution — are answers balanced across positions 1-4?

**Sample output — print 10 diverse questions:**
- Sample 2-3 from each topic/source combination if possible
- For each, print: article title (first 80 chars), question, all 4 options, gold label, difficulty
- Pick a mix of easy and hard questions
- Truncate article to first 200 chars just to show what kind of text it is

**Cross-article analysis (for constructing insufficient context condition):**
- How many articles share the same topic? Same source?
- For insufficient context swaps, we need articles that are similar-enough in genre but contain different information. Print the topic × source matrix.

Save summary to `experiments/quality_dataset_profile.md` (concise, dense format — no fluff).

---

## Constraints & Reminders
- Use the venv Python: `.venv\Scripts\python`
- GGUF model location: check `C:\Users\evama\.ollama\models\blobs\` for the largest file
- `gold_label` is 1-indexed in QuALITY — convert to 0-indexed internally everywhere
- JSON output format must match v2 structure so compute_signals.py works with minimal changes
- Save all scripts to the right directories (src/ for library, experiments/ for scripts, analysis/ for signal computation)
- Keep code clean: type hints, docstrings on public functions, clear module structure
- If anything fails, report clearly what broke — don't silently skip
