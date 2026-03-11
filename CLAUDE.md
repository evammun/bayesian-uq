# CLAUDE.md — Project Context for Claude Code

## Project Overview

This is a research project developing a black-box Bayesian framework for uncertainty quantification (UQ) in LLM outputs. The framework combines paraphrase-based input diversification with sequential adaptive stopping using Dirichlet posterior updating. It is being developed as an academic paper and potential foundation for an AI consulting offering.

**Authors:** Eva Martin (lead researcher) and Professor Luigi (supervisor, Bayesian inference and computational neuroscience background).

## Core Concept

- Query LLMs with paraphrased versions of the same question (+ answer reordering for multiple-choice)
- Each response updates a Dirichlet posterior over the answer choices (increment pseudo-count by 1)
- After each update, compute exceedance probability via Monte Carlo (Gamma sampling trick)
- Stop when confidence threshold is met; continue querying if not
- Black-box only: text-in, text-out, no logits or model internals required
- Structured output enforced via JSON schema (enum of valid answer choices)

## Two Experiments

**Experiment 1 (Standard benchmarks):** Run framework on MMLU and TruthfulQA. Compare efficiency and calibration against baselines: single query, fixed-budget same-prompt, fixed-budget with paraphrasing, Adaptive-Consistency.

**Experiment 2 (Broken-premise detection):** Novel contribution. A paired dataset of ~100-200 MMLU questions, each with a matched broken-premise variant. The broken version modifies one element to invalidate the premise while keeping answer options identical. Six types of breaks: invalid premise, contradictory setup, category error, temporal violation, mathematical impossibility, entity-framing mismatch. We analyse whether the posterior distribution (convergence speed, entropy, exceedance probability) differs between well-formed and broken-premise versions.

## Tech Stack

- **Python** with uv for package management
- **Ollama** for local model inference (primary model: Qwen 3 8B Q4)
- **NumPy/SciPy** for Dirichlet posterior computation
- **Structured output** via Ollama's JSON schema support
- **Target models:** Small local models — Qwen 3 8B, Gemma 3 4B/12B, Phi-4, Ministral 8B (all quantised)

## Project Structure

```
src/bayesian_uq/          Core library
  dirichlet.py             Posterior updates, exceedance probability computation
  paraphrase.py            Paraphrase generation and quality filtering
  query.py                 LLM query wrapper with structured output enforcement
  stopping.py              Stopping criteria (exceedance, entropy, credible interval)
  pipeline.py              Main sequential sampling loop

experiments/
  exp1_mmlu/               Standard benchmark experiment scripts
  exp2_broken_premise/     Broken-premise paired dataset experiments

data/
  mmlu/                    MMLU test set (test.csv)
  broken_pairs/            Paired broken-premise dataset (our novel contribution)
  paraphrases/             Pre-generated paraphrases per question

notebooks/                 Exploratory analysis and visualisation
results/                   Experiment outputs (git-ignored)
paper/                     LaTeX/manuscript drafts
```

## Key Design Decisions

- All LLM queries return exactly one of K fixed choices via structured output (JSON schema with enum). No free-text responses.
- Paraphrases are generated offline by a strong external model (Claude or GPT-4o), filtered by embedding similarity (meaning preservation) and lexical distance (surface diversity), and stored for reuse.
- Answer choice ordering is shuffled on every query to integrate out position bias.
- The Dirichlet prior is initialised as Dirichlet(1,1,...,1) — uniform, one pseudo-count per option.
- Exceedance probability is computed via Monte Carlo: draw 10,000 samples from the Dirichlet (using the Gamma normalisation trick), count how often the leading answer's component is the largest.

## Coding Style

- Clear, readable Python. Prioritise simplicity over cleverness.
- Type hints on function signatures.
- Docstrings on all public functions.
- Keep core library (src/) clean and modular. Experiment scripts can be scrappier.
- When in doubt, ask — don't assume.

## Important Context

- Eva has strong Python skills (MSc in Data Science & AI) but is newer to Bayesian statistics — prefer clear variable names and comments over terse mathematical notation in code.
- The project targets local models on a laptop with an NVIDIA RTX 3070 (8GB VRAM). Performance and memory constraints matter.
- The primary baseline paper is Adaptive-Consistency (Aggarwal et al., EMNLP 2023) — we extend their approach with paraphrase-based input diversification.
- This is a real research project aimed at publication. Code quality matters because experiments need to be reproducible.


Copy

# Experiment Context

## What we're investigating

LLMs give single answers with no confidence measure. When you ask a question, you don't know if the model is certain, guessing, or confidently wrong. We're building a framework that wraps around any LLM (black-box, text-in text-out, no logits needed) and produces calibrated uncertainty estimates.

The key insight: if you ask the same question multiple ways (paraphrases + answer reordering) and the model gives consistent answers, it's probably right. If it gives inconsistent answers, it's uncertain — and that inconsistency is a useful signal.

## How it works mechanically

1. Start with a multiple-choice question (4 options: A, B, C, D)
2. Initialise a Dirichlet(1,1,1,1) prior over the four answer choices
3. For each query: pick a paraphrase of the question, shuffle the answer order, query the model with structured output (JSON schema forcing one of A/B/C/D), map the response back to canonical answer ordering
4. Update the Dirichlet posterior by incrementing the pseudo-count for the chosen answer
5. Compute exceedance probability: draw 10,000 samples from the Dirichlet (via Gamma trick), count how often the leading answer wins. This is our confidence measure.
6. If exceedance exceeds threshold (e.g. 0.95), stop — we're confident enough. Otherwise query again with a different paraphrase.
7. Result: a posterior distribution over answers, a confidence level, and a count of how many queries were needed.

Easy questions converge in 3-5 queries. Hard or ambiguous questions use the full paraphrase budget without converging — which is itself a useful signal.

## Two experiments

**Experiment 1 — Standard benchmarks (MMLU Redux 2.0):** Run the framework on verified-correct multiple-choice questions. Measure: does it get the right answer? Is the confidence well-calibrated? How many queries does it save compared to a fixed budget? Baselines: single query, same-prompt resampling (Adaptive-Consistency), fixed-budget paraphrasing.

**Experiment 2 — Broken-premise detection (novel contribution):** We created paired questions: each pair has a valid MMLU question and a matched broken-premise variant. The broken version modifies one element to invalidate the premise while keeping the four answer options identical. Example:

- Valid: "What is the primary function of the mitochondria in a cell?" → Answer: Energy production
- Broken: "What is the primary function of the mitochondria in a Monocercomonoides cell?" → Monocercomonoides has no mitochondria. No answer is correct, but the model is forced to pick one.

Six types of premise breaks: existence violations, contradictory setups, category errors, temporal violations, mathematical impossibilities, entity-framing mismatches.

The hypothesis: on broken-premise questions, the posterior will converge more slowly, have lower exceedance, and higher entropy than on matched valid questions — because different paraphrases activate different reasoning pathways and the model becomes inconsistent. This inconsistency is invisible from a single query but visible through the Dirichlet posterior.

We've already confirmed this manually: Qwen 3 8B answers "B" (Energy production) consistently for the valid mitochondria question, but when the broken version is paraphrased as "Monocercomonoides, the first known eukaryote to completely lack certain organelles..." it switches to "A" (Protein synthesis). Different phrasing, different answer — exactly the signal the framework detects.

Key methodological point (from Luigi): we only analyse broken-premise results for pairs where the model confidently gets the valid version RIGHT. This rules out the alternative explanation that the model is just confused because it doesn't know the topic. We're specifically testing: does the model know the domain but fail to detect the broken premise?

## Technical constraints

- Black-box only: structured output via JSON schema, no logits, no hidden states
- Local models via Ollama on an NVIDIA RTX 3070 (8GB VRAM)
- Primary test model: Qwen 3 8B Q4 quantisation
- Paraphrases pre-generated offline by a strong model (Claude API), stored and reused
- All results stored as detailed JSON logs so analysis can be re-run without re-querying models

## Performance and Runtime Considerations

- **Think about runtime before writing loops.** If a task involves 5,000+ items and an API call or model query per item, estimate the total wall-clock time before starting. If it's over 10 minutes, consider parallelisation.
- **Use parallel workers for I/O-bound tasks** like API calls, model queries, and file downloads. Python's `concurrent.futures.ThreadPoolExecutor` for API calls, `ProcessPoolExecutor` for CPU-bound work. Default to 4-8 workers for API calls (respect rate limits).
- **Batch sensibly.** When batching items for API calls, keep batches small enough that one failure doesn't lose much progress (10-20 items), but large enough to be efficient. Never put 200+ items in a single batch.
- **Always save incrementally.** Any long-running script that produces output should write results after every batch or iteration, not just at the end. Design for resumability — check what's already done before starting, skip completed work.
- **Estimate and display progress.** Long-running scripts should print: items completed, items remaining, elapsed time, and estimated time remaining. Use tqdm or manual progress logging.
- **Set timeouts when appropriate.** API calls and model queries should have explicit timeouts. If something is taking 10x longer than expected, it's stuck — don't wait, retry or skip.