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
