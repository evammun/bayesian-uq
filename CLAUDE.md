# CLAUDE.md — Project Context for Claude Code

## Project Overview

This is a research project developing **pre-action uncertainty quantification** for LLM-based systems. The core question: before a local LLM acts on a user query + retrieved context, can we detect whether it actually understood the situation — and route accordingly?

The project builds on prior work (archived in `v2_mmlu_archive/`) that developed paraphrase-based logprob analysis for MCQA uncertainty. The new direction applies these techniques to realistic RAG and tool-calling scenarios rather than academic benchmarks.

**Authors:** Eva Martin (lead researcher) and Professor Luigi (supervisor, Bayesian inference and computational neuroscience background).

**Status:** Early-stage pivot. Brainstorming and design phase. No experiments yet.

## The Core Idea

In production LLM deployments (customer service, knowledge assistants, RAG-augmented agents), the model receives a user query + retrieved context and must decide what to do: answer directly, call a tool, ask for clarification, or escalate. At that decision point, nobody checks whether the model actually understood the query or the context.

We propose a **pre-action verification layer** that uses paraphrased comprehension probes + logprob analysis to measure three types of uncertainty before the model acts:

1. **Query comprehension uncertainty** — Does the model understand what the user is asking? Tested by paraphrasing the user query and checking whether the model's interpretation is stable across phrasings.

2. **Context sufficiency uncertainty** — Does the model recognise whether the RAG-retrieved context actually contains the information needed? Tested by probing comprehension of the retrieved context and comparing model behaviour with sufficient vs insufficient context.

3. **Action selection uncertainty** — Should the model answer, call a tool, ask for clarification, or escalate? Tested by extracting logprobs at the action-selection decision point across paraphrased inputs.

## How Prior Work Maps Onto This

The v2 MMLU work established:
- Paraphrase-based logprob extraction pipeline (query LLM with paraphrased inputs, extract logprob distributions, aggregate)
- Uncertainty signals: entropy, agreement, epistemic/aleatoric decomposition, confidence variance, rank stability, answer coverage
- The "fragile confidence" finding: high single-prompt confidence + low cross-paraphrase consistency = dangerous failure mode
- Scaffolding absorption: CoT and structured output absorb uncertainty into token scaffolding before the decision token

All of this transfers. The model changes from "Qwen answering MCQs" to "Qwen acting as a RAG-augmented assistant." The logprob machinery is identical. The evaluation framework is identical. We just need a different experimental setup and dataset.

## What Needs To Be Figured Out

- **Dataset design:** What does the evaluation dataset look like? Options: adapt existing QA datasets (Natural Questions, SQuAD, MS MARCO) with sufficient/insufficient context variants, build a synthetic business-scenario dataset, or both.
- **Comprehension probe generation:** How are the probes created? Auto-generated MCQ from query + context? Templated from the action space? Simpler consistency checks?
- **Ground truth:** What counts as "correct understanding"? For action selection, ground truth is cleaner (there's a correct action). For comprehension, need a proxy.
- **Tool-calling evaluation:** How to set up a realistic tool-calling scenario with defined tools and clear correct/incorrect action choices.
- **Scope:** Is this one paper or two? Comprehension + context sufficiency is one story; tool-calling verification might be a separate one.

## Architecture (from v2 — to be adapted)

### Tech Stack
- **Python** with Pydantic for data models, NumPy/SciPy for analysis
- **Ollama** (or llama-cpp-python) for local model inference
- **Primary model:** Qwen 3 8B Q4 (`qwen3:8b-q4_K_M`) — or newer equivalent
- **YAML** configs for experiment definitions

### Project Structure
```
src/pre_action_uq/       Core library (to be built)
data/                     Evaluation datasets
results/                  Experiment output
paper/                    Drafts, notes, lit review
experiments/configs/      YAML experiment configs
v2_mmlu_archive/          All prior MMLU work (code, data, results, paper drafts)
```

## Prior Work Archive

Everything from the MMLU-based project is preserved in `v2_mmlu_archive/`, including:
- `v1_sampling_archive/` — Original Dirichlet sampling approach
- `src/bayesian_uq/` — v2 logprob extraction pipeline (query.py, pipeline.py, analysis.py, config.py)
- `analysis/` — Signal computation and exploration notebooks
- `paper/` — Brainstorm docs, lit review, running notes, uncertainty signals spec
- `data/` — MMLU Redux questions, paraphrases
- `results/` — Full experiment results (10 conditions, ~2.7GB)
- `dashboard/` — Streamlit monitoring dashboard

## Coding Style

- Clear, readable Python. Prioritise simplicity over cleverness.
- Type hints on function signatures.
- Docstrings on all public functions.
- Keep core library (src/) clean and modular. Experiment scripts can be scrappier.
- When in doubt, ask — don't assume.

## Important Context

- Eva has strong Python skills (MSc in Data Science & AI) but is newer to Bayesian statistics.
- The project targets local models on a laptop with an NVIDIA RTX 3070 (8GB VRAM). Performance and memory constraints matter.
- This is a real research project aimed at publication. Code quality matters because experiments need to be reproducible.
- The commercial angle: "a verification layer that makes your cheap local models more reliable without changing your deployment."

## Documentation Habits

- **Running log (`paper/paper_running_notes.md`):** Update during every heavy work session. Captures: work done that day, decision points, experiments run.
- **Brainstorm (`paper/brainstorm.md`):** Capture all ideas during brainstorming sessions, even speculative ones. We may not act on them but want to remember them.
- **Both docs must be concise.** Dense note format, not polished prose. These get read by LLMs regularly — context size matters. Bullet points, shorthand, compressed summaries. No beautifying.

## Performance and Runtime Considerations

- **Think about runtime before writing loops.** Estimate wall-clock time before starting anything that touches 1000+ items with API calls.
- **Use parallel workers for I/O-bound tasks.** ThreadPoolExecutor for API calls, ProcessPoolExecutor for CPU-bound work.
- **Always save incrementally.** Long-running scripts should write results after every batch, not just at the end. Design for resumability.
- **Estimate and display progress.** Use tqdm or manual progress logging.
- **Set timeouts.** API calls and model queries should have explicit timeouts.
