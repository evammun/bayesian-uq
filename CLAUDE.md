# CLAUDE.md — Project Context for Claude Code

## Project Overview

Research project on **uncertainty quantification for local LLMs** — can we measure how confident a model really is by probing it multiple times with shuffled/paraphrased inputs, treating the resulting logprob distributions as Bayesian evidence, and using that to make smarter routing decisions (answer directly vs escalate to a more expensive mode)?

**Authors:** Eva Martin (lead researcher) and Professor Luigi (supervisor, Bayesian inference and computational neuroscience background).

**Status:** Experiments running on CSC Mahti (A100 GPUs). 3 models × multiple conditions. Core pipeline complete, adaptive sampling framework built, analysis ongoing.

## What We're Doing

We present the same MCQ question to a local LLM multiple times with **shuffled answer orderings** (and optionally paraphrased question text). Each query yields a full probability vector [P(A), P(B), P(C), P(D)] from first-token logprobs, mapped back to canonical order. These N probability vectors are then aggregated into a posterior belief about the correct answer.

The key contribution is the **adaptive sampling framework**: instead of always running all N permutations, we monitor the posterior as evidence accumulates and stop early when confident enough (saving compute) or escalate to a more expensive inference mode (CoT or think) when uncertain. The research questions are about which posterior aggregation method best serves this stopping decision, and how the uncertainty signals behave across models and reasoning modes.

**Dataset:** QuALITY (4609 long-form reading comprehension MCQs with article context). Chosen because it requires genuine comprehension of provided context — a proxy for RAG scenarios.

**Models:** Qwen 3 8B, Qwen 3.5 9B, Gemma 4 E4B (all Q4_K_M quantised, run via llama-cpp-python).

**Conditions (factorial):** prompt_mode (direct/CoT) × shuffle (on/off) × think (on/off). CoT and think use a two-pass pipeline: Pass 1 generates reasoning, Pass 2 extracts logprobs conditioned on the reasoning.

## Key Findings So Far

- **Scaffolding absorption:** CoT reasoning absorbs uncertainty into scaffolding tokens, spiking logprobs to near-1.0. Think mode collapses them entirely (MSP=1.000). Two-pass pipeline recovers informative logprobs.
- **Per-query overconfidence:** Individual logprob vectors have MSP ≈ 0.91, ECE = 0.186. Mean across permutations is well-calibrated. This distinction is critical for posterior aggregation.
- **Fragile confidence:** High single-prompt confidence + low cross-permutation consistency = dangerous failure mode. Invisible to any single metric.
- **Adaptive stopping works:** At τ=0.95 with temperature-scaled Product posterior, avg_N=2.76 (of 10), accuracy maintained, 15% of hard questions escalated to think mode.

## Architecture

### Tech Stack
- **Python** with Pydantic data models, NumPy/SciPy for analysis
- **llama-cpp-python** for local inference with low-level logit extraction
- **YAML** configs for experiment definitions
- **Streamlit** dashboard for monitoring and analysis
- **CSC Mahti** (A100 40GB) and vast.ai (RTX 5090) for compute

### Project Structure
```
src/pre_action_uq/       Core library (inference.py, pipeline.py, config.py)
data/                     QuALITY dataset + paraphrases
results/                  Experiment output (JSON, one file per condition)
paper/                    Brainstorm, running notes, lit review
experiments/configs/      YAML experiment configs (local + mahti)
dashboard/                Streamlit monitoring + analysis dashboard
v2_mmlu_archive/          Prior MMLU work (code, data, results, paper drafts)
```

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
