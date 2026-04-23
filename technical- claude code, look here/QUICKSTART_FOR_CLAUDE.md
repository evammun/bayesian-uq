# Quickstart for next Claude instance

## The one-paragraph context
Eva's research project on **pre-action uncertainty quantification** for local LLMs. Before a RAG-augmented model acts on a query+context, can we tell if it understood? Reframes UQ as *input-side* (did it understand?) vs output-side (is the answer right?). Target: EMNLP 2026. Supervisor: Prof. Luigi.

## The current phase
Data collection done. All 4609 QuALITY MCQ questions × 4 main conditions + earlier runs.

Current work = **adaptive sampling framework**:
- Bayesian posterior stopping (implemented in dashboard tab 7)
- Temperature scaling T=3.0 for calibrating overconfident per-query logprobs (ECE 0.186→0.012)
- Joint (T, τ) optimization with Pareto frontier (dashboard)
- Signal-augmented stopping (designed, not yet implemented: 2nd Gap, argmax flips, epistemic, conf variance)
- New adaptive pipeline with timing instrumentation (TODO)
- Multi-model expansion (TODO — need to audit Qwen3-specific code)

Still pending (analysis): AUROC table, selective prediction curves, easy/hard split, N-query sensitivity, trace content analysis.

## Key files to read in order
1. `CLAUDE.md` (repo root) — project rules, tech stack, coding style
2. `paper/brainstorm.md` — current design, three uncertainty types, context conditions, analysis TODO (§7b), adaptive escalation (§8, IN PROGRESS)
3. `paper/paper_running_notes.md` — decision log and what's been done (most recent at bottom, April 23 = adaptive sampling session)
4. `src/pre_action_uq/pipeline.py` — experiment runner, resume logic, paraphrase/shuffle handling
5. `src/pre_action_uq/inference.py` — llama-cpp-python low-level logit extraction (the VRAM-saving trick)
6. `dashboard/app.py` — Streamlit monitoring (auto-refresh 2min). Signal Battery = tab 6. **Adaptive Sampling = tab 7** (new: posterior stopping, joint T×τ optimization, Pareto frontier).
7. `experiments/dump_dashboard_summary.py` — CLI summary of all signals (no dashboard needed)
8. `experiments/configs/mahti/*.yaml` — experiment configs

## Tech stack
- **Python** + Pydantic + numpy + pandas
- **llama-cpp-python** (NOT Ollama) — low-level `llama_get_logits()` with `logits_all=False` (fits 8GB GPU)
- **Qwen 3 8B Q4_K_M** (`qwen3:8b-q4_K_M`)
- **Streamlit** dashboard, YAML experiment configs
- **Venv:** `Reconciliation` (conda) on Eva's Windows laptop
- **Compute:** CSC Mahti (A100 40GB, SLURM gpusmall 36h max) + historically vast.ai RTX 5090

## The signal battery (the main science)
9 signals across 5 categories:
- **Single-query:** MSP, 2nd Gap (best single-query), Answer Coverage
- **Multi-query:** Epistemic decomposition (mutual info), Conf Var, Agree-Conf Gap
- **Cross-mode:** CoT vs direct disagreement → 44.9% acc (vs 80.4% agree) — **strongest signal**
- **Position:** Loyalty under shuffle
- **Reasoning:** Trace length (think wrong = +1976 chars), CoT P1/P2 disagreement

## Critical technical gotchas (don't re-discover these)
- `logits_all=False` + `llama_get_logits()` — 4.7GB VRAM saved
- Chat template with `Answer:` as **assistant prefill** — without it logprobs are garbage
- `model._ctx.memory_clear(True)` — `reset()` alone doesn't clear KV cache
- CoT anti-leak via **conciseness** ("3-4 bullets") not explicit instruction
- Think mode has scaffolding absorption: ~99% confidence on everything, uninformative
- Paraphrase adds minimal value vs shuffle (epistemic 0.056 vs 0.136). Shuffle disrupts position bias; paraphrase doesn't change cognitive path.
- RNG seeding uses **stable `_original_index` map** built BEFORE filtering (so resume preserves shuffle perms)
- 3-layer dedup: pre-loop filter + in-loop skip + per-question set
- **Per-query logprobs are overconfident** (MSP≈0.91, ECE=0.186). Mean probs across permutations are better calibrated. Temperature scaling T=3.0 needed for Bayesian posterior to work (ECE→0.012). Without it, posterior concentrates in 1-2 samples and cap_rate≈0%.
- **Qwen3-specific code** (audit before multi-model): chat template in inference.py, `/think` tag handling in generate_think(), model filename pattern. 2 hard items, 4 easy.

## Running the dashboard
```bash
conda activate Reconciliation
cd "C:/Users/evama/Dropbox/Family Room/Projects/bayesian-uq"
streamlit run dashboard/app.py
```

## Checking Mahti jobs
```bash
ssh mahti.csc.fi "squeue -u evamar"
ssh mahti.csc.fi "cd /scratch/project_2018384/bayesian-uq && grep -o '\"question_id\"' results/<file>.json | wc -l"
```
Project dir on Mahti: `/scratch/project_2018384/bayesian-uq`. Username: `evamar`.

## Resubmitting a timed-out Mahti job
```bash
ssh mahti.csc.fi "cd /scratch/project_2018384/bayesian-uq && sbatch --time=36:00:00 --export=CONFIG=experiments/configs/mahti/<config>.yaml scripts/mahti/slurm_single.sh"
```
Auto-resume happens inside `slurm_single.sh` via `ls -S` picking largest matching file.

## User preferences (important)
- **Concise**: dense notes, no prose padding. Context size matters for LLM consumption.
- **Ask before big scripts** — explain approach first
- **Proactively flag performance issues**
- **Remind to push to GitHub** at end of heavy sessions
- **Source files are precious** — Dropbox, treat as read-only
- Dashboard styling: Inter font, Playfair for h1, teal/rose
- Eva has Python/DS background, newer to Bayesian stats — frame stats accordingly

## Claude Code workflow (USE THIS)
- **Opus** for: architecture, judgment calls, synthesis, reviewing code, anything with high blast radius if wrong
- **Sonnet** for: implementation, boilerplate, data exploration, building tests — treat as a **capable intern nearing end of internship**
  - Trust but verify: don't hand-hold (no step-by-step instructions), DO check the work
  - Have Sonnet **write tests for its own code** — build in verification
  - Give it context and intent, let it figure out the how
  - Review output before shipping — Sonnet is good but not infallible
- Pattern: Opus designs + specifies → Sonnet implements → Opus reviews → ship
- Don't over-delegate understanding: Opus must understand the problem before briefing Sonnet

## Paper skeleton (target: EMNLP 2026, June submission)
Intro → Related work → Method → Setup (QuALITY, Qwen 8B, context conditions) → Results (do signals predict sufficiency? does paraphrasing help? 2D space? easy vs hard?) → Discussion

## What's parked (future work)
- **C5 condition**: topically-relevant-but-unanswerable context (Garden of Eden problem — model falls back to pretraining)
- **Partial (C3) and counterfactual (C4)** context conditions
- **Tool-calling / action selection (RQ3)** via BFCL
- Multi-model expansion (2 models TBD — Luigi testing his side)
- With/without-context delta as a signal

## What's in progress
- Signal-augmented stopping (2nd Gap, flips, epistemic, conf var — designed, code TODO)
- New adaptive pipeline with timing instrumentation (architecture done, code TODO)
- Multi-model code audit (Qwen3-specific items identified)

## Archive
Everything from the v1/v2 MMLU-based work is in `v2_mmlu_archive/`. Don't touch unless explicitly asked.
