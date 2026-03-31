# Bayesian UQ Paper — Running Notes & Decision Log

**Authors:** Eva Martin (lead researcher), Professor Luigi (supervisor)
**Working title:** Token-Level Uncertainty Quantification for LLMs via Paraphrase-Based Logprob Analysis
**Started:** March 2026
**Status:** Experiment 1 in progress — 4 direct conditions complete, cot_noshuffle_nopara complete, 3 CoT multi-query conditions + 3 think chunks running on cloud GPU (Vast.ai RTX 5090). cot_structured dropped.

---

## 1. Project Genesis

### 1.1 Background and Motivation

The project originated from a dual motivation: academic (a publishable paper demonstrating research capability) and commercial (a foundation for an AI consulting offering focused on making small local LLMs more reliable for enterprise deployment).

Luigi drafted an initial project overview document: "Bayesian Uncertainty Quantification for LLMs via Multi-Model Fusion with Paraphrase-Based Sampling." The core observation: when you ask an LLM a question, you get one answer with no indication of confidence. The proposed framework: ask the question multiple times, in different ways, using Bayesian statistics to build a picture of how certain the answer is, and stop asking as soon as confidence is sufficient.

The commercially compelling framing: "We can make your cheap local models dramatically more reliable without changing anything about how you deploy them." This works because the framework is black-box — text-in, text-out, no model internals required. It works with any API, any model.

### 1.2 Luigi's Original Framework

Luigi's draft proposed combining four ideas from the literature that had never been integrated:

1. **Paraphrase-based input diversification** — asking the same question with different wordings, since LLMs are sensitive to phrasing (SPUQ, Gao et al., EACL 2024)
2. **Sequential adaptive stopping** — stop querying when confident enough, don't waste compute on easy questions (Adaptive-Consistency, Aggarwal et al., EMNLP 2023)
3. **Bayesian posterior aggregation** — use a Dirichlet posterior over answer choices, updated after each query, with exceedance probability as the stopping criterion
4. **Answer reordering** — shuffle multiple-choice answer positions across queries to integrate out position bias (Cox et al., AAAI 2025)

The framework is entirely black-box: it treats the LLM as a function that takes text and returns text. No logits, no model internals, no fine-tuning. This was positioned as both a feature (universal applicability) and a practical necessity (many deployment scenarios only offer API access).

### 1.3 Initial Scope Decisions

From the first brainstorming session, we made several scope decisions:

- **Single paper focus:** The draft described ~4 papers' worth of extensions. We scoped down to: baseline framework + paraphrasing + stopping + one novel experiment.
- **Small local models:** Target Qwen3 8B Q4 on a laptop RTX 3070. This is the realistic deployment scenario — companies running small quantised models locally for cost/privacy reasons.
- **MMLU as the benchmark:** Standard, well-understood, allows comparison with published numbers.
- **No multi-model fusion yet:** Luigi's draft mentioned routing across multiple models. Deferred to future work — the single-model case is the foundational result.

---

## 2. The Broken-Premise Idea (Eva's Contribution)

### 2.1 Origin

During the first brainstorming session, Eva proposed what became the project's most distinctive contribution: a paired dataset of "broken-premise" questions. The idea: take a well-formed MMLU question, modify one element to invalidate the premise, and see whether the uncertainty framework can detect the difference.

The breakthrough example: the mitochondria question. Valid: "What is the primary function of the mitochondria in a cell?" Broken: "What is the primary function of the mitochondria in a *Monocercomonoides* cell?" (Monocercomonoides is the first known eukaryote to completely lack mitochondria.) The question looks identical in structure but has no correct answer.

This is fundamentally different from existing "bullshit detection" benchmarks like BullshitBench, which use obvious gibberish ("chromatic aberration of Docker image layers"). The broken-premise questions look and read like perfectly normal exam questions — every word is real, the grammar is fine, the options make sense in isolation. Only domain knowledge reveals the problem.

### 2.2 Taxonomy of Premise Violations

We developed a six-type taxonomy with 25+ examples:

- **Type 1 — Existence violations:** Asking about properties of entities that lack them (e.g., RBC mitochondria, sound in vacuum, insulin post-pancreatectomy)
- **Type 2 — Contradictory setups:** Internally conflicting information (e.g., adiabatic expansion with heat transfer, bowling strike on second throw)
- **Type 3 — Category errors:** Valid concepts applied to wrong domains (e.g., PCR on protein samples, bacterial autosomal inheritance)
- **Type 4 — Temporal/contextual violations:** References to non-existent time/context (e.g., $500 bill "currently in active circulation," Linnaean 1735 system with "Domain" rank)
- **Type 5 — Mathematical impossibilities:** Quantitatively contradictory setups (e.g., mutually exclusive events with P(E)+P(F) > 1, division by zero)
- **Type 6 — Entity-framing mismatches:** Entities don't belong to assumed category (e.g., "Which charity is not a stock brokerage?" when none are charities)

### 2.3 Expected Posterior Patterns

The paired design enables three diagnostic patterns:

- **Pattern A (Confident Ignorance):** Model answers the broken version identically to the valid version. Fast convergence, high confidence, wrong answer. The model lacks the knowledge to detect the break. Framework honestly reports confident ignorance.
- **Pattern B (Detectable Confusion):** Model gives inconsistent answers across paraphrases on the broken version but is consistent on the valid version. Slow convergence, high entropy, low confidence. The gold case — the framework flags the problematic question.
- **Pattern C (Successful Detection):** Model consistently recognises the broken premise. Flat posterior / refusal.

The key methodological point: we filter to pairs where the model gets the valid version right with high confidence. This rules out domain ignorance — if the model can't answer the original, we can't interpret the broken-version results.

### 2.4 Current Status

The broken-premise experiment (Experiment 2) is designed but deferred until Experiment 1 results are complete. The taxonomy and paired-question approach will be a separate contribution.

---

## 3. Implementation History

### 3.1 V1: Sampling-Based Framework (March 4-12)

**Architecture:** Dirichlet posterior with conjugate updates. Each model query returns a categorical answer (A/B/C/D) via structured JSON output. The answer increments the corresponding pseudo-count. After each update, compute exceedance probability via Monte Carlo (10,000 Gamma samples). Stop when exceedance exceeds threshold (0.95) or budget exhausted (up to 100 queries).

**Stack:** Python with uv, Ollama for local inference (Qwen3 8B Q4_K_M), structured JSON output via Ollama's `format` parameter, NumPy/SciPy for Dirichlet computation, Streamlit dashboard for monitoring.

**Key V1 Results:**

Accuracy across conditions was stable at 76-77% — the experimental variables didn't change how often the model was right. But they changed whether the framework could *tell the difference*:

| Condition | Accuracy | AUROC | Avg queries (correct) | Avg queries (incorrect) |
|-----------|----------|-------|-----------------------|------------------------|
| No shuffle, no para | 76.4% | 0.511 | 5.4 | 6.8 |
| No shuffle, para on | 76.2% | 0.549 | 7.9 | 15.9 |
| Shuffle + para | 77.1% | 0.568 | 11.2 | 31.6 |

The headline finding: without input diversification, the framework converges to 0.95 exceedance on everything equally (AUROC barely above random). With paraphrasing and shuffling, the posterior becomes diagnostic — incorrect answers take 3× longer to converge.

**Key insight from V1:** The adaptive stopping criterion censors the calibration data. Almost every question eventually converges to 0.95 exceedance regardless of correctness. The diagnostic signal is expressed as *convergence speed* (how long it takes to decide), not *final confidence* (which is censored to ~0.95 by design). This is a genuine methodological contribution — most UQ papers report calibration on fixed-budget posteriors.

**V1 Problems:**
- Extremely slow: 5-8 hours per experimental condition (up to 100 queries per question, each taking 2+ seconds)
- Think mode caused models to hang on certain questions
- The JSON schema enforcement was a hidden confound (the model had to generate `{"answer": "B"}` which absorbed uncertainty into the JSON scaffolding tokens before the answer token)
- A Windows localhost→IPv6 DNS resolution bug added ~2 seconds per request

### 3.2 The Logprob Pivot (March 16)

**The realisation:** Ollama (v0.12.11+) exposes token log-probabilities. Instead of asking the model 100 times and counting votes, we can ask once and read the full probability distribution over A/B/C/D from the answer token's logprobs.

**Why this doesn't invalidate paraphrasing:** Logprobs from a single forward pass tell you "given *this exact prompt*, how does the model distribute probability across answers?" Paraphrase-based analysis tells you "across *many different ways of asking*, how consistently does the model give the same answer?" These measure different things. A model can assign 95% to B on one phrasing and 95% to A on another. The single-pass logprobs look confident both times; the *inconsistency across phrasings* is the diagnostic signal.

This is the epistemic vs. aleatoric uncertainty distinction: logprobs capture within-prompt confidence, paraphrase variation captures across-prompt robustness (a proxy for epistemic uncertainty).

**Critical discovery:** Structured JSON output kills the logprob signal. With JSON schema enforcement, the model produces `{"answer": "B"}` — by the time it reaches the answer token, the JSON scaffolding has absorbed all uncertainty. At the answer position, logprobs are spiked (99.99% on one letter). Without JSON enforcement, asking for a single letter and setting `num_predict: 1`, the logprobs contain meaningful spread (e.g., B: 61%, D: 30%, A: 8%, C: 0.1%).

**The fix:** Use Ollama's `/api/generate` endpoint (text completion) rather than `/api/chat`. End the prompt with `Answer:` and let the model complete with a single token. Set `top_logprobs: 20` to capture probabilities for all four answer letters.

### 3.3 V2: Logprob-Based Framework (March 16 onwards)

**Architecture:** Each paraphrase (or original question) queried exactly once. No sampling, no Dirichlet, no Monte Carlo, no stopping criterion. Extract the probability distribution [P(A), P(B), P(C), P(D)] from the first output token's logprobs, normalised over just the four answer letters. Aggregate 11 distributions (1 original + 10 paraphrases) per question using mean probabilities.

**Speed improvement:** ~100× faster than V1. A full 5,330-question run with paraphrasing takes ~90 minutes instead of 8+ hours. The no-paraphrase baseline (1 query per question) takes ~20 minutes.

**Additional speed fix:** Changed the Ollama URL from `localhost` to `127.0.0.1` to avoid Windows IPv6 DNS resolution, saving ~2 seconds per request (from 2.2s to 0.19s).

**Data richness:** Each query now produces a full probability vector instead of a single categorical vote. The results file contains both raw logprobs (for auditing) and normalised probabilities (for analysis).

---

## 4. Current Experimental Design

### 4.1 Dataset

**MMLU-Redux 2.0** (Gema et al., NAACL 2025): 5,700 manually re-annotated questions across all 57 MMLU subjects. We filter to `error_type = "ok"` (verified correct questions), yielding 5,330 questions. This is the corrected version of the original MMLU, which has an estimated 6.49% error rate in ground truth labels.

Rationale for Redux over original MMLU: cleaner labels mean any uncertainty we detect is genuinely from the model, not from noisy ground truth. Redux is also the version recommended by the benchmark maintainers and is gaining adoption in the evaluation community.

### 4.2 Paraphrase Generation

10 paraphrases per question generated offline using the Anthropic API (Claude Sonnet). Prompt instructs: preserve all factual content exactly, vary sentence structure and vocabulary, each paraphrase must be a standalone exam question, do not include answer choices.

5,101 questions paraphrased via the API; remaining 229 completed by Claude Code (same model, same instructions). Quality validated: checked for coverage (all 5,330 questions have 10 paraphrases), count verification, original text matching, duplicate detection, and absence of answer choices in paraphrase text.

**Known limitation:** The paraphrases are relatively conservative — mostly synonym substitution and sentence restructuring rather than deep reframing. Manual experiments showed that more aggressive paraphrases (e.g., "Monocercomonoides, the first known eukaryote to completely lack certain organelles...") can crack questions that conservative paraphrases can't. This is noted as a limitation and potential follow-up: paraphrase depth as an experimental variable.

### 4.3 Experimental Variables

Three-dimensional factorial design plus prompt mode:

| Variable | Levels | Rationale |
|----------|--------|-----------|
| **Prompt mode** | direct / cot / cot_structured | Does chain-of-thought reasoning improve accuracy and/or uncertainty quality? |
| **Answer shuffling** | on / off | Does randomising answer order integrate out position bias and improve calibration? |
| **Paraphrasing** | on / off | Does input diversification improve the framework's ability to distinguish correct from incorrect? |

Total: 3 × 2 × 2 = 12 experimental conditions. All use the same model (Qwen3 8B Q4_K_M), same random seed (2024), same temperature (0.7).

**Query counts per condition:**
- Paraphrase on + shuffle on: 11 queries (original + 10 paraphrases, each with different answer order)
- Paraphrase on + shuffle off: 11 queries (original + 10 paraphrases, same answer order)
- Paraphrase off + shuffle on: 11 queries (original text × 11 different answer orderings)
- Paraphrase off + shuffle off: 1 query (identical prompt → identical logprobs, repetition pointless)

### 4.4 Prompt Modes

Detailed rationale in `paper/cot_prompting_rationale.md`. Summary:

**No system prompt** in any condition. The standard MMLU evaluation protocol (EleutherAI lm-eval-harness) uses no system prompt. The "You are a helpful assistant" framing is non-standard and adds prompt sensitivity as a confound. Removed.

**Direct mode:** Prompt ends with `Answer:` — model completes with single letter. `num_predict: 1`. Logprobs extracted from this single token.

**CoT mode:** Prompt says "Consider each option, then state your answer as a single letter." Model generates free-form reasoning, then the answer letter. `num_predict: 500`. Logprobs extracted from the last A/B/C/D token. The key insight: in the original V1 design, we explored using JSON schema with a `reasoning` field before the `answer` field, forcing the model to reason before committing (leveraging autoregressive left-to-right generation). V2 uses free-text CoT since we dropped JSON enforcement.

**CoT-structured mode:** Similar to CoT but with a more explicit instruction: "Briefly evaluate each option, then state your final answer as a single letter (A, B, C, or D)." Forces the model to consider all four options.

### 4.5 Hypotheses

**H1 (Paraphrase effect on discrimination):** Paraphrase-based sampling produces distributions that better discriminate correct from incorrect answers than single-prompt logprobs alone. Measured by AUROC of max(mean_probs) as a classifier for correctness.

**H2 (Shuffling effect on calibration):** Answer shuffling reduces position bias and improves the calibration of aggregated probability estimates. Measured by comparing the distribution of mean_probs for correct vs incorrect answers.

**H3 (CoT effect):** Chain-of-thought prompting changes both accuracy and the distribution of logprobs. Prediction: CoT may improve accuracy but could reduce the diagnostic value of logprobs by making the model more consistently confident (whether right or wrong).

**H4 (Convergence speed as diagnostic — from V1):** In the sampling framework, the number of queries needed to reach a confidence threshold is itself a diagnostic signal. Correct answers converge faster than incorrect ones. This was demonstrated in V1; the V2 analog is the variance/agreement across paraphrase distributions.

**H5 (Combined signal):** The combination of within-prompt confidence (logprobs) and across-prompt consistency (paraphrase agreement) provides better uncertainty estimation than either alone.

---

## 5. Key References

### Primary Baselines and Related Work

- **Adaptive-Consistency:** Aggarwal et al., EMNLP 2023. Sequential stopping with same-prompt resampling. Direct baseline for V1. Extended by our paraphrase-based approach.
- **SPUQ:** Gao et al., EACL 2024. Paraphrase-based uncertainty quantification with fixed budget. Validates the paraphrase approach. We add adaptive stopping (V1) and logprob analysis (V2).
- **Self-Consistency:** Wang et al., NeurIPS 2022. The foundational paper on sampling multiple reasoning paths and taking majority vote. Critically, they tried sequence-level log probability weighting and found it unhelpful — but nobody tested *token-level* logprobs at the answer decision point. This is our key novelty claim.
- **"Just Rephrase It!":** Beker et al., 2024. Rephrasing + majority vote for uncertainty estimation, explicitly framed as a workaround for when logprobs are unavailable. We show that logprobs + rephrasing together are better than either alone.
- **BayesPE:** Tonolini, ACL 2024. Bayesian prompt ensembles. Needs a validation set. We don't.
- **Cox et al.:** AAAI 2025. Paraphrasing vs temperature for uncertainty. Validates paraphrase approach over simple temperature sampling.

### Dataset

- **MMLU-Redux 2.0:** Gema et al., NAACL 2025 (arXiv:2406.04127). Corrected version of MMLU with error annotations. 5,700 re-annotated questions, ~6.49% original error rate.
- **MMLU-Pro:** Wang et al., NeurIPS 2024 (arXiv:2406.01574). 10-option variant, more reasoning-focused. Found MMLU scores vary 4-5% across prompt styles; CoT helps on MMLU-Pro but not original MMLU.

### Uncertainty Estimation

- **Semantic Entropy:** Farquhar et al., Nature 2024. Clusters model generations by meaning before computing entropy. High semantic entropy = diverse meanings = genuine uncertainty. We work at the token level rather than sequence level.
- **Semantic Entropy Probes:** Kossen et al., 2024. Approximates semantic entropy from single-generation hidden states.

### Prompting and Evaluation

- **Chain-of-Thought:** Wei et al., NeurIPS 2022. CoT prompting elicits reasoning. Kojima et al., NeurIPS 2022 ("Let's think step by step").
- **CoT Unfaithfulness:** Bentham et al., TMLR 2024. CoT reasoning may be post-hoc rationalisation rather than genuine deliberation.
- **Instructor Library Benchmarks (2024):** "Bad Schemas Could Break Your LLM Structured Outputs." Adding a reasoning field to JSON schema boosted accuracy 60% on GSM8k. Field ordering matters — reasoning must come before answer.
- **Prompt Repetition:** arXiv:2512.14982, March 2025. Repeating the prompt improves non-reasoning LLM performance. Noted for future work.
- **lm-eval-harness:** EleutherAI. Standard evaluation framework. MMLU default uses no system prompt.

### Luigi's Broader Vision

- **Rephrase and Respond:** Deng et al., 2024. Model rephrases the question before answering → near-perfect accuracy on some tasks.
- Luigi's discussion document (`paper/discussion_summary_v2.md`) positions the work toward agentic tool-calling: every decision in an LLM agent is token prediction with inspectable logprobs. The MMLU experiment establishes the foundational signal; tool-calling is the high-stakes application.

---

## 6. Technical Decisions and Rationale

### 6.1 Why Qwen3 8B Q4

Target is "realistic small local model deployment." 8B parameters, Q4 quantisation, fits in 8GB VRAM on consumer hardware (RTX 3070). Published MMLU-Redux score of 86.4% (FP16, thinking enabled). Our observed score: ~76% (Q4, no thinking, direct prompting). The ~10 point gap is explained by ~5 points from Q4 quantisation and ~5 points from disabling thinking mode. This is consistent with published quantisation research on Qwen3.

### 6.2 Why MMLU-Redux 2.0 over original MMLU

Original MMLU has ~6.49% ground truth errors (57% in Virology alone). Using verified-correct questions means any uncertainty we detect is genuinely from the model, not from label noise. Also, when building broken-premise pairs (Experiment 2), starting from verified-correct originals means confusion is caused by our modification, not by a dodgy original.

### 6.3 Why We Dropped JSON Schema Enforcement

JSON schema enforcement via Ollama's `format` parameter constrains the model's token distribution before generation. This absorbs uncertainty into the JSON scaffolding tokens. At the answer token position, logprobs are spiked (99.99%) because the model has already been funneled through `{"answer": "`. The unconstrained completion approach (prompt ending with `Answer:`, `num_predict: 1`) preserves the model's genuine uncertainty in the logprobs.

### 6.4 Why We Use /api/generate Instead of /api/chat

The `/api/chat` endpoint applies chat templates which add system/assistant/user framing tokens. For the standard MMLU evaluation protocol (which uses no system prompt and expects direct completion), `/api/generate` with `raw: true` is more appropriate. This also avoids chat template variations across models.

### 6.5 Why Temperature 0.7

Logprobs are deterministic for a given prompt regardless of temperature — temperature affects which token gets *sampled* but the probability distribution is computed from the forward pass. However, temperature does affect the *shape* of the distribution that gets reported in logprobs (higher temperature = softer distribution). We use 0.7 as a reasonable middle ground; the literature hasn't converged on an optimal temperature for logprob-based UQ.

**Note:** This is actually a point worth investigating. Temperature 0.0 (greedy) would give the raw model distribution. Temperature 0.7 softens it. The effect on AUROC should be tested.

### 6.6 Why Parallel Workers for Direct Mode Only

Ollama supports `OLLAMA_NUM_PARALLEL=3` for concurrent request processing. Direct mode queries (`num_predict: 1`) are very fast (~200ms each) with significant HTTP overhead between requests. Running 3 concurrent requests fills the parallel slots and eliminates idle gaps (3× speedup). CoT modes use streaming and generate many more tokens, so the GPU is busier and the parallelism benefit is smaller.

---

## 7. Results So Far (Preliminary)

### 7.1 V1 Results (Sampling Framework — Now Archived)

Full 5,330-question runs completed for 3 conditions. See Section 3.1 for the table. Key finding: paraphrasing improves AUROC from 0.511 to 0.568 and creates a 3× convergence speed ratio between correct and incorrect answers.

### 7.2 V2 Results (Logprob Framework — In Progress)

First completed run: direct mode, no shuffle, no paraphrases (baseline). Even with single-query logprobs and no diversification, the correct/incorrect separation is already visible — correct answers cluster at high max(prob), incorrect answers spread across the range.

Additional runs in progress. Full 12-condition factorial expected to complete within 1-2 days.

---

## 8. Open Questions and Future Work

### For This Paper

- **Temperature effect:** Should we test different temperatures? The logprob distribution shape depends on temperature.
- **Paraphrase quality:** Our paraphrases are conservative (synonym substitution). More aggressive reframing might improve discrimination. Worth testing if time permits.
- **Number of paraphrases:** We use 10. Is there diminishing returns? A sensitivity analysis (2, 5, 10, 20 paraphrases) would strengthen the paper.
- **Other models:** Running the same framework on Gemma 3, Phi-4, Mistral 8B would test generalisability.
- **Prompt sensitivity:** The MMLU-Pro paper found 4-5% accuracy variation across prompt styles. Our "Answer:" completion format is one choice among many.

### For Future Work (Scope Explicitly Deferred)

- **Broken-premise experiment (Experiment 2):** Designed and ready, awaiting Experiment 1 completion.
- **Agentic tool-calling extension:** Luigi's vision for high-stakes application. The MMLU answer token is structurally identical to a tool-name token. If logprob-based UQ works for multiple-choice answers, it should transfer to tool-calling decisions.
- **Pre/post grammar logprob comparison:** llama-server's `post_sampling_probs` toggle could disentangle model uncertainty from format-constraint effects.
- **Multi-model fusion:** Route questions to different models based on uncertainty. Originally in Luigi's draft, deferred to keep the paper focused.
- **Self-paraphrasing:** Instead of external paraphrase generation, have the model rephrase its own questions. Tests whether surface-level rewording creates genuine variation (circularity concern).
- **Prompt repetition:** Recent finding (arXiv:2512.14982) that repeating the prompt improves performance. Potentially complementary to paraphrasing.

---

## 9. Paper Structure (Preliminary Outline)

1. **Introduction:** LLMs provide answers without calibrated confidence. Existing UQ approaches use either logprobs (within-prompt) or sampling (across-prompts) but not both. We show the combination is more powerful.

2. **Related Work:** Self-Consistency, Adaptive-Consistency, SPUQ, "Just Rephrase It!", Semantic Entropy, BayesPE. Position our contribution: first to combine token-level logprobs with paraphrase-based sampling for UQ.

3. **Method:**
   - Paraphrase generation and quality
   - Logprob extraction at the answer token
   - Answer shuffling for position bias control
   - Aggregation across paraphrases (mean probabilities, agreement)
   - Uncertainty metrics: within-prompt entropy, across-prompt agreement, epistemic uncertainty decomposition

4. **Experiments:**
   - Dataset: MMLU-Redux 2.0 (5,330 verified questions)
   - Model: Qwen3 8B Q4 (representative small local model)
   - Factorial design: prompt mode × shuffle × paraphrases
   - Metrics: accuracy, AUROC, calibration, agreement

5. **Results:**
   - Paraphrasing improves discrimination (AUROC) without changing accuracy
   - Shuffling provides additional improvement
   - CoT effects on both accuracy and uncertainty quality
   - Analysis of failure modes: confidently wrong vs genuinely uncertain

6. **Discussion:**
   - Practical implications for LLM deployment
   - The convergence speed signal (from V1) vs logprob signal (V2)
   - Limitations: paraphrase quality, single model, temperature effects
   - Future work: broken-premise detection, agentic tool-calling, multi-model

7. **Conclusion**

---

## 10. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Mar 3 | Scope to single paper: framework + paraphrasing + one novel experiment | Luigi's draft had ~4 papers of scope |
| Mar 3 | Use MMLU as primary benchmark | Standard, comparable, well-understood |
| Mar 4 | Develop broken-premise paired dataset (Eva's idea) | Most distinctive contribution, tests diagnostic uncertainty signatures |
| Mar 4 | Use uv over conda for environment management | Faster, cleaner, less tangled |
| Mar 4 | Target Qwen3 8B Q4 as primary model | Realistic small local deployment scenario |
| Mar 6 | Switch from MMLU to MMLU-Redux 2.0 | Original MMLU has ~6.49% error rate; Redux is corrected and recommended |
| Mar 6 | Generate paraphrases offline via Anthropic API (Sonnet) | Reproducible, consistent, cheap (~$5), better quality than self-paraphrasing |
| Mar 8 | Remove system prompt from all queries | Standard MMLU evaluation uses no system prompt; reduces confound |
| Mar 10 | Add CoT via JSON schema field ordering rather than native think mode | Bounded runtime, generalisable across models, controllable |
| Mar 10 | Three prompt modes: direct, cot, cot_structured | Tests whether structured reasoning improves accuracy and/or uncertainty quality |
| Mar 16 | **PIVOT: Switch from sampling to logprobs (V1 → V2)** | Logprobs give full distribution in one query; 100× speedup; richer data |
| Mar 16 | Drop JSON schema enforcement | JSON scaffolding absorbs uncertainty before answer token; kills logprob signal |
| Mar 16 | Use /api/generate with raw:true instead of /api/chat | Standard completion format, avoids chat template effects |
| Mar 16 | Smart query scheduling: 1 query for no-para+no-shuffle | Identical prompt → identical logprobs, repetition pointless |
| Mar 16 | Fix localhost → 127.0.0.1 | Windows IPv6 DNS resolution bug, 2s→0.19s per request |
| Mar 16 | Store raw logprobs alongside normalised probs | Audit trail; don't hide bugs behind normalisation |
| Mar 17 | Deleted `prompts/compute_signals_csv.md` | Eva decided future analysis changes go directly to the analysis script or brainstorm, not a separate prompt spec file |
| Mar 17 | Added failure mode taxonomy to brainstorm.md (Section 10) | 18 model states across 5 families (A: knowledge failures, B: reasoning failures, C: prompt sensitivity, D: correct modes, E: edge cases), each mapped to predicted metric signatures |
| Mar 17 | **CoT structured prompt rewrite** | Changed from "Briefly evaluate each option..." to structured ✓/✗ format with one-shot example (abstract algebra). Responses dropped from 1100-1600 chars to 302-1016 chars, time from ~17s to ~8.8s per question |
| Mar 17 | Added system message to CoT chat payload | "You are a concise exam grader..." — behavioural anchor to reduce waffling |
| Mar 17 | **Implemented two-pass CoT pipeline** | Pass 1: /api/chat with stop sequence `\nAnswer:` for bounded reasoning. Pass 2: /api/generate with raw=True, num_predict=1 for answer token logprob extraction. Solves both verbosity and logprob spike problems |
| Mar 17 | num_predict for CoT Pass 1 kept at 4000 (safety net) | The stop sequence does the work; num_predict is just a circuit breaker. Context window (2048) is the real cap |
| Mar 17 | COT_CONTEXT_SIZE reverted to 2048 | Briefly raised to 4608 by Claude Code, reverted because context window itself acts as circuit breaker (~1900 tokens of generation headroom with ~150 token prompts) |
| Mar 17 | Pass 2 fallback logic: check for answer letters anywhere in top_logprobs | If Pass 2 top token is LaTeX/formatting, answer letters may still be in top-20. Falls back to Pass 1 stream logprobs if no answer letters found at all |
| Mar 17 | **CoT accuracy drop confirmed: 68.6% vs 75.5% direct** | Reasoning-induced error — model overrides correct associative knowledge with faulty step-by-step evaluation. Known phenomenon in small models (CCoT, Renze & Guven 2024). Reported as finding, not fixed. |
| Mar 17 | **Decision: add thinking-mode condition** | Three-way comparison needed: direct (no reasoning) vs CoT structured (forced external reasoning) vs thinking mode (native trained reasoning). Tests whether reasoning-induced errors are specific to prompt-based CoT or inherent to reasoning. New config: `exp1_full_think_noshuffle_nopara.yaml` |
| Mar 17 | Clarified two-pass attribution | Prompt engineering stopped waffling (1500→300 chars). Two-pass recovered real logprob distributions ([1,0,0,0]→real spread). Independent contributions, initially conflated. |

---

## 11. Useful Quotes and Framings

**From Luigi's discussion document:**
> "Self-Consistency tried weighting by sequence-level log probability and it didn't help. They concluded model probabilities are poorly calibrated. But 'sequence-level log probability doesn't help' and 'token-level logprobs don't help' are very different claims. Nobody tested the second one."

> "'Just Rephrase It!' is framed as a workaround for closed-source models where logprobs are unavailable. The implicit assumption is: if you had logprobs, you wouldn't need rephrasing. This assumption is wrong."

> "What you actually want is both signals together: logprobs to detect *when* the model is uncertain, and rephrasing to test *whether* its confidence is robust or fragile."

**From Eva's manual experiments (Day 1):**
The Monocercomonoides question: Qwen3 8B was confident about B (Energy production) across 5 identical queries. But when paraphrased as "Monocercomonoides, the first known eukaryote to completely lack certain organelles," it shifted to A (Protein synthesis). The model *almost* caught the broken premise in its thinking traces — mentioning mitosomes, iron-sulfur clusters, saying "none of the options would be correct" — but forced-choice format suppressed the uncertainty.

This single demonstration encapsulates the paper's thesis: single-prompt confidence is not reliable; the inconsistency across phrasings is the signal.

---

## 12. CoT Experiments — The Verbosity Problem and Two-Pass Solution (March 17)

### 12.1 The Problem: CoT Models Waffle

First CoT pilot run (cot_structured, 4 questions) revealed two problems:

**Problem 1 — Verbosity:** With the original prompt ("Briefly evaluate each option, then state your final answer as a single letter"), responses were 1100-1637 chars (~300-400 tokens of reasoning). At ~17s per question, a full noshuffle_nopara run would take ~25 hours, and the 11-query shuffle/para conditions would be ~6 days each.

**Problem 2 — Logprob spike:** Every question showed `[1 missing letters]` — only the chosen answer letter appeared in top_logprobs, the other 3 were absent. Canonical probs were `[1.0, 0.0, 0.0, 0.0]` on every query. By the time the model writes "the answer is B" at the end of its reasoning, the logprob distribution at the answer token is completely spiked. This is the **scaffolding absorption effect** (see brainstorm.md Section 8) — the reasoning chain commits the model before the answer token, exactly as JSON schema enforcement did in V1.

**Implication for the framework:** If CoT logprobs are always [1,0,0,0], then Tier I within-prompt signals (MSP, entropy, second-choice gap) become uninformative for CoT conditions. Tier II across-prompt signals (agreement, epistemic uncertainty, rank stability) still work because they measure whether different paraphrases lead to different answers, not how confident each individual answer is.

### 12.2 Prompt Engineering Iterations

**Attempt 1 — Softer instructions:** Changed prompt to "For each option, a few words on why or why not." No improvement — 18.8s/q, similar response lengths. Qwen 8B Q4 ignores adjective-based brevity instructions.

**Attempt 2 — Structured ✓/✗ format with one-shot example:** Rewrote prompt to show exact format expected, with a concrete maths example demonstrating 3-5 words per option:
```
Which option is correct? Use this exact format:
A) ✓/✗ reason
B) ✓/✗ reason
...
Answer: X

Example:
Q: Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
  A) 0  B) 2  C) 1  D) 3
A) ✗ x²+0 = x² is reducible
B) ✓ x²+2 has no roots in Z_3
C) ✗ x²+1 = (x+1)(x+2) in Z_3
D) ✗ 3 ∉ Z_3
Answer: B
```

Result: 8.8s/q, responses 302-1016 chars. 2× faster than old prompt. The format works ~35% of the time (short ✓/✗ responses), but ~65% of questions the model still waffles with full paragraphs. Abstract algebra and maths-heavy questions are worst-case. Research confirms this pattern: output length mirrors example length more reliably than any instruction, but Q4 quantisation specifically degrades instruction-following compliance (the Ionio.ai IFEval benchmark finding).

Also added a system message to the chat payload: "You are a concise exam grader. For each MCQ, state ✓ or ✗ with a few words per option, then Answer: X. Never write derivations, proofs, or step-by-step working."

### 12.3 The Two-Pass Pipeline

**Architecture:**
- **Pass 1:** Send question via `/api/chat` with the structured prompt and `stop: ["\nAnswer:"]`. The model reasons until it writes "\nAnswer:", then Ollama halts generation. We keep the reasoning text but don't need logprobs from this pass.
- **Pass 2:** Send question + choices + reasoning + "Answer:" via `/api/generate` with `raw: true, num_predict: 1, logprobs: true, top_logprobs: 20`. Single-token completion extracts the answer with full logprob distribution, identical mechanism to direct mode.

**Why two passes:** Ollama's stop sequences consume the triggering text and strip it from the output — no logprobs are generated for the stop tokens. So you can't use a stop sequence and extract answer logprobs in the same call. The two-pass approach separates reasoning generation (with stop sequence for brevity) from answer extraction (with logprobs for uncertainty).

**Fallback:** If the stop sequence doesn't fire (model finishes naturally or hits token limit), fall back to single-pass extraction from the streamed logprobs (find last A/B/C/D token, same as before). If Pass 2's top token has no answer letters in top_logprobs, fall back to Pass 1 stream logprobs.

### 12.4 Two-Pass Pilot Results

| Metric | Single-pass (old prompt) | Single-pass (✓/✗ prompt) | Two-pass |
|--------|-------------------------|--------------------------|----------|
| Time/question | ~17s | ~8.8s | ~10s |
| Response chars | 1118-1637 | 302-1016 | 192-311 |
| Accuracy (4q) | 3/4 | 2/4 | 3/4 |
| Missing letters | 3 of 4 missing | 3 of 4 missing | **0 missing** |
| Logprob spread | [1.0, 0.0, 0.0, 0.0] | [1.0, 0.0, 0.0, 0.0] | Real distributions |

**Key finding:** Pass 2 logprobs have all 4 letters present. Example: `A: -0.00 B: -18.37 C: -18.06 D: -17.53` instead of `A: -0.00 B: --- C: --- D: ---`. The probabilities are still heavily spiked (the model is very confident), but having all 4 letters visible is a qualitative improvement over single-pass CoT where alternatives were completely absent from top-20.

**Why this works:** In single-pass CoT, the answer token is generated in the assistant's chat-template context immediately after the model's own reasoning. The model is maximally committed. In Pass 2, the reasoning text is fed back as raw text input (no chat template), and the model completes "Answer:" as a text-continuation task. The raw completion mode loosens the commitment just enough for all 4 answer letters to appear in top_logprobs.

**Unexpected finding:** Q3 showed a reasoning-logprob disagreement. The reasoning text concluded "Answer: A" but Pass 2 logprobs picked D (0.863 probability). This happens because the code strips the "Answer: A" line from the reasoning before sending to Pass 2, so Pass 2 sees the per-option evaluations but NOT the reasoning's stated conclusion, and independently reaches a different answer. This is the Wang et al. (2024) "My Answer is C" phenomenon — text reasoning and logprob distributions can disagree.

### 12.5 Full Run Observations (Early)

Started full cot_structured noshuffle_nopara run (5,330 questions). First 10 questions revealed:

- Stop sequence fires on all questions — two-pass working consistently
- Logprob distributions are meaningfully spread (e.g., `[0.003, 0.112, 0.226, 0.660]` on uncertain questions)
- Some questions silently skipped (questions 4, 6, 7, 8 missing from output) — extraction failures being investigated. Likely cause: Pass 2 top token is LaTeX or formatting token with no answer letters in top-20, AND Pass 1 stream logprobs also fail extraction
- Early accuracy looks lower than direct mode on abstract algebra (these are hard questions for an 8B model)

### 12.6 Research on CoT Verbosity in Small Local Models

Conducted research on taming verbose reasoning (saved as separate document). Key findings relevant to our experiments:

- **Qwen3 thinking mode is the dominant verbosity driver** — inflates output 3-5× vs Qwen2.5. We already have think=false in all configs, but the Qwen3 architecture is inherently chattier even in non-thinking mode.
- **Q4 quantisation degrades instruction following** — the Ionio.ai IFEval benchmark found Q4 causes "unacceptable losses for production-level deployments" of instruction compliance. Q5_K_M retains ~95-97% of instruction-following quality. This may explain why the model ignores our structured format on ~65% of questions. Worth noting in paper limitations.
- **Output length mirrors example length** — academic research on Concise CoT (Renze & Guven, 2024) found that few-shot example length controls output length more reliably than any instruction. Our structured example is already short, but the model overrides it on harder questions.
- **Token-budget prompts partially work** — the TALE framework (Han et al., 2024) found "solve in ~50 tokens" compressed CoT, but budgets that are too small cause the model to IGNORE the constraint and produce MORE tokens.
- **Anti-verbosity sampling parameters** — `repeat_penalty: 1.2` and `presence_penalty: 0.5` can reduce waffling. Not currently used in our experiments (would change the experimental condition).

### 12.7 Implications for the Paper

The two-pass pipeline is not just engineering convenience — it has scientific implications:

1. **Three levels of logprob entropy:** Direct mode (raw logprobs, most spread) → Two-pass CoT (reasoning as context, moderate spread) → Single-pass CoT (inline reasoning, spiked). This is the scaffolding absorption spectrum from brainstorm Section 8, now with empirical evidence.

2. **Reasoning-logprob disagreement rate** as a novel uncertainty signal: How often does the model's text reasoning conclude one answer while its raw logprob distribution favours another? High disagreement = model is uncertain about its own reasoning.

3. **Format mismatch as a feature:** Pass 2 uses raw text completion (no chat template) while the reasoning was generated in chat mode. This format mismatch could produce systematically different logprob distributions, which is itself interesting — it reveals how much the chat template contributes to the scaffolding absorption effect.

---

## 13. Failure Mode Taxonomy (March 17)

Added a comprehensive taxonomy to brainstorm.md (Section 10) covering every distinct state a model can be in when answering a question. 18 modes across 5 families:

**Family A — Knowledge Failures (4 modes):** Confident Ignorance, Systematic Parametric Error, Partial Knowledge, Knowledge Gap.

**Family B — Reasoning Failures (3 modes):** CoT Derailment, Distractor Seduction, Calculation Error.

**Family C — Prompt Sensitivity Failures (3 modes):** Framing Sensitivity, Position Bias Override, Off-Label Escape.

**Family D — Correct Answer Modes (7 modes):** Textbook Knowledge, Solid Knowledge, Correct via Elimination, Right but Fragile, Lucky Guess, Right via Position, CoT Rescue.

**Family E — Edge Cases (3 modes):** Genuinely Ambiguous Question, Right Answer Wrong Reason, Broken-Premise Response.

Each mode mapped to a diagnostic table of 14 metric columns showing predicted H/M/L signatures. Key insights:

- **A1 (Confident Ignorance) ≡ D1 (Textbook Knowledge)** — indistinguishable from metrics alone. Fundamental limitation of any single-model UQ method.
- **AGap (agreement-confidence gap) is the unique diagnostic for A2** — validates the NMR-inspired signal.
- **Position Loyalty is the unique diagnostic for C2** — validates shuffling as diagnostic instrument.
- **Answer Coverage is the unique diagnostic for C3** — validates our novel local-model signals.
- **Mirror pairs** (B1↔D7, C2↔D6, D4↔C1, D5↔A4) reveal what each experimental condition buys you.
- **The taxonomy validates the experimental design** — each condition is required to detect specific failure modes.

See brainstorm.md Section 10 for the full diagnostic tables and detection strategies.

---

## 14. CoT Accuracy Drop and the Case for Three Reasoning Levels (March 17, evening)

### 14.1 The Reasoning-Induced Error Discovery

Early results from the cot_structured full run revealed a significant accuracy drop: **68.6% (CoT structured) vs 75.5% (direct mode)** — a 7-point penalty for adding reasoning. This wasn't just noise. Individual questions showed a clear pattern: the model would get the answer right in direct mode (where raw associative knowledge comes through in the logprobs) but then reason itself into the wrong answer in CoT mode.

**Example — Business ethics "race to the bottom" question:** Direct mode correctly associates "race to the bottom" + "globalisation" + "social/environmental" and picks the right answer with 0.991 confidence. CoT structured mode forces the model to evaluate each option step-by-step. When it reaches the correct answer, it evaluates "Bottom is incorrect term" and rejects it, overriding its own correct intuition with faulty sequential reasoning. It picks D instead, with 100% confidence.

This is a textbook case of **unfaithful reasoning / reasoning-induced error** from the CoT literature (Bentham et al., TMLR 2024). The model knows the answer associatively but can't articulate why, and the forced step-by-step format corrupts the signal.

### 14.2 Why This Happens (Research Context)

Several factors compound:

- **Small models + CoT on knowledge questions = known accuracy drop.** The CCoT paper (Renze & Guven 2024) found this specifically: on smaller models, forcing reasoning on questions that don't require reasoning hurts accuracy. MMLU is mostly knowledge recall, not multi-step derivation.
- **Concise prompts may make it worse.** Our ✓/✗ format forces binary judgements with minimal deliberation. A longer reasoning chain might self-correct ("Wait, 'race to the bottom' is actually a well-known phrase..."). The concise format doesn't allow for that recovery. There's an inherent tension: we compressed reasoning to save time, but compression removes the self-correction that makes CoT valuable.
- **Q4 quantisation compounds it.** Qwen3 8B loses ~5 MMLU points at Q4 vs FP16. Official non-thinking MMLU-Redux: 79.5% (FP16). Our direct: 75.5% (Q4). CoT structured: 68.6%. Both penalties stack.

### 14.3 Implication: Three Reasoning Levels, Not Two

This forces a natural three-way comparison that the paper should make:

| Level | Mode | Reasoning type | Expected behaviour |
|-------|------|---------------|-------------------|
| 1 | **Direct** | None — raw associative knowledge | Highest accuracy on knowledge-recall. Clean logprobs. No reasoning to corrupt the signal. |
| 2 | **CoT structured** | Forced external reasoning via prompt | Lower accuracy on knowledge-recall. Model reasons itself out of correct intuitions. But provides different uncertainty signal. |
| 3 | **Thinking mode** | Model's native internal reasoning via `<think>` tokens | Unknown — key question. Qwen3 was *trained* for this mode. The thinking tokens are architectural, not a prompt hack. May avoid reasoning-induced errors because the model learned to self-correct in think chains. |

**Why this comparison matters for the paper:**

The direct-vs-CoT accuracy gap alone is a finding, but adding thinking mode turns it into a principled study of how reasoning modality affects both accuracy and uncertainty calibration. Three concrete questions:

1. **Does native thinking avoid reasoning-induced errors?** If thinking mode gets 75%+ accuracy (matching direct), it means the problem isn't reasoning per se — it's *forced external reasoning via prompting* that corrupts the signal. The model's trained reasoning pathway works; the prompt-based one doesn't.

2. **Does thinking mode produce different uncertainty signatures?** Thinking mode's logprobs at the answer token will be conditioned on potentially thousands of hidden think tokens. How does that compare to direct (no context) and CoT (visible context)?

3. **Which combination gives the best AUROC?** Maybe direct mode is best for accuracy, but thinking mode is best for uncertainty calibration, and CoT structured is best for cross-condition disagreement signals. The framework's value is combining all three.

**Practical note:** Thinking mode for the noshuffle_nopara condition is just 1 query per question. Even at ~25s/q (500+ think tokens before the answer), that's ~37 hours — long but feasible. The pipeline already supports `think: true` in configs. Need to create a new YAML config: `exp1_full_think_noshuffle_nopara.yaml`.

### 14.4 Correctly Attributing the Two-Pass Wins

Important clarification from discussion: the two-pass pipeline and the prompt engineering solve *different problems*, and we were initially conflating them.

**What stopped the waffling:** The prompt changes — system message ("never write derivations"), structured ✓/✗ format, one-shot maths example. These brought responses from 1500 chars to 300 chars.

**What recovered real logprob distributions:** The two-pass architecture. Pass 1's stop sequence cuts off before the answer letter; Pass 2 extracts the answer via fresh single-token completion, giving distributions like [0.003, 0.112, 0.226, 0.660] instead of [1.0, 0.0, 0.0, 0.0].

**These are independent contributions.** You could have the concise prompt with single-pass (short waffle, spiked logprobs) or verbose prompt with two-pass (long waffle, real logprobs). We happen to have both, which gives us short waffle AND real logprobs.

### 14.5 Paper Narrative

This isn't a bug to fix — it's a finding to report. The story:

*"Direct mode outperforms CoT on knowledge-recall MCQs for small quantized models. CoT introduces reasoning-induced errors where the model overrides correct intuitions. Despite lower accuracy, CoT provides a different uncertainty signal — and the disagreement between direct and CoT answers is itself a powerful uncertainty indicator. The framework's value is combining multiple prompting strategies, each revealing different aspects of the model's knowledge."*

The accuracy drop actually *strengthens* the case for the multi-condition framework. If CoT always agreed with direct mode, there'd be no point running both.

---

## 15. CoT Prompt Refinement and Answer Leakage Fix (March 18)

### 15.1 The Premature Commitment Problem in cot_structured

The ✓/✗ format in `cot_structured` forces sequential per-option commitment: the model evaluates A (✗), then B, then C, then D. By the time it marks A as ✗, it's committed before seeing the full picture. This isn't structured reasoning — it's forced premature judgment. On questions where cross-option comparison matters, this produces worse answers than no reasoning at all.

### 15.2 The `cot` Prompt Evolution

The `cot` prompt went through several iterations to find the right balance between genuine reasoning and brevity:

1. **"Consider each option and how they relate to each other"** — too verbose. Median 1788 chars, 41.7s/q. Model writes textbook-length answers.
2. **"Concisely, a couple of sentences of reasoning"** — better. Median 353 chars, 36.3s/q. But still inconsistent.
3. **"BE CONCISE. 3-4 bullet points of reasoning only"** — best. ~250 chars, genuine topic-level observations without option-by-option commitment. Real logprob distributions emerged (Q13: B=0.572, C=0.428).

**Critical addition:** "do NOT name the answer letter in your reasoning." This eliminated answer leakage where the model would write "Conclusion: B is correct" in its reasoning, which pre-committed Pass 2. Before this fix, 2/20 questions had letter leakage. After: 0/20.

**Final `cot` prompt:**
```
BE CONCISE. 3-4 bullet points of reasoning only — do NOT name the answer letter in your reasoning.

End with: Answer: X
```

No system message, no one-shot example, no format template. The model reasons freely in bullet points about the topic, then Pass 2 extracts logprobs from a fresh "Answer:" completion.

### 15.3 Decision: Drop cot_structured

`cot_structured` is fundamentally flawed by the premature commitment design. It performs worse than direct mode (68.6% vs 75.5%) and doesn't test anything interesting that `cot` doesn't test better. Dropped from the paper. The three-way comparison is now: direct / cot / think mode.

---

## 16. Endpoint Switching Bug and Fix (March 18)

### 16.1 The 3.4× Slowdown

Profiling revealed that every two-pass CoT query triggered a ~3s Ollama model reload because Pass 1 used `/api/chat` and Pass 2 used `/api/generate`. Switching between endpoints forces Ollama to tear down and rebuild the model context. With 22 API calls per question (11 queries × 2 passes), this added ~6s of pure reload overhead per question.

### 16.2 The Fix

Changed `_complete_answer_token` (Pass 2) to use `/api/chat` with assistant prefill instead of `/api/generate`. Both passes now use the same endpoint, eliminating reloads.

**Pass 2 payload (new):** Send the question as a user message, then the model's reasoning + "Answer:" as an assistant prefill message, with `num_predict: 1`. The model completes the assistant turn with a single token.

**Result:** 9.3s/q down from 31.7s. 3.4× speedup from a one-line change. `_get_top_logprobs` already handled both response formats, so extraction worked immediately.

---

## 17. Full CoT Run Analysis at 1,600 Questions (March 17 evening)

### 17.1 Key Numbers

| Metric | CoT structured | Direct (baseline) |
|--------|---------------|-------------------|
| Accuracy | 68.6% | 75.5% |
| Mean confidence | 0.974 | 0.907 |
| Questions > 0.99 confidence | 86.4% | 56.2% |
| Extraction failures | 0 | 0 |

### 17.2 The Overconfidence Finding

CoT is simultaneously less accurate AND more confident than direct mode. The confidence-accuracy relationship is bimodal: questions above 0.99 confidence are 73.8% accurate, but questions at 0.95-0.99 are only 41.3% accurate. There's a cliff — the model either knows it or doesn't, with a dangerous middle zone of moderate confidence and terrible accuracy.

### 17.3 Subject-Level Patterns

CoT helps on reasoning/logic tasks: elementary maths (+7pp), formal logic (+6pp), astronomy (+2pp).
CoT hurts on knowledge recall: college maths (-11pp), business ethics (-8pp), clinical knowledge (-6pp).

This confirms the CCoT research — forcing small models to reason on knowledge-recall questions introduces errors rather than helping.

---

## 18. Think Mode Implementation (March 18-19)

### 18.1 Architecture: Direct + Think

Think mode uses `prompt_mode: "direct"` with `think: true`. The model reasons internally via Qwen3's trained `<think>` blocks, then outputs a short visible answer. This required routing through `/api/chat` (instead of `/api/generate` which doesn't support think), with `num_predict: 4000` (think tokens count against the budget) and `num_ctx: 12288` (THINK_CONTEXT_SIZE — think chains can be very long).

**Prompt:** Simple and direct — no system message, no format template:
```
{question}
{choices}
Answer with a single letter.
End with: Answer: X
```

### 18.2 Dual Logprob Extraction

Think mode captures TWO logprob distributions per question:

1. **committed_canonical_probs** (Pass 1, single-pass): Extracted from the answer token in the streamed output after the model has written "Answer: B". Post-commitment — the model has already stated its answer.
2. **canonical_probs** (Pass 2, pre-commitment): From a fresh completion with the answer stripped. The model sees its own thinking + visible reasoning but NOT the answer letter, then Pass 2 extracts logprobs from "Answer:" completion.

The difference between these two distributions quantifies how much stating the answer collapses uncertainty — the same scaffolding absorption effect, measured within a single query.

**New QueryResult fields:** `committed_canonical_probs`, `committed_display_letter_logprobs`, `committed_canonical_logprobs`, `committed_display_answer`, `committed_canonical_answer` — all optional, populated only for think mode.

### 18.3 Extraction Mode Fix

`direct + think` uses `/api/chat` and the model may output multiple visible tokens ("The answer is B" rather than just "B"). The extraction mode needed to use CoT-style last-answer-token search, not direct-mode first-token extraction. Fixed by computing `extraction_mode = "cot"` when `config.think and config.prompt_mode == "direct"`.

---

## 19. Cloud GPU Deployment (March 19)

### 19.1 Setup

Rented an RTX 5090 (32GB VRAM, 1447 GB/s bandwidth) on Vast.ai at $0.26/hr to run the multi-query CoT conditions (11 queries per question = ~44 hours each on laptop). Instance runs Ollama, code pulled from GitHub.

### 19.2 The KV Cache Reload Bug

First cloud run showed alternating KV cache sizes between Pass 1 (2048) and Pass 2 (default 32768 on the 5090). Every request triggered a model reload — Ollama tears down and rebuilds the runner when context size changes. Fixed by explicitly setting `num_ctx` in `_complete_answer_token` to match Pass 1's context size.

### 19.3 Parallelisation Experiments (The HTTP Overhead Story)

Extensive testing to maximise GPU utilisation (which sat at 6-30% across all configurations):

| Setup | Total questions/min | GPU% |
|-------|-------------------|------|
| 1 experiment, sequential, NUM_PARALLEL=1 | 2.2 | 17% |
| 3 experiments, sequential, NUM_PARALLEL=3 | 3.0 | 6-12% |
| 1 experiment, 8 parallel workers, NUM_PARALLEL=8 | 2.0 | 10% |
| 3 experiments, 8 workers each, NUM_PARALLEL=8 | 1.8 | 6-17% |

**The bottleneck is HTTP round-trip latency, not GPU.** Each two-pass query does 0.94s of GPU work inside ~6s of wall time. The remaining 5s is Python → HTTP → Ollama → GPU → Ollama → HTTP → Python overhead. More parallelism creates contention at the Ollama scheduling layer without meaningful GPU speedup. The 5090's 1447 GB/s bandwidth is irrelevant — the GPU is idle 75%+ of the time waiting for requests.

**Winner:** 3 experiments running simultaneously with sequential queries and NUM_PARALLEL=3. Not optimal per-experiment, but best total throughput.

### 19.4 Think Mode on the Cloud — Dual Ollama Instances

Think mode queries take ~120s each (500+ hidden think tokens) and timeout when competing with CoT experiments for Ollama slots. Solution: run a second Ollama instance on port 11435, dedicated to think mode.

**Code change:** Made `DEFAULT_OLLAMA_URL` read from `OLLAMA_URL` environment variable on non-Windows systems:
```python
DEFAULT_OLLAMA_URL = _os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") if _os.name != "nt" else "http://127.0.0.1:11434"
```
Windows keeps the hardcoded URL (to avoid the IPv6 DNS resolution bug). Linux reads from env var, defaulting to the same URL.

Think experiment split into 3 chunks (~1776 questions each) running in parallel on Ollama 2 (NUM_PARALLEL=3).

### 19.5 Final Cloud Configuration

- **Ollama 1** (port 11434, NUM_PARALLEL=3): 3 CoT experiments — cot_shuffle_para, cot_noshuffle_para, cot_shuffle_nopara
- **Ollama 2** (port 11435, NUM_PARALLEL=3): 3 think chunks — think_chunk1, think_chunk2, think_chunk3
- **Auto-pull:** Python script on laptop downloads results via Jupyter file server (port 8384 → external 40066) every 5 minutes
- **GPU utilisation:** ~25-30%, VRAM 14GB/32GB
- **Estimated cost:** ~$23-30 for all experiments

### 19.6 Experiments Running on Laptop Concurrently

`exp1_full_cot_noshuffle_nopara` completed on laptop (~14 hours, 11.6s/q). This is the single-query CoT baseline.

---

## 20. Updated Decision Log (March 18-19)

| Date | Decision | Rationale |
|------|----------|-----------|
| Mar 18 | **Refined `cot` prompt: bullet points + no answer leakage** | "BE CONCISE. 3-4 bullet points. do NOT name the answer letter." Eliminated 100% of answer leakage, produced genuine pre-commitment logprob distributions |
| Mar 18 | No system message for `cot` mode | System messages constrain reasoning style; `cot` should be free-form deliberation |
| Mar 18 | **Fixed endpoint switching bug: Pass 2 now uses /api/chat** | Switching between /api/chat and /api/generate caused ~3s model reload per pass. Using /api/chat for both passes eliminated reloads. 3.4× speedup (31.7s → 9.3s/q). |
| Mar 18 | **Dropped cot_structured from paper** | Premature commitment design (✓/✗ per option) forces the model to judge each option before reasoning through all of them. Produces worse accuracy than no reasoning. Not a useful experimental condition. |
| Mar 18 | Dual logprob extraction for think mode | Captures committed (post-answer) and pre-commitment (answer-stripped) distributions from the same query. Quantifies scaffolding absorption within a single query. |
| Mar 18 | Think mode extraction uses "cot" mode internally | Direct+think routes through /api/chat, may produce multi-token visible answers. Needs last-answer-token search, not first-token extraction. |
| Mar 19 | Rented RTX 5090 on Vast.ai for multi-query CoT conditions | 11 queries/question × 5330 questions = too slow on laptop. Cloud GPU 4× faster per query. |
| Mar 19 | **Fixed KV cache reload bug on cloud** | Pass 2 was missing `num_ctx`, defaulting to 32768 on 5090. Constant 2048↔32768 context switching caused reloads on every request. |
| Mar 19 | HTTP overhead is the real bottleneck, not GPU | 0.94s GPU work inside 6s wall time per request. Parallelisation doesn't help — creates scheduling contention. |
| Mar 19 | Run 3 experiments simultaneously, sequential queries, NUM_PARALLEL=3 | Best measured total throughput (3.0 q/min) despite GPU underutilisation |
| Mar 19 | Dual Ollama instances on cloud | Think mode (120s queries) timeouts when competing with CoT for slots. Second Ollama on port 11435 with env var `OLLAMA_URL`. |
| Mar 19 | Think experiment split into 3 chunks | ~1776 questions each, running in parallel on Ollama 2. Reduces think wall-clock from ~178h to ~60h. |
| Mar 19 | `OLLAMA_URL` env var (Linux only) | Windows keeps hardcoded `127.0.0.1:11434` to avoid IPv6 bug. Linux reads env var for flexible multi-instance deployment. |
| Mar 19 | **Parallelising think chunks doesn't help** | Same lesson as CoT: more concurrent streams on the same GPU doesn't increase total throughput because the bottleneck is HTTP round-trip latency, not GPU compute. Reverted to single think experiment on Ollama 2. |
| Mar 19 | **Decision: rewrite inference layer for future experiments** | Ollama's HTTP overhead wastes 75%+ of GPU time. For Experiment 2 (broken-premise), other models, and temperature sensitivity, switch to llama-server (llama.cpp's built-in server with continuous batching) or vLLM. Same GGUF model files, eliminates HTTP bottleneck. Deferred until current experiments complete. |

---

## 21. Architecture Bottleneck and Future Rewrite (March 19)

### 21.1 The Fundamental Problem

Ollama's architecture imposes unavoidable HTTP overhead: each query requires Python → HTTP → Ollama scheduler → GPU → Ollama scheduler → HTTP → Python. Measured at 0.94s GPU work inside 6s wall time — the GPU is idle 84% of the time. No amount of parallelisation fixes this because the bottleneck is round-trip latency, not GPU compute. Confirmed across every configuration tested:

- More Ollama parallel slots → scheduling contention, no speedup
- Multiple Ollama instances → same GPU, same bottleneck
- Parallel workers in Python → requests queue at Ollama, net slowdown
- Think mode chunks → slower than sequential due to contention

The RTX 5090 (1447 GB/s, 109 TFLOPS) is being used at 6-30% — generating one token at a time with gaps between requests.

### 21.2 The Fix (For Future Experiments)

Switch from Ollama to **llama-server** (llama.cpp's built-in HTTP server) or **vLLM**. Both support continuous batching — processing tokens from multiple concurrent requests in a single GPU forward pass. This would saturate the GPU and eliminate the scheduling overhead.

**llama-server** is the path of least resistance: same GGUF model files (already downloaded), OpenAI-compatible API with logprob support, minimal code changes (swap the API URL and adjust response parsing).

**Deferred** until current Experiment 1 results are complete. The current runs are producing valid data — just slowly. Analyse first, rewrite second.

### 21.3 Current Experiment Status (March 19 evening)

**Completed:**
- direct_noshuffle_nopara ✓ (laptop)
- direct_noshuffle_para ✓ (laptop)
- direct_shuffle_nopara ✓ (laptop)
- direct_shuffle_para ✓ (laptop)
- cot_noshuffle_nopara ✓ (laptop, ~14hrs)
- cot_structured_noshuffle_nopara ✓ (laptop — dropped from paper but data exists)

**Running on cloud (Vast.ai RTX 5090, ~$0.26/hr):**
- cot_shuffle_para — Ollama 1 (port 11434), ~30s/q with 11 queries, ETA ~44hrs
- cot_noshuffle_para — Ollama 1, same timing
- cot_shuffle_nopara — Ollama 1, same timing
- direct_noshuffle_nopara_think — Ollama 2 (port 11435), ~60-120s/q, ETA ~100hrs

**Not yet started:**
- Think mode with shuffle/para conditions (deferred — too slow on current architecture)

**Dropped:**
- All cot_structured conditions except the completed noshuffle_nopara (premature commitment flaw)
