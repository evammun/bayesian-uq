# Paper Brainstorm — Running Ideas File

**Last updated:** 2026-03-17 (updated: CoT two-pass pipeline, failure mode taxonomy, prompt engineering for small models)
**Purpose:** Evolving collection of big-picture ideas, structural thinking, and strategic decisions. Come back to this with fresh eyes after experiment progress. Separate from `paper_running_notes.md` (decision log) and `uncertainty_signals_spec.md` (implementation spec).

---

## 1. The Paper We're Writing

### Working Titles (pick later)
- "Anatomy of Uncertainty: A Systematic Comparison of Confidence Signals in Small Local LLMs"
- "Beyond Single-Prompt Confidence: Dissecting Uncertainty Signals in Quantized Language Models"
- "The Uncertainty Playbook: What Local LLMs Tell You About What They Don't Know"
- "How Uncertain Is Your Local LLM? A Practitioner's Guide to Confidence Signals"

### Core Thesis
Existing UQ methods use either logprobs (within-prompt confidence) or paraphrasing (across-prompt consistency) but treat them as alternatives. We show they measure independent dimensions of uncertainty, and the combination is more powerful than either alone. We also discover novel signals — off-label probability mass, per-question position sensitivity — that are uniquely accessible from local models. All experiments run on consumer hardware with a quantized 8B model.

### What Makes This Different From "Just Another UQ Paper"
1. **Systematic comparison, not single-method advocacy.** Most papers propose one metric, test it, and claim victory. We test ~25 signals on the same data and show which are redundant and which are complementary.
2. **Small local models, not frontier APIs.** Our setting is a quantized 8B model on a laptop GPU. This is where UQ matters most (weaker models need better uncertainty estimation) and where the literature is thinnest.
3. **Novel signals from local-model access.** Off-label probability mass (what else the model wanted to say besides A/B/C/D) and per-question position sensitivity (is the model tracking content or position?) are only accessible when you have full top-N logprobs from a local model.
4. **The 2D uncertainty space.** Confidence × consistency as independent axes. The "fragile confidence" quadrant (high confidence, low consistency) is the most dangerous failure mode and is invisible to any single metric.

### Target Venue
**EMNLP 2026** (primary target, submission ~June 2026)
- Where the UQ/evaluation papers live: Adaptive-Consistency (EMNLP 2023), CISC (ACL 2025), BayesPE (ACL 2024), SPUQ (EACL 2024)
- Strong empirical-methods track, welcomes systematic analyses
- 8 pages + references + appendix

**Fallback options:**
- ACL Findings (slightly lower bar, still strong venue)
- COLM 2026 (language-model-focused conference, might suit the practitioner angle)
- NeurIPS Datasets & Benchmarks track (if we lean into the evaluation/methodology framing)

---

## 2. Paper Structure (Draft Outline)

### 1. Introduction
- LLMs give answers without calibrated confidence
- Existing approaches: logprobs OR paraphrasing, never both
- We present a systematic anatomy of uncertainty signals from local LLMs
- Key contributions: (a) 2D uncertainty space showing confidence and consistency are independent, (b) novel signals from full top-N logprobs (answer coverage, hesitation profile), (c) position sensitivity as per-question diagnostic, (d) all on consumer hardware with quantized model

### 2. Related Work
Three streams:
- Logprob-based confidence: Kadavath 2022, Plaut 2024, CISC (Taubenfeld 2025)
- Paraphrase-based UQ: SPUQ (Gao 2024), Just Rephrase It (Yang 2024), Mapping from Meaning (Cox 2025)
- Closest neighbours: CAPE (Jiang 2023), BayesPE (Tonolini 2024) — vary instruction templates, not question stems
- Position bias: Zheng et al. ICLR 2024 (population-level; we use per-question)
- Constrained decoding: Park et al. NeurIPS 2024 (theoretical support for our JSON finding)

### 3. Method: The Signal Taxonomy
- Framework: paraphrase generation, logprob extraction, answer shuffling, aggregation
- Within-prompt signals (S1-S7): MSP, entropy, gap, shape, **answer coverage**, **hesitation profile**
- Across-prompt signals (S8-S18): agreement, mean confidence, epistemic/aleatoric, **confidence variance**, **rank stability**, **simplex spread**
- Position signals (S19-S22): **position loyalty**, **content loyalty**, **factorial decomposition**
- The 2D uncertainty space: confidence × consistency, four quadrants
- Formal definitions for each signal (readers can implement from these)

### 4. Experimental Setup
- Datasets: MMLU-Redux 2.0 (+ others, see Section 3 of this doc)
- Model: Qwen3 8B Q4_K_M via Ollama (+ ideally a second model)
- Factorial design: 3 prompt modes × 2 shuffle × 2 para = 12 conditions
- Why this design: each factor isolates a specific source of variation

### 5. Results
**5.1 The Signal Landscape**
- AUROC table for all signals
- Correlation matrix / PCA — how many independent uncertainty dimensions?
- Key finding: novel signals (answer coverage, position loyalty) add information beyond standard metrics

**5.2 The 2D Uncertainty Space** (headline result)
- Scatter plot, quadrant analysis, per-quadrant accuracy
- Combined AUROC beats any single signal
- The "fragile confidence" quadrant: only detectable through combination

**5.3 Position Sensitivity as Diagnostic**
- Factorial decomposition: position-driven vs content-driven inconsistency
- Position loyalty predicts incorrectness
- Reframing: shuffling is a diagnostic instrument, not just bias mitigation

### 6. Analysis
- Per-subject variation
- Paraphrase count sensitivity (how many do you need?)
- **Effect of CoT on uncertainty signals — the "scaffolding absorption" finding** (see below)
- Constrained decoding finding (JSON kills the signal)
- Cross-dataset generalisability (if multiple datasets)
- Aggregation method comparison: does mean-probs vs majority-vote vs geometric-mean matter?

### 7. Discussion
- Practitioner recommendations: the decision tree ("if you have logprobs, do X; add paraphrasing for Y; add shuffling for Z")
- Why small models matter more: weaker models need better UQ
- Limitations: single model (or two), conservative paraphrases, fixed temperature
- Future work: broken-premise detection, agentic tool-calling, multi-model fusion

### 8. Conclusion

**Appendix:** Full signal definitions, per-subject tables, additional plots, full AUROC results by condition.

---

## 3. Dataset Strategy

### The Problem With MMLU-Redux Only
Reviewers will push back on single-dataset results. The standard expectation at EMNLP/ACL is 2-3 datasets minimum. CAPE uses MMLU + TruthfulQA + HellaSwag + CommonsenseQA. Plaut et al. use ARC + HellaSwag + MMLU + TruthfulQA + WinoGrande. Even "Just Rephrase It" uses ARC-Challenge + OpenBookQA.

A single-dataset paper can work if the analysis is deep enough (and ours is), but adding 1-2 more datasets substantially strengthens the submission and largely eliminates the "is this a quirk of MMLU?" concern.

### Recommended Datasets

**MMLU-Redux 2.0 (already have) — Primary**
- 5,330 verified-correct questions, 57 subjects
- Broad knowledge benchmark, the most widely used
- Our 12-condition factorial runs here

**TruthfulQA (Multiple Choice) — Strongly Recommended** ★
- 817 questions designed to elicit common misconceptions
- Multiple-choice format: 4-5 options per question
- WHY THIS IS PERFECT FOR US: TruthfulQA is specifically about questions where models are *confidently wrong* because the incorrect answer matches common misconceptions. Our uncertainty signals should shine here — the model might be confident on a single prompt ("common knowledge says X"), but paraphrasing might crack that false confidence. This is exactly the "fragile confidence" quadrant.
- CAPE and Plaut et al. both use it → enables direct comparison
- Small enough to run all 12 conditions relatively quickly
- Paraphrases would need to be generated (~$1 via Anthropic API)

**ARC-Challenge — Recommended**
- 1,172 science questions requiring reasoning (grade-school science)
- Multiple-choice, 4 options
- WHY IT'S USEFUL: Tests whether our signals work on reasoning-heavy questions vs pure recall. MMLU is mixed; ARC-Challenge is specifically the hard reasoning subset. Our CoT conditions should show bigger effects here.
- "Just Rephrase It" uses it → enables comparison
- Also small enough for full factorial
- Paraphrases would need to be generated

**GPQA (Graduate-Level QA) — Optional Stretch**
- 448 very hard graduate-level science questions
- WHY: Shows what happens when the model is thoroughly out of its depth. Most questions should land in the "thorough confusion" or "honest uncertainty" quadrants. If our signals can detect "this model has no idea" reliably, that's practically useful.
- Small dataset, quick to run, but may not add much if the model gets <30% accuracy (floor effects)

**HellaSwag — Probably Skip**
- Sentence completion, not really MCQA in the traditional sense
- Requires different prompt formatting
- Less relevant to the "practitioner asking a local model questions" framing

### Minimum Viable Dataset Strategy
1. **MMLU-Redux** — full 12-condition factorial (already running)
2. **TruthfulQA** — full 12-condition factorial (small dataset, fast to run)
3. **ARC-Challenge** — at minimum the key conditions (direct+shuffle+para, direct+noshuffle+nopara as baseline)

This gives us three datasets spanning knowledge recall (MMLU), misconception detection (TruthfulQA), and scientific reasoning (ARC). The analysis can then show whether the signal rankings and the 2D space structure hold across all three, or whether different datasets favour different signals.

### What's Needed to Add a New Dataset
For each new dataset:
1. Download and format into `data/questions.json` format (question_id, text, choices, correct_answer, subject)
2. Generate paraphrases via Anthropic API (~10 per question, ~$1-5 depending on dataset size)
3. Validate paraphrases (same pipeline as MMLU)
4. Create YAML configs for each condition
5. Run experiments
6. Analysis code should work identically (same signal computations)

Steps 1-4 are a day's work per dataset. Step 5 is runtime-dependent (TruthfulQA: ~1 hour for all 12 conditions; ARC: ~2-3 hours).

### Dataset Priority If Time Is Short
If we can only add one: **TruthfulQA.** It's the most interesting for our story (confidently-wrong answers are where UQ matters most), it's small (fast to run), and both CAPE and Plaut et al. use it (direct comparison).

---

## 4. Second Model Question

### The Problem
Single-model papers get "is this a quirk of Qwen3 8B?" at review. Two models largely defuses this.

### Options
- **Gemma 3 4B** — smaller model, Google family (different training data/methodology from Qwen). Would show signals work across model families AND across model sizes. Q4 quantization fits in 8GB VRAM.
- **Phi-4 Mini (3.8B)** — Microsoft, strong for its size. Different architecture lineage.
- **Mistral 7B** — well-known baseline model. Slightly older but widely benchmarked.
- **Qwen3 8B Q8** — same model, different quantization. Isolates quantization effects but doesn't address the "different model" concern.

### Recommendation
**Gemma 3 4B Q4** as the second model. Different family (Google vs Alibaba), different size (4B vs 8B), likely different accuracy (~65-70% on MMLU). If the same signals work for both, the generalisability claim is much stronger. And it's smaller, so experiments run faster.

Don't need full 12-condition factorial on the second model. Run the key conditions: direct+noshuffle+nopara (baseline), direct+shuffle+para (full diversification), and maybe one CoT condition. Enough to show the patterns hold.

### Priority
This is important but secondary to finishing MMLU-Redux conditions and adding TruthfulQA. If we have to cut something, cut the second model before cutting the second dataset.

---

## 5. Open Questions / Things to Decide Later

- **Paper title.** Don't decide until results are in. The framing might shift depending on what the data shows.
- **Luigi's framing.** The paper has evolved from "paraphrase-aggregated logprobs" to "anatomy of uncertainty signals." Need to check Luigi is on board with the broader framing. His agentic vision still fits as motivation and future work.
- **How many signals in the main paper vs appendix?** 25 signals is a lot. The main text should focus on maybe 10-12 key ones (the baselines + the novel ones + the 2D combination). Full catalogue in appendix.
- **Constrained decoding: section or just a paragraph?** It's an interesting finding but might dilute the main story. Could be a paragraph in Discussion rather than a full subsection.
- **Broken-premise experiment.** Still deferred to future work. But if the paper becomes "anatomy of uncertainty," a brief pilot (e.g., 50 broken-premise pairs) showing that the 2D space separates valid from broken questions would be compelling. Don't commit to this yet.
- **The Hou et al. epistemic/aleatoric mapping.** We need to argue that variation across semantic-preserving paraphrases captures epistemic (knowledge robustness), not aleatoric (input ambiguity). This is defensible but needs careful writing. Could be a paragraph in Method or a brief theoretical aside.

---

## 6. Key Figures We'll Need

1. **The 2D uncertainty space scatter plot** — confidence × consistency, coloured by correctness. This is the paper's signature figure.
2. **AUROC comparison bar chart** — all signals ranked by AUROC, with the 2D combination at the top.
3. **Position sensitivity diagnostic** — per-question position loyalty vs content loyalty scatter.
4. **Answer coverage histogram** — distribution of off-label mass across correct vs incorrect questions.
5. **Signal correlation heatmap** — shows which signals are redundant and which are independent.
6. **Paraphrase count sensitivity curve** — AUROC vs number of paraphrases.
7. **Per-subject heatmap** — AUROC by MMLU subject for key signals.
8. **Selective prediction curves** — accuracy vs coverage (fraction of questions answered) for different signals.

---

## 7. Analysis Output Decision

**Decision (2026-03-16):** Start with the compute layer only. Output a single CSV with all signals, one row per question per condition. Eva will explore in Power BI first to discover what views matter, then we'll decide on a permanent interactive tool (Dash, Jupyter, or keep Power BI).

**Why:** Eva thinks better by playing with data than by designing dashboards in the abstract. Power BI gives the cross-filtering and drill-down that Streamlit can't. Once the interesting views are clear from exploration, we can decide whether to build something reproducible in Python or stay in Power BI.

**The CSV schema (one row per question per condition):**
- Metadata: question_id, subject, correct_answer, condition, prompt_mode, shuffle, para
- Result: final_answer, is_correct, num_queries
- Within-prompt signals (from first query): msp, single_entropy, second_gap, answer_coverage, hesitation_mass, top_token_is_answer
- Aggregated signals: mean_confidence, agreement, total_uncertainty, aleatoric, epistemic, confidence_variance, rank_stability, aggregated_answer_coverage, second_gap_agg, vote_entropy
- Position signals (shuffle conditions only): position_loyalty, content_loyalty, position_content_ratio
- Per-query detail: kept in separate JSON for drill-down, not in the main CSV

**The existing Streamlit dashboard stays as-is** for experiment monitoring. Analysis is a separate workflow.

---

## 8. The Scaffolding Absorption Effect (2026-03-16)

### The Core Insight
We already discovered that JSON schema enforcement kills the logprob signal — by the time the model reaches the answer token in `{"answer": "B"}`, the scaffolding has absorbed all uncertainty. But the same principle applies to CoT reasoning.

When a CoT model writes "Let me consider each option... Based on my analysis, the correct answer is B", the reasoning chain IS scaffolding. By the time it produces "B", it has already committed in its own text. The answer-token logprobs will be spiked — not 99.99% like JSON, but probably 90-99% on one letter.

### What This Means for the Paper

**This is a feature, not a bug.** It reveals something fundamental about WHERE uncertainty lives in different prompting modes:

- **Direct mode:** Uncertainty is visible in within-prompt logprobs (Tier I signals work). The model hasn't committed — it's making one decision.
- **CoT modes:** Uncertainty is absorbed by the reasoning chain, so within-prompt logprobs at the answer position are spiked (Tier I signals degrade). But uncertainty is visible in across-paraphrase variation (Tier II signals work) because different paraphrases → different reasoning chains → sometimes different answers.

### The Paper Story

This gives us a natural "spectrum of uncertainty absorption":
1. Direct + raw (no template): Maximum uncertainty visible at the answer token. Richest logprobs.
2. Direct + chat template: Slight absorption from the template framing. (We're using raw for direct, so this doesn't apply.)
3. CoT: Moderate absorption. The model's reasoning commits before the answer.
4. JSON schema: Total absorption. The structure dictates everything except the answer letter.

Each step in this spectrum moves uncertainty from "within-prompt logprobs" to "across-prompt variation." The framework adapts — it uses whichever signal dimension is still informative.

### Implications for Analysis
- Expect Tier I AUROC to be high for direct, low for CoT
- Expect Tier II AUROC to be informative for both (but relatively MORE important for CoT)
- The comparison between direct and CoT AUROC leaderboards directly demonstrates this effect
- The `cot_response_length` signal might capture some of the absorbed uncertainty — longer reasoning = harder question, even if the final logprobs are spiked

### Implication for the 2D Space
The 2D confidence × consistency space might look very different for direct vs CoT:
- Direct: genuine spread on both axes
- CoT: compressed on the confidence axis (everything looks confident), spread on the consistency axis
- This means the 2D space visualisation should be done PER PROMPT MODE, not pooled

---

## 8b. The Agreement Limitation & Distributional Consistency (Eva, 2026-03-16)

### The NMR Example
Question `mmlu_redux_college_chemistry_0023` — spin-lattice relaxation, hard physics. Correct answer: A (3.74 T). Model picks B (5.19 T). **Agreement = 1.00** (all 11 paraphrases unanimously wrong). Mean confidence = 0.502.

This is the canonical failure mode of consistency-based UQ: the model has a systematic parametric error, and no amount of rephrasing shakes it loose because the mistake lives in the weights, not the prompt.

### What Agreement Misses
Agreement = 1.00 hides real distributional instability. Looking at the actual probability vectors:
- P(A) bounces between 0.10 and 0.35 across paraphrases
- P(C) swings from 0.15 to 0.27
- B wins every time but by varying margins (0.44 to 0.59)

Argmax agreement is a lossy compression — it throws away all this information. Two situations both score 1.00: "all 11 give B probability 0.95" vs "all 11 give B probability 0.51". Those are completely different.

### Two New Signals From This Insight

**Mean pairwise JSD**: average Jensen-Shannon divergence between all pairs of individual probability vectors. Different from epistemic uncertainty (which is JSD from the mean). Catches distributional drift that argmax agreement misses. The reframe: argmax agreement asks "do the paraphrases agree on the answer?" — pairwise JSD asks "do the paraphrases agree on the model's uncertainty?"

**Agreement-confidence gap**: `agreement - mean_confidence`. A large positive gap (e.g. 1.00 - 0.502 = 0.498) is a red flag: the model consistently barely picks the same answer. This catches the "systematically wrong with moderate confidence" pattern that neither agreement nor confidence alone would flag.

### Paper Angle
This makes a great case study in the results section. The NMR question is a vivid example of WHY you need both axes of the 2D space, and why looking at probability vectors (not just argmax) matters. Agreement-based methods (self-consistency, Just Rephrase It) would say "this is reliable." Our distributional signals say "wait, the model isn't actually sure."

Also raises the question: do questions where agreement is high but the answer is wrong cluster by subject or question type? Are there domains where systematic parametric errors produce misleading agreement? (Hypothesis: calculation-heavy questions requiring specific equations are most vulnerable.)

---

## 9. The "Who Wants to Be a Millionaire" Idea (Eva, 2026-03-16)

### Observation
The pipeline already tracks "missing letters" — answer options that don't even appear in the top-20 logprobs. When 2-3 letters are missing, the model has effectively narrowed the field to 1-2 options. The PATTERN of missing letters across paraphrases is rich:
- Option C missing on 11/11 queries → robust elimination, the model is sure C is wrong regardless of phrasing
- Option C missing on 7/11 queries → fragile elimination, some phrasings make C seem plausible
- No missing letters on any query → all four options competitive, genuine uncertainty

### As an Uncertainty Signal (computable now, no new experiments)
- Missing letter count per query → certainty signal (more missing = more confident, but also more dangerous if wrong)
- Consistent eliminations across paraphrases → robust knowledge
- Fragile eliminations → fragile knowledge
- Effective option count (how many options is the model actually choosing between?) → continuous version

### As an Experimental Technique (future work — requires new experiments)
**The 50:50 Lifeline:** Use the first pass to identify options the model has consistently eliminated across paraphrases. Re-query the model with a reduced option set (e.g., just A, B, D after eliminating C). Does accuracy improve? Does confidence become more calibrated?

This is iterative uncertainty-guided option elimination. The model itself tells you which options to remove. Nobody is doing this.

**Why it might work:** Removing clearly-wrong distractors changes the model's attention landscape. With 4 options, the model has to distribute attention across all of them. With 2, it can focus. This might improve both accuracy and calibration on hard questions.

**Why it might not work (or be more subtle):** The model might already be ignoring the eliminated options. Removing them explicitly might not change anything — or it might confuse the model if it expects 4 options. Needs testing.

**Paper framing:** "The framework doesn't just measure uncertainty — it can ACT on it. Options consistently eliminated across paraphrases can be removed, and the model re-queried with a reduced choice set, mimicking expert test-taking strategy."

**Priority:** Future work / potential extension. The analysis-only version (missing letter patterns as signals) goes in the current paper. The re-querying experiment is a natural follow-up.

### The Full "Who Wants to Be a Millionaire?" Analogy (Eva's framing)

The entire uncertainty framework maps beautifully onto WWTBAM lifelines:

**50:50 — Option Elimination**
The model's logprobs already tell you which options it's ruling out (missing from top-20 across paraphrases). Two versions:
- *Analysis-only (current paper):* Identify consistently eliminated options, re-normalise over survivors, check if calibration improves.
- *Experimental (future work):* Actually re-query with reduced options. Does the model do better when distractors are removed?

**Ask the Audience — Resample with Variation**
When the model is uncertain on one prompt, don't just trust that one answer. "Ask the audience" = query the same model multiple ways: different paraphrases, different answer orderings, different temperatures, with/without CoT. Each "audience member" is a different view of the same question. Aggregate their responses. This is literally what our framework does — paraphrasing + shuffling IS asking the audience.

**Phone a Friend — Escalate to a Bigger Model**
When the framework detects genuine, persistent uncertainty (low confidence AND low consistency even after paraphrasing and shuffling), the model should ACCEPT it doesn't know and escalate. Queue the question for: a bigger model, thinking/reasoning mode, a human reviewer, or a different approach entirely. This is the cost-aware cascading idea from Luigi's framework (Section 5.6 of his doc) — but framed more intuitively.

**Use in the paper:** This analogy could work as a framing device in the Introduction or Discussion. "Our framework equips local LLMs with the equivalent of game-show lifelines..." — it's memorable, accessible, and maps precisely onto the technical contributions. Even reviewers have watched WWTBAM. Consider a figure showing the three lifelines mapped to technical components.

**Note:** This is a communication device, not the paper's structure. The technical structure stays as outlined in Section 2. But the analogy could make the Introduction much more engaging than the standard "LLMs lack calibrated confidence..." opening.

---

## 9. Elevator Pitch (Draft)

"When you ask a local LLM a question, it gives you an answer but no indication of whether it's guessing. We show that small, quantized models running on consumer hardware contain rich uncertainty information that nobody is extracting. By systematically combining within-prompt confidence from logprobs with across-prompt consistency from paraphrasing, and by repurposing answer shuffling as a diagnostic instrument rather than just a bias fix, we identify 25 uncertainty signals — several genuinely novel — and show that their combination substantially outperforms any individual metric at distinguishing correct from incorrect answers. The result is a practical playbook for anyone running local LLMs who needs to know when to trust the output."

---

## 9b. The Three Reasoning Levels — Why We Need Direct, CoT, AND Thinking Mode

### The Discovery (March 17)

Early cot_structured results revealed a 7-point accuracy drop vs direct mode (68.6% vs 75.5%). Individual questions showed the model reasoning itself OUT of correct answers — getting the right answer by pattern-matching in direct mode, then rejecting it via faulty step-by-step evaluation in CoT.

This is reasoning-induced error (Bentham et al., TMLR 2024; Renze & Guven CCoT 2024). On knowledge-recall questions (most of MMLU), the model knows the answer associatively but can't articulate why. Forced external reasoning corrupts the signal.

### The Three-Way Comparison

| Level | How it reasons | What it reveals |
|-------|---------------|-----------------|
| **Direct** | Doesn't — raw pattern matching, single token | Pure associative knowledge. Highest accuracy on recall questions. Clean logprobs with genuine spread. |
| **CoT structured** | Forced by prompt to evaluate each option | Sequential analytical knowledge. Lower accuracy when reasoning corrupts intuition. Logprobs conditioned on (possibly wrong) reasoning. |
| **Thinking mode** | Model's trained internal `<think>` tokens | Native reasoning pathway. Key question: does it avoid reasoning-induced errors because Qwen3 was trained specifically for this? Think tokens are architectural, not a prompt hack. |

### Why This Matters for the Paper

**1. It's a controlled study of reasoning modality.**
Same model, same questions, same everything — except how the model reasons. If thinking mode matches direct-mode accuracy (~75%), the problem isn't reasoning per se but *forced external reasoning via prompting*. If thinking mode also drops, reasoning genuinely hurts on knowledge-recall MCQs for small models.

**2. Different reasoning modes produce different uncertainty signatures.**
Direct-mode logprobs capture associative confidence. CoT logprobs capture post-hoc reasoned confidence. Thinking-mode logprobs capture trained-reasoning confidence. These three "views" of the same question are independent signals that the framework can combine.

**3. Cross-mode disagreement is itself a diagnostic.**
Questions where direct and CoT disagree are inherently more uncertain — the model's intuition and its reasoning conflict. Add thinking mode and you get a three-way vote. Questions where all three agree are maximally trustworthy. Questions where they disagree in specific patterns map onto our failure mode taxonomy (B1: CoT derailment, D7: CoT rescue).

**4. Practical guidance for practitioners.**
"Should I use CoT with my 8B model?" is a real question people face. Our data can answer it: "For knowledge recall, no — it hurts accuracy. For multi-step reasoning (the questions where direct mode is uncertain), yes — CoT adds diagnostic value."

### Connection to the Scaffolding Absorption Spectrum

This extends brainstorm Section 8. The spectrum is now:

Direct (no scaffolding → max logprob signal) → Two-pass CoT (reasoning as context → moderate absorption) → Single-pass CoT (inline reasoning → heavy absorption) → Thinking mode (thousands of hidden tokens → ???) → JSON schema (format tokens → total absorption)

Where thinking mode falls on this spectrum is an open empirical question. If think tokens absorb uncertainty the same way CoT tokens do, thinking-mode logprobs will be spiked. If the model's trained reasoning pathway preserves uncertainty better, the logprobs might be more spread than CoT.

---

## 10. Model State Taxonomy — "Diagnostic Pathology of LLM Behaviour"

### Motivation

To validate that our signal suite is comprehensive, we enumerate every distinct state a model can be in when answering a question — correct or incorrect, confident or uncertain — then map each state to the metric signature it should produce. This serves three purposes:

1. **Completeness check:** If a failure mode has no distinguishing metric signature, we're missing something.
2. **Paper structure:** The Results section can be organised around "which failure modes does each signal catch?" rather than just raw AUROC numbers.
3. **Practitioner value:** A clinician uses symptoms to diagnose diseases. A practitioner uses metric signatures to diagnose why their model is unreliable on a particular question.

### The Failure Modes

We identify 12 distinct failure modes, organised into four families based on where the problem originates.

**Family A — Knowledge Failures (the model's weights are the problem)**

**A1. Confident Ignorance ("Apples are blue")**
The model has learned something factually wrong and is fully committed to it. No amount of rephrasing shakes it because the error lives in the weights, not the prompt. This is the hardest failure mode for any UQ method to catch.
- *Example:* A question about NMR physics where the model confidently picks the wrong equation.
- *Distinguishing feature:* Looks identical to a correct, well-known answer from the model's perspective.

**A2. Systematic Parametric Error ("Consistently barely wrong")**
A subtler version of A1. The model picks the same wrong answer every time, but with moderate rather than high confidence. The correct answer gets non-trivial probability mass. The NMR example (Section 8b) is the canonical case — agreement=1.00, confidence=0.50, wrong.
- *Distinguishing feature:* The gap between agreement and confidence reveals the fragility.

**A3. Partial Knowledge ("I've narrowed it to two")**
The model has eliminated 2 of 4 options (genuine knowledge!) but picks the wrong survivor. Knows enough to narrow the field but not enough to finish.
- *Example:* "Is the capital of Australia Sydney or Canberra?" — eliminates Melbourne and Brisbane, picks Sydney.
- *Distinguishing feature:* Bimodal distribution, high missing letter count, moderate confidence split between two options.

**A4. Knowledge Gap ("I have no idea")**
The question is completely outside the model's training distribution. It has no basis for choosing and is essentially guessing.
- *Example:* A highly specialised graduate-level question on a niche topic.
- *Distinguishing feature:* Near-uniform distribution, no strong preference for any option.

**Family B — Reasoning Failures (the model has the knowledge but misapplies it)**

**B1. Overthinking / CoT Derailment**
The model's "gut instinct" (direct mode) is correct, but when it reasons step-by-step, it talks itself into the wrong answer. The reasoning chain introduces errors or follows a plausible but incorrect logical path.
- *Example:* A math problem where the direct logprobs favour the right answer but CoT reasoning makes an arithmetic error mid-chain.
- *Distinguishing feature:* Direct-mode correct, CoT-mode incorrect. Only visible in cross-condition comparison.

**B2. Distractor Seduction**
One of the wrong options is designed to be appealing (common misconception, superficially plausible, or shares keywords with the question). The model "knows" the right answer in some sense but the distractor pulls it away on certain phrasings.
- *Example:* "What's the powerhouse of the cell?" with a trick option referencing mitochondrial DNA function.
- *Distinguishing feature:* Oscillation between correct and one specific wrong answer across paraphrases. Bimodal distribution, LOW rank stability (rankings flip).

**B3. Calculation / Multi-Step Error**
Questions requiring precise computation. The model has the right conceptual framework but makes execution errors (wrong formula application, arithmetic mistakes, unit conversion errors).
- *Distinguishing feature:* CoT traces show correct setup but wrong execution. Confidence may be moderate-to-high because the model "knows how to do it." Similar to A1 but specific to procedural knowledge.

**Family C — Prompt Sensitivity Failures (the model's answer depends on HOW you ask)**

**C1. Framing Sensitivity ("Fragile knowledge")**
The model's answer changes depending on how the question is worded, even though all paraphrases are semantically equivalent. The underlying knowledge is real but loosely held — specific wordings activate different associations.
- *Example:* A question where formal academic phrasing gets the right answer but colloquial phrasing triggers a common misconception.
- *Distinguishing feature:* High epistemic uncertainty, high confidence variance, low agreement. The signature of paraphrase-based UQ working as intended.

**C2. Position Bias Override**
The model has genuine (perhaps weak) knowledge of the correct answer, but position bias overrides it. When the correct answer happens to be in position A (the favoured position), the model gets it right. When it's in position D, the model picks whatever is in position A instead.
- *Distinguishing feature:* High position loyalty, low content loyalty. The factorial decomposition (S22) directly measures this. Agreement may be high in noshuffle conditions but drops in shuffle conditions.

**C3. Off-Label Escape ("I don't want to answer this")**
The model's strongest impulse is to NOT pick a letter at all — it wants to explain, hedge, or ask for clarification. We force it to pick a letter, but its top token isn't an answer letter. The forced answer is unreliable because the model was coerced.
- *Example:* An ambiguous question, a question requiring qualifications, or a question outside the expected format.
- *Distinguishing feature:* Low answer coverage (S5), high hesitation mass (S6), top token is NOT an answer letter (S7). These are our novel local-model signals.

**Family D — Correct Answer Modes (model gets it right, but HOW it gets there varies)**

**D1. Textbook Knowledge ("The sun is hot")**
The model knows the answer with certainty. Every paraphrase, every answer ordering — same answer, high confidence, no wobble. This is the gold standard. The knowledge is deeply encoded in the weights and completely robust to surface variation.
- *Example:* "What is the chemical formula for water?" → H₂O, every time, 95%+ confidence.
- *Distinguishing feature:* High everything good, low everything bad. The reference signature.

**D2. Solid Knowledge ("Knows it well")**
The model gets it right consistently but isn't maximally confident. Confidence is moderate-to-high (60-85%), agreement is perfect or near-perfect. The model has the knowledge but other options aren't completely dismissed — they retain some plausibility.
- *Example:* A well-known fact in a domain the model has moderate training coverage on. Gets it right but gives 15-25% to the runner-up.
- *Distinguishing feature:* Correct, moderate-to-high confidence, high agreement. Healthy knowledge with proportionate confidence.

**D3. Correct via Elimination ("Process of elimination")**
The model arrives at the right answer not because it strongly recognises the correct option, but because it confidently rules out the wrong ones. The correct answer wins by default rather than by recognition. Missing letter count is high (2-3 eliminated), but confidence on the winner is only moderate.
- *Example:* An obscure question where three options are obviously wrong but the fourth isn't obviously right.
- *Distinguishing feature:* High missing letter count, moderate confidence on winner, correct. Answer coverage may be moderate (the model is sure some things are wrong, less sure what's right).

**D4. Right but Fragile ("Shaky correct")**
The model gets the right answer on most paraphrases but not all. Aggregate answer is correct, but the knowledge is shaky. Some rephrasings tip it to a wrong answer.
- *Example:* A question where the model knows the answer in one framing but a slightly different phrasing activates a competing association.
- *Distinguishing feature:* Correct aggregate, but agreement < 1.0, high confidence variance, moderate epistemic uncertainty. A "D4 today might be C1 tomorrow" — small weight changes could flip it.

**D5. Lucky Guess ("Right for the wrong reasons")**
The model picks the correct answer but with low confidence. It's essentially guessing among options it can't distinguish and happens to land on the right one. Not reliable.
- *Example:* A graduate-level question outside the model's training. It picks C with 30% confidence and happens to be right.
- *Distinguishing feature:* Correct but low confidence, low agreement, high entropy. Indistinguishable from A4 (Knowledge Gap) except the answer happens to be right.

**D6. Right via Position ("Got lucky with ordering")**
In noshuffle conditions, the correct answer happens to be in the model's favoured position. The model picks it, but it's tracking position not content. In shuffle conditions, this mode becomes C2 (Position Bias Override) and the model starts getting it wrong.
- *Distinguishing feature:* Correct in noshuffle, incorrect in shuffle conditions. High position loyalty. Only visible when comparing across shuffle conditions.

**D7. CoT Rescue ("Thinking helped")**
The model's gut instinct (direct mode) is wrong, but CoT reasoning corrects it. The reasoning chain catches the error and arrives at the right answer. The inverse of B1.
- *Example:* A multi-step question where the immediately obvious answer is wrong but step-by-step reasoning reveals the correct answer.
- *Distinguishing feature:* Direct-mode wrong, CoT-mode correct. Only visible in cross-condition comparison. CoT confidence likely high (the model reasoned its way to certainty).

**Family E — Edge Cases and Ambiguous States**

**E1. Genuinely Ambiguous Question**
The question itself is poorly written, has multiple defensible answers, or requires context not provided. The model's uncertainty is appropriate — it SHOULD be uncertain. This isn't a failure of the model; it's a failure of the question.
- *Example:* Questions flagged as having errors in the MMLU-Redux validation process.
- *Distinguishing feature:* High entropy, low agreement, but potentially high answer coverage (the model engages with the question, just can't decide). Similar to A4 but the model may show higher per-query confidence (it has opinions, they just conflict).

**E2. Right Answer, Wrong Reason (CoT reveals misconception)**
The model picks the correct answer but its reasoning trace (if available) is wrong. It arrived at the right letter through faulty logic that happened to converge on the correct answer.
- *Example:* Gets a biology question right but the CoT trace cites a wrong mechanism that coincidentally supports the right answer.
- *Distinguishing feature:* Correct answer, potentially high confidence, but CoT trace analysis would reveal errors. From logprobs alone, indistinguishable from D1/D2. A future-work detection challenge.

**E3. Broken-Premise Response (Experiment 2 only)**
The question's premise is invalid (e.g., asking about mitochondria in an organism that has none). No answer is correct, but the model is forced to pick one. The "correct" behaviour is uncertainty.
- *Example:* "What is the primary function of the mitochondria in a Monocercomonoides cell?"
- *Distinguishing feature:* We hypothesise: lower confidence, lower agreement, higher entropy than the matched valid question. The distributional signature of "this question doesn't make sense" should differ from "I don't know the answer" (A4). Validating this is the core of Experiment 2.

---

### The Diagnostic Table

Each row is a failure mode ("disease"). Each column is a metric ("diagnostic test"). Cells show the expected metric value: **H** (high), **M** (moderate), **L** (low), **~** (near-zero or uninformative), **VAR** (variable/unstable). For binary metrics, **Y/N**.

**Key metrics used as columns:**

| Abbrev | Full Signal | What It Measures |
|--------|-------------|------------------|
| Conf | Mean confidence (S10) | How peaked is the aggregated distribution? |
| Agr | Agreement rate (S8) | Do all paraphrases pick the same answer? |
| Ent | Total uncertainty / entropy (S11) | How spread is the aggregated distribution? |
| Epi | Epistemic uncertainty (S13) | How much does the distribution shift across paraphrases? |
| Ale | Aleatoric uncertainty (S12) | How spread is each individual query's distribution? |
| CVar | Confidence variance (S14) | How stable is confidence across paraphrases? |
| Gap2 | Aggregated second-choice gap (S15) | How decisive is the aggregated answer? |
| ACov | Answer coverage (S5) | Does the model want to answer at all? |
| Miss | Missing letter count (S7b) | How many options eliminated from top-20? |
| PJSD | Mean pairwise JSD (S15b) | How much do individual distributions differ from each other? |
| AGap | Agreement-confidence gap (S15c) | Agreement minus confidence — the "barely winning" flag |
| PLoy | Position loyalty (S19) | Does the model track position or content? |
| RkSt | Rank stability (S16) | Do option rankings stay consistent? |
| Shape | Distribution shape (S4) | Peaked / bimodal / spread / flat? |

---

#### Family A — Knowledge Failures

| Mode | Correct? | Conf | Agr | Ent | Epi | Ale | CVar | Gap2 | ACov | Miss | PJSD | AGap | PLoy | RkSt | Shape |
|------|----------|------|-----|-----|-----|-----|------|------|------|------|------|------|------|------|-------|
| **A1. Confident Ignorance** | N | **H** | **H** | **L** | **L** | **L** | **L** | **H** | **H** | **H** | **L** | **L** | **L** | **H** | Peaked |
| **A2. Systematic Parametric** | N | **M** | **H** | **M** | **L** | **M** | **L-M** | **M** | **M-H** | **M** | **M** | **H** ★ | **L** | **H** | Peaked-soft |
| **A3. Partial Knowledge** | N | **M** | **M-H** | **M** | **L-M** | **M** | **M** | **L** | **M-H** | **H** | **L-M** | **M** | **L** | **M-H** | Bimodal |
| **A4. Knowledge Gap** | N | **L** | **L** | **H** | **M-H** | **H** | **M** | **L** | **M-L** | **L** | **M-H** | **~** | **L** | **L** | Flat |

**Detection strategy:**
- **A1** is the nightmare — almost invisible to any single metric. The ONLY reliable signals are external (we know the correct answer). Within the framework, the best hope is that answer_coverage or hesitation signals are slightly lower than for truly known facts, but this needs empirical validation. If the model is confidently wrong AND high-coverage AND stable, no UQ method can help. This is a fundamental limitation worth stating honestly in the paper.
- **A2** is caught by **AGap** (agreement-confidence gap) — the headline signal from the NMR insight. Agreement is high but confidence is only moderate, and that gap is diagnostic. PJSD also helps: the distributions wobble even though argmax doesn't flip.
- **A3** is caught by **shape** (bimodal), **Miss** (high — 2 options eliminated), and **Gap2** (low — close race between survivors).
- **A4** is caught by everything — low confidence, high entropy, flat shape. This is the easy case.

#### Family B — Reasoning Failures

| Mode | Correct? | Conf | Agr | Ent | Epi | Ale | CVar | Gap2 | ACov | Miss | PJSD | AGap | PLoy | RkSt | Shape |
|------|----------|------|-----|-----|-----|-----|------|------|------|------|------|------|------|------|-------|
| **B1. CoT Derailment** | N (CoT) | **H** (CoT) | **H** (CoT) | **L** (CoT) | **L** (CoT) | **L** (CoT) | **L** | **H** (CoT) | **H** | **H** | **L** | **L** | — | **H** | Peaked |
| **B2. Distractor Seduction** | N/VAR | **M** | **L-M** | **M** | **H** | **M** | **H** | **L** | **M** | **M** | **H** | **M** | **L** | **L** ★ | Bimodal |
| **B3. Calculation Error** | N | **M-H** | **M-H** | **L-M** | **L** | **L-M** | **L** | **M** | **H** | **M-H** | **L** | **M** | **L** | **M-H** | Peaked-soft |

**Detection strategy:**
- **B1** is ONLY detectable via **cross-condition signals** (S23-S25). Within the CoT condition alone, it looks like A1 (confidently wrong). You need the direct-mode result to reveal the discrepancy. This is a strong argument for running both prompt modes.
- **B2** is caught by **RkSt** (rank stability) — rankings flip between correct and distractor. Also **Epi** (high — distributions shift), **CVar** (high — confidence swings), **PJSD** (high — distributions differ pairwise). This is the classic "fragile confidence" quadrant target.
- **B3** looks similar to A1/A2 from the metrics — the model is confident in its wrong calculation. Potentially distinguishable by **CoT trace analysis** (future work) but hard from logprobs alone. Similar fundamental limitation as A1 for procedural errors.

#### Family C — Prompt Sensitivity Failures

| Mode | Correct? | Conf | Agr | Ent | Epi | Ale | CVar | Gap2 | ACov | Miss | PJSD | AGap | PLoy | RkSt | Shape |
|------|----------|------|-----|-----|-----|-----|------|------|------|------|------|------|------|------|-------|
| **C1. Framing Sensitivity** | VAR | **M** | **L** ★ | **M** | **H** ★ | **M** | **H** ★ | **M** | **M** | **M** | **H** | **L-M** | **L** | **L-M** | Variable |
| **C2. Position Bias Override** | VAR | **M** | **M** (noshuffle) / **L** (shuffle) | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **M** | **H** ★ | **M** | Variable |
| **C3. Off-Label Escape** | VAR | **L-M** | **M** | **M-H** | **M** | **M-H** | **M** | **L-M** | **L** ★ | **L** | **M** | **M** | **L** | **M** | Spread |

**Detection strategy:**
- **C1** is the poster child for paraphrase-based UQ. Caught by **Epi**, **CVar**, and **Agr**. This is exactly what the framework is designed to detect.
- **C2** is caught specifically by **PLoy** (position loyalty) and the factorial decomposition. Agreement drops specifically in shuffle conditions. Without shuffling, this failure mode is invisible.
- **C3** is caught by the **novel local-model signals**: **ACov** (answer coverage), hesitation mass, top-token identity. These are our unique contribution — no cloud API or black-box method can detect this.

#### Family D — Correct Answer Modes

| Mode | Correct? | Conf | Agr | Ent | Epi | Ale | CVar | Gap2 | ACov | Miss | PJSD | AGap | PLoy | RkSt | Shape |
|------|----------|------|-----|-----|-----|-----|------|------|------|------|------|------|------|------|-------|
| **D1. Textbook Knowledge** | Y | **H** | **H** | **L** | **L** | **L** | **L** | **H** | **H** | **H** | **L** | **L** | **L** | **H** | Peaked |
| **D2. Solid Knowledge** | Y | **M-H** | **H** | **L-M** | **L** | **L-M** | **L** | **M-H** | **H** | **M-H** | **L** | **L** | **L** | **H** | Peaked-soft |
| **D3. Correct via Elimination** | Y | **M** | **H** | **M** | **L** | **M** | **L** | **M** | **M** | **H** ★ | **L** | **M** | **L** | **H** | Bimodal-ish |
| **D4. Right but Fragile** | Y (agg) | **M** | **M** | **M** | **M-H** | **M** | **H** | **M** | **M** | **M** | **M** | **L-M** | **L** | **M** | Variable |
| **D5. Lucky Guess** | Y | **L** | **L** | **H** | **M-H** | **H** | **M** | **L** | **M-L** | **L** | **M-H** | **~** | **L** | **L** | Flat-ish |
| **D6. Right via Position** | Y (noshuffle) | **M** | **H** (noshuffle) | **M** | **L** (noshuffle) | **M** | **L** | **M** | **M** | **M** | **L** | **M** | **H** ★ | **M** | Variable |
| **D7. CoT Rescue** | Y (CoT) | **H** (CoT) | **H** (CoT) | **L** (CoT) | **L** (CoT) | **L** (CoT) | **L** | **H** (CoT) | **H** | **H** | **L** | **L** | — | **H** | Peaked |

**Detection strategy:**
- **D1** is the reference class. Any deviation from this signature warrants investigation.
- **D2** is healthy — the model has proportionate confidence. No action needed.
- **D3** is interesting — high missing letters + moderate confidence = "I know what's wrong, just not sure what's right." The elimination pattern (S17b, consistent eliminations) is the signature.
- **D4** matters for selective prediction (AUARC) — correct today, might be wrong with small perturbations. CVar and Epi flag these.
- **D5** matters for coverage decisions — accepting these inflates accuracy now but is unreliable. Low confidence is the flag.
- **D6** is only visible cross-condition. Within noshuffle, it looks like D2. The PLoy signal in shuffle conditions reveals it.
- **D7** is the mirror of B1. Only visible cross-condition. Arguments for "when to use CoT" — specifically on questions where direct mode is uncertain.

#### Family E — Edge Cases and Ambiguous States

| Mode | Correct? | Conf | Agr | Ent | Epi | Ale | CVar | Gap2 | ACov | Miss | PJSD | AGap | PLoy | RkSt | Shape |
|------|----------|------|-----|-----|-----|-----|------|------|------|------|------|------|------|------|-------|
| **E1. Genuinely Ambiguous** | N/A | **M** | **L** | **M-H** | **H** | **M** | **H** | **L** | **M** | **L** | **H** | **~** | **L** | **L** | Spread |
| **E2. Right Answer, Wrong Reason** | Y | **M-H** | **H** | **L-M** | **L** | **L-M** | **L** | **M** | **H** | **M** | **L** | **L** | **L** | **H** | Peaked-soft |
| **E3. Broken-Premise** | N/A | **L** ★ | **L** ★ | **H** ★ | **H** ★ | **H** | **H** | **L** | **L-M** ★ | **L** | **H** ★ | **~** | **L** | **L** | Spread/Flat |

**Detection strategy:**
- **E1** looks like C1 (framing sensitivity) but the root cause is different — the question is bad, not the model. Distinguishing E1 from C1 requires external question-quality assessment (which MMLU-Redux provides via its validation annotations).
- **E2** is undetectable from logprobs alone. Requires CoT trace analysis. Fundamental limitation.
- **E3** is the subject of Experiment 2. We predict the metric signature based on the hypothesis that breaking a question's premise produces distributional instability. The ★ markers show the signals we expect to be most diagnostic. If confirmed, this validates the framework's ability to detect "this question doesn't make sense" — a capability with major practical implications for real-world deployment.

---

### Key Insights From the Table

**1. A1 (Confident Ignorance) and D1 (Textbook Knowledge) are indistinguishable. This is a fundamental limit.**
A1's metric signature is identical to D1. This is an honest limitation to state in the paper. The only ways to catch it are: (a) cross-model disagreement (a second model doesn't share the same wrong knowledge), (b) broken-premise detection (our Experiment 2 — the model's consistency breaks down when the question is subtly invalid), or (c) external ground truth. No amount of paraphrasing or shuffling helps when the error is baked into the weights. Similarly, E2 (Right Answer Wrong Reason) is indistinguishable from D2 (Solid Knowledge).

**2. AGap (Agreement-Confidence Gap) is the unique diagnostic for A2 (Systematic Parametric Error).**
No other single metric catches this. Agreement says "reliable." Confidence says "moderate." Only their gap reveals the problem. This validates the NMR-inspired signal as a genuine contribution.

**3. Position Loyalty is the unique diagnostic for C2 (Position Bias Override).**
Without shuffling and without measuring per-question position sensitivity, this failure mode is completely invisible. Reinforces the "shuffling as diagnostic instrument" framing.

**4. Answer Coverage / Off-Label signals are the unique diagnostic for C3 (Off-Label Escape).**
Cloud APIs and black-box methods can't detect when the model doesn't want to answer. Our local-model top-20 logprobs are the only source.

**5. Cross-condition signals are the unique diagnostic for B1 (CoT Derailment).**
Within any single prompting mode, this failure looks like confident correctness or confident ignorance. Only comparing across modes reveals the problem.

**6. Most failure modes are caught by 2-3 signals working together, not by any single metric.**
This is the core argument for the signal suite and the 2D (or higher-dimensional) uncertainty space. A "decision tree" for practitioners might look like:
- High agreement + high confidence + wrong → A1 (undetectable, accept the limitation) or B3 (check CoT trace)
- High agreement + moderate confidence + wrong → A2 (check AGap)
- Low agreement + moderate confidence → C1 (framing sensitivity — paraphrasing is earning its keep)
- High position loyalty → C2 (position bias — shuffling is earning its keep)
- Low answer coverage → C3 (model doesn't want to answer — local model signals earning their keep)
- Direct correct, CoT wrong → B1 (CoT derailment)

**7. The table validates our experimental design.**
Each experimental condition (shuffle/noshuffle, para/nopara, direct/CoT) is specifically required to detect certain failure modes. No single condition catches everything. The 12-condition factorial is justified not as brute-force exploration but as targeted diagnostic coverage.

**8. Mirror pairs reveal what each experimental condition buys you.**
Several modes come in pairs that are only distinguishable via cross-condition comparison:
- **B1 (CoT Derailment) ↔ D7 (CoT Rescue):** Same cross-condition signal, opposite direction. Quantifying how many questions fall into each tells you whether CoT helps or hurts on balance — and more importantly, whether you can PREDICT which mode a question will fall into before running CoT.
- **C2 (Position Bias Override) ↔ D6 (Right via Position):** Same mechanism, opposite luck. In noshuffle, D6 looks correct and C2 looks wrong. Only shuffling reveals both are position-driven.
- **D4 (Right but Fragile) ↔ C1 (Framing Sensitivity):** The same underlying phenomenon (phrasing-dependent knowledge). D4 happens to aggregate to the correct answer; C1 aggregates to wrong. They're on a continuum — D4 is "barely correct," C1 is "barely wrong."
- **D5 (Lucky Guess) ↔ A4 (Knowledge Gap):** Identical metric signatures, different outcomes. The model is equally clueless in both cases. Only the coin flip differs.

**9. The taxonomy organises along two dimensions: correctness × reliability.**
Modes can be mapped onto a 2×2 (or 2×3):

|  | Reliable | Shaky | Unreliable |
|---|---|---|---|
| **Correct** | D1, D2 | D3, D4, D7 | D5, D6 |
| **Incorrect** | A1, B3 | A2, A3, B2 | A4, B1, C1, C2, C3 |

The framework's practical value is detecting the off-diagonal: reliable-but-wrong (A1 — hard) and unreliable-but-correct (D5, D6 — useful for coverage decisions).

---

### Paper Use

This taxonomy can structure the Results discussion. Instead of just reporting "AUROC for signal X is 0.72," we can say "signal X specifically catches modes A2 and C1, which account for N% of errors, while failing to distinguish A1 from D1." The full table belongs in the Appendix with a simplified version (key modes × key signals) in the main Results.

The taxonomy motivates the experimental design: each condition exists to detect specific modes. It motivates Experiment 2: broken premises (E3) should produce a distinctive distributional signature different from normal uncertainty (A4). And it motivates future work: the undetectable modes (A1, E2) point to where cross-model methods or CoT trace analysis could add value.

**Potential paper figure:** A simplified version of the diagnostic table as a heatmap, with modes on rows and signals on columns. Colour-coded by expected value (red=high, blue=low). Then overlay the ACTUAL observed signatures from the data to validate or refute the predictions.
