# Uncertainty Signals Specification — The Full Playbook

**Purpose:** Comprehensive specification of every uncertainty signal the analysis module should compute. This is the implementation reference for Claude Code when building analysis scripts. Signals are organised by source (where the information comes from) and tagged by novelty status relative to the existing literature.

**Paper framing:** "Anatomy of Uncertainty in Small Local LLMs" — a systematic inventory and comparison of all uncertainty signals extractable from a quantized 8B model on consumer hardware, showing which signals are redundant, which are complementary, and what practitioners gain from combining them.

**Key thesis:** Within-prompt confidence (logprobs) and across-prompt consistency (paraphrases) are independent dimensions of uncertainty. Their combination substantially outperforms either alone. Further decomposition into position-driven and content-driven inconsistency reveals distinct failure modes invisible to any single metric.

---

## What Data We Have Per Question

From each experimental condition, each question produces up to 11 queries (1 original + 10 paraphrases). Each query stores:

- `canonical_probs`: normalised `[P(A), P(B), P(C), P(D)]` over the four answer options
- `raw_logprobs`: **the FULL top-20 logprobs from Ollama** — not just A/B/C/D but all 20 most probable tokens at the answer position, with their logprob values. This is richer than what most papers use.
- `answer_permutation`: which canonical answer appeared in which display position
- `display_letter_logprobs`: raw logprobs for just the answer letters
- `canonical_logprobs`: same, mapped to canonical indices
- `thinking_trace`: full CoT reasoning text (for CoT modes)
- `raw_response`: the model's full text output
- `query_text`: the exact prompt sent

The four conditions per prompt mode (shuffle × para) provide controlled variation:
- `noshuffle_nopara`: 1 query, no variation (pure baseline)
- `shuffle_nopara`: 11 queries, same text, 11 different answer orderings (isolates position effects)
- `noshuffle_para`: 11 queries, 11 different wordings, same answer order (isolates content effects)
- `shuffle_para`: 11 queries, different wordings AND different orderings (full variation)

---

## I. Within-Prompt Signals (Single Query, Single Forward Pass)

These signals come from one prompt and one model response. They establish the "what you get without any diversification" baseline. Every other category improves on these.

### I.1 Answer-Token Distribution (the 4-option normalised probabilities)

**Standard in literature. Our baseline to beat.**

**S1. MSP — Max Single-token Probability**
- `max(canonical_probs)`
- The simplest confidence score. Plaut et al. (2024) report AUROC 60–69% for this across 15 chat models. This is our primary single-prompt baseline.
- Novelty: Standard. Kadavath et al. (2022), Plaut et al. (2024).

**S2. Single-query Entropy**
- `H(canonical_probs) = -sum(p * log2(p))`
- Information-theoretic measure of spread. Correlated with MSP but not identical — entropy captures the full distribution shape, not just the peak.
- Novelty: Standard.

**S3. Second-choice Gap**
- `sorted_probs[0] - sorted_probs[1]`
- How close is the race between the top two answers? A question at [0.6, 0.3, 0.05, 0.05] is more contested than [0.6, 0.15, 0.15, 0.1] even though MSP is identical. Captures "decisiveness" beyond raw confidence.
- Novelty: Occasionally used but rarely reported as a standalone AUROC signal for MCQA.

**S4. Distribution Shape Category**
- Classify each query's 4-option distribution into one of:
  - **Peaked**: one answer > 80% mass
  - **Bimodal**: two answers share > 80% mass combined, both > 15%
  - **Spread**: three or more answers above 10%
  - **Flat**: max prob < 35% (near-uniform)
- Different shapes predict different failure modes. Peaked-and-wrong is confident ignorance. Bimodal means the model is torn between two specific options. Spread means genuine confusion.
- Novelty: Novel as a categorical uncertainty signal for MCQA. Distribution shape is used in calibration research but not as a per-question diagnostic category.

### I.2 Full Vocabulary Signal (the top-20 logprobs beyond A/B/C/D)

**This is our local-model advantage. Cloud APIs typically don't expose this. Novel territory.**

We request `top_logprobs: 20` from Ollama. Only 4 of those slots are answer letters. The other ~16 slots contain whatever else the model wanted to say. From inspecting actual results, these include tokens like " Let", " The", " I", " Okay" (the model wants to reason/explain), "\n\n", " ?" (formatting/hesitation), " Option" (referencing the choices), etc.

**S5. Answer Coverage (off-label probability mass)** ★ NOVEL
- `sum(exp(logprob) for tokens matching A/B/C/D) / sum(exp(logprob) for all top-20 tokens)`
- More precisely: total raw probability mass on answer-letter tokens vs total mass in top-20.
- A confident model puts nearly all mass on answer letters. A confused or reluctant model "leaks" mass to non-answer tokens — it wants to explain, hedge, or do something other than pick a letter.
- This is fundamentally different from entropy or MSP, which only measure how probability is distributed *among* the answer options. Answer coverage measures whether the model even wants to answer in the expected format.
- Novelty: **Not found in any paper in the literature review.** The UQ literature extracts logprobs for answer tokens and normalises. Nobody examines the probability mass on non-answer tokens at the decision point. This is uniquely accessible from local models with top-N logprobs.
- Connection to broken-premise work (future): on broken-premise questions, the model might show very low answer coverage because it wants to explain why none of the answers apply.
- Connection to constrained decoding finding: JSON schema forces 100% answer coverage by construction, which is precisely why it kills the uncertainty signal.

**S6. Hesitation Profile** ★ NOVEL
- Categorise the non-answer tokens in the top-20 into semantic types:
  - **Reasoning starters**: " Let", " The", " I", " To", " Okay", " Well" — model wants to think/explain
  - **Formatting/hedging**: " \n", " ?", " ?\n", " \n\n" — model is hesitating
  - **Meta-references**: " Option", " (" — model is referencing the answer structure
- Compute the probability mass in each category.
- Hypothesis: high mass on reasoning starters correlates with questions where CoT would change the answer (the model "needs to think"). High mass on hedging correlates with genuine confusion.
- Novelty: **Completely novel.** Nobody analyses the semantic content of non-selected tokens at MCQA decision points.
- Implementation: Simple token-matching on the top-20 list. Categorisation can be manual (small fixed vocabulary of common tokens) or automated.

**S7. Top-token Identity**
- Binary: is the model's #1 most probable token an answer letter?
- If the most probable token is " Let" or " The" rather than " B", the model's primary impulse was to explain rather than answer. Even if we force a single token and it ends up being "B", the model's first choice wasn't to answer directly.
- Compute from: `raw_logprobs[0]["token"]` — check if it matches an answer letter.
- Novelty: **Novel as an uncertainty signal.** The "My Answer is C" paper (Wang et al., 2024) documents misalignment between logprobs and text output, but in the context of evaluation methodology, not as an uncertainty measure.

**S7b. Missing Letter Count** ★ NOVEL
- How many of A, B, C, D are completely absent from the top-20 logprobs?
- Range: 0 (all four present) to 3 (only one answer letter appears).
- A high missing count is a *certainty* signal: the model is so sure about 1-2 options that the others don't even register. If 3 letters are missing, the model has effectively narrowed it to a single answer.
- But the relationship with correctness is non-trivial: a model that eliminates 3 options and is RIGHT is maximally confident and correct. A model that eliminates 3 options and is WRONG is maximally confidently wrong (dangerous).
- Compute from: count of answer letters NOT found in `raw_logprobs[0]["top_logprobs"]`.
- Novelty: **Novel as an explicit uncertainty signal.** The pipeline already tracks this for diagnostics but nobody has used it analytically. The information is already in the answer_coverage signal (S5), but the integer count is simpler and more interpretable.

**S7c. Effective Option Count**
- The inverse: `4 - missing_letter_count`. How many options is the model actually considering?
- 4 = all options in play (uncertain). 1 = one option dominates so completely the others are invisible.
- Can also be computed continuously from the probability vector using the "effective number" formula: `exp(H(probs))` where H is Shannon entropy. This gives a smooth version: 1.0 = peaked on one, 4.0 = uniform.

---

## II. Across-Prompt Signals (Multiple Queries, Same Question)

These signals compare the model's behaviour across different versions of the same question. This is where paraphrasing and shuffling add value beyond single-prompt methods.

### II.1 Vote-Based (argmax only — ignoring confidence magnitudes)

**These are the self-consistency family. Established in literature.**

**S8. Agreement Rate**
- Fraction of queries where `argmax(canonical_probs)` matches the overall final answer `argmax(mean_probs)`.
- The "Just Rephrase It" / self-consistency signal. Ignores confidence magnitudes entirely.
- Already implemented in `analysis.py`.
- Novelty: Established (Wang et al. 2022, Beker et al. 2024).

**S9. Vote Entropy**
- Entropy of the vote-count distribution across queries. If 11 queries all vote B, entropy is 0. If votes are split 6B/3D/2A, entropy is higher.
- Different from agreement rate: agreement only checks whether votes match the *winner*. Vote entropy captures the full dispersion pattern.
- Novelty: Used in self-consistency literature but rarely compared head-to-head with logprob-based signals on the same data.

### II.2 Distribution-Based (using full probability vectors)

**These are the logprob-aggregation signals. Our core methodological contribution.**

**S10. Mean Confidence (max of mean probs)**
- `max(mean(canonical_probs_list, axis=0))`
- The paper's primary aggregated confidence measure. A question where the model averages [0.85, 0.08, 0.05, 0.02] across paraphrases has mean confidence 0.85.
- Already computed (implicitly) in `compute_question_metrics()`.
- Novelty: The specific combination of mean-aggregated logprob vectors across question-stem paraphrases is novel (see lit landscape — CAPE and BayesPE vary instructions, not question content).

**S11. Total Uncertainty (entropy of the mean distribution)**
- `H(mean_probs)`
- Already implemented as `mean_of_dist_entropy`.

**S12. Aleatoric Uncertainty (mean per-query entropy)**
- `mean([H(p) for p in canonical_probs_list])`
- Average within-query spread. If this is high, the model is uncertain *on every phrasing*, not just on some.
- Already implemented as `mean_entropy`.

**S13. Epistemic Uncertainty (mutual information)**
- `H(mean_probs) - mean([H(p)])` = Total - Aleatoric.
- How much the distribution *shifts* across paraphrases. High epistemic = fragile knowledge.
- Already implemented. Equivalent to multi-distribution JSD.
- Novelty: The epistemic/aleatoric decomposition via paraphrasing is conceptually established (Hou et al. ICML 2024), but applying it to per-option logprob vectors across question-stem paraphrases is novel. Note the Hou et al. reversed mapping issue — they assign input variation to aleatoric. We argue semantic-preserving paraphrases capture epistemic uncertainty because they test knowledge robustness, not input ambiguity.

**S14. Confidence Variance**
- `std([max(p) for p in canonical_probs_list])`
- A model consistently at 75% is different from one swinging between 95% and 40%. This captures *stability* of confidence.
- Novelty: Surprisingly absent from the UQ literature as a standalone signal. People report mean confidence but not its variance.

**S15. Second-choice Gap (aggregated)**
- From the mean probability vector: `mean_probs_sorted[0] - mean_probs_sorted[1]`
- How decisive is the framework's combined answer? Close races in the aggregated distribution suggest genuine ambiguity that persists even after averaging.

**S15b. Mean Pairwise JSD** ★ NOVEL
- Average Jensen-Shannon divergence between ALL pairs of per-query probability vectors.
- Different from epistemic uncertainty (S13), which is JSD from the mean. Pairwise JSD measures how much individual distributions differ from EACH OTHER, not from the centroid.
- Motivation: the NMR example (mmlu_redux_college_chemistry_0023). Agreement=1.00, all 11 paraphrases pick B, but P(A) bounces between 0.10 and 0.35 across paraphrases. Argmax agreement completely misses this distributional instability. Pairwise JSD catches it.
- With 11 queries, there are 55 pairs — same as rank_stability, trivially fast.
- Novelty: **Novel.** The UQ literature uses epistemic uncertainty (MI / JSD from mean) but not pairwise distributional divergence across paraphrases.

**S15c. Agreement-Confidence Gap** ★ NOVEL
- `agreement - mean_confidence`. A large positive gap means the model always picks the same answer but isn't confident about it (agreement high, confidence moderate). The NMR example: agreement=1.00, confidence=0.502, gap=0.498.
- This is the "consistently barely winning" red flag. It catches the worst failure mode: the model has a systematic parametric error (same wrong answer every time) but the logprobs reveal it's not actually sure.
- Not redundant with either agreement or confidence alone — it's their interaction that's diagnostic.
- Novelty: **Novel as an explicit signal.** The idea that the GAP between categorical agreement and probabilistic confidence is itself a diagnostic has not been explored.

**S16. Answer Rank Stability (Kendall's tau)**
- Mean pairwise Kendall's tau correlation of option rankings across queries.
- If the model always ranks B > D > A > C regardless of paraphrase, tau ≈ 1. If rankings reshuffle, tau is low.
- Captures relative assessment stability: a model that always ranks the same way but varies in magnitude is more trustworthy than one where the ranking itself changes.
- Novelty: Kendall's tau is standard in information retrieval but not used as an MCQA uncertainty signal.

**S17. Aggregated Answer Coverage** ★ NOVEL
- Mean of per-query answer coverage (signal S5) across all 11 queries.
- Also: variance of answer coverage across queries. If coverage is consistently low, the model consistently wants to do something other than answer. If it varies, the model's willingness to answer depends on phrasing.

**S17b. Missing Letter Patterns Across Paraphrases** ★ NOVEL
- For each of the 4 canonical answers, count how many of the 11 queries have that letter missing from the top-20.
- Key derived signals:
  - **Mean missing count**: average number of missing letters per query across paraphrases.
  - **Consistent eliminations**: number of options missing on ALL (or >80% of) queries. If option C is missing on 11/11 queries, it's consistently eliminated regardless of phrasing = robust elimination.
  - **Fragile eliminations**: number of options missing on SOME queries but not others. Option C missing on 7/11 queries = fragile elimination, the model isn't sure whether C is relevant.
  - **Elimination stability**: for each option, the variance across queries of whether it's missing (binary). Low variance = consistent (always present or always missing). High variance = the option's relevance depends on phrasing.
- Connection to the "Who Wants to Be a Millionaire" idea: options with consistent elimination across paraphrases are candidates for removal in an iterative re-querying scheme (future work — see brainstorm.md).
- Novelty: **Novel.** Nobody analyses patterns of absent tokens across paraphrases.

**S17c. Effective Option Count (aggregated)**
- Mean of per-query effective option counts (`exp(H(probs))`) across all queries.
- Also: the effective option count computed from the mean probability vector.
- If the mean effective count is 1.5, the model is essentially choosing between 1-2 options across all phrasings. If it's 3.5, most options remain competitive.

**S18. Distribution Trajectory / Simplex Spread** ★ NOVEL
- The 11 probability vectors live on a 3-simplex (4 options, probabilities sum to 1). The *geometry* of how they spread on this simplex is informative.
- Metrics:
  - **Centroid distance**: mean distance from each query's vector to the centroid (mean vector). Large = high dispersion.
  - **Convex hull volume**: volume of the convex hull of the 11 points on the simplex. Captures the total "territory" the model's distributions cover.
  - **Directional spread**: is the variation along one axis (A vs B) or spread across all dimensions? PCA of the probability vectors — if the first PC explains 90% of variance, the model is oscillating between two options. If variance is spread across PCs, it's genuinely confused across all options.
- Novelty: **Novel.** Nobody characterises the geometry of per-question probability vector sets on the simplex.

---

## III. Position Sensitivity Signals (from Shuffle Conditions)

**These reframe answer shuffling from "bias mitigation" to "diagnostic instrument."** ★ NOVEL CONTRIBUTION

The shuffle conditions (same text, different answer orderings) isolate how much the model's answer depends on *where* options appear versus *what* they say.

### III.1 Per-Question Position Diagnostics

**S19. Position Loyalty Score** ★ NOVEL
- From the shuffle_nopara condition (same text, 11 different orderings).
- For each canonical answer option, compute the variance of its assigned probability across the 11 orderings. When the correct answer is shuffled from position A to position D, does its probability stay high (content-tracking) or does it drop (position-sensitive)?
- Aggregate: mean variance across all four canonical options, or focus on the correct answer specifically.
- Low position loyalty = model tracks content (genuine knowledge). High position loyalty = model's confidence depends on display position (heuristic-driven).
- Novelty: **Genuinely novel.** Zheng et al. (ICLR 2024) document position bias as a population-level phenomenon. We use per-question position sensitivity as an individual uncertainty signal.

**S20. Position Preference Profile** ★ NOVEL
- For each question in the shuffle condition, compute: across 11 orderings, what fraction of probability mass goes to each display position (1st, 2nd, 3rd, 4th) regardless of content?
- A model with strong position bias will show a consistent position preference profile (e.g., always favouring position A). A model tracking content will show no position pattern.
- Can be summarised as the entropy of the position-preference distribution: low entropy = strong position preference, high entropy = no position preference.

**S21. Content Loyalty Score**
- From the noshuffle_para condition (different text, same answer order).
- Variance of canonical_probs across paraphrases with positions held fixed.
- Isolates content sensitivity (framing effects) from position sensitivity.

### III.2 Factorial Decomposition

**S22. Position vs Content Attribution** ★ NOVEL
- Compare the variance of probability vectors in the shuffle_nopara condition (position variation only) against the noshuffle_para condition (content variation only).
- For each question, compute the ratio: `position_variance / (position_variance + content_variance)`.
  - Ratio near 1: inconsistency is mostly position-driven (model is relying on heuristics)
  - Ratio near 0: inconsistency is mostly content-driven (model has fragile knowledge)
  - Ratio near 0.5: both contribute equally
- The shuffle_para condition provides the combined variance — check whether it's approximately additive (position + content) or shows interaction effects.

### III.3 Why This Matters

This reframes the entire role of answer shuffling in UQ research. The current framing: "shuffle to mitigate bias, then analyse." Our framing: "the degree to which shuffling changes the answer is itself the most direct measure of whether the model is relying on knowledge or heuristics." The shuffle doesn't just fix a problem — it *measures* a problem. And the measurement is per-question, not per-model.

---

## IV. The 2D Uncertainty Space (Headline Analytical Contribution)

### IV.1 The Two Axes

**Axis 1 — Confidence:** How peaked is the model's aggregated distribution?
- Primary operationalisation: Mean confidence (S10).
- Alternatives: 1 - Total uncertainty (S11), aggregated second-choice gap (S15).

**Axis 2 — Consistency:** How stable is the distribution across paraphrases?
- Primary operationalisation: Agreement rate (S8) or 1 - epistemic uncertainty (S13).
- Alternatives: Rank stability (S16), 1 - confidence variance (S14).
- Analysis should compare all candidates and report which gives best separation.

### IV.2 Four Quadrants

| | High Consistency | Low Consistency |
|---|---|---|
| **High Confidence** | **Trustworthy** — sure and robust. Highest accuracy. | **Fragile Confidence** — looks reliable on one prompt but breaks under rephrasing. Most dangerous: only detectable via paraphrasing. |
| **Low Confidence** | **Honest Uncertainty** — consistently admits it doesn't know. Least dangerous. | **Thorough Confusion** — doesn't know and changes its mind with every phrasing. |

The key claim: **Fragile Confidence (high confidence, low consistency) is the failure mode that justifies the entire framework.** These are questions where a single prompt gives a confident-looking answer, but paraphrasing reveals the confidence is an artefact of that specific phrasing. Only the combination of signals catches this.

### IV.3 Analysis Plan

1. 2D scatter plot coloured by correctness. Visually demonstrate the four quadrants.
2. Per-quadrant accuracy table: n, accuracy, proportion of total.
3. **AUROC comparison table** — the paper's key quantitative result:
   - Each signal individually → AUROC
   - Logistic regression on Axis 1 + Axis 2 → combined AUROC
   - Logistic regression on all signals → ceiling AUROC
   - Show that the 2D combination beats every individual signal.
4. AUARC (selective prediction): if you refuse the most uncertain questions, how does remaining accuracy improve?
5. Calibration reliability diagrams (optional).

---

## V. Cross-Condition Signals (Comparing Across Experimental Conditions)

These require results from multiple experimental conditions (e.g., direct + CoT, or shuffle + noshuffle). They're computed per-question across conditions, not within a single condition.

**S23. Direct vs CoT Agreement** ★ NOVEL
- Does the model's answer change when it's allowed to reason? For each question, compare the final answer from the direct condition against the CoT condition.
- Questions where CoT flips the answer are inherently more uncertain — the model's immediate intuition and its reasoned conclusion disagree.
- Can also compare the confidence levels: does CoT increase or decrease confidence?

**S24. Cross-Mode Confidence Shift**
- `mean_confidence_CoT - mean_confidence_direct` for each question.
- Positive = CoT makes the model more confident. Negative = CoT makes it less confident.
- Hypothesis: questions where CoT reduces confidence are ones where reasoning reveals complications the model's "gut" didn't notice.

**S25. Sensitivity Profile Vector** ★ NOVEL
- For each question, construct a binary or continuous vector: [responds to shuffle?, responds to paraphrase?, responds to CoT?]
- This creates a per-question "fingerprint" of what kinds of perturbation affect the model's answer. Questions sensitive to everything are maximally uncertain. Questions sensitive to nothing are (probably) well-known.
- Can be used for clustering: do questions cluster into recognisable types by their sensitivity profile?

---

## VI. Stretch Goals (Implement If Time Allows)

### VI.1 CoT Reasoning Consistency
- Embedding similarity of reasoning traces across paraphrases (requires sentence-transformer model).
- Reasoning-answer alignment: does the trace support the same answer as the logprobs?
- Connection: "Deep Think with Confidence" (2025) aggregates trace confidence for selection, but we'd use trace consistency as an uncertainty signal.

### VI.2 Paraphrase Count Sensitivity
- Subsample from 10 paraphrases: compute all signals at N = 2, 3, 5, 7, 10.
- Plot AUROC as a function of N. Practical guidance: "how many paraphrases do you need?"
- Pure analysis on existing data. No new experiments.

### VI.3 Per-Subject Breakdown
- AUROC and signal distributions by MMLU subject (57 subjects).
- Where does UQ add the most value? Hypothesis: subjects at ~60-80% accuracy benefit most (ceiling/floor effects at extremes).
- Heatmap: subjects × signals → AUROC.

### VI.4 Temperature Sensitivity (Requires New Experiments)
- Run the baseline condition at T = 0.0, 0.3, 0.7, 1.0.
- Show how temperature affects both accuracy and signal quality.
- Low priority — only if all other analyses are complete.

### VI.5 Paraphrase Distance × Stability Curve
- For each question, plot answer stability (e.g., probability of correct answer, or agreement) against the semantic distance of each paraphrase from the original (stored as `embedding_similarity` in paraphrase data).
- Find the "breaking point": at what semantic distance does the model's answer start to change?
- Questions with low breaking points have fragile knowledge. Questions that remain stable across distant paraphrases have robust knowledge.
- Requires the embedding similarity scores from paraphrase generation (already stored).

### VI.6 Prompt Perplexity as Uncertainty Prior
- How surprising is the question itself to the model? High prompt perplexity = unfamiliar topic = prior expectation of higher uncertainty.
- Requires vLLM's `prompt_logprobs` or llama.cpp's `llama-perplexity` tool. Not available via Ollama currently.
- Would be a powerful signal if accessible — effectively measures "has the model seen this kind of content in training?"

### VI.7 Signal Redundancy Analysis
- Correlation matrix of ALL signals across all questions.
- PCA: how many independent dimensions of uncertainty are there really?
- Helps the paper's narrative: if 20 signals collapse to 3 principal components, the playbook's practical recommendation is simpler than it looks.

---

## Implementation Plan

### Per-question signal computation (`analysis.py`)

Extend the existing `compute_question_metrics()` or add a new `compute_full_signal_suite()` function that takes:
- `canonical_probs_list`: list of probability vectors (existing input)
- `raw_logprobs_list`: list of full top-20 logprob entries (new — for signals S5-S7, S17)
- `answer_permutations`: list of permutations per query (new — for signals S19-S22)

Returns a dict with ALL per-question signals.

### Cross-question analyses (separate scripts)

```
analysis/
  signals.py               Extended per-question signal computation
  auroc_comparison.py       AUROC table for every signal, 2D combination, logistic regression
  uncertainty_space.py      2D scatter plots, quadrant analysis, selective prediction curves
  position_analysis.py      Tier III position sensitivity decomposition
  cross_condition.py        Tier V cross-condition signals (needs multiple result files)
  sensitivity.py            Paraphrase count sensitivity (Tier VI.2)
  subject_breakdown.py      Per-subject analysis (Tier VI.3)
  redundancy.py             Signal correlation matrix, PCA (Tier VI.7)
```

### Data access pattern

Per-question results stored in JSON. Key fields per query:
```json
{
  "canonical_probs": [0.34, 0.25, 0.40, 0.01],
  "raw_logprobs": [{"token": " C", "logprob": -0.92, "top_logprobs": [...20 entries...]}],
  "answer_permutation": [2, 0, 3, 1],
  "display_letter_logprobs": {"A": -1.08, "B": -2.30, "C": -0.92, "D": -4.14},
  "canonical_logprobs": {0: -2.30, 1: -1.08, 2: -0.92, 3: -4.14},
  "thinking_trace": "..."
}
```

The `raw_logprobs[0]["top_logprobs"]` array contains all 20 tokens with their logprobs — this is what signals S5-S7 need.

---

## Master AUROC Comparison Table

The paper's key table. Every signal gets an AUROC for "can it separate correct from incorrect answers?"

| # | Signal | Source | Category | Novelty |
|---|--------|--------|----------|---------|
| S1 | MSP (single query) | noshuffle_nopara | Single-prompt | Standard |
| S2 | Single-query entropy | noshuffle_nopara | Single-prompt | Standard |
| S3 | Second-choice gap | noshuffle_nopara | Single-prompt | Minor |
| S4 | Distribution shape | noshuffle_nopara | Single-prompt | Novel framing |
| S5 | Answer coverage | noshuffle_nopara | Single-prompt (top-20) | **Novel** |
| S6 | Hesitation profile | noshuffle_nopara | Single-prompt (top-20) | **Novel** |
| S7 | Top-token identity | noshuffle_nopara | Single-prompt (top-20) | **Novel** |
| S8 | Agreement rate | multi-query | Vote-based | Standard |
| S9 | Vote entropy | multi-query | Vote-based | Minor |
| S10 | Mean confidence | multi-query | Aggregated logprob | Novel combination |
| S11 | Total uncertainty | multi-query | Information-theoretic | Established framework |
| S12 | Aleatoric uncertainty | multi-query | Information-theoretic | Established framework |
| S13 | Epistemic uncertainty (MI) | multi-query | Information-theoretic | Novel application |
| S14 | Confidence variance | multi-query | Stability | **Novel** |
| S15 | Aggregated 2nd-choice gap | multi-query | Aggregated logprob | Minor |
| S16 | Rank stability (tau) | multi-query | Ordinal | **Novel for MCQA** |
| S17 | Aggregated answer coverage | multi-query | Full vocabulary | **Novel** |
| S18 | Simplex spread (PCA) | multi-query | Geometric | **Novel** |
| S19 | Position loyalty | shuffle condition | Position diagnostic | **Novel** |
| S20 | Position preference entropy | shuffle condition | Position diagnostic | **Novel** |
| S21 | Content loyalty | para condition | Content diagnostic | Novel application |
| S22 | Position/content ratio | factorial | Decomposition | **Novel** |
| — | **2D combined** | multi-query | **Composite** | **Novel** |
| — | **All-signal combined** | all conditions | **Composite** | **Novel** |

**Goal:** Show that (a) the novel signals (S5, S14, S16, S19) contribute AUROC gains beyond standard signals, (b) the 2D combination beats any individual signal, and (c) adding position diagnostics provides further gain. The all-signal combined model sets the ceiling.

---

## Novelty Summary

**What exists and we replicate as baselines:**
- MSP, entropy, self-consistency/agreement (Kadavath 2022, Plaut 2024, Wang 2022)
- Epistemic/aleatoric decomposition framework (Hou et al. 2024)

**What exists in adjacent form but we apply differently:**
- Paraphrase-based UQ (SPUQ, Just Rephrase It) — but they're black-box, we add logprobs
- Logprob-based MCQA confidence (Plaut et al.) — but they use single prompts, we aggregate
- Position bias (Zheng et al. 2024) — population-level; we use per-question diagnostic
- Distribution shape analysis — used in calibration, not as per-question uncertainty signal

**What is genuinely new (not found in the literature review):**
- Off-label probability mass / answer coverage (S5) — examining non-answer tokens at the decision point
- Hesitation profile (S6) — semantic categorisation of non-answer tokens
- Position loyalty as per-question uncertainty (S19) — reframing shuffling as diagnostic
- 2D confidence × consistency space with quadrant analysis (Tier IV)
- Factorial decomposition of position vs content effects on uncertainty (S22)
- Distribution trajectory / simplex geometry (S18)

**What is new but could exist and we should double-check:**
- Confidence variance across paraphrases (S14) — suspiciously absent from literature, verify
- Rank stability via Kendall's tau for MCQA (S16) — verify not used elsewhere
- Cross-condition agreement as uncertainty signal (S23-S25) — verify
