# Brief: Melo et al. (2026) — Epistemic UQ to Improve Decisions from Black-Box Models

**Paper:** "Epistemic Uncertainty Quantification To Improve Decisions From Black-Box Models"
**Authors:** Sebastien Melo, Gael Varoquaux, Marine le Morvan (Inria Saclay)
**Venue:** ICLR 2026
**Code:** https://github.com/soda-inria/epistemic-uq-decision

## What They Did

Decomposed epistemic loss into **calibration loss + grouping loss (GL)** — the hidden within-bin heterogeneity that ECE completely misses. A model can be well-calibrated on average but systematically overconfident on some subgroups and underconfident on others, with the effects cancelling out.

Built asymptotically consistent estimators using honest regression trees (Wager & Athey, 2017) for:
1. **Grouping Loss** — variance of true probabilities within model confidence bins
2. **Pointwise excess decision risk** — per-sample epistemic risk score

Applied to LLM cascades: route queries to stronger models only when epistemic risk is high. Evaluated on Folktexts benchmark (tabular-to-text, binary classification) with 27 LLMs from 1B–70B params.

**Key results:**
- GL decreases with model scale and instruction tuning
- Cascade: +6% accuracy at 46% of cost vs largest model
- Full epistemic risk (CL+GL) vs calibration-only (CL): up to +15% accuracy on high-GL datasets

## How It Compares to Our Approach

| Dimension | Melo et al. | Our work |
|-----------|-------------|----------|
| **Detecting epistemic uncertainty** | Learned partition on input features (honest tree) | Permutation-based probe (shuffle answers, observe logprob variance) |
| **Data requirement** | Requires labeled data to build tree | Label-free — probes model's own response variance |
| **Confidence extraction** | Single query, binary logprobs P(yes)/P(no) | N queries, 4-way logprob vectors [P(A),P(B),P(C),P(D)] |
| **Aggregation** | Frequentist: partition-based residual estimation | Bayesian: posterior aggregation (Product, Dirichlet, MoM) |
| **Recalibration** | Platt scaling | Temperature scaling (T=3.0) |
| **Compute routing** | Cascade across model sizes (1B->70B) | Cascade across inference modes (direct->CoT->think) |
| **Task domain** | Tabular-to-text, binary classification (Folktexts) | Reading comprehension MCQ (QuALITY, 4-way) |

**Same core insight, different detection method.** Their GL captures exactly what our permutation-based variance captures: the model assigns similar confidence to inputs where true probabilities differ. Two independent approaches converging on the same phenomenon.

**Complementary, not overlapping.** They need labeled data and input features to build their tree; we need no labels and extract uncertainty purely from the model's own response variance. The approaches could be combined.

## Learnings for Us

1. **Theoretical vocabulary.** Their decomposition (Epistemic = Calibration + Grouping) provides formal grounding for our empirical finding that per-query ECE=0.186 but aggregated MSP is well-calibrated. We can cite this to frame our temperature scaling as addressing grouping loss.

2. **"Calibration is insufficient" is now an ICLR-published claim.** Strengthens our argument that single-query confidence is misleading and multi-query probing reveals hidden uncertainty.

3. **Cascade comparison point.** Their +6% accuracy at 46% cost is a natural comparison for our adaptive stopping results (avg_N=2.76/10, accuracy maintained, ~15% escalated to think). Different compute-saving strategies, similar motivation.

4. **Instruction tuning overconfidence.** They confirm what we observe with CoT/think scaffolding absorption — instruction-tuned models produce overconfident logprobs. Our two-pass pipeline is a different solution to the same problem.

5. **References to chase:**
   - Chen et al. (2024) "Reconfidencing LLM Uncertainty from the Grouping Loss Perspective" (EMNLP 2024) — precursor paper, same group
   - Bickford Smith et al. (2025) "Rethinking aleatoric and epistemic uncertainty" (ICML) — criticizes common UQ definitions
   - Shorinwa et al. (2025) and Xia et al. (2025) — surveys on UQ for LLMs

6. **Their limitation is our strength.** Binary classification only, requires labeled data, needs input features for the tree. Our permutation approach works on multi-class, requires no labels, and uses no input features — just the model's own outputs. Worth highlighting in related work.
