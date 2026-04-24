# Analysis: Adaptive Sampling Results (2026-04-24)

**Source:** `paper/results/adaptive_sampling_export.json` — 3 models x 5 methods x 9 temperatures x 7 tau values, with think-mode escalation (cost=7.1x).

## Summary of Optimal Settings

| Model | Method | T* | tau* | acc_esc | avg_N | cap_rate |
|-------|--------|---:|-----:|--------:|------:|---------:|
| **Gemma 4** | Product | 5.0 | 0.99 | 77.0% | 3.65 | 8.9% |
| | Sum | 4.5 | 0.70 | 77.0% | 3.99 | 11.6% |
| | Dirichlet MLE | 5.0 | 0.80 | 77.0% | 3.35 | 13.0% |
| | MoM | 3.0 | 0.99 | 76.9% | 3.79 | 21.5% |
| | MoM+Bayes | 5.0 | 0.95 | 77.0% | 4.17 | 23.4% |
| **Qwen 3** | Product | 5.0 | 0.99 | 77.1% | 5.67 | 19.6% |
| | Sum | 2.0 | 0.85 | 77.1% | 5.84 | 19.7% |
| | Dirichlet MLE | 5.0 | 0.85 | 77.2% | 4.74 | 26.2% |
| | MoM | 3.0 | 0.99 | 77.0% | 4.45 | 29.0% |
| | MoM+Bayes | 3.5 | 0.99 | 77.1% | 6.18 | 42.9% |
| **Qwen 3.5** | Product | 2.5 | 0.90 | 83.9% | 4.16 | 11.6% |
| | Sum | 2.0 | 0.60 | 83.8% | 4.69 | 17.1% |
| | Dirichlet MLE | 1.5 | 0.80 | 83.9% | 3.57 | 14.7% |
| | MoM | 5.0 | 0.90 | 83.9% | 3.78 | 20.1% |
| | MoM+Bayes | 1.5 | 0.85 | 83.9% | 3.60 | 15.9% |

Baselines: Gemma 4 = 76.8%, Qwen 3 = 76.4%, Qwen 3.5 = 83.3%.

## Key Findings

### 1. All methods converge to similar acc_esc

The most striking finding: **all 5 methods achieve nearly identical accuracy+escalation** for each model. Gemma 4 and Qwen 3 cluster at 77.0-77.2%, Qwen 3.5 at 83.8-83.9%. The differences are in the *path* — how many queries needed and how many escalated — not the destination.

This suggests the information ceiling is set by the data and model, not the aggregation method. The methods differ in *efficiency* (how quickly they reach confidence) not *capability* (what accuracy they can reach).

### 2. Optimal temperature varies wildly across methods — and that's a concern

**The T* values are suspiciously high and inconsistent:**

- Product consistently wants T*=5.0 for Gemma/Qwen 3, but only 2.5 for Qwen 3.5
- Sum wants T*=4.5 for Gemma 4 but only 2.0 for Qwen 3 and Qwen 3.5
- MLE hits the ceiling at T*=5.0 for Gemma/Qwen 3 but needs only T*=1.5 for Qwen 3.5
- MoM wants T*=3.0 for Gemma/Qwen 3 but T*=5.0 for Qwen 3.5 (inverted!)
- MoM+Bayes: T*=5.0 for Gemma, T*=3.5 for Qwen 3, T*=1.5 for Qwen 3.5

**Concern: T*=5.0 means the Pareto optimizer is hitting the grid boundary.** We searched T in [1.0, 5.0] and multiple methods landed at the ceiling. This means the true optimum might be even higher — the search space may be too narrow. But T=5.0 is *extreme* temperature scaling. At T=5, a logprob vector of [0.9, 0.05, 0.03, 0.02] becomes approximately [0.37, 0.22, 0.20, 0.21] — nearly uniform. This raises the question: are we solving overconfidence or destroying signal?

**Interpretation:** The T* values are high because the Pareto optimization maximizes acc_esc, which rewards escalation (sending uncertain questions to think mode). High T makes the posterior *more* uncertain, triggering more escalation, which is beneficial when think mode is substantially more accurate. The optimizer is essentially learning to escalate aggressively.

**Why Qwen 3.5 is different:** Qwen 3.5 is fundamentally more accurate (83.3% baseline vs 76.4-76.8%). Its logprobs are more informative, so less temperature softening is needed. The model's native confidence is already closer to well-calibrated. This is consistent with our earlier finding that better models need less temperature correction.

### 3. MoM vs MoM+Bayes — the hypothesis partially holds

The original hypothesis: MoM and MoM+Bayes should need less temperature scaling because concentration estimation already accounts for overconfidence.

**Results:**
- MoM: T*=3.0 for Gemma/Qwen 3 (lower than Product/MLE at 5.0) — **hypothesis supported**
- MoM+Bayes: T*=5.0 for Gemma, T*=3.5 for Qwen 3 — **mixed**
- For Qwen 3.5: MoM goes high (T*=5.0) while MoM+Bayes goes low (T*=1.5) — **reversed!**

MoM's variance-based approach does seem to handle overconfidence more naturally (lower T* on weaker models). But MoM+Bayes doesn't consistently behave as expected — the Bayesian marginalization over concentration doesn't reliably reduce T* dependence.

### 4. Compute efficiency: Product wins on Gemma 4, MLE/MoM+Bayes win on Qwen 3.5

**Lowest avg_N at optimal settings (fewest queries before stopping):**
- Gemma 4: MLE (3.35), Product (3.65), MoM (3.79)
- Qwen 3: MoM (4.45), MLE (4.74), Product (5.67)
- Qwen 3.5: MLE (3.57), MoM+Bayes (3.60), MoM (3.78)

MLE and MoM consistently stop early. Product is fast on Gemma 4 but slower on Qwen 3 (5.67 queries — almost 60% of n_max). Sum is consistently the least efficient (highest avg_N).

### 5. Think-mode escalation adds 0.3-0.7% accuracy

Across all models and methods, the gap between base accuracy and +think accuracy is small:
- Gemma 4: base ~76.5-76.6% -> +think ~77.0% (delta +0.4%)
- Qwen 3: base ~76.2-76.5% -> +think ~77.0-77.2% (delta +0.5-0.7%)
- Qwen 3.5: base ~83.3% -> +think ~83.6-83.9% (delta +0.3-0.6%)

The escalation benefit is modest but consistent. Think mode helps most when cap_rate is high (more questions escalated).

### 6. Raw (T=1) vs Calibrated — dramatic difference

At T=1, Product barely stops at all: avg_N=1.14-2.27, cap_rate<3%. The raw logprobs are so overconfident that Product exceeds the tau threshold on the first query for nearly every question. It's making fast but poorly-informed decisions.

After calibration (T=5.0), Product slows to avg_N=3.65-5.67 and escalates 9-20% of questions — much more deliberate stopping that leads to better accuracy+escalation outcomes.

## Recommendations

1. **Extend temperature grid to T=7 or T=10.** Multiple methods are hitting the T=5.0 ceiling. We need to know if the true optimum is beyond our search space. If it is, that changes the interpretation significantly.

2. **Report results at fixed tau (e.g., tau=0.90) across methods** rather than at Pareto-optimal (T*, tau*). The current optima pick different tau values per method, making comparison unfair. At tau=0.90, we can compare how much temperature helps each method.

3. **MLE is the most efficient method** in terms of avg_N for comparable accuracy. It stops earliest across all three models. This should be highlighted.

4. **Product's extreme T* is an artifact of the optimization target.** When we optimize for acc_esc with expensive think-mode, the optimizer *wants* to escalate — and high T drives escalation. Consider optimizing for a different objective that penalizes unnecessary escalation more heavily.

5. **Qwen 3.5's lower T* values validate the framework.** A better-calibrated model needs less correction. This is the expected behavior and suggests temperature scaling is doing something meaningful, not just gaming the metric.
