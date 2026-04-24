# The Numbskull Spud's Guide to Posterior Aggregation

*Five methods for deciding when you've heard enough — from playground intuition to Bayesian marginalization.*

---

# Level 1: ELI10

## The setup: why ask the same question ten times?

Imagine you have a friend who's really good at pub quizzes but has a weird habit: sometimes they blurt out the wrong answer even when they know the right one. Not because they're lying — they just get flustered, or the way you phrased the question threw them off, or they focused on a red herring in the question. But if you ask them the same question several times — maybe rewording it slightly each time, or listing the multiple-choice options in a different order — their *pattern* of answers tells you much more than any single answer does.

That's what we're doing with an AI model. We give it a reading comprehension question (like "What was the main reason the colony failed?") with four answer choices: A, B, C, D. But each time we ask, we shuffle the order of those four choices. The model doesn't just say "A" — it gives us a confidence score for every option. Something like: "I'm 90% sure it's A, 5% B, 3% C, 2% D."

Here's the problem: **the model is a showoff**. When it says "90% sure," it's actually only right about 75% of the time. It's overconfident on every single answer. If you trusted that 90% at face value, you'd think you barely need to ask twice. But the model is wrong a quarter of the time it says that — that's terrible.

So we ask multiple times, collect these confidence scores, and try to figure out: **do I have enough evidence to trust the answer, or should I keep asking?** And if the model stays confused after many tries, we escalate — we let it "think out loud" using a more expensive, slower reasoning mode.

The question is: how do you combine those multiple sets of confidence scores into a single "how sure am I?" number? That's what the five methods do.

## Method 1: Product (Multiply everything together)

Think of it like a jury trial. Each time you ask the model, it's like calling a new witness. The first witness says "I'm pretty sure it's A." The second witness independently says "I'm also pretty sure it's A." Each new witness who agrees makes you more confident.

Mathematically, you multiply the confidence scores together. If two witnesses each say 80% chance it's A, the combined evidence is stronger than either alone.

**The catch:** Remember, each witness is overconfident. If you naively multiply 90% × 90%, you get an absurdly high combined confidence after just two witnesses, and you stop investigating way too early. So we apply a "humility filter" (called temperature scaling) that deflates each witness's confidence before multiplying. Instead of 90%, each witness effectively says something like 60%. Now you need to hear from several witnesses before you're convinced. The system actually *works* — it stops early on easy questions (maybe 2-3 asks) and keeps going on hard ones.

**The worry:** That humility filter has a dial (the temperature T). We set it by testing on practice questions. If the model changes, or the questions are different from the practice ones, the dial might be wrong.

## Method 2: Sum (Stack up the votes)

Instead of multiplying, just add. Each time the model says "90% A, 5% B, 3% C, 2% D," toss those numbers into a running tally. After three asks, your tally for A might be 1 + 0.9 + 0.85 + 0.92 = 3.67, while B's tally is 1 + 0.05 + 0.08 + 0.04 = 1.17. (We start each tally at 1 so nobody starts at zero.) The bigger the gap between the leader and the rest, the more confident we are.

**Good news:** No humility filter needed. Adding up doesn't cause the explosive overconfidence that multiplying does.

**Bad news:** Sum is blind to *consistency*. Here's a thought experiment with a simpler two-choice question (A or B):
- Scenario 1: You ask twice, and both times the model says "50% A, 50% B." Tally: A=3, B=3.
- Scenario 2: You ask twice, and the first time it says "100% A, 0% B," and the second time "0% A, 100% B." Tally: A=3, B=3.

Same tallies! But these are completely different situations. In Scenario 1, the model genuinely doesn't know. In Scenario 2, the model is confidently contradicting itself — something is very wrong. Sum can't tell the difference.

## Method 3: Dirichlet MLE (Fit the pattern)

This one tries to figure out: what *kind* of answerer is the model acting like? Imagine watching someone answer the same question ten times. Are they:
- (a) Always giving roughly the same confident answer? → Reliable, trust them.
- (b) Always giving wishy-washy "I dunno" answers? → Unsure, but at least consistent.
- (c) Wildly swinging between confident answers? → Unreliable, don't trust them.

The MLE method looks at all N answers and fits a mathematical model that captures both the *average* answer and the *consistency*. It's like saying "this person acts like a 'type (a)' answerer with 85% tendency toward A" or "this person acts like a 'type (c)' answerer who's all over the place."

**Good news:** Can distinguish consistent from inconsistent — unlike Sum.

**Bad news:** With only 2-3 asks, there's not enough data to reliably fit the pattern. It's like trying to judge someone's personality after meeting them once.

## Method 4: MoM (Quick consistency check)

A simpler version of Method 3. Instead of fitting a full mathematical model, it just asks one question: **how much do the answers vary?**

If the model gives almost the same confidence scores every time, the variance is low → consistent → trust the average. If the scores bounce around wildly, variance is high → inconsistent → don't trust it.

It's a one-line formula. Fast and simple. But like MLE, with only 2-3 data points, the variance estimate is noisy — you might get unlucky and think the model is consistent when it's not (or vice versa).

## Method 5: MoM + Bayes (Honest about what you don't know)

This is Method 4 wearing a seatbelt. It computes the same variance as MoM, but then asks: "Given that I've only seen N answers, how *sure* am I about this variance estimate?"

With 2 observations, the answer is: not sure at all. The variance could plausibly be anywhere. So the method considers *all possible* consistency levels, weighted by how likely each one is given the data. This blurs out the confidence estimate — with little data, it stays cautious. With lots of data, it converges to plain MoM.

**This is the key idea: uncertainty about uncertainty.** We're not just uncertain about the *answer* — we're uncertain about *how uncertain we are*. Method 5 is the only one that accounts for both layers. For an adaptive stopping system — where stopping too early on flimsy evidence is dangerous — this conservatism is exactly what you want.

## Why five methods?

Each method makes a different trade-off:

| Method | Core idea | Main weakness |
|---|---|---|
| **Product** | Multiply likelihoods, Bayesian updating | Needs a calibration dial (temperature) |
| **Sum** | Add votes, simple accumulation | Blind to consistency vs. contradiction |
| **MLE** | Fit the full pattern from data | Noisy with few observations |
| **MoM** | Quick variance check | Treats noisy variance as certain |
| **MoM+Bayes** | Variance check + honest about uncertainty | Most complex |

The concentration parameter — a single number capturing "how consistent and confident is the model?" — is the thread connecting the last four methods. Product sidesteps it with temperature. The rest try to estimate it, with increasing sophistication.

---

# Level 2: ELI15

## Probability vectors and the overconfidence problem

When our model answers a 4-choice MCQ, it produces a **probability vector**: four non-negative numbers that sum to 1. For example:

$$\mathbf{p} = [0.91,\ 0.05,\ 0.03,\ 0.01]$$

This says: "91% confident it's answer A, 5% for B, 3% for C, 1% for D." We extract this from the model's logprobs — the log-probabilities it assigns to each answer token.

We shuffle the answer labels and ask again. After 3 queries, we might have (all mapped back to canonical order):

| Query | P(A) | P(B) | P(C) | P(D) |
|-------|-------|-------|-------|-------|
| 1 | 0.91 | 0.05 | 0.03 | 0.01 |
| 2 | 0.85 | 0.08 | 0.04 | 0.03 |
| 3 | 0.88 | 0.06 | 0.04 | 0.02 |

The model is consistently pointing at A, with high confidence each time. The **maximum softmax probability (MSP)** — the largest value in each row — averages about 0.91 across our dataset. But the model's accuracy is only ~75%. This gap is the **overconfidence problem**, measured by Expected Calibration Error (ECE = 0.186).

However, the **mean** probability vector $\bar{\mathbf{p}} = [0.88, 0.063, 0.037, 0.02]$ is much better calibrated. When the mean says 88% for A, the model is right close to 88% of the time. The mean across shuffled permutations absorbs the position-bias noise that inflates individual query confidence.

So: individual queries are unreliable narrators, but their aggregate tells the truth. The five methods are five ways to perform that aggregation while also producing a **stopping criterion** — a number between 0 and 1 that says "I'm this confident the leader is correct."

## Method 1: Product (Temperature-Calibrated Bayesian Posterior)

Start with a uniform prior: each answer is equally likely, $P(k) = 0.25$ for $k \in \{A, B, C, D\}$.

Each query's probability vector $\mathbf{p}^{(n)}$ is treated as a likelihood. Bayes' rule says:

$$P(k \mid \text{data}) \propto P(k) \times \prod_{n=1}^{N} p_k^{(n)}$$

In practice, we work in log-space to avoid underflow:

$$\log P(k \mid \text{data}) = \log P(k) + \sum_{n=1}^{N} \log p_k^{(n)} + \text{const}$$

Using our three example vectors:

| | log P(A) | log P(B) | log P(C) | log P(D) |
|---|---|---|---|---|
| Prior | -1.386 | -1.386 | -1.386 | -1.386 |
| +Query 1 | -0.094 | -3.00 | -3.51 | -4.61 |
| +Query 2 | -0.163 | -2.53 | -3.22 | -3.51 |
| +Query 3 | -0.128 | -2.81 | -3.22 | -3.91 |
| **Sum** | **-1.771** | **-9.726** | **-11.346** | **-13.416** |

After normalizing: $P(A \mid \text{data}) \approx 0.99985$. After just 3 queries, the posterior is already 99.98% sure it's A. The stopping threshold $\tau = 0.95$ would have been crossed after query 1.

This is the problem: overconfident inputs produce an overconfident posterior that converges immediately. Adaptive stopping becomes meaningless — everything stops at N=1.

**Temperature scaling** fixes this. We replace each $p_k^{(n)}$ with:

$$\tilde{p}_k^{(n)} = \frac{\exp(\log p_k^{(n)} / T)}{\sum_j \exp(\log p_j^{(n)} / T)}$$

With $T = 3.0$, that first vector $[0.91, 0.05, 0.03, 0.01]$ becomes approximately $[0.53, 0.18, 0.15, 0.14]$. The confidence drops from 91% to 53%. Now the posterior needs multiple consistent observations before reaching $\tau$.

At $T = 3.0$ (ECE-optimal), the adaptive framework works: easy questions stop after 2-3 queries (avg $N = 2.76$), hard questions use all 10, and about 15% of the hardest questions get escalated to a more expensive reasoning mode.

## Method 2: Sum (Dirichlet Pseudo-Counts)

Instead of multiplying, we add. Start with a Dirichlet prior $\boldsymbol{\alpha} = (1, 1, 1, 1)$ — a uniform distribution over probability vectors. Each observation adds fractional counts:

$$\boldsymbol{\alpha} \leftarrow \boldsymbol{\alpha} + \mathbf{p}^{(n)}$$

After our 3 queries:

$$\boldsymbol{\alpha} = (1 + 0.91 + 0.85 + 0.88,\ 1 + 0.05 + 0.08 + 0.06,\ 1 + 0.03 + 0.04 + 0.04,\ 1 + 0.01 + 0.03 + 0.02)$$
$$= (3.64,\ 1.19,\ 1.11,\ 1.06)$$

The total $S = \sum \alpha_k = 7.0$. The expected value $E[X_k] = \alpha_k / S$ gives $(0.52, 0.17, 0.16, 0.15)$ — much softer than the raw average, because the prior (adding 1 to each) pulls toward uniform.

The confidence metric isn't just $\max(E[X_k])$. It's the **exceedance probability**: given this Dirichlet, what's the probability that the leading component is truly the largest? This requires integrating over the Dirichlet distribution, which the copula approximation handles (details in Level 3).

**The consistency blindness problem.** Consider a 2-choice simplification:

- Scenario A: Two observations of $[0.5, 0.5]$ → $\boldsymbol{\alpha} = (2.0,\ 2.0)$
- Scenario B: One observation of $[1.0, 0.0]$, one of $[0.0, 1.0]$ → $\boldsymbol{\alpha} = (2.0,\ 2.0)$

Identical $\boldsymbol{\alpha}$. Yet Scenario A represents genuine ignorance (model consistently unsure), while Scenario B represents contradiction (model confidently disagrees with itself). This is a fundamental limitation of any method that only tracks running sums.

## Method 3: Dirichlet MLE (Minka 2000)

Instead of *assuming* how evidence accumulates (multiplicatively or additively), MLE asks: "What Dirichlet distribution would have been most likely to generate these N probability vectors?"

Given observations $\{\mathbf{p}^{(1)}, \ldots, \mathbf{p}^{(N)}\}$, find $\boldsymbol{\alpha}$ that maximises:

$$\log L(\boldsymbol{\alpha}) = N \left[ \log \Gamma\!\left(\sum_k \alpha_k\right) - \sum_k \log \Gamma(\alpha_k) \right] + \sum_k (\alpha_k - 1) \sum_{n=1}^{N} \log p_k^{(n)}$$

The key quantity is the **sufficient statistic**: $\bar{\ell}_k = \frac{1}{N}\sum_n \log p_k^{(n)}$, the average of log-probabilities for each component.

For our example: $\bar{\ell}_A = \frac{1}{3}(\log 0.91 + \log 0.85 + \log 0.88) = -0.128$. This is close to zero, meaning the model consistently assigns high probability to A. Meanwhile $\bar{\ell}_B = -2.78$, $\bar{\ell}_C = -3.32$, $\bar{\ell}_D = -4.01$ — the non-A components are consistently low.

Minka's algorithm iteratively updates $\boldsymbol{\alpha}$ using the digamma function $\psi$:

$$\alpha_k^{\text{new}} = \psi^{-1}\!\left(\psi\!\left(\sum_j \alpha_j\right) + \bar{\ell}_k\right)$$

After convergence, the fitted concentration $\alpha_0 = \sum \alpha_k$ tells us about consistency:
- High $\alpha_0$: the N vectors cluster tightly — model is consistent.
- Low $\alpha_0$: the N vectors are spread out — model is inconsistent.

The confidence metric is again copula exceedance on the fitted $\boldsymbol{\alpha}$.

**With regularisation:** At $N = 2$, the MLE can be unstable. We add a uniform pseudo-observation (prior_strength=1.0) and label-smooth the inputs ($\epsilon = 10^{-3}$) to keep the log-likelihood finite.

## Method 4: MoM (Method of Moments)

Rather than iterating to find $\boldsymbol{\alpha}$, MoM uses a direct formula based on the variance of the observations.

For a Dirichlet with concentration $\alpha_0$ and mean $\boldsymbol{\mu}$:

$$\text{Var}(X_k) = \frac{\mu_k(1 - \mu_k)}{\alpha_0 + 1}$$

The ratio $R = \frac{\text{Var}(X_k)}{\mu_k(1 - \mu_k)} = \frac{1}{\alpha_0 + 1}$ is the same for all components. We estimate it with the **pooled variance ratio**:

$$\hat{R} = \frac{\sum_k s_k^2}{\sum_k \hat{\mu}_k(1 - \hat{\mu}_k)}$$

where $s_k^2 = \frac{1}{N-1}\sum_n (p_k^{(n)} - \hat{\mu}_k)^2$ is the sample variance and $\hat{\mu}_k$ is the sample mean.

For our example:
- $\hat{\mu}_A = 0.88$, $s_A^2 = \frac{1}{2}[(0.91-0.88)^2 + (0.85-0.88)^2 + (0.88-0.88)^2] = 0.0009$
- Numerator (all components): $\sum s_k^2 = 0.0009 + 0.00023 + 0.00003 + 0.0001 = 0.00127$
- Denominator: $\sum \hat{\mu}_k(1-\hat{\mu}_k) = 0.88 \times 0.12 + 0.063 \times 0.937 + 0.037 \times 0.963 + 0.02 \times 0.98 = 0.221$
- $\hat{R} = 0.00127 / 0.221 = 0.00574$
- $\hat{\alpha}_0 = (1 - 0.00574) / 0.00574 \approx 173$

A concentration of 173 is very high — the model is extremely consistent across these 3 queries. This makes intuitive sense: $[0.91, 0.05, 0.03, 0.01]$ and $[0.85, 0.08, 0.04, 0.03]$ are nearly identical.

Then $\boldsymbol{\alpha} = \hat{\alpha}_0 \cdot \hat{\boldsymbol{\mu}} = (152.2, 10.9, 6.4, 3.5)$, and exceedance probability on this Dirichlet is essentially 1.0.

**The pooling trick matters.** If we tried to estimate $R$ for each component separately, component D ($\hat{\mu}_D = 0.02$) would give a ratio dominated by noise — dividing a tiny variance by a tiny denominator. By summing both numerator and denominator across components, the near-zero components contribute proportionally little, and the estimate is driven by components with actual signal.

## Method 5: MoM + Bayes (Bayesian Wrapper)

MoM gave us $\hat{\alpha}_0 = 173$. But should we trust that number based on 3 observations?

The sample variance from $N = 3$ has $N - 1 = 2$ degrees of freedom per component. With $K = 4$ components pooled, that's $K(N-1) = 8$ degrees of freedom total. A chi-squared distribution with 8 df has a 95% confidence interval ratio of about 4:1. Our point estimate could easily be off by a factor of 2 in either direction.

MoM+Bayes puts a prior on $\alpha_0$ — a Gamma(shape=2, scale=5) distribution, which peaks around $\alpha_0 = 5$ and has mean 10. This says: "before seeing any data, I expect models to be moderately consistent." Then it computes:

$$P(\alpha_0 \mid \hat{R}, N) \propto \underbrace{L(\hat{R} \mid \alpha_0)}_{\text{chi-squared likelihood}} \times \underbrace{\pi(\alpha_0)}_{\text{Gamma prior}}$$

on a grid of 80 possible $\alpha_0$ values from 0.5 to 300. For each grid point, it computes the copula exceedance as if that were the true $\alpha_0$. The final confidence is the **weighted average** of these exceedances:

$$P(\text{leader best}) = \sum_g \text{exc}(\alpha_{0,g} \cdot \hat{\boldsymbol{\mu}}) \times P(\alpha_{0,g} \mid \hat{R}, N) \times \Delta_g$$

At $N = 2$ (just 4 df), the posterior on $\alpha_0$ is wide — many values are plausible — so the exceedance gets blurred toward 0.5 (uncertainty). At $N = 10$ (36 df), the posterior is tight and converges to plain MoM.

This is "uncertainty about uncertainty": even if $\hat{R}$ suggests high consistency, the method knows it might be a fluke when $N$ is small, and hedges accordingly.

## Comparison

| Method | What it estimates | Hyperparameters | Handles overconfidence via | Consistency-aware? | Uncertainty-aware? |
|---|---|---|---|---|---|
| **Product** | $P(k \mid \text{data})$ via Bayes' rule | Temperature $T$ | Temperature scaling | No (relies on calibration) | No |
| **Sum** | Dirichlet pseudo-count accumulation | None | Linear accumulation | No (blind to dispersion) | No |
| **MLE** | MLE of Dirichlet $\boldsymbol{\alpha}$ | Regularisation strength | Fits concentration from data | Yes ($\alpha_0$) | No (point estimate) |
| **MoM** | $\alpha_0$ via variance ratio | Clamp bounds | Estimates $\alpha_0$ directly | Yes | No (point estimate) |
| **MoM+Bayes** | Posterior over $\alpha_0$ | Prior shape/scale | Marginalises over $\alpha_0$ | Yes | Yes |

---

# Level 3: Full Technical

## Notation and setup

We have $K = 4$ answer choices. A single query to the model produces a probability vector $\mathbf{p}^{(n)} \in \Delta^{K-1}$, the $(K{-}1)$-simplex, extracted from first-token logprobs and mapped to canonical label order. We collect $N$ such vectors (typically $N_{\max} = 10$) under shuffled label orderings. Define:

- $\hat{\mu}_k = \frac{1}{N}\sum_{n=1}^{N} p_k^{(n)}$ — sample mean
- $s_k^2 = \frac{1}{N-1}\sum_{n=1}^{N}(p_k^{(n)} - \hat{\mu}_k)^2$ — sample variance
- $\bar{\ell}_k = \frac{1}{N}\sum_{n=1}^{N}\log p_k^{(n)}$ — mean log-probability (sufficient statistic for Dirichlet MLE)

The stopping criterion takes the form: compute a confidence metric $c_N \in [0, 1]$ after each query $n = 1, \ldots, N$. If $c_N > \tau$, stop and return the leading answer. If $c_N \leq \tau$ after $N_{\max}$ queries, escalate.

## Temperature scaling (used by Product)

The raw logprob vector $\mathbf{p}$ is overconfident: MSP $\approx 0.91$, accuracy $\approx 0.75$, ECE $= 0.186$. Temperature scaling applies:

$$\tilde{p}_k = \frac{\exp(\log p_k / T)}{\sum_j \exp(\log p_j / T)} = \frac{p_k^{1/T}}{\sum_j p_j^{1/T}}$$

This is equivalent to raising each probability to the power $1/T$ and renormalising. For $T > 1$, the distribution is flattened; for $T < 1$, sharpened. The effect on entropy:

$$H(\tilde{\mathbf{p}}) = \frac{1}{T}H_T(\mathbf{p}) + \log Z_T$$

where $H_T(\mathbf{p}) = -\sum_k p_k^{1/T} \log p_k$ is a generalised entropy and $Z_T = \sum_k p_k^{1/T}$. For $T > 1$, entropy increases — uncertainty is honestly reported.

Optimal $T$ minimises ECE on a calibration set. For our QuALITY dataset with Qwen 3 8B (direct + shuffle), $T^* = 3.0$ achieves ECE $= 0.012$ (from 0.186). This is found via grid search over $T \in [1, 10]$.

**KL divergence impact:** Temperature scaling reduces the KL divergence between the model's prediction and the true posterior:

$$D_{\text{KL}}(\delta_y \| \tilde{\mathbf{p}}) = -\log \tilde{p}_y$$

where $y$ is the true label. Since $\tilde{p}_y < p_y$ for the overconfident leading component, the penalty for errors *decreases* less than the penalty for correct answers, improving calibration.

## Method 1: Product — Bayesian sequential updating

### Derivation

Treat each query as an independent observation (conditional on the question). The posterior after $N$ observations under a $\text{Dir}(\boldsymbol{\beta})$ prior over the categorical parameter $\boldsymbol{\theta}$, with each observation modelled as a categorical draw with parameter $\tilde{\mathbf{p}}^{(n)}$... but this isn't quite right. The observations aren't categorical *draws* — they're full probability vectors used as *likelihoods*.

More precisely, we define the update rule:

$$\log P(k \mid \mathbf{p}^{(1:N)}) = \log P(k) + \sum_{n=1}^{N} \log \tilde{p}_k^{(n)} - \log Z_N$$

where $Z_N = \sum_j \exp\left(\log P(j) + \sum_n \log \tilde{p}_j^{(n)}\right)$ is the normalising constant. With uniform prior $P(k) = 1/K$:

$$P(k \mid \mathbf{p}^{(1:N)}) = \frac{\prod_n \tilde{p}_k^{(n)}}{\sum_j \prod_n \tilde{p}_j^{(n)}}$$

This is a **product of experts** model, where each query acts as an independent expert with softmax likelihood. The temperature-scaled version ensures no single expert dominates.

### Stopping criterion

$$c_N^{\text{prod}} = \max_k P(k \mid \mathbf{p}^{(1:N)})$$

### Rate of concentration

For the leading answer $k^*$ with true mean $\mu_{k^*}$ after temperature scaling:

$$\log P(k^* \mid \mathbf{p}^{(1:N)}) \approx N \cdot \mathbb{E}[\log \tilde{p}_{k^*}] - \log Z_N$$

The gap between the leader and the runner-up grows linearly in $N$ (in log-space), so the posterior concentrates *exponentially* in $N$. The temperature controls the *rate* — higher $T$ means slower concentration, requiring more queries.

## Method 2: Sum — Dirichlet pseudo-count accumulation

### Model

Treat the observations as fractional pseudo-counts for a Dirichlet posterior. Starting from $\text{Dir}(\mathbf{1})$ (uniform prior on the simplex):

$$\boldsymbol{\alpha}^{(N)} = \mathbf{1} + \sum_{n=1}^{N} \mathbf{p}^{(n)}$$

This is interpretable as a Dirichlet-Multinomial model where each observation contributes fractional counts proportional to the probability vector, rather than a single count at the MAP category.

The total concentration is $S_N = K + N$ (since $\sum_k p_k^{(n)} = 1$ for each observation). Evidence accumulates at rate 1 per query — linear, not exponential. This is why Sum doesn't need temperature calibration: the posterior can't concentrate faster than $O(N)$.

### Copula exceedance (stopping criterion)

The confidence metric is the probability that the leading component $\theta_{k^*}$ exceeds all others under $\text{Dir}(\boldsymbol{\alpha})$:

$$c_N^{\text{sum}} = P(\theta_{k^*} > \theta_j\ \forall j \neq k^*) = \int_{\Delta^{K-1}} \mathbf{1}[\theta_{k^*} > \max_{j \neq k^*} \theta_j]\ \text{Dir}(\boldsymbol{\theta} \mid \boldsymbol{\alpha})\ d\boldsymbol{\theta}$$

**Exact computation** is intractable for $K > 2$. We use a **damped Gaussian-copula approximation** (Luigi's method):

**Step 1: Pairwise Beta exceedances.** For each competitor $j \neq k^*$, the marginal comparison $\theta_{k^*} > \theta_j$ involves a bivariate marginal of the Dirichlet. Using the well-known result that for $\text{Dir}(\boldsymbol{\alpha})$, the ratio $\theta_{k^*}/(\theta_{k^*} + \theta_j) \sim \text{Beta}(\alpha_{k^*}, \alpha_j)$:

$$p_j = P(\theta_{k^*} > \theta_j) = 1 - I_{0.5}(\alpha_{k^*}, \alpha_j)$$

where $I_x(a, b)$ is the regularised incomplete Beta function.

**Step 2: Probit transform.** Convert each pairwise exceedance to a standard normal quantile:

$$a_j = \Phi^{-1}(p_j)$$

**Step 3: Dirichlet-implied correlations.** The events $\{\theta_{k^*} > \theta_j\}$ and $\{\theta_{k^*} > \theta_l\}$ are positively correlated because they share $\theta_{k^*}$. Under the Dirichlet, the correlation between $\theta_j$ and $\theta_l$ (for $j, l \neq k^*$) is:

$$\text{Cor}(\theta_j, \theta_l) = \frac{-\alpha_j \alpha_l}{(S - \alpha_j)(S - \alpha_l)} \cdot \frac{S + 1}{1} \cdot \frac{1}{S+1} = \frac{-\alpha_j \alpha_l}{S^2(S+1)/(S+1)}$$

More precisely, for two components of a Dirichlet:

$$\text{Cov}(\theta_j, \theta_l) = \frac{-\alpha_j \alpha_l}{S^2(S + 1)}$$

The exceedance events are correlated through the shared dependence on $\theta_{k^*}$. We approximate the correlation between the probit-transformed events as:

$$\rho_{jl} \approx \frac{\alpha_j \alpha_l}{(S - \alpha_{k^*})^2}$$

derived from the conditional covariance of $(\theta_j, \theta_l)$ given $\theta_{k^*}$.

**Step 4: First-order correction with damping.**

$$c_N^{\text{sum}} \approx \prod_{j \neq k^*} p_j + d(K) \sum_{j < l, \, j,l \neq k^*} \rho_{jl} \cdot \phi(a_j) \cdot \phi(a_l) \cdot \prod_{m \neq k^*, j, l} p_m$$

where $\phi$ is the standard normal PDF and $d(K)$ is a $K$-dependent damping factor calibrated to achieve RMSE $< 0.01$ against Monte Carlo estimates. For $K = 4$, $d(4) \approx 0.8$, giving RMSE $\approx 0.005$.

The first term is the **independence approximation** (product of marginal exceedances). The second term is the **first-order Gaussian copula correction** accounting for positive correlation between the pairwise events.

## Method 3: Dirichlet MLE (Minka 2000)

### Log-likelihood

Assuming $\mathbf{p}^{(1)}, \ldots, \mathbf{p}^{(N)} \overset{\text{iid}}{\sim} \text{Dir}(\boldsymbol{\alpha})$:

$$\log L(\boldsymbol{\alpha}) = N \left[\log \Gamma(S) - \sum_{k=1}^{K} \log \Gamma(\alpha_k)\right] + \sum_{k=1}^{K}(\alpha_k - 1) \cdot N \bar{\ell}_k$$

where $S = \sum_k \alpha_k$ and $\bar{\ell}_k = \frac{1}{N}\sum_n \log p_k^{(n)}$.

### Minka's fixed-point iteration

Taking the gradient and setting it to zero:

$$\psi(\alpha_k) = \psi(S) + \bar{\ell}_k$$

where $\psi = \Gamma'/\Gamma$ is the digamma function. Since $\psi$ is monotonically increasing and concave, its inverse $\psi^{-1}$ exists and the fixed-point iteration:

$$\alpha_k^{(t+1)} = \psi^{-1}\!\left(\psi(S^{(t)}) + \bar{\ell}_k\right)$$

converges monotonically. In practice, $\psi^{-1}$ is computed via Newton's method on $\psi(x) - y = 0$, using the trigamma function $\psi'(x)$ as the derivative.

**Initialisation:** MoM estimate (see Method 4) or $\alpha_k = 1$ for all $k$.

**Convergence:** Typically 10-20 iterations to relative tolerance $10^{-6}$.

### Regularisation

At small $N$ or when some $p_k^{(n)} \approx 0$, the MLE can degenerate. We apply:

1. **Label smoothing:** $p_k^{(n)} \leftarrow (1 - \epsilon) p_k^{(n)} + \epsilon/K$ with $\epsilon = 10^{-3}$, ensuring $\bar{\ell}_k > -\infty$.
2. **Prior pseudo-observation:** Add a uniform vector $\mathbf{1}/K$ with weight `prior_strength` $= 1.0$ to the sufficient statistics:
   $$\bar{\ell}_k^{\text{reg}} = \frac{N \bar{\ell}_k + \text{prior\_strength} \cdot \log(1/K)}{N + \text{prior\_strength}}$$

### Stopping criterion

$$c_N^{\text{MLE}} = \text{copula\_exceedance}(\hat{\boldsymbol{\alpha}})$$

using the same approximation as Method 2.

## Method 4: MoM (Method of Moments)

### Derivation

For $\mathbf{X} \sim \text{Dir}(\alpha_0 \boldsymbol{\mu})$ with $\boldsymbol{\mu} = \boldsymbol{\alpha}/\alpha_0$ and $\alpha_0 = \sum_k \alpha_k$:

$$\text{Var}(X_k) = \frac{\mu_k(1 - \mu_k)}{\alpha_0 + 1}$$

This gives the **variance ratio** identity:

$$R \equiv \frac{\text{Var}(X_k)}{\mu_k(1 - \mu_k)} = \frac{1}{\alpha_0 + 1} \quad \text{(same for all } k\text{)}$$

### Pooled estimator

Estimating $R$ per component and taking a median is unstable: when $\hat{\mu}_k \approx 0$, both $s_k^2$ and $\hat{\mu}_k(1 - \hat{\mu}_k)$ are near zero, and their ratio is dominated by noise. The pooled estimator:

$$\hat{R} = \frac{\sum_{k=1}^{K} s_k^2}{\sum_{k=1}^{K} \hat{\mu}_k(1 - \hat{\mu}_k)}$$

is a **ratio of sums**, not a sum of ratios. Components with $\hat{\mu}_k \approx 0$ contribute negligibly to both numerator and denominator, naturally downweighting them. This is analogous to a precision-weighted average.

**Consistency of $\hat{R}$:** As $N \to \infty$, $s_k^2 \xrightarrow{p} \text{Var}(X_k)$ and $\hat{\mu}_k(1 - \hat{\mu}_k) \xrightarrow{p} \mu_k(1 - \mu_k)$. By the continuous mapping theorem (the denominator is bounded away from zero since at least one $\mu_k$ is bounded away from both 0 and 1):

$$\hat{R} \xrightarrow{p} \frac{\sum_k \text{Var}(X_k)}{\sum_k \mu_k(1 - \mu_k)} = \frac{1}{\alpha_0 + 1} = R$$

So $\hat{R}$ is a consistent estimator of $R$.

### Concentration estimate

$$\hat{\alpha}_0 = \frac{1 - \hat{R}}{\hat{R}}$$

Clamped to $[1, 200]$ to handle degenerate cases ($\hat{R} \leq 0$ or $\hat{R} \geq 1$).

### Stopping criterion

Set $\hat{\boldsymbol{\alpha}} = \hat{\alpha}_0 \cdot \hat{\boldsymbol{\mu}}$ and compute:

$$c_N^{\text{MoM}} = \text{copula\_exceedance}(\hat{\boldsymbol{\alpha}})$$

## Method 5: MoM + Bayes (Bayesian Wrapper)

### Motivation

MoM produces a point estimate $\hat{\alpha}_0$. At $N = 2$, the pooled sample variance $\sum_k s_k^2$ is based on $K(N-1) = 4$ degrees of freedom. The 95% confidence interval for a $\chi^2_4$ variate spans a factor of $\approx 12$ (from $\chi^2_{4, 0.025}/4 = 0.121$ to $\chi^2_{4, 0.975}/4 = 2.77$). This means $\hat{\alpha}_0$ could plausibly be anywhere from $\sim 0.6\hat{\alpha}_0$ to $\sim 8\hat{\alpha}_0$. Reporting a single exceedance from the point estimate ignores this massive uncertainty.

### Sampling distribution of $\hat{R}$

Under the Dirichlet model, for component $k$:

$$\frac{(N-1)s_k^2}{\text{Var}(X_k)} \approx \chi^2(N-1)$$

This is approximate because $p_k^{(n)}$ is bounded in $[0, 1]$ rather than Gaussian, but for the moderate sample sizes and the Beta-like marginals of the Dirichlet, the chi-squared approximation is adequate.

Pooling across $K$ independent components (the marginals of a Dirichlet are not independent, but the chi-squared approximation treats the pooled statistic as having $K(N-1)$ df):

$$\frac{\hat{R}}{R} \cdot K(N-1) \approx \chi^2(K(N-1))$$

Equivalently, letting $\nu = K(N-1)$:

$$\hat{R} \mid \alpha_0 \sim \frac{R}{\nu} \cdot \chi^2(\nu) = \frac{1}{(\alpha_0 + 1)\nu} \cdot \chi^2(\nu)$$

The PDF of $\hat{R}$ given $\alpha_0$:

$$f(\hat{R} \mid \alpha_0) = \frac{\nu}{R} \cdot f_{\chi^2(\nu)}\!\left(\frac{\hat{R} \cdot \nu}{R}\right) = (\alpha_0 + 1)\nu \cdot f_{\chi^2(\nu)}\!\left(\hat{R}(\alpha_0 + 1)\nu\right)$$

where $f_{\chi^2(\nu)}$ is the chi-squared PDF with $\nu$ degrees of freedom.

### Prior

$$\pi(\alpha_0) = \text{Gamma}(\alpha_0 \mid \text{shape}=2,\ \text{scale}=5)$$

This has mode $= (2-1) \times 5 = 5$, mean $= 10$, variance $= 50$. It encodes the belief that models are moderately consistent ($\alpha_0 \sim 5\text{--}15$) but allows for both near-random ($\alpha_0 \approx 1$) and very consistent ($\alpha_0 > 100$) behaviour.

The prior is deliberately vague and is overwhelmed by data at $N \geq 4$. At $N = 2$, it provides a soft regularisation that is strictly preferable to MoM's hard clamp $[1, 200]$ — a clamp is simply a uniform prior on $[1, 200]$ with hard edges, which is less principled.

### Posterior computation

On a grid $\{g_1, \ldots, g_M\}$ of $M = 80$ points log-spaced from 0.5 to 300:

$$w_i = f(\hat{R} \mid \alpha_{0,i}) \cdot \pi(\alpha_{0,i}) \cdot \Delta_i$$

where $\Delta_i$ is the grid spacing (in log-space, $\Delta_i = g_i \cdot \Delta\log g$). Normalise: $\tilde{w}_i = w_i / \sum_j w_j$.

### Marginalised exceedance

For each grid point, compute the copula exceedance using $\boldsymbol{\alpha}_i = \alpha_{0,i} \cdot \hat{\boldsymbol{\mu}}$:

$$\text{exc}_i = \text{copula\_exceedance}(\alpha_{0,i} \cdot \hat{\boldsymbol{\mu}})$$

The marginalized stopping criterion:

$$c_N^{\text{Bayes}} = \sum_{i=1}^{M} \tilde{w}_i \cdot \text{exc}_i$$

### Behaviour at limiting N

**$N = 2$:** $\nu = K(N-1) = 4$. The chi-squared posterior on $\alpha_0$ is wide. Even if $\hat{R}$ is small (suggesting high consistency), the posterior places substantial mass on low $\alpha_0$ values. The copula exceedance at low $\alpha_0$ is closer to $1/K = 0.25$, dragging down the marginalised exceedance. **Effect:** the system almost never stops at $N = 2$, regardless of the point estimate — exactly the conservatism needed for adaptive stopping.

**$N = 10$:** $\nu = 36$. The chi-squared distribution concentrates: $\hat{R}/R$ has a 95% CI of roughly $[0.67, 1.40]$. The posterior on $\alpha_0$ is tight around the MoM estimate, and $c_N^{\text{Bayes}} \approx c_N^{\text{MoM}}$. The Bayesian wrapper adds negligible overhead when data is plentiful.

**Crossover:** At $N \approx 4\text{--}5$, the data dominates the prior and the Bayesian method transitions from "cautious" to "data-driven."

### Connection to Bayesian model comparison

The marginalised exceedance can be interpreted as a **Bayesian model average**. Define two "models": $M_1 = $ "answer $k^*$ is correct" and $M_0 = $ "some other answer is correct." The exceedance at each $\alpha_0$ is a model-conditional posterior probability. Marginalising over $\alpha_0$ gives:

$$P(M_1 \mid \text{data}) = \int P(M_1 \mid \alpha_0, \text{data}) \cdot P(\alpha_0 \mid \text{data})\ d\alpha_0$$

This is the standard Bayesian marginal likelihood framework, with $\alpha_0$ as a nuisance parameter that is integrated out rather than optimised. The grid integration is a discrete approximation to this integral.

### Computational cost

Per query: one pooled-$R$ computation ($O(NK)$), one 80-point grid evaluation (each requiring a copula exceedance at $O(K^2)$), and one normalisation. Total: $O(NK + MK^2)$ per stopping check. With $K = 4$ and $M = 80$, this is $\sim 1300$ flops — negligible compared to a single LLM forward pass.

## Properties of the Dirichlet distribution (reference)

For $\mathbf{X} \sim \text{Dir}(\boldsymbol{\alpha})$ with $S = \sum_k \alpha_k$:

- **Mean:** $E[X_k] = \alpha_k / S$
- **Variance:** $\text{Var}(X_k) = \frac{\alpha_k(S - \alpha_k)}{S^2(S + 1)}$
- **Covariance:** $\text{Cov}(X_j, X_k) = \frac{-\alpha_j \alpha_k}{S^2(S + 1)}$ for $j \neq k$
- **Mode** (for $\alpha_k > 1$): $\text{mode}(X_k) = \frac{\alpha_k - 1}{S - K}$
- **Marginals:** $X_k \sim \text{Beta}(\alpha_k, S - \alpha_k)$
- **Aggregation:** $(X_A, X_B + X_C + X_D) \sim \text{Dir}(\alpha_A, \alpha_B + \alpha_C + \alpha_D)$, i.e., $\frac{X_A}{X_A + X_B} \sim \text{Beta}(\alpha_A, \alpha_B)$

The concentration parameter $\alpha_0 = S$ controls the "peakiness" of the distribution on the simplex. As $\alpha_0 \to \infty$ with $\boldsymbol{\mu} = \boldsymbol{\alpha}/\alpha_0$ fixed, $\text{Dir}(\alpha_0 \boldsymbol{\mu}) \to \delta_{\boldsymbol{\mu}}$ (point mass at the mean). As $\alpha_0 \to 0$, mass concentrates on the vertices of the simplex.

## Summary comparison

| | **Product** | **Sum** | **MLE** | **MoM** | **MoM+Bayes** |
|---|---|---|---|---|---|
| **What it estimates** | $P(k \mid \text{data})$ | Dirichlet $\boldsymbol{\alpha}$ (additive) | Dirichlet $\boldsymbol{\alpha}$ (MLE) | $\alpha_0$ (point) | $P(\alpha_0 \mid \text{data})$ |
| **Aggregation** | Multiplicative (log-sum) | Additive (pseudo-counts) | Fit to data | Variance ratio | Marginalised variance ratio |
| **Confidence metric** | $\max P(k)$ | Copula exceedance | Copula exceedance | Copula exceedance | Marginalised exceedance |
| **Hyperparameters** | $T$ (temperature) | None | prior_strength, $\epsilon$ | Clamp bounds | Gamma prior (shape, scale) |
| **Handles overconfidence** | Temperature scaling | Linear accumulation | Fits from data | Estimates from data | Estimates + marginalises |
| **Consistency-aware** | No | No | Yes | Yes | Yes |
| **Uncertainty about $\alpha_0$** | N/A | N/A | No | No | **Yes** |
| **Small-$N$ behaviour** | Overconfident (without $T$) | Conservative | Noisy | Point estimate | Conservative |
| **Large-$N$ behaviour** | Concentrates exponentially | Concentrates as $O(N)$ | Converges to true $\boldsymbol{\alpha}$ | Converges to true $\alpha_0$ | Converges to MoM |
| **Computational cost** | $O(NK)$ | $O(NK + K^2)$ | $O(NK + IK)$, $I$ iterations | $O(NK + K^2)$ | $O(NK + MK^2)$ |
| **Key assumption** | Queries independent given $T$ | Evidence is fractional counts | iid Dirichlet model | iid Dirichlet model | iid Dirichlet + Gamma prior |
| **What it ignores** | Dispersion in data | Dispersion in data | Uncertainty in $\hat{\boldsymbol{\alpha}}$ | Uncertainty in $\hat{\alpha}_0$ | Higher moments of $R$ |
| **Best when** | Good calibration data available | Quick baseline, no tuning | $N \geq 5$, need shape info | $N \geq 3$, quick estimate | $N$ is small, safety matters |
