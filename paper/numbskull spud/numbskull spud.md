# The Numbskull Spud's Guide to Posterior Aggregation

*Five methods for deciding when you've heard enough — from playground intuition to Bayesian marginalization.*

---

# Level 1: ELI10 — The Story

## Why ask the same question ten times?

Imagine you have a friend who's really good at pub quizzes but has a weird habit: sometimes they blurt out the wrong answer even when they know the right one. Not because they're lying — they just get flustered, or the way you phrased the question threw them off, or they focused on a red herring. But if you ask them the same question several times — maybe rewording it slightly each time, or listing the multiple-choice options in a different order — their *pattern* of answers tells you much more than any single answer does.

That's what we're doing with an AI model. We give it a reading comprehension question with four answer choices (A, B, C, D), but each time we shuffle the order. The model doesn't just say "A" — it gives us a confidence score for every option: "I'm 90% sure it's A, 5% B, 3% C, 2% D."

Here's the problem: **the model is a showoff.** When it says "90% sure," it's actually only right about 75% of the time. It's overconfident on every single answer. If you trusted that 90% at face value, you'd stop after one ask. But the model is wrong a quarter of the time — that's terrible.

So we ask multiple times, collect these confidence scores, and try to figure out: **do I have enough evidence to trust the answer, or should I keep asking?** If the model stays confused after many tries, we escalate — we let it "think out loud" using a more expensive reasoning mode.

The question: how do you combine those multiple sets of confidence scores into one "how sure am I?" number? That's what the five methods do.

## Method 1: Product — Multiply everything together

Think of it like a jury. Each time you ask the model, you call a new witness. Each witness independently says "I think it's A." The more witnesses who agree, the more confident you get — you multiply the evidence together.

**The catch:** Each witness is overconfident. If you naively multiply 90% × 90%, you're absurdly sure after two witnesses and stop way too early. So we add a "humility filter" (temperature) that deflates each witness before multiplying. Instead of 90%, each effectively says 55%. Now you need several agreeing witnesses before you're convinced.

**The worry:** That humility dial is set by testing on practice questions. If the model or domain changes, the dial might be wrong.

## Method 2: Sum — Stack up the votes

Instead of multiplying, just add. Each time the model says "90% A, 5% B, 3% C, 2% D," toss those numbers into a running tally. After three asks, A's tally is way ahead of the rest.

**Good news:** No humility dial needed. Adding doesn't cause the explosive overconfidence that multiplying does.

**Bad news:** Sum is blind to *consistency*. Two observations of [50%, 50%] gives the same tally as one observation of [100%, 0%] followed by [0%, 100%]. The first is "genuinely unsure." The second is "confidently contradicting itself." Sum can't tell the difference.

## Method 3: MLE — Fit the pattern

This one asks: what *kind* of answerer is the model? Imagine watching someone answer the same question ten times:
- Always the same confident answer? → Reliable.
- Always wishy-washy? → Unsure but consistent.
- Wildly swinging between confident answers? → Unreliable.

MLE fits a mathematical model that captures both the *average* answer and the *consistency*.

**Good news:** Can distinguish consistent from inconsistent.
**Bad news:** With only 2-3 observations, there's not enough data to fit reliably.

## Method 4: MoM — Quick consistency check

A simpler version of MLE. Instead of fitting a full model, just measure: **how much do the answers vary?** Low variance → consistent → trust it. High variance → inconsistent → don't.

One-line formula instead of an iterative fit. But with 2-3 data points, the variance estimate is noisy.

## Method 5: MoM + Bayes — Honest about what you don't know

Method 4 wearing a seatbelt. It computes the same variance, but then asks: "Given that I've only seen N answers, how *sure* am I about this variance estimate?"

With 2 observations, the answer is: not sure at all. So it considers *all possible* consistency levels, weighted by how likely each is given the data. With little data, it stays cautious. With lots of data, it converges to plain MoM.

**This is the key idea: uncertainty about uncertainty.** We're not just uncertain about the *answer* — we're uncertain about *how uncertain we are*. For an adaptive stopping system where stopping too early is dangerous, this conservatism is exactly what you want.

## Why five?

| Method | Core idea | Main weakness |
|---|---|---|
| **Product** | Multiply witness evidence | Needs a calibration dial |
| **Sum** | Add votes | Blind to consistency |
| **MLE** | Fit the full pattern | Noisy with few observations |
| **MoM** | Quick variance check | Treats noisy estimate as certain |
| **MoM+Bayes** | Variance check + honest uncertainty | Heaviest computation |

The **concentration parameter** — a single number capturing "how consistent and confident is the model?" — is the thread connecting methods 2–5. Product sidesteps it with temperature. The rest estimate it, with increasing sophistication.

---

# Level 2: ELI15 — The Numbers

## Probability vectors and the overconfidence problem

When our model answers a 4-choice question, it produces a **probability vector** — four numbers that sum to 1:

$$\mathbf{p} = [0.91,\ 0.05,\ 0.03,\ 0.01]$$

We shuffle the answer labels and ask again. After 3 queries (mapped back to canonical order):

| Query | P(A) | P(B) | P(C) | P(D) |
|-------|------|------|------|------|
| 1 | 0.91 | 0.05 | 0.03 | 0.01 |
| 2 | 0.85 | 0.08 | 0.04 | 0.03 |
| 3 | 0.88 | 0.06 | 0.04 | 0.02 |

The model is consistently pointing at A with high confidence. The **maximum softmax probability (MSP)** — just the largest number in each row — averages about 0.91 across our dataset. But the model's accuracy is only ~75%. This gap is the **overconfidence problem**, measured by **Expected Calibration Error (ECE = 0.186)**. ECE works by binning predictions by confidence (all the 80–90% ones together, all the 90–100% ones together, etc.) and checking how often they're actually right. A perfectly calibrated model would have ECE = 0. Ours is 0.186, meaning predictions are off by about 19 percentage points on average.

However, the **mean** vector $\bar{\mathbf{p}} = [0.88, 0.063, 0.037, 0.02]$ is much better calibrated. Individual queries are unreliable narrators, but their aggregate tells the truth.

## Method 1: Product — multiply, then normalize

Start with a uniform belief: each answer is equally likely (25% each). Each query's probabilities act as evidence — multiply them in.

With our three vectors, the multiplication for answer A gives: $0.91 \times 0.85 \times 0.88 = 0.681$. For B: $0.05 \times 0.08 \times 0.06 = 0.00024$. After normalizing so all four sum to 1, answer A gets **99.98%** of the posterior. After just 3 queries!

That's the overconfidence problem in action. The posterior concentrates immediately. The stopping threshold τ = 0.95 would trigger after query 1. Adaptive stopping is useless.

**Temperature scaling** fixes this by squishing each vector toward uniform before multiplying. With T=3.0, that first vector [0.91, 0.05, 0.03, 0.01] becomes roughly [0.52, 0.20, 0.17, 0.12]. Now the model says "I lean toward A" instead of "I'm certain it's A." The posterior needs several consistent observations before reaching τ.

At T=3.0 (ECE-optimal): easy questions stop after 2–3 queries (avg N = 2.76), hard ones use all 10, and ~15% of the hardest get escalated.

## Method 2: Sum — add to a running tally

Start with a tally of (1, 1, 1, 1) — a uniform "Dirichlet prior." (A **Dirichlet distribution** is a probability distribution over probability vectors — it's a way of expressing uncertainty about *which* probability vector is the "true" one. The tally $\boldsymbol{\alpha}$ parameterizes it: bigger numbers in the tally mean more confidence in that component. **Pseudo-counts** are the entries in this tally — they act like "fake observations" that encode our prior belief. Starting at (1,1,1,1) means "I've seen one fake vote for each answer, so I start with no preference.") Each query adds its probabilities:

$$\boldsymbol{\alpha} = (1 + 0.91 + 0.85 + 0.88,\ 1 + 0.05 + 0.08 + 0.06,\ \ldots) = (3.64,\ 1.19,\ 1.11,\ 1.06)$$

Total = 7.0. The expected proportion for A is 3.64/7.0 = 52% — much softer than the raw 88% average, because the prior pulls toward uniform.

The confidence metric isn't just "how far ahead is the leader." It's the **exceedance probability**: given this tally, what's the probability that A is *truly* the best answer? "Exceedance" just means "the probability that one thing exceeds all the others" — here, that A's true probability exceeds B's, C's, and D's simultaneously. This accounts for the fact that with a total of only 7 pseudo-counts, there's still meaningful uncertainty. (The **copula** approximation handles the computation — a copula is a technique for combining multiple pairwise comparisons into a joint probability. Details in Level 3.)

### The consistency blindness problem

With a 2-choice simplification:
- **Scenario A:** Two observations of [0.5, 0.5] → tally = (2.0, 2.0)
- **Scenario B:** One of [1.0, 0.0], one of [0.0, 1.0] → tally = (2.0, 2.0)

Identical tallies. But A is "genuinely unsure" while B is "confidently contradicting itself." Sum can't see this because it only tracks totals, not spread.

## The key concept: concentration

Picture the model's 3 answers as 3 darts on a dartboard. "Concentration" measures how tightly they cluster.

**High concentration** (darts clustered): the model gives nearly the same answer every time. It might be wrong, but it's *consistently* wrong or right. The average is trustworthy.

**Low concentration** (darts scattered): the model contradicts itself. Even if the average looks good, any individual answer might be a fluke. Don't trust it.

This is what Product and Sum don't measure. Product assumes calibration handles everything. Sum only sees the average dart position, not the cluster tightness. The next three methods all try to measure concentration from the data.

Formally, concentration is the parameter $\alpha_0$ in a Dirichlet distribution. Think of $\alpha_0$ as "how many effective observations the model acts like it has." High $\alpha_0$ means the model behaves as if it's seen lots of evidence (tight cluster). Low $\alpha_0$ means it behaves like it's guessing (wide scatter).

## Method 3: MLE — fit the full Dirichlet

The computer tries to find the Dirichlet distribution that would have been most likely to produce our observed vectors. It iterates through candidate distributions, adjusting until it finds the best fit.

The output: a fitted $\boldsymbol{\alpha}$ vector that captures both the *direction* (which answer leads) and the *concentration* (how tightly the observations cluster). For our example, the fitted α would have high concentration — the model is very consistent.

Confidence = copula exceedance on the fitted α.

**Needs N ≥ 2** (can't fit a distribution from one point). With regularization to handle small-N instability.

## Method 4: MoM — the one-line shortcut

Instead of iteratively fitting, use a direct formula based on variance.

For a Dirichlet, the variance of each component relates to concentration by:

$$\text{Var}(X_k) = \frac{\mu_k(1 - \mu_k)}{\alpha_0 + 1}$$

The ratio $R = \text{variance} / [\mu(1-\mu)]$ is the same for all components, and equals $1/(\alpha_0 + 1)$. Estimate R from our data, solve for $\alpha_0$.

**Worked example with our 3 vectors:**

Sample means: $\hat{\mu} = [0.88, 0.063, 0.037, 0.02]$

Sample variances: $s_A^2 = 0.0009$, $s_B^2 = 0.00023$, $s_C^2 = 0.00003$, $s_D^2 = 0.0001$

Pool them (sum numerators, sum denominators — this naturally downweights near-zero components):

$$\hat{R} = \frac{0.0009 + 0.00023 + 0.00003 + 0.0001}{0.88 \times 0.12 + 0.063 \times 0.937 + 0.037 \times 0.963 + 0.02 \times 0.98} = \frac{0.00127}{0.221} = 0.0057$$

$$\hat{\alpha}_0 = \frac{1 - 0.0057}{0.0057} \approx 173$$

A concentration of 173 is very high — the model is extremely consistent. This makes sense: the three vectors barely differ. Set $\boldsymbol{\alpha} = 173 \times [0.88, 0.063, 0.037, 0.02]$ and compute exceedance → essentially 1.0.

## Method 5: MoM + Bayes — but can we trust 3 darts?

MoM gave us $\hat{\alpha}_0 = 173$. But we only had 3 observations. With $K=4$ components and $N=3$, our variance estimate has $K(N-1) = 8$ **degrees of freedom**. (Degrees of freedom roughly means "how many independent pieces of information went into the estimate." With 3 data points, each component's variance uses $N-1 = 2$ independent deviations from the mean. Pooling 4 components gives $4 \times 2 = 8$.) That's not much — our point estimate could easily be off by a factor of 2.

MoM+Bayes puts a **prior** on $\alpha_0$ — a Gamma distribution (a bell-curve-like shape for positive numbers, useful for things that can't be negative like concentration) that says "I expect moderate consistency before seeing data" — then computes a **posterior** (the updated belief after seeing the data) that balances the prior with the evidence.

For each of 80 possible $\alpha_0$ values (from 0.5 to 300), it asks: "If the true concentration were this value, how likely is the R̂ we observed?" Then it weights the exceedance at each $\alpha_0$ by that likelihood.

At N=2 (only 4 degrees of freedom): the posterior is wide → many $\alpha_0$ values are plausible → the weighted exceedance is blurred toward uncertainty → the system stays cautious.

At N=10 (36 degrees of freedom): the posterior is tight → only one $\alpha_0$ fits → converges to plain MoM.

## Comparison

| | Estimates | Hyperparams | Consistency-aware? | Uncertainty about $\alpha_0$? |
|---|---|---|---|---|
| **Product** | P(answer) via Bayes | Temperature T | No | N/A |
| **Sum** | Dirichlet pseudo-counts | None | No | N/A |
| **MLE** | Dirichlet α (fitted) | Regularization | Yes | No (point estimate) |
| **MoM** | $\alpha_0$ via variance | Clamp bounds | Yes | No (point estimate) |
| **MoM+Bayes** | Posterior over $\alpha_0$ | Gamma prior | Yes | **Yes** |

---

# Level 3: The Full Picture

This section contains the same math as a paper appendix — but walked through step by step, with our running example threaded through every derivation. If an equation appears, we'll compute it with real numbers right after so you can see what it actually does.

## Notation

We have $K = 4$ answer choices. A single query produces a probability vector $\mathbf{p}^{(n)}$ on the **simplex** — that's just the fancy name for "the set of all vectors whose components are non-negative and sum to 1." A probability vector with 4 entries lives on the 3-simplex (3 because the 4th component is determined by the other three). We collect $N$ such vectors under shuffled orderings.

The key statistics we'll need:

| Symbol | What it is | Our example (N=3) |
|--------|-----------|-------------------|
| $\hat{\mu}_k$ | Sample mean of component $k$ | [0.880, 0.063, 0.037, 0.020] |
| $s_k^2$ | Sample variance of component $k$ | [0.00090, 0.00023, 0.00003, 0.00010] |
| $\bar{\ell}_k$ | Mean of $\log p_k$ across queries (the **sufficient statistic** for Dirichlet MLE — "sufficient" because it captures everything the data can tell us about the Dirichlet parameters, without needing to store the raw observations) | [-0.128, -2.78, -3.32, -4.01] |

The stopping criterion always takes the same form: compute a confidence $c_N \in [0, 1]$ after each query. If $c_N > \tau$, stop and return the leading answer. If $c_N \leq \tau$ after $N_{\max}$ queries, escalate.

## 3.1 Temperature scaling

In Level 2 we saw that raw logprobs are overconfident. Here's the formal mechanism.

Temperature scaling raises each probability to the power $1/T$ and renormalizes:

$$\tilde{p}_k = \frac{p_k^{1/T}}{\sum_j p_j^{1/T}}$$

For $T > 1$, this flattens the distribution (more uniform). For $T < 1$, it sharpens it. Let's see what $T = 3$ does to our first vector:

| Component | Raw $p_k$ | $p_k^{1/3}$ | After normalizing |
|-----------|----------|-------------|-------------------|
| A | 0.91 | 0.969 | 0.520 |
| B | 0.05 | 0.368 | 0.198 |
| C | 0.03 | 0.311 | 0.167 |
| D | 0.01 | 0.215 | 0.115 |
| **Sum** | 1.00 | **1.863** | 1.000 |

The 91% became 52%. The model is now more honest about its uncertainty. The **entropy** increases from 0.44 to 1.22. Entropy measures how "spread out" a distribution is — it's $-\sum p_k \log p_k$. A distribution that puts all its mass on one answer has entropy 0 (totally certain). A uniform distribution [0.25, 0.25, 0.25, 0.25] has maximum entropy $\log 4 = 1.39$ (maximally uncertain). Our temperature-scaled vector at 1.22 is much closer to "honest uncertainty" than the raw 0.44.

Why $T = 3$ specifically? We search over $T \in [1, 10]$ and pick the value that minimizes Expected Calibration Error (ECE) on our dataset. ECE measures the gap between predicted confidence and actual accuracy — at $T = 3$, ECE drops from 0.186 to 0.012.

## 3.2 Method 1: Product — Bayesian sequential updating

### The update rule

With a uniform prior $P(k) = 1/K = 0.25$, Bayes' rule after $N$ observations gives:

$$P(k \mid \mathbf{p}^{(1:N)}) = \frac{\prod_{n=1}^{N} \tilde{p}_k^{(n)}}{\sum_{j=1}^{K} \prod_{n=1}^{N} \tilde{p}_j^{(n)}}$$

In words: for each answer $k$, multiply together all $N$ temperature-scaled probabilities for that answer, then normalize so they sum to 1. This is called a **product of experts** model — a framework where multiple independent "experts" (here, each shuffled query) each cast a vote, and their opinions are combined by multiplication rather than averaging. The key property: if *any* expert strongly disagrees with an answer, the product drives that answer's score to near zero. Consensus is required.

We work in **log-space** to avoid **numerical underflow** (when you multiply many small numbers together, the result can become so tiny that the computer rounds it to zero — by adding logarithms instead of multiplying raw numbers, we keep everything in a numerically safe range):

$$\log P(k \mid \text{data}) = \sum_{n=1}^{N} \log \tilde{p}_k^{(n)} - \log Z_N$$

where $Z_N$ is just whatever constant makes everything sum to 1.

### Walking through our example (with T=3)

After temperature-scaling all three vectors, the log-probabilities are roughly:

| | $\log \tilde{p}_A$ | $\log \tilde{p}_B$ | $\log \tilde{p}_C$ | $\log \tilde{p}_D$ |
|---|---|---|---|---|
| Query 1 | -0.65 | -1.62 | -1.79 | -2.16 |
| Query 2 | -0.72 | -1.48 | -1.69 | -1.87 |
| Query 3 | -0.68 | -1.55 | -1.74 | -2.03 |
| **Sum** | **-2.05** | **-4.65** | **-5.22** | **-6.06** |

Component A has the least negative sum by a large margin. After exponentiating and normalizing: $P(A) \approx 0.90$, $P(B) \approx 0.07$, $P(C) \approx 0.02$, $P(D) \approx 0.005$. With temperature scaling, 3 queries gives 90% instead of the unscaled 99.98%. The system needs a few more queries to cross $\tau = 0.95$.

### Why it concentrates exponentially

The gap between the leader and the runner-up grows by roughly $\mathbb{E}[\log \tilde{p}_{A}] - \mathbb{E}[\log \tilde{p}_{B}]$ with each new query. In log-space, that gap grows *linearly* in $N$ — which means the posterior concentrates *exponentially* in $N$. Temperature controls the rate: higher $T$ means smaller per-query gaps, so more queries needed.

### Stopping criterion

$$c_N^{\text{prod}} = \max_k P(k \mid \mathbf{p}^{(1:N)})$$

Simply: the posterior probability of the leading answer. Stop when it exceeds $\tau$.

## 3.3 The copula exceedance — shared by Sum, MLE, MoM, and MoM+Bayes

This is the most complex piece of math in the whole framework, so let's take it slowly. It's used by four of the five methods, so understanding it once pays off four times.

**What's a copula?** In general, a copula is a mathematical tool for describing how multiple random variables are correlated with each other — it separates the question "what does each variable look like on its own?" from the question "how do they move together?" Here we use a **Gaussian copula**: we convert each pairwise comparison into a standard normal variable, model their joint behavior as a multivariate normal (which is easy to work with), and then convert back. It's an approximation, but a good one.

**What's exceedance?** Simply the probability that one quantity exceeds (is bigger than) all others. "Exceedance probability of A" = "probability that A is truly the best."

### What we're computing

Given a Dirichlet distribution with parameters $\boldsymbol{\alpha} = (\alpha_1, \alpha_2, \alpha_3, \alpha_4)$, we want:

$$P(\theta_{k^*} > \theta_j \text{ for all } j \neq k^*)$$

"What's the probability that the leading component is truly the largest?" Here $k^*$ is whichever component has the biggest $\alpha_k$ (the leader), and $\theta_1, \ldots, \theta_4$ are random variables drawn from $\text{Dir}(\boldsymbol{\alpha})$.

This integral has no closed form for $K > 2$. (For $K = 2$, it reduces to a single Beta comparison — easy. But for $K = 4$, the leader has to beat three opponents simultaneously, and the three comparisons are correlated.) We need an approximation. Luigi's damped Gaussian-copula method gets within RMSE ≈ 0.005 of the true answer for $K = 4$.

### Step 1: Pairwise comparisons (the building blocks)

Start with something we *can* compute exactly. For any two components $k^*$ and $j$ of a Dirichlet, there's a well-known result: the ratio $\theta_{k^*} / (\theta_{k^*} + \theta_j)$ follows a $\text{Beta}(\alpha_{k^*}, \alpha_j)$ distribution. So:

$$p_j = P(\theta_{k^*} > \theta_j) = 1 - I_{0.5}(\alpha_{k^*}, \alpha_j)$$

where $I_x(a, b)$ is the **regularized incomplete Beta function** — a standard function available in any stats library (scipy.special.betainc in Python). You don't need to know its internals; it's the CDF (cumulative distribution function) of the Beta distribution. Here it answers: "given that A has $\alpha_A$ pseudo-counts and B has $\alpha_B$ pseudo-counts, what's the probability that A's true proportion exceeds B's?"

**With our Sum example** ($\boldsymbol{\alpha} = (3.64, 1.19, 1.11, 1.06)$, leader = A):

| Comparison | $\alpha_{k^*}$ | $\alpha_j$ | $p_j = P(A > j)$ |
|------------|-----------|---------|-----------|
| A vs B | 3.64 | 1.19 | 0.908 |
| A vs C | 3.64 | 1.11 | 0.918 |
| A vs D | 3.64 | 1.06 | 0.924 |

So A beats each competitor individually with ~91–92% probability.

### Step 2: If the comparisons were independent (the naive estimate)

If "A beats B" and "A beats C" and "A beats D" were independent events, we'd just multiply:

$$P(\text{A beats all}) \approx 0.908 \times 0.918 \times 0.924 = 0.770$$

But they're **not** independent. They all share $\theta_A$ — if $\theta_A$ happens to be large, A is more likely to beat *everyone*. This positive correlation means the true probability is higher than the naive product.

### Step 3: Measuring the correlation

The Dirichlet distribution gives us the covariance structure. For two components $j$ and $l$ (both competitors of the leader):

$$\text{Cov}(\theta_j, \theta_l) = \frac{-\alpha_j \alpha_l}{S^2(S + 1)}$$

where $S = \sum_k \alpha_k$ is the total. For our example, $S = 7.0$.

But we need the correlation between the *events* "A beats B" and "A beats C", not between $\theta_B$ and $\theta_C$ themselves. The Gaussian copula approach handles this by converting to a common scale.

First, we apply the **probit transform** — converting a probability into the corresponding point on the standard normal (bell curve) scale. If $p_j = 0.908$, the probit asks: "at what point on the bell curve does 90.8% of the area lie to the left?" The function $\Phi^{-1}$ is the inverse of the normal CDF (in Python: scipy.stats.norm.ppf):

$$a_j = \Phi^{-1}(p_j)$$

| Comparison | $p_j$ | $a_j = \Phi^{-1}(p_j)$ |
|-----------|------|------|
| A vs B | 0.908 | 1.329 |
| A vs C | 0.918 | 1.392 |
| A vs D | 0.924 | 1.433 |

Then compute the approximate correlation in probit space. The key formula involves the shared dependence on $\theta_A$:

$$\rho_{jl} \approx \frac{\text{Cov}(\theta_j, \theta_l)}{\sqrt{\text{Var}_j \cdot \text{Var}_l}}$$

where the variances account for the leader's influence. For our $\boldsymbol{\alpha}$, the correlations $\rho_{jl}$ between the three competitor pairs are all small positive numbers (around 0.02–0.05) — the competitors are weakly correlated.

### Step 4: The correction term

The Gaussian copula correction adds a term for each pair of competitors $(j, l)$ that accounts for their correlation:

$$\text{correction} = \sum_{j < l} \rho_{jl} \cdot \phi(a_j) \cdot \phi(a_l) \cdot \prod_{m \neq j, l} p_m$$

where $\phi$ is the standard normal PDF (the bell curve height at each probit value).

What this says in English: for each pair of correlated comparisons, compute how much the bell curves overlap (the $\phi \cdot \phi$ term), scale by the correlation strength ($\rho_{jl}$), and multiply by the probability that A beats everyone *else* ($\prod_{m \neq j,l} p_m$).

### Step 5: Damping and final answer

The first-order Gaussian copula tends to over-correct, especially for larger $K$. A calibrated damping factor $d(K)$ brings it in line with Monte Carlo truth:

$$d(K) = 0.637 + 0.206 \cdot e^{-0.587(K - 3)}$$

For $K = 4$: $d(4) \approx 0.752$.

The final exceedance probability:

$$P(\text{leader best}) = \underbrace{\prod_{j \neq k^*} p_j}_{\text{independence assumption}} + d(K) \cdot \underbrace{\text{correction}}_{\text{correlation adjustment}}$$

For our example: $0.770 + 0.752 \times (\text{small correction}) \approx 0.78$.

So after 3 queries using the Sum method, the exceedance probability is about 78% — the system is fairly confident A is correct, but hasn't yet crossed $\tau = 0.95$.

**The exceedance also handles batches.** The function accepts an $(M, K)$ matrix — $M$ different $\boldsymbol{\alpha}$ vectors at once — and vectorizes the whole computation. This matters for MoM+Bayes, which evaluates exceedance at 80 grid points per stopping check.

## 3.4 Method 2: Sum — Dirichlet pseudo-count accumulation

The model is simple — we covered it in Level 2. Starting from $\text{Dir}(\mathbf{1})$:

$$\boldsymbol{\alpha}^{(N)} = \mathbf{1} + \sum_{n=1}^{N} \mathbf{p}^{(n)}$$

The total concentration is always $S_N = K + N$, since $\sum_k p_k^{(n)} = 1$. Evidence accumulates at rate 1 per query — **linear**, not exponential like Product. This is why Sum doesn't need temperature: the posterior physically can't concentrate faster than $O(N)$.

The stopping criterion:

$$c_N^{\text{sum}} = \text{copula\_exceedance}(\boldsymbol{\alpha}^{(N)})$$

## 3.5 Method 3: Dirichlet MLE — fitting the distribution shape

### What MLE is doing conceptually

Unlike Sum (which assumes each probability vector contributes one pseudo-count) or Product (which assumes each is a likelihood), MLE asks: "If these $N$ vectors were genuinely drawn from *some* Dirichlet distribution, which Dirichlet fits best?"

The answer depends on two things: the *direction* (which answer leads) and the *concentration* (how tightly the vectors cluster). MLE estimates both from the data.

### The log-likelihood

Given $N$ observations assumed to be drawn from $\text{Dir}(\boldsymbol{\alpha})$, the log-likelihood is:

$$\log L(\boldsymbol{\alpha}) = N \left[\log \Gamma(S) - \sum_{k=1}^{K} \log \Gamma(\alpha_k)\right] + \sum_{k=1}^{K}(\alpha_k - 1) \cdot N \bar{\ell}_k$$

The $\Gamma$ (Gamma function) is a generalization of the factorial — $\Gamma(n) = (n-1)!$ for integers, but it also works for non-integers. It appears here because the Dirichlet distribution's normalization constant is built from Gamma functions. The $\Gamma$ terms ensure everything integrates to 1. The important part for intuition is the second sum: $(\alpha_k - 1) \cdot N\bar{\ell}_k$, where $\bar{\ell}_k = \frac{1}{N}\sum_n \log p_k^{(n)}$ is the average log-probability for component $k$.

**What this means:** the log-likelihood is high when each $\alpha_k$ is large for components where the observations consistently assign high probability (large $\bar{\ell}_k$, i.e., close to zero since logs of numbers near 1 are near zero), and small for components where the observations consistently assign low probability (very negative $\bar{\ell}_k$).

For our example: $\bar{\ell}_A = -0.128$ (barely negative — the model consistently puts high mass on A), while $\bar{\ell}_D = -4.01$ (very negative — D consistently gets almost no mass).

### Minka's fixed-point iteration

Setting the gradient to zero gives:

$$\psi(\alpha_k) = \psi(S) + \bar{\ell}_k$$

where $\psi$ is the **digamma function** — the derivative of $\log \Gamma$. If the Gamma function is a generalized factorial, the digamma is its "rate of change on a log scale." It's a smooth, monotonically increasing function that maps positive numbers to real numbers. We don't need to compute it by hand — it's a standard library function (scipy.special.digamma). The key property: because $\psi$ is monotonically increasing, the equation above has exactly one solution for each $\alpha_k$, and we can find it by iterating:

$$\alpha_k^{\text{new}} = \psi^{-1}\!\left(\psi(S^{\text{current}}) + \bar{\ell}_k\right)$$

In words: given the current total concentration $S$, compute where each component's $\alpha_k$ should be based on the sufficient statistics $\bar{\ell}_k$. The inverse digamma $\psi^{-1}$ is computed via **Newton's method** — a standard root-finding technique that repeatedly improves a guess by using the function's slope (here, the slope comes from the **trigamma function** $\psi'$, which is just the second derivative of $\log \Gamma$). Each Newton step gives a better estimate; a few iterations suffice.

This converges in about 10–20 iterations. Each iteration is cheap — just digamma evaluations and inversions.

### Regularization for small N

When $N = 2$ or when some $p_k^{(n)} \approx 0$, two things can go wrong: $\log p_k^{(n)} \to -\infty$ (blowing up the sufficient statistics), and the MLE can degenerate (fitting an extreme distribution from minimal data).

Two fixes:
1. **Label smoothing:** mix each probability with a tiny uniform component: $p_k \leftarrow (1 - \epsilon) p_k + \epsilon/K$ where $\epsilon = 10^{-3}$. In practice, this replaces any exact-zero probability with a tiny positive number (0.00025), which ensures $\log p_k$ never blows up to $-\infty$.
2. **Prior pseudo-observation:** add a uniform vector $\mathbf{1}/K$ to the sufficient statistics with weight 1. This pulls the fit toward uniform when data is scarce, and has negligible effect when data is plentiful.

### Stopping criterion

$$c_N^{\text{MLE}} = \text{copula\_exceedance}(\hat{\boldsymbol{\alpha}})$$

Same copula as Sum, but on the MLE-fitted $\boldsymbol{\alpha}$ instead of the pseudo-count $\boldsymbol{\alpha}$.

## 3.6 Method 4: MoM — the variance ratio shortcut

### Deriving the formula

For a Dirichlet with concentration $\alpha_0$ and mean direction $\boldsymbol{\mu}$ (so $\boldsymbol{\alpha} = \alpha_0 \boldsymbol{\mu}$):

$$\text{Var}(X_k) = \frac{\mu_k(1 - \mu_k)}{\alpha_0 + 1}$$

This is a standard Dirichlet property. The key insight: if we divide both sides by $\mu_k(1 - \mu_k)$, we get:

$$R \equiv \frac{\text{Var}(X_k)}{\mu_k(1 - \mu_k)} = \frac{1}{\alpha_0 + 1}$$

This ratio $R$ is the **same for all components** $k$. It only depends on the concentration $\alpha_0$. If we can estimate $R$, we can recover $\alpha_0 = (1 - R)/R$.

### The pooled estimator

We could estimate $R$ separately for each component and take a median. But when $\hat{\mu}_k \approx 0$ (like our component D at 0.02), both the numerator $s_k^2$ and the denominator $\hat{\mu}_k(1 - \hat{\mu}_k)$ are near zero, and their ratio is dominated by noise.

The pooled estimator sums both numerators and denominators across components:

$$\hat{R} = \frac{\sum_{k=1}^{K} s_k^2}{\sum_{k=1}^{K} \hat{\mu}_k(1 - \hat{\mu}_k)}$$

This is a ratio of sums, not a sum of ratios. Components with $\hat{\mu}_k \approx 0$ contribute negligibly to both top and bottom, naturally downweighting themselves.

### Why this estimator works (consistency)

As $N \to \infty$: $s_k^2 \to \text{Var}(X_k)$ and $\hat{\mu}_k(1 - \hat{\mu}_k) \to \mu_k(1 - \mu_k)$. By the **continuous mapping theorem** — which says that if your inputs converge to the right values, then any smooth function of those inputs also converges to the right value — and because the denominator is bounded away from zero (at least one $\mu_k$ is well away from 0 and 1, so we're not dividing by something that vanishes):

$$\hat{R} \to \frac{\sum_k \text{Var}(X_k)}{\sum_k \mu_k(1 - \mu_k)} = \frac{\sum_k \mu_k(1-\mu_k)/(\alpha_0+1)}{\sum_k \mu_k(1-\mu_k)} = \frac{1}{\alpha_0 + 1} = R$$

So $\hat{R}$ is a consistent estimator of $R$. The common factor $\mu_k(1-\mu_k)$ cancels in the ratio, leaving exactly what we want.

### Stopping criterion

Set $\hat{\boldsymbol{\alpha}} = \hat{\alpha}_0 \cdot \hat{\boldsymbol{\mu}}$ (estimated concentration times estimated mean direction):

$$c_N^{\text{MoM}} = \text{copula\_exceedance}(\hat{\alpha}_0 \cdot \hat{\boldsymbol{\mu}})$$

The clamp $\hat{\alpha}_0 \in [1, 200]$ handles edge cases where $\hat{R} \leq 0$ or $\hat{R} \geq 1$.

## 3.7 Method 5: MoM + Bayes — marginalizing over concentration uncertainty

This is the most involved method, so let's build it piece by piece.

### The problem with point estimates at small N

In Level 2 we saw that MoM gave us $\hat{\alpha}_0 = 173$ from 3 observations. But how much should we trust that number?

The sample variance from $N = 3$ observations has $N - 1 = 2$ degrees of freedom per component. With $K = 4$ components pooled, that's $\nu = K(N-1) = 8$ total degrees of freedom.

To understand what 8 degrees of freedom means in practice: a chi-squared variable with 8 df has a 95% confidence interval that spans from about 0.34× to 2.73× its expected value. So our estimated $\hat{R}$ could plausibly be anywhere from $0.34 \times 0.0057 = 0.0019$ to $2.73 \times 0.0057 = 0.016$. Translating back to $\alpha_0$: anywhere from roughly 60 to 520. That's a huge range — yet MoM reports a single number (173) as if it's certain.

At $N = 10$, the pooled df is $K(N-1) = 36$. The 95% interval shrinks to roughly $0.67\times$ to $1.40\times$ the expected value. Now the estimate is trustworthy.

The Bayesian wrapper accounts for this uncertainty explicitly.

### Step 1: The sampling distribution of R̂

We need to know how much our estimate $\hat{R}$ can bounce around relative to the true $R$. This is where the **chi-squared distribution** comes in.

**What's a chi-squared distribution?** If you take $\nu$ independent standard normal random variables, square them, and add them up, the result follows a $\chi^2(\nu)$ distribution. The parameter $\nu$ is the degrees of freedom. A chi-squared variable is always positive, has mean $\nu$, and gets more concentrated (relative to its mean) as $\nu$ increases. It shows up naturally whenever you're estimating variances from data, because sample variances are essentially sums of squared deviations.

Under the Dirichlet model, the pooled sample variance $\sum_k s_k^2$ approximately follows a scaled chi-squared. The key relationship:

$$\frac{\hat{R}}{R_{\text{true}}} \cdot \nu \approx \chi^2(\nu)$$

where $R_{\text{true}} = 1/(\alpha_0 + 1)$ is the true variance ratio and $\nu = K(N-1)$ is the degrees of freedom.

What this says in English: the ratio of our estimate to the truth, scaled by degrees of freedom, follows a chi-squared distribution. When $\nu$ is small (few observations), the chi-squared is wide and $\hat{R}$ bounces around a lot relative to truth. When $\nu$ is large, the chi-squared concentrates around $\nu$ and our estimate is precise.

This lets us write the likelihood — how probable is our observed $\hat{R}$ given a specific true $\alpha_0$:

$$L(\hat{R} \mid \alpha_0) = \frac{\nu}{R_{\text{true}}} \cdot f_{\chi^2}\!\left(\frac{\hat{R} \cdot \nu}{R_{\text{true}}};\ \nu\right)$$

where $f_{\chi^2}(\cdot; \nu)$ is the chi-squared PDF and $R_{\text{true}} = 1/(\alpha_0 + 1)$.

**Why chi-squared?** For any set of $N$ observations from a distribution with variance $\sigma^2$, the sample variance $s^2$ satisfies $(N-1)s^2/\sigma^2 \sim \chi^2(N-1)$. We're pooling $K$ such terms, giving $K(N-1)$ degrees of freedom. This is approximate (Dirichlet marginals aren't Gaussian), but adequate for the moderate sample sizes we work with.

### Step 2: The prior

We place a Gamma prior on $\alpha_0$:

$$\pi(\alpha_0) = \text{Gamma}(\alpha_0 \mid \text{shape}=2,\ \text{scale}=5)$$

This has mode = 5, mean = 10, and substantial spread. It says: "Before seeing data, I think models are moderately consistent, but I'm not very sure."

**Why Gamma?** It's a natural choice for a prior on a positive quantity ($\alpha_0$ can't be negative). The **Gamma distribution** is defined by two parameters (shape and scale) and is always positive. With shape > 1, it has a single peak and a long right tail. Our shape=2, scale=5 gives a distribution that peaks at 5 but stretches out toward larger values — we're not ruling out very high concentration, just saying it's less likely a priori.

**Is the prior sensitive?** Not much. At $N \geq 4$, the likelihood from 4+ observations overwhelms the prior. The prior matters most at $N = 2$, where it provides the regularization that MoM's clamp does crudely.

### Step 3: The posterior on α₀

Bayes' theorem says: the probability of a parameter given the data is proportional to "how likely the data would be if this parameter were true" (the likelihood) times "how likely we thought this parameter was before seeing data" (the prior):

$$P(\alpha_0 \mid \hat{R}, N) \propto \underbrace{L(\hat{R} \mid \alpha_0)}_{\text{likelihood: how well does this } \alpha_0 \text{ explain our } \hat{R}\text{?}} \cdot \underbrace{\pi(\alpha_0)}_{\text{prior: how plausible was this } \alpha_0 \text{ before seeing data?}}$$

The $\propto$ ("proportional to") means we compute the right side for every candidate $\alpha_0$, then normalize so it all sums to 1. We evaluate this on a grid of $M = 80$ values of $\alpha_0$, log-spaced from 0.5 to 300. At each grid point $g_i$:

1. Compute the true $R$ for that $\alpha_0$: $R_i = 1/(g_i + 1)$
2. Compute the likelihood: evaluate the chi-squared PDF at $\hat{R} \cdot \nu / R_i$ with $\nu$ degrees of freedom
3. Multiply by the Gamma prior density at $g_i$
4. Store the unnormalized weight $w_i$

Then normalize: $\tilde{w}_i = w_i / \sum_j w_j$.

**With our example** ($\hat{R} = 0.0057$, $N = 3$, $\nu = 8$): the likelihood peaks where $R_{\text{true}} \approx \hat{R}$, i.e., where $\alpha_0 \approx 173$. But at $\nu = 8$, the peak is broad — significant weight extends down to $\alpha_0 \approx 50$ and up past 300. The Gamma prior gently pulls toward moderate values, slightly lowering the peak.

### Step 4: Marginalizing the exceedance

**Marginalization** means "averaging over something you don't know." We don't know the true $\alpha_0$, but we have a posterior that tells us which values are plausible. So instead of picking one $\alpha_0$ and computing the exceedance (which is what MoM does), we compute the exceedance at *every* plausible $\alpha_0$ and take a weighted average — weighted by how plausible each value is. This "integrates out" the nuisance parameter $\alpha_0$.

Now the payoff. For each grid point $g_i$, we can compute the copula exceedance as if $\alpha_0 = g_i$ were the true concentration:

$$\text{exc}_i = \text{copula\_exceedance}(g_i \cdot \hat{\boldsymbol{\mu}})$$

This calls the same copula function from Section 3.3, with $\boldsymbol{\alpha} = g_i \cdot [0.88, 0.063, 0.037, 0.02]$.

At high $g_i$ (say 200), the Dirichlet is extremely concentrated → the exceedance is near 1.0 (A is almost certainly best).

At low $g_i$ (say 2), the Dirichlet is diffuse → the exceedance is closer to 0.25 (almost any answer could be best).

The **marginalized exceedance** is the weighted average:

$$c_N^{\text{Bayes}} = \sum_{i=1}^{M} \tilde{w}_i \cdot \text{exc}_i$$

In English: "For every plausible concentration level, compute how confident I'd be that A is best. Then average those confidences, weighted by how plausible each concentration level is given my data."

When $N$ is small: the weights $\tilde{w}_i$ are spread across many grid points, including low-concentration points where $\text{exc}_i$ is small. This **drags the average down** → the system stays cautious.

When $N$ is large: the weights concentrate on a narrow range of $\alpha_0$ values, and $c_N^{\text{Bayes}} \approx c_N^{\text{MoM}}$.

### The crossover

At $N = 2$ ($\nu = 4$): the posterior is very wide. Even if $\hat{R}$ suggests high consistency, the method hedges — it almost never stops at $N = 2$.

At $N = 4$-$5$ ($\nu = 12$-$16$): the data starts to dominate the prior. The system transitions from "cautious" to "data-driven."

At $N = 10$ ($\nu = 36$): the posterior is tight. The 95% interval on $\hat{R}/R$ is roughly $[0.67, 1.40]$. The Bayesian wrapper adds essentially nothing over plain MoM — it has converged.

This is exactly the behavior you want for adaptive stopping. The wrapper is most conservative precisely when the evidence is weakest, and steps aside when the evidence is strong.

### Computational cost

The 80-point grid requires 80 copula evaluations per stopping check. Each copula call is $O(K^2) \approx 16$ operations. Total: roughly 1300 floating-point operations per check — negligible compared to even one LLM forward pass.

## Dirichlet distribution — quick reference

For $\mathbf{X} \sim \text{Dir}(\boldsymbol{\alpha})$ with $S = \sum_k \alpha_k$:

| Property | Formula |
|----------|---------|
| Mean | $E[X_k] = \alpha_k / S$ |
| Variance | $\text{Var}(X_k) = \frac{\alpha_k(S - \alpha_k)}{S^2(S + 1)} = \frac{\mu_k(1-\mu_k)}{S+1}$ |
| Covariance ($j \neq k$) | $\text{Cov}(X_j, X_k) = \frac{-\alpha_j \alpha_k}{S^2(S + 1)}$ |
| Mode ($\alpha_k > 1$) | $\frac{\alpha_k - 1}{S - K}$ |
| Marginals | $X_k \sim \text{Beta}(\alpha_k, S - \alpha_k)$ |
| Aggregation | $\frac{X_A}{X_A + X_B} \sim \text{Beta}(\alpha_A, \alpha_B)$ |

The concentration $\alpha_0 = S$ controls peakiness. As $\alpha_0 \to \infty$ with $\boldsymbol{\mu} = \boldsymbol{\alpha}/\alpha_0$ fixed: $\text{Dir}(\alpha_0\boldsymbol{\mu}) \to \delta_{\boldsymbol{\mu}}$ (point mass). As $\alpha_0 \to 0$: mass concentrates on the simplex vertices.

## Summary comparison

| | **Product** | **Sum** | **MLE** | **MoM** | **MoM+Bayes** |
|---|---|---|---|---|---|
| **Estimates** | $P(k \mid \text{data})$ | Dirichlet $\boldsymbol{\alpha}$ (additive) | Dirichlet $\boldsymbol{\alpha}$ (MLE) | $\alpha_0$ (point) | $P(\alpha_0 \mid \text{data})$ |
| **Confidence metric** | $\max P(k)$ | Copula exceedance | Copula exceedance | Copula exceedance | Marginalized exceedance |
| **Hyperparameters** | $T$ (temperature) | None | prior_strength, $\epsilon$ | Clamp bounds | Gamma prior |
| **Consistency-aware** | No | No | Yes | Yes | Yes |
| **Uncertainty about $\alpha_0$** | N/A | N/A | No | No | **Yes** |
| **Small-$N$** | Overconfident (without $T$) | Conservative | Noisy | Point estimate | Conservative |
| **Large-$N$** | Concentrates exponentially | Concentrates as $O(N)$ | Converges to true $\boldsymbol{\alpha}$ | Converges to true $\alpha_0$ | Converges to MoM |
| **Cost** | $O(NK)$ | $O(NK + K^2)$ | $O(NK + IK)$ | $O(NK + K^2)$ | $O(NK + MK^2)$ |
| **Best when** | Good calibration data | Quick baseline | $N \geq 5$ | $N \geq 3$, fast | Small $N$, safety matters |

---

# Empirical Results: How the Five Methods Actually Perform

*Data from the QuALITY dataset (4609 long-form reading comprehension MCQs). Three models: Gemma 4 E4B, Qwen 3 8B, Qwen 3.5 9B — all Q4_K_M quantised. 10 shuffle permutations per question, adaptive stopping with N_max = 10. Product uses T = 3.0 (ECE-optimal); other methods use raw logprobs. Qwen 3.5 data is from 4260/4609 questions (4 runs still in progress at time of export).*

## The headline numbers (τ = 0.95)

| Method | Gemma 4 (base 76.8%) | Qwen 3 (base 76.4%) | Qwen 3.5 (base 82.7%) |
|--------|----------------------|----------------------|------------------------|
| **Product** | avg_N=1.9, acc=76.3%, cap=2.1% | avg_N=3.3, acc=76.4%, cap=5.6% | avg_N=5.3, acc=82.8%, cap=18.0% |
| **Sum** | avg_N=6.2, acc=76.8%, cap=17.4% | avg_N=6.7, acc=76.4%, cap=22.7% | avg_N=7.4, acc=82.7%, cap=31.4% |
| **MLE** | avg_N=7.2, acc=76.6%, cap=28.4% | avg_N=7.4, acc=76.3%, cap=33.7% | avg_N=6.4, acc=82.8%, cap=26.5% |
| **MoM** | avg_N=3.3, acc=76.5%, cap=15.3% | avg_N=3.7, acc=76.2%, cap=19.9% | avg_N=3.4, acc=82.4%, cap=16.4% |
| **MoM+Bayes** | avg_N=3.4, acc=76.6%, cap=16.1% | avg_N=3.9, acc=76.4%, cap=21.8% | avg_N=4.1, acc=82.7%, cap=22.1% |

Where **avg_N** = mean queries per question, **acc** = accuracy of the stopping answer, **cap** = fraction of questions escalated (hit N_max without reaching τ).

## What the numbers mean

### Product: fastest but fragile

Product stops the earliest — just 1.9 queries on Gemma — because temperature-scaled logprobs are still somewhat overconfident for very confident predictions. The real problem is the **escalation rate**: on Gemma, only 2.1% of questions get flagged as hard. That's way too few. The model gets about 23% of questions wrong, but Product only identifies 2% as needing help. Most incorrect answers sail through uncaught.

Worse: Product's behaviour is wildly **inconsistent across models**. On Gemma it stops almost immediately (1.9 queries), on Qwen 3.5 it needs 5.3. This happens because temperature scaling is a one-size-fits-all correction. T=3 was optimised for Gemma's overconfidence level; Qwen 3.5's logprobs have a different distribution, so the same T produces different stopping dynamics.

On Gemma, post-escalation accuracy (esc2) is actually **lower** than baseline (76.4% vs 76.8%). Escalation hurts — probably because the tiny fraction of questions that do get escalated are genuinely ambiguous, and the think mode doesn't help with ambiguity, it helps with complexity.

### Sum and MLE: thorough but expensive

Sum and MLE use the most queries (6–7.4) and achieve the highest escalation rates. They're doing more work to be more certain.

But MLE is consistently **dominated** — it's almost always the slowest method (most queries) without any accuracy advantage over Sum. The iterative Dirichlet fit is expensive and adds noise at small N, but doesn't buy anything once the copula exceedance takes over. MLE is the method you'd use if you had plentiful data and wanted the best possible Dirichlet fit; for adaptive stopping with N ≤ 10, it's overkill.

Sum's lack of consistency awareness turns out to be less of a problem than expected — because most questions in the dataset either have consistently high model confidence or consistently low model confidence. The pathological case (high mean, high variance) is real but relatively rare.

### MoM: fast and consistent, but blunt

MoM hits a consistent avg_N of 3.3–3.7 across all three models. It's the most **stable** stopper — it doesn't care about the model's intrinsic confidence level the way Product does, because it's measuring the *data's* consistency, not the logprobs' magnitude.

Its weakness shows in the Pareto comparison: the best MoM operating point achieves lower accuracy than the other methods because it treats its noisy $\hat{\alpha}_0$ as gospel and occasionally stops on a wrong variance estimate. On Gemma, the best MoM Pareto point matches baseline exactly (76.78%) — no improvement at all.

### MoM+Bayes: the Goldilocks method

MoM+Bayes adds just 0.1–0.3 extra queries over plain MoM — the Bayesian wrapper makes it slightly more cautious at small N, sampling a few more observations before committing. The payoff:

| Model | Best MoM (Pareto) | Best MoM+Bayes (Pareto) | Δ |
|-------|-------------------|-------------------------|---|
| Gemma 4 | 76.78%, N=3.15 | 76.94%, N=3.61 | **+0.16%** |
| Qwen 3 | 76.83%, N=3.99 | 77.11%, N=4.35 | **+0.28%** |
| Qwen 3.5 | 83.38%, N=3.82 | 83.47%, N=3.61 | **+0.09%** |

Small but consistent gains on every model. And on Qwen 3.5, MoM+Bayes actually achieves its best accuracy with **fewer** queries than MoM — the wrapper's conservatism at N=2 prevents premature stopping on what would have been wrong answers, saving the cost of escalation.

Across the Pareto frontier, MoM+Bayes matches or beats Product's peak accuracy on every model while requiring no temperature hyperparameter. It's the only method whose stopping behavior **adapts automatically to sample quality** — cautious at N=2 when the variance estimate is noisy, decisive at N=5 when the estimate firms up.

## The efficiency picture

Measuring accuracy-per-query at Pareto-optimal settings:

| Method | Gemma 4 | Qwen 3 | Qwen 3.5 |
|--------|---------|--------|----------|
| Product | 0.211 | 0.136 | 0.199 |
| Sum | 0.124 | 0.116 | 0.164 |
| MLE | 0.107 | 0.104 | 0.166 |
| MoM | 0.244 | 0.193 | 0.218 |
| **MoM+Bayes** | **0.213** | **0.177** | **0.231** |

MoM wins raw efficiency because it's the fastest, but its accuracy ceiling is lower. MoM+Bayes trades a small efficiency penalty for meaningfully better accuracy — the best cost/quality tradeoff across models.

## Escalation rates: who catches hard questions?

The escalation rate (cap%) tells you what fraction of questions the method flags as "too uncertain — send to a more expensive reasoning mode." This is critical for routing: too low means you miss hard questions; too high means you waste expensive compute on easy ones.

| Method | Gemma 4 | Qwen 3 | Qwen 3.5 |
|--------|---------|--------|----------|
| Product | **2.1%** ⚠️ | 5.6% | 18.0% |
| Sum | 17.4% | 22.7% | 31.4% |
| MLE | 28.4% | 33.7% | 26.5% |
| MoM | 15.3% | 19.9% | 16.4% |
| MoM+Bayes | 16.1% | 21.8% | 22.1% |

Product's 2.1% on Gemma is a red flag — it means 97.9% of questions are considered "easy enough" to answer directly, yet the model is wrong on ~23% of them. Most errors slip through undetected. MoM and MoM+Bayes sit in a healthy middle ground: enough escalation to catch genuinely hard questions (15–22%) without drowning in false alarms.

## Remaining concerns

1. **Qwen 3.5 data is incomplete.** Only 4260 of 4609 questions — 4 runs were still in progress. Numbers may shift by a percentage point or two when all data is in.

2. **No CoT/think results yet.** We're comparing direct-mode shuffle results. The escalation gains (esc2) assume "send to think mode" helps — but we don't yet have the think-mode data to confirm. The -0.4% esc2 result for Product on Gemma hints that think mode may not always help.

3. **MLE should probably be dropped.** It's dominated on cost (always slowest) and adds no accuracy. The iterative fitting adds complexity for no benefit in the N ≤ 10 regime. The paper should report it for completeness but recommend against it in practice.

4. **Temperature sensitivity.** Product is the only method requiring a hyperparameter that must be recalibrated per model. For deployment, this is a significant practical disadvantage — you need a held-out calibration set, and if the domain shifts, T may be stale.

---

# Temperature Scaling: Not Just a Product Trick

The raw results above tell a clear story — but they also contain a hidden flaw. Until now, temperature scaling was treated as a Product-specific fix for overconfident logprobs. But the same overconfidence problem infects *every* method. Temperature scaling turns out to be a preprocessing step that all five methods should use, for different but related reasons.

## The core problem: illusory consistency

Remember what temperature does. A raw vector like [0.91, 0.05, 0.03, 0.01] lives near the simplex vertex — almost all mass on one answer. Temperature scaling with T=3 transforms it to [0.52, 0.20, 0.17, 0.12] — still favouring A, but honest about uncertainty.

Now imagine two shuffled queries producing:
- Query 1: [0.91, 0.05, 0.03, 0.01]
- Query 2: [0.90, 0.06, 0.03, 0.01]

The sample variance of the A component is $s_A^2 = 0.00005$. Tiny! The MoM variance ratio gives $\hat{R} \approx 0$, which means $\hat{\alpha}_0 \to \infty$ — the system concludes the model is *extremely* consistent and stops immediately.

But is the model genuinely consistent? Or are these two vectors just both jammed against the vertex by overconfident logprobs, hiding real uncertainty? If the model actually assigns 75% to A (its true calibrated confidence), then after temperature scaling those vectors would become something like [0.52, 0.20, ...] and [0.50, 0.22, ...] — and the variance would be 10× larger.

This is **illusory consistency**: overconfident logprobs compress all vectors toward the simplex vertices, making the spread between them appear tiny even when the model genuinely wavers across shuffles. Any method that measures consistency from raw logprobs will be fooled.

## How temperature affects each method — and what we'd expect

### Product: temperature is essential (T* ≈ 3.0)

Product multiplies probability vectors together. Without temperature, [0.91]^3 = 0.75 for just the leading component — the posterior concentrates exponentially, and after 2-3 queries the stopping threshold is blown past regardless of consistency. Temperature is the only thing that makes Product usable. We already know T* ≈ 3.0 from ECE optimization (ECE drops from 0.186 to 0.012).

**Prediction:** T* ≈ 3.0 across models, possibly slightly higher for the most overconfident model (Qwen 3).

### Sum: temperature should barely matter (T* ≈ 1.0)

Sum accumulates pseudo-counts: $\alpha_k = 1 + \sum_n p_k^{(n)}$. Temperature changes the *magnitude* of each contribution (scaled vectors sum to 1 either way), but the *ordering* doesn't change — if A leads in raw vectors, A leads in scaled vectors. The total concentration $S = K + N$ is fixed regardless of T because probability vectors always sum to 1.

The only way temperature helps Sum is indirect: if raw vectors are so overconfident that the exceedance computation saturates (pairwise Beta comparisons hit near-1.0 probabilities where the copula correction vanishes). In practice, this should be a small effect.

**Prediction:** T* = 1.0 or very close. If T* is significantly above 1.0, it means the overconfidence is so extreme that even linear accumulation is distorted — that would be a surprising and important finding.

### MLE: temperature probably helps modestly (T* ≈ 1.5–2.5)

MLE fits the Dirichlet shape from the *variation* across vectors. It already accounts for overconfidence somewhat — if vectors cluster near [0.9, 0.05, 0.03, 0.02], the MLE fits a high-concentration Dirichlet pointing in that direction. But the sufficient statistics $\bar{\ell}_k = \frac{1}{N}\sum_n \log p_k^{(n)}$ are computed from raw log-probabilities. When raw probs are near 0 or 1, the logs are extreme (−0.1 or −4.6), and the fitted α can be dominated by these extremes.

Temperature scaling would moderate the log-probabilities, giving the MLE a more balanced signal to fit from. But MLE already has regularisation (label smoothing + prior pseudo-observation), which partially compensates.

**Prediction:** T* somewhere between 1.5 and 2.5. Lower than Product because the MLE's iterative fit partly self-corrects.

### MoM: temperature directly fixes the variance estimate (T* ≈ 1.5–2.5)

This is where the illusory consistency problem bites hardest. MoM computes $\hat{R} = \sum s_k^2 / \sum \hat{\mu}_k(1-\hat{\mu}_k)$. When raw vectors are vertex-compressed:
- The numerator (sample variance) is artificially small
- The denominator ($\hat{\mu}(1-\hat{\mu})$) is also small (because $\hat{\mu} \approx 0.9$ gives $0.9 \times 0.1 = 0.09$, not that small)
- But the numerator shrinks *faster* than the denominator, because vertex compression reduces spread more than it reduces the mean

Temperature scaling inflates the numerator (variance increases as vectors spread away from vertices) while keeping the denominator more stable. The result: $\hat{R}$ increases, $\hat{\alpha}_0$ decreases, and the system needs more evidence before stopping.

**Prediction:** T* ≈ 1.5–2.5. Lower than Product's 3.0 because the concentration estimate already partially accounts for overconfidence — it just needs the raw variance to be in a reasonable range rather than artificially compressed. The key insight: *temperature fixes the shape of each vector; concentration fixes the agreement between vectors. You need less of one when you have the other.*

### MoM+Bayes: least temperature needed (T* ≈ 1.0–2.0)

MoM+Bayes already has a built-in mechanism for handling unreliable variance estimates — the Gamma prior and chi-squared likelihood explicitly model how much $\hat{R}$ can bounce around given $N$ observations. When $\hat{R}$ is suspiciously low (as it is with vertex-compressed vectors), the prior pulls $\alpha_0$ toward moderate values, and the wide posterior drags down the marginalized exceedance.

Temperature scaling would still help by giving the variance estimate a less distorted input signal. But the Bayesian machinery already does part of the job that temperature does for the other methods.

**Prediction:** T* ≈ 1.0–2.0, the lowest of any method (or tied with Sum). If MoM+Bayes + optimal T approaches Product + optimal T in cost-efficiency, that's the strongest result: the concentration estimation is doing real work that temperature alone can't replicate.

## The complementarity story

This gives us a clean narrative for the paper:

> **Temperature** calibrates the *magnitude* of each observation — how peaked or spread each probability vector is.
>
> **Concentration** calibrates the *agreement* between observations — how consistently the model points in the same direction across shuffles.
>
> These are complementary corrections. Product relies entirely on temperature. MoM/MoM+Bayes rely primarily on concentration. The optimal temperature for a concentration-aware method is lower than for a concentration-blind method — because you need less of one correction when you have the other.

If the data confirms that MoM+Bayes achieves the same accuracy as Product at lower T (and therefore with less sensitivity to the hyperparameter), it makes a compelling deployment argument: **concentration estimation reduces your dependence on calibration data.**

## Empirical results: what actually happened

*[To be filled with calibrated results from dashboard export. Key questions: Does T* vary across methods as predicted? Does MoM+Bayes need less T than Product? Does Sum's T* stay near 1.0? How much does calibration improve each method?]*

## Bottom line

For the paper's recommended configuration: **MoM+Bayes with τ ∈ [0.90, 0.99]**. No temperature tuning needed in the raw regime, consistent 3.4–4.4 queries across models, healthy escalation rates, and the best accuracy-per-query tradeoff. The Bayesian wrapper earns its keep at small N — exactly where adaptive stopping decisions are made. With temperature calibration, the recommendation may strengthen further — or the gap may close with simpler methods. The data will tell.
