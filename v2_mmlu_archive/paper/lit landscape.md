# Uncertainty quantification for LLMs: a comprehensive literature review

**The specific combination of question-stem paraphrasing with per-answer token-level logprob extraction for MCQA uncertainty estimation is a genuinely novel contribution, but two papers — CAPE and BayesPE — occupy adjacent territory and require careful differentiation.** After exhaustive search across 60+ queries covering all major venues (2022–2026), no paper was found that does exactly what is proposed: extracting per-answer-option logprobs (P(A), P(B), P(C), P(D)) from a local model across multiple paraphrases of the question itself and aggregating the probability vectors for uncertainty estimation. However, the gap is narrower than it might appear. The two most proximate threats vary *instruction templates* rather than question content, and the paraphrase-based UQ literature is overwhelmingly black-box by design. The constrained decoding finding — that JSON schema enforcement kills the logprob signal — appears to be a genuinely under-studied phenomenon with emerging but scattered evidence.

---

## 1. The two closest threats to novelty: CAPE and BayesPE

**CAPE (Calibrating Language Models via Augmented Prompt Ensembles)** by Jiang, Ruan, Huang et al. at the ICML 2023 Deployable Generative AI Workshop is the single most threatening prior work. CAPE uses GPT-4-generated paraphrased instruction templates combined with answer-option permutation, extracts per-option logprobs in both ENUM format (P(A), P(B), etc.) and ITEM format (summed log-likelihood of option text), and averages across ensemble members with equal weighting. It is applied to MCQA benchmarks including **MMLU, HellaSwag, TruthfulQA, and CommonsenseQA**, reporting ECE, accuracy, and Brier score improvements. Critically, CAPE paraphrases the *instruction template* (e.g., "Answer the following question" → "Select the correct answer"), not the question stem itself. It also focuses on calibration (ECE) rather than uncertainty-based selective prediction (AUROC). It requires no labeled validation data.

**BayesPE (Bayesian Prompt Ensembles)** by Tonolini, Aletras, Massiah, and Kazai at ACL 2024 Findings (Amazon Research) extends this general idea with Bayesian weighting. BayesPE generates ~10 semantically equivalent instruction prompts, extracts token-level class probabilities p(y|a_i, x) for each variant, and learns per-prompt weights via variational inference on a small labeled validation set (**20–50 examples**). It targets text classification (sentiment, topic) rather than MCQA, and is designed for black-box API access. Metrics include NLL, ECE, Brier score, and calibration curves. BayesPE requires supervised weight learning, operates on classification labels rather than A/B/C/D options, and varies instructions rather than question content.

**Key differentiation points for your contribution:** (a) You paraphrase the *question content itself*, which tests whether the model's knowledge is robust to surface-form variation of the semantic content — fundamentally different from instruction-template variation, which only tests prompt sensitivity; (b) you focus specifically on MCQA per-option logprob vectors; (c) you require no labeled validation data; (d) you evaluate with AUROC for selective prediction, not just calibration metrics; (e) you operate as a white-box method on local models. These distinctions are defensible but must be explicitly argued. Cite both papers prominently.

---

## 2. Paraphrase-based uncertainty estimation: a black-box literature

The paraphrase-based UQ literature is dominated by methods designed for closed-source models that deliberately avoid logprobs. **SPUQ** (Gao et al., EACL 2024) is the foundational perturbation-based framework, combining input paraphrasing with dummy tokens, system message changes, and temperature variation. It aggregates via text similarity (ROUGE, BERTScore, SBERT) or verbalized confidence — never logprobs — and achieves **~50% ECE reduction** on TruthfulQA and TriviaQA. **"Just Rephrase It!"** (Yang, Chen, Pitas; arXiv 2405.13907, 2024) provides theoretical grounding, showing that rephrasing produces a "tempered" version of the inaccessible categorical distribution. It tests four rephrasing strategies (rewording, expansion, permutation, LLM-based) on ARC-Challenge and OpenBookQA, reducing ECE by **10–40%** with purely black-box consistency checking.

**"Mapping from Meaning"** (Cox et al., AAAI 2025) provides the strongest theoretical case for why paraphrasing improves calibration, modeling prompt sensitivity as generalization error and showing that a sample from a *new paraphrase* provides more marginal information than an additional sample from the same paraphrase. It reports AUROC up to **87.4** for GPT-3.5 with 12 paraphrases × 4 samples. Crucially, it decomposes uncertainty into epistemic (cross-paraphrase) and aleatoric (within-paraphrase) components — but uses embedding-space variance, not logprobs. The **"Consistency Hypothesis"** paper (Xiao et al., UAI 2025) formalizes the implicit assumption underlying all these methods, testing three mathematical formulations and finding that geometric/harmonic mean aggregations of similarities outperform baselines across 8 benchmarks. A **Scenario-independent UE framework** (WWW 2025) proposes factor analysis to disentangle scenario-related noise from semantic content in paraphrased queries, and is explicitly compatible with any existing UE method including logprob-based ones — making it a potential complement to your approach rather than a competitor.

The consistent finding across this literature: **paraphrasing captures epistemic uncertainty that sampling alone misses**, but all methods leave logprob information on the table. Your work bridges this gap.

---

## 3. Logprob-based MCQA confidence: rich but single-prompt

**Kadavath et al. (Anthropic, 2022)** established the foundation: large base LMs are well-calibrated on MCQA when answers are formatted as lettered options and token probabilities are extracted. RLHF models are miscalibrated but fixable with temperature T=2.5. They introduced P(True) (self-evaluation) and P(IK) (trained predictor), reporting calibration curves and AUROC across MMLU, BIG-Bench, and TriviaQA. This remains the canonical reference for logprob-based MCQA evaluation.

**Plaut, Nguyen & Trinh (UC Berkeley, 2024; submitted ICLR 2025)** studied 15 chat-finetuned LLMs and found that maximum softmax probabilities (MSPs) of first-token predictions are consistently miscalibrated but still encode useful uncertainty information — wrong answers have smaller MSPs. Average AUROC of **60–69%** for best models across ARC-Challenge, HellaSwag, MMLU, TruthfulQA, and WinoGrande. They showed a strong correlation between QA accuracy and MSP correctness prediction but no correlation between accuracy and calibration error, suggesting **discrimination will improve with capabilities but calibration will not**.

A critical challenge for first-token logprob methods was identified by **Wang et al. (2024)** in "My Answer is C": first-token log probabilities and actual text outputs are misaligned in instruction-tuned models, with **mismatch rates exceeding 60%**. **Cerón et al. (2025)** proposed a "prefilling attack" — structured prefixes like "The correct option is:" — that substantially improves accuracy, calibration, and consistency of FTP-based evaluation. **Chen et al. (UC Berkeley, 2024)** corroborated this concern, showing model correctness in MCQ vs. free-generation formats is weakly correlated. **Boseak (2025)** compared raw log-likelihood, length-normalized log-likelihood, and softmax-based choice probability across ARC, BoolQ, and HellaSwag, finding no single scoring method universally best.

More sophisticated approaches include **LogU** (Ma et al., 2025), which reinterprets raw logits as Dirichlet distribution parameters for single-pass uncertainty decomposition, and **"Deep Think with Confidence"** (2025), which aggregates token-level confidence over overlapping spans rather than entire reasoning traces, showing that retaining the top 10% highest-confidence traces yields **+5.27 percentage point gains** over majority voting. **Joshi et al. (2025)** analyzed calibration across transformer layers, discovering a "confidence correction phase" in later layers and a low-dimensional direction in the residual stream that, when perturbed, improves ECE/MCE without degrading accuracy.

**The gap your work fills:** All logprob-based MCQA methods use a single prompt per question. None aggregate probability vectors across multiple paraphrases of the same question.

---

## 4. Self-consistency has been meaningfully extended

Wang et al. (ICLR 2023) famously found that sequence-level log probability doesn't improve self-consistency — majority voting matched or exceeded probability-weighted voting. This finding has been **directly challenged** by **CISC (Confidence Improves Self-Consistency)** from Taubenfeld et al. (ACL Findings 2025, Google Research/Hebrew University). CISC shows that the original finding reflected a normalization problem, not a fundamental limitation. Raw sequence probability scores are too clustered, but after within-question normalization, confidence signals become highly useful for weighted majority voting. Testing across 9 instruction-tuned LLMs on MATH, GSM8K, CommonsenseQA, and StrategyQA, CISC reduces required samples by **>40%**. The P(True) method (from Kadavath et al.) achieves best results. Counterintuitively, the most *calibrated* confidence method proved least effective — **discrimination matters more than calibration** for self-consistency weighting.

Other extensions include **Soft Self-Consistency** (Wang et al., ACL 2024), replacing discrete majority voting with continuous likelihood scores for agent tasks; **Universal Self-Consistency** (Chen et al., ICML 2024 Workshop), using LLMs themselves to select the most consistent answer; **Dynamic Self-Consistency** (Wan et al., 2024), adaptively sampling based on agreement; and **"Learning When to Sample"** (2026), training a confidence-aware framework to decide between single-path and multi-path reasoning, maintaining accuracy while using **up to 80% fewer tokens**.

None of these self-consistency extensions combine paraphrasing with token-level logprobs, but CISC's finding that properly normalized confidence signals do help is directly relevant — it suggests that aggregating logprobs across paraphrases could provide even richer confidence signals than within-prompt normalization alone.

---

## 5. Epistemic-aleatoric decomposition: a maturing theoretical landscape

**Hou et al. (ICML 2024 Oral)** is the most directly relevant decomposition paper. Their "Input Clarification Ensembling" generates clarifications of ambiguous inputs, feeds them to the LLM, and decomposes uncertainty via **H(Y|X) = E[H(Y|X,C)] + I(Y;C|X)**, where C represents clarifications. Critically, they *reverse* the standard BNN mapping: mutual information I(Y;C|X) captures **aleatoric** uncertainty (input ambiguity), while conditional entropy captures **epistemic** uncertainty. This reversal occurs because they vary *inputs* rather than *model parameters*. They report AUROC for mistake detection and ambiguity detection on AmbigQA, NQ, and GSM8K.

**Ling et al. (NAACL 2024)** formulate ICL as BNNs with in-context demonstrations as latent variables, using the standard decomposition where I(Y;Θ|X) = epistemic and E[H(Y|X,Θ)] = aleatoric. They test on classification tasks (EMOTION, CoLA, AG_News, SST2). **Yadkori et al. ("To Believe or Not to Believe Your LLM," NeurIPS 2024, DeepMind)** derive a mutual information metric to detect when only epistemic uncertainty is large, using iterative prompting where the LLM is conditioned on its own previous responses.

The **Spectral Uncertainty** framework (arXiv 2025) proposes a generalized decomposition using functional Bregman information and von Neumann entropy, claiming state-of-the-art on both ambiguity detection and correctness prediction. **Huo et al. (2025)** extend beyond binary decomposition using the chain rule of conditional entropy for multi-factor decomposition: H[y|x,w] = I(y;δ₁|x,w) + I(y;δ₂|δ₁,x,w) + ... + H[y|δ₁,...,δₖ,x,w].

Two critical theoretical papers temper enthusiasm: **Wimmer et al. (UAI 2023)** identify incoherencies in the standard MI-based decomposition, showing conditional entropy is neither a true expectation nor a bound on true aleatoric uncertainty. **"The Illusion of Certainty"** (2025) establishes theoretical bounds showing current UQ methods correlate with true epistemic uncertainty only in the absence of aleatoric uncertainty — under ambiguity, both entropy and MI lose consistent interpretation.

**For your work:** The information-theoretic decomposition via paraphrasing is well-established conceptually but has not been applied to MCQA logprob vectors specifically. Hou et al.'s reversed mapping (paraphrase variation → aleatoric, not epistemic) is important to address — you should discuss whether question paraphrasing reveals epistemic uncertainty (knowledge gaps robust to surface form) or aleatoric uncertainty (genuine question ambiguity), and argue that for semantic-preserving paraphrases, variation in logprob vectors reflects epistemic rather than aleatoric uncertainty.

---

## 6. Constrained decoding destroys logprob signals: emerging but scattered evidence

**Park et al. ("Grammar-Aligned Decoding," NeurIPS 2024)** formally proved that grammar-constrained decoding (GCD) distorts the LLM's probability distribution. When high-probability tokens are masked for grammar violations, remaining tokens are renormalized, amplifying relative differences. They demonstrated this with KL divergence measurements and proposed ASAp (Adaptive Sampling with Approximate Expected Futures) to progressively correct the distortion. Their key example: if GCD masks token "0," forcing the sequence "00000" with joint probability 0.45⁵ ≈ 10⁻⁸, the greedy token probability of 0.45 vastly overestimates the true grammatical mass.

**Schall & de Melo (RANLP 2025)** provide the most direct empirical evidence for your finding. Across **11 models**, their log probability analysis explicitly reveals that constrained decoding forces models away from their preferred natural language patterns into **lower-confidence structured alternatives**. Instruction-tuned models are particularly affected on generation tasks. **Tam et al. (EMNLP Industry 2024)** showed significant performance degradation under JSON-mode on reasoning tasks (GSM8K, Last Letter Concatenation), with stricter constraints causing greater degradation. They recommend the NL-to-Format strategy: generate freely, then convert. **DOMINO** (Beurer-Kellner et al., ICML 2024) identified subword misalignment as an additional source of distribution distortion, proposing subword-aligned constrained decoding. **AdapTrack** (2025) introduced backtracking-based constrained decoding with theoretical proofs that its distribution matches the model's true conditional distribution.

**None of the major UQ surveys discuss how constrained decoding affects uncertainty signal reliability.** This represents a significant blind spot. Your observation that JSON schema enforcement kills the logprob signal aligns perfectly with Park et al.'s theoretical analysis and Schall & de Melo's empirical findings. Reporting this formally — specifically in the context of uncertainty estimation — would itself be a contribution.

---

## 7. Surveys position the landscape clearly

Seven major surveys provide comprehensive coverage of the field:

| Survey | Venue | Key taxonomy |
|--------|-------|-------------|
| Shorinwa et al. | ACM Computing Surveys, 2025 | Token-level, self-verbalized, semantic-similarity, mechanistic interpretability |
| Geng et al. | NAACL 2024 | White-box vs. black-box confidence estimation + calibration methods |
| Huang et al. | arXiv, Oct 2024 | Theory-meets-practice: Bayesian, information-theoretic, ensemble |
| Xia et al. | ACL Findings, 2025 | Verbalizing, logit-based, semantic clustering, multi-sample (with experiments) |
| Da & Liu et al. | KDD 2025 | Aleatoric vs. epistemic × token vs. response level + conformal prediction |
| He et al. | Information Fusion, 2025 | Source-oriented: data, model, user uncertainty |
| Ye et al. | ICLR 2025 | Instruction-following UQ (beyond factuality tasks) |

Common open problems across all surveys: computational cost of multi-sample methods, long-form generation UQ, calibration degradation from RLHF/alignment, black-box access limitations, multimodal extension, and claim-level decomposition. **No survey identifies the intersection of paraphrasing and logprob analysis as a research direction**, confirming that your contribution addresses an unrecognized gap.

---

## Conclusion: a defensible but nuanced novelty claim

The proposed method — extracting per-answer logprobs from a local model across multiple question-stem paraphrases and aggregating probability vectors for MCQA uncertainty estimation — occupies a genuine gap at the intersection of two well-established but separate research streams. The paraphrase-based stream (SPUQ, "Just Rephrase It," Mapping from Meaning) is overwhelmingly black-box; the logprob-based stream (Kadavath, Plaut et al., Boseak) uses single prompts. **CAPE and BayesPE are the two papers that come closest** by combining prompt variants with logprobs, but both vary instruction templates rather than question content — a fundamentally different source of variation that tests prompt sensitivity rather than knowledge robustness.

The strongest positioning for your contribution requires five elements: (1) clearly differentiate question-stem paraphrasing from instruction-template variation, arguing the former probes knowledge robustness while the latter probes prompt sensitivity; (2) cite CAPE and BayesPE explicitly and explain the distinction; (3) evaluate with AUROC for selective prediction, not just ECE, since CISC showed discrimination matters more than calibration; (4) leverage the epistemic-aleatoric decomposition framework from Hou et al. but argue that semantic-preserving paraphrases capture epistemic rather than aleatoric uncertainty; and (5) report the constrained decoding finding formally, as it has no direct precedent in the UQ literature despite strong supporting evidence from the constrained generation literature. The information-theoretic decomposition (mutual information between paraphrase variation and output logprob distribution) provides a principled theoretical framework that connects to Ling et al. (NAACL 2024), Yadkori et al. (NeurIPS 2024), and the Spectral Uncertainty framework, giving the work theoretical depth beyond empirical contribution alone.