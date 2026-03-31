# Uncertainty quantification at the LLM decision boundary: a literature review

**The core question—can we measure whether a local LLM understood its input before it acts—represents a genuine gap in the literature.** Extensive work exists on output-side uncertainty (is the answer correct?) but virtually none on input-processing uncertainty (did the model comprehend the query and context?). This review maps the research landscape across nine interconnected areas, identifies the closest prior work, and catalogs datasets suitable for experimental validation. The proposed approach—paraphrased comprehension probes with logprob analysis on Qwen3 8B—sits at an intersection no existing paper occupies: pre-generation comprehension verification for three-way action selection (proceed/clarify/escalate) in RAG-augmented local deployments.

---

## Part A: What the literature covers and where the gaps are

### 1. Uncertainty quantification in RAG remains an unsolved problem

The most important recent finding is that **existing UQ methods systematically fail in RAG settings**. Soudani, Kanoulas, and Hasibi (ACL 2025 Findings; arXiv 2505.07459) proposed an axiomatic framework with five formal constraints that UQ methods should satisfy in retrieval-augmented contexts. Testing both white-box methods (predictive entropy, semantic entropy) and black-box methods (eigenvalue-based), they found no existing method satisfies all five axioms, and that UQ performance often *deteriorates* when non-parametric knowledge enters the prompt. This directly motivates the search for novel UQ signals in RAG.

Several papers have attempted to address RAG-specific uncertainty. FRANQ (Fadeeva et al., arXiv 2505.21072, 2025) explicitly disentangles factuality from faithfulness by decomposing RAG output into atomic claims and conditioning the UQ strategy on whether each claim is grounded in retrieved context or parametric knowledge. The R2C framework (Soudani et al., arXiv 2510.11483) introduces perturbation-based consistency measurement for multi-step retrieval-augmented reasoning, improving AUROC by over 5% on average. QuCo-RAG (Min et al., arXiv 2512.19134) takes a radically different approach, shifting from model-internal signals to corpus statistics from pre-training data, arguing that internal signals are fundamentally unreliable due to LLM miscalibration. The URAG benchmark (arXiv 2603.19281) reformulates open-ended RAG into multiple-choice for principled conformal prediction, finding that **accuracy-uncertainty correlations break under retrieval noise**.

Dynamic retrieval methods have used uncertainty as a trigger signal. FLARE (Jiang et al., EMNLP 2023) pioneered uncertainty-triggered retrieval using token probability thresholds, while DRAGIN (Su et al., ACL 2024) integrated multiple signals including entropy and attention weights. Self-RAG (Asai et al., ICLR 2024 Oral) trains special reflection tokens (ISREL, ISSUP, ISUSE) that function as implicit confidence mechanisms, deciding when retrieval is needed and whether retrieved passages support the output. Lookback Lens (Chuang et al., EMNLP 2024) introduced the "lookback ratio"—the ratio of attention on context tokens versus generated tokens—as a lightweight hallucination detector. ReDeEP (ICLR 2025) uses mechanistic interpretability to decouple external context utilization from parametric knowledge, finding hallucinations occur when Knowledge FFNs overemphasize parametric memory while Copying Heads fail to retain external knowledge.

The key gap: **all these methods operate on output-side uncertainty**. None probes whether the model comprehended its input before generating.

### 2. Context sufficiency detection reveals a paradox at the heart of RAG

The Google "Sufficient Context" paper (Joren et al., ICLR 2025; arXiv 2411.06037) is the single most relevant prior work. It formalizes "sufficient context" as whether retrieved information contains enough to answer the query—crucially, this is a function of (question, context) that does not require a ground-truth answer. The paper's most striking finding: **RAG paradoxically reduces abstention**. When context is insufficient, models hallucinate rather than abstaining—adding partial context actually suppresses the model's tendency to say "I don't know." Larger models (Gemini 1.5 Pro, GPT-4o) excel with sufficient context but hallucinate with insufficient context. Combining sufficient-context signals with confidence improves correct-answer fraction by **2–10%** across models. The autorater approach for classifying sufficient versus insufficient context was tested on models in the 7–8B parameter range (Llama 3.1 8B, Mistral 7B), confirming feasibility for Qwen3 8B.

Zhou et al. (arXiv 2509.01476, 2025) extend this finding, showing that when all retrieved documents are irrelevant, RALMs paradoxically refuse questions they *could* have answered from parametric knowledge—irrelevant retrieval suppresses useful parametric recall. Yoran et al. (ICLR 2024; arXiv 2310.01558) demonstrated that retrieval augmentation can negatively impact performance, especially in multi-hop reasoning where irrelevant evidence causes cascading errors, though even 1,000 training examples suffice to train robustness.

The gap this reveals for the proposed research: the Sufficient Context paper operates as a post-hoc stratification tool and selective generation mechanism. It does not use comprehension probes or logprob analysis to measure whether the model *understood* the context—it classifies whether the context *contains* sufficient information. The proposed paraphrase-probe approach would complement this by testing the model's actual processing of the context, not just the context's informational content.

### 3. Query comprehension verification is the least explored area

This is where the literature is thinnest—and where the proposed research's novelty is strongest. The S2AF framework (Neural Networks, 2025) introduces "understanding self-consistency" using a self-question-and-answer closed loop: generate, question, answer, evaluate. It is the closest existing work to the comprehension probe concept, but operates post-hoc rather than pre-generation. CoCA (Confidence before Answering; arXiv 2603.05881, 2025) proposes generating confidence *before* the answer through joint optimization of confidence and answer tokens—directly implementing "verify before acting" but through training rather than probing.

Intent detection research reveals the difficulty LLMs face with query understanding. Work from the EMNLP 2024 Industry Track shows fine-tuned small models still outperform LLMs on intent classification, and LLMs struggle particularly with out-of-scope detection. One study found GPT-4 correct only **30% of the time on ambiguous queries**. EAGLE (arXiv 2509.01564) demonstrates that hidden representations across intermediate layers encode richer confidence signals than final-output methods, supporting the use of internal state analysis.

A critical negative result: Huang and Chen (ICLR 2024; arXiv 2310.01798) showed **LLMs cannot reliably self-correct reasoning without external feedback**, and self-correction can even degrade performance. This argues against pure prompting-based comprehension verification and favors the proposed approach of using logprob analysis as an external signal rather than relying on the model's self-assessment.

### 4. Tool-call validation and agent uncertainty are rapidly developing

SAGE-Agent (Suri et al., arXiv 2511.08798, 2025) is the closest existing work to the proposed decision boundary for agentic systems. It models structured uncertainty over tool-call parameters as a POMDP with Expected Value of Perfect Information (EVPI), using aspect-based cost modeling to decide when clarification questions are needed versus when to proceed. Its When2Call component addresses "should I proceed or ask?" directly, improving coverage on ambiguous tasks by **7–39%** while reducing clarification questions by 1.5–2.7×. However, it focuses on parameter-level disambiguation rather than whether the model understood the overall query and context.

UALA (Han et al., arXiv 2401.14016, 2024) implements uncertainty-gated action selection where the agent only invokes tools when uncertain, substantially reducing tool calls and tokens while improving performance on HotpotQA and StrategyQA. The UQ in LLM Agents framework paper (arXiv 2602.05073, 2025) provides formal foundations, modeling agent trajectories as stochastic processes and identifying four challenges: multi-turn cascade uncertainty, source attribution, intrinsic action multiplicity, and evaluation beyond pointwise accuracy. SELAUR (arXiv 2602.21158) integrates token-level uncertainty into RL reward design for agent learning.

The Berkeley Function Calling Leaderboard (BFCL; Patil et al., NeurIPS 2024) includes "function relevance detection"—determining whether provided functions are suitable for the query. Its V2/V3 versions contain **875 irrelevance detection entries** (model should not call any function) and relevance detection entries (model should ask for clarification when parameters are missing), directly testing the abstention and clarification branches of the decision boundary.

For safety-critical deployments, "Towards Verifiably Safe Tool Use" (arXiv 2601.08012, 2025) proposes deterministic information-flow constraints rather than probabilistic safeguards, arguing model-based judges provide no guarantees. This complements the probabilistic decision boundary with a hard constraint layer.

### 5. Abstention research shows reasoning models paradoxically struggle most

AbstentionBench (Kirichenko et al., Meta; NeurIPS 2025 D&B Track; arXiv 2506.09038) is the definitive benchmark, spanning **20 datasets, 35,000+ queries, and 6 abstention scenarios** (unknown answers, underspecification, false premises, subjective interpretations, outdated information). Its most alarming finding: **reasoning fine-tuning degrades abstention by ~24% on average**. Models express uncertainty in reasoning chains but still produce definitive final answers. Increasing reasoning token budget—which typically boosts accuracy—*further worsens abstention*. Model scale has almost no effect. This disconnect between internal uncertainty and expressed confidence is a fundamental problem that the proposed logprob-based approach could help address by detecting the internal uncertainty that the model's output masks.

The "Know Your Limits" survey (Wen et al., TACL 2025) comprehensively organizes abstention methods across the LLM lifecycle: pretraining (data augmentation), alignment (R-Tuning, DPO), and inference (token-likelihood, semantic entropy, consistency sampling, verbalized confidence, multi-LLM collaboration). R-Tuning (Zhang et al., NAACL 2024 Outstanding Paper; arXiv 2311.09677) showed refusal-aware instruction tuning—training with uncertainty expressions on uncertain examples—creates a meta-skill that generalizes to out-of-domain tasks. A recent paper (arXiv 2603.21172, 2026) proves **entropy alone is insufficient for safe selective prediction**, proposing combined entropy-plus-correctness-probe signals.

The emerging consensus: calibration requires multiple complementary signals. The "Emergence of Semantic Calibration" paper proves base LLMs are semantically calibrated from pretraining, but RLHF and chain-of-thought *break* this calibration—directly relevant to the challenge of using instruction-tuned Qwen3 8B.

### 6. Paraphrase-based methods provide the methodological foundation

Semantic entropy (Kuhn, Gal, and Farquhar; ICLR 2023; arXiv 2302.09664; extended in Nature 2024) is the foundational method. By sampling multiple outputs, clustering semantically equivalent ones via bidirectional NLI, and computing entropy over meaning clusters, it separates genuine semantic uncertainty from lexical variation. The Nature extension demonstrated reliable "confabulation" detection across models and tasks without task-specific data. Semantic Entropy Probes (Kossen et al., NeurIPS 2024; arXiv 2406.15927) reduce this to near-zero overhead by training linear probes on hidden states from a single generation to approximate semantic entropy. Kernel Language Entropy (Nikitin et al., NeurIPS 2024) generalizes semantic entropy with soft kernel-based similarity, providing more fine-grained estimates.

**SPUQ (Gao et al., EACL 2024; arXiv 2403.02509) is the closest methodological predecessor** to the proposed approach. It generates input perturbations (paraphrasing, dummy tokens, perturbed system messages) and measures output consistency, reducing Expected Calibration Error by ~50%. However, SPUQ measures *output consistency under input perturbation* for answer correctness—not whether the model *understood* its input. The proposed contribution of measuring input-processing uncertainty is genuinely novel.

Tanneru et al. (AISTATS 2024) used "sample probing" (paraphrasing input questions) and "model probing" to measure explanation consistency, finding verbalized confidence averages **94.46%** (massive overconfidence) while probing uncertainty correlates with explanation faithfulness. A chemistry-domain paper (NeurIPS 2024 Workshop) used question rephrasing to test whether LLMs truly understand molecular representations—remarkably close to the proposed concept but domain-specific. The Uncertainty Profiles paper (arXiv 2505.07309, 2025) formally decomposes LLM uncertainty into four sources including "Surface Form Uncertainty" (input comprehension failures), explicitly acknowledging input processing as a distinct uncertainty source but without proposing the paraphrase-probe solution.

The Decomposing Uncertainty paper (Hou et al., ICML 2024) creates ensembles through input clarification to disentangle aleatoric (input ambiguity) from epistemic (knowledge gaps) uncertainty—providing theoretical grounding for the paraphrase-based approach.

### 7. Logprob reliability depends heavily on the model and extraction method

The evidence on logprob reliability is nuanced and directly informs the proposed pipeline design. Kadavath et al. (Anthropic, 2022; arXiv 2207.05221) established that larger models are well-calibrated on multiple choice, but **RLHF deteriorates calibration** (fixable with temperature scaling at T≈2.5). Tian et al. (EMNLP 2023; arXiv 2305.14975) found **verbalized confidence is typically better calibrated than logprobs** for RLHF models, reducing ECE by ~50%—but this varies by model, and Xiong et al. (ICLR 2024; arXiv 2306.13063) showed both approaches remain far from ideal (AUROC 0.5–0.6). Yang et al. (arXiv 2412.14737, 2024) demonstrated reliability depends heavily on prompt formulation, with tiny LLMs favoring simple formats and large LLMs benefiting from complex ones.

MARS (Bakman et al., ACL 2024) improved on raw logprobs by weighting token probabilities by semantic importance—tokens more important to meaning receive higher weight. The SAR method (Duan et al., ACL 2024) shifts attention to task-relevant tokens, mitigating the conflation of semantic uncertainty with lexical/syntactic uncertainty. LM-Polygraph (Vashurin et al., TACL 2025) provides a comprehensive benchmark of 20+ UQ methods, finding **consistency-based methods consistently outperform logit-based or verbalized proxies**.

For the Qwen3 8B pipeline specifically: Ollama supports logprob extraction via API. The two-pass CoT approach (direct mode, think mode) is well-motivated by AbstentionBench's finding that reasoning traces contain uncertainty signals the final answer suppresses. Combining logprob analysis with paraphrase consistency addresses the known limitations of either signal alone.

### 8. Pre-action verification establishes the architectural pattern

Chain-of-Verification (Dhuliawala et al., Meta; ACL 2024 Findings; arXiv 2309.11495) established the four-step verify-then-generate pattern: draft response → plan verification questions → answer them independently → generate verified response. It doubled precision on Wikidata list tasks and improved FActScore from **55.9 to 71.4** on biographies. However, CoVe verifies the *draft output*, not input comprehension. Reflexion (Shinn et al., NeurIPS 2023; arXiv 2303.11366) introduced verbal self-reflection with episodic memory, but requires multiple trial-and-error episodes. Self-RAG's reflection tokens are the closest to a pre-action verification layer that is *part of* generation, but require end-to-end training.

Self-consistency (Wang et al., ICLR 2023) provides the philosophical foundation: if multiple reasoning paths converge on the same answer, confidence is higher. The proposed research inverts this insight—if multiple paraphrased comprehension probes elicit consistent understanding signals, the model likely comprehended its input.

### 9. Faithfulness metrics reveal the attribution problem

The distinction between faithfulness and factuality is increasingly recognized as critical. Wallat et al. (arXiv 2412.18004, 2024) showed up to **57% of RAG citations are "post-rationalized"**—models generate from parametric memory then find supporting documents post-hoc. Citation correctness ≠ citation faithfulness. Trust-Score (ICLR 2025; arXiv 2409.11242) evaluates four dimensions: grounded refusals, claim recall, citation support, and citation relevance, finding even GPT-4 and Claude-3.5-Sonnet heavily rely on parametric knowledge in RAG. ALCE (Gao et al., EMNLP 2023) showed the best models lack complete citation support 50% of the time. FActScore (Min et al., EMNLP 2023) and SAFE (Wei et al., NeurIPS 2024) provide automated evaluation via atomic fact decomposition.

EvidenceRL (arXiv 2603.19532) and Context-DPO (Bi et al., ACL 2025 Findings) treat grounding as a training objective rather than post-hoc correction, using RL and DPO respectively to align models toward context-faithful generation. GaRAGe (ACL 2025 Findings) provides 2,366 questions with human-curated grounding annotations including insufficient-information cases requiring deflective responses.

---

## How the proposed research fills a genuine gap

The literature reveals a clear void at the intersection of five established areas. The table below maps what exists against what the proposed research contributes:

| Capability | Closest existing work | What's missing |
|---|---|---|
| Verify input comprehension *before* acting | CoCA (2025), S2AF (2025) | Neither uses paraphrase probes or logprob analysis; both rely on self-assessment |
| Paraphrase-based uncertainty on *input processing* | SPUQ (EACL 2024) | SPUQ measures output consistency, not input comprehension |
| Three-way action selection (proceed/clarify/escalate) | SAGE-Agent (2025), BFCL relevance detection | SAGE-Agent handles parameter disambiguation only; no comprehension verification |
| UQ specifically designed for RAG | Soudani et al. (ACL 2025), FRANQ | All focus on output uncertainty; none probe context comprehension |
| Logprob-based decision boundary for local models | LM-Polygraph, SEPs | Not integrated into action-selection pipelines |

The proposed pipeline—paraphrase-aggregated logprob extraction with two-pass CoT on Qwen3 8B—addresses the unique combination of: (1) pre-generation comprehension verification, (2) logprob-based signals complemented by consistency analysis, (3) three-way decision routing, and (4) practical deployment on local hardware. No existing paper occupies this intersection.

---

## Part B: Datasets and benchmarks for experimental validation

### Tier 1 datasets offer native sufficient/insufficient context pairing

**MuSiQue-Full** (Trivedi et al., TACL 2022; arXiv 2108.00573) is the strongest candidate. It contains **~50K questions** with explicit answerable/unanswerable contrast pairs—unanswerable versions created by replacing a supporting paragraph with an insufficient one. Supporting paragraph annotations enable precise context degradation. It includes sub-question decompositions and was used in the Google Sufficient Context paper. Six reasoning graph shapes test genuine multi-hop reasoning rather than shortcuts. Context sufficiency prediction is a built-in task. Fully compatible with local 8B models.

**SQuAD 2.0** (Rajpurkar et al., ACL 2018; arXiv 1806.03822) remains foundational with **150K questions** including 50K+ adversarially-crafted unanswerable questions. The `is_impossible` flag provides binary labels directly mapping to the decision boundary. Single-paragraph contexts are compact for logprob analysis. The limitation is extractive rather than generative format, but it provides clean signal for comprehension probing.

**RGB** (Chen et al., AAAI 2024; arXiv 2309.01431) is purpose-built for RAG robustness across four testbeds: noise robustness (varying noise ratios 0–0.8), negative rejection (all documents irrelevant), information integration (answers span multiple documents), and counterfactual robustness (documents contain errors). Tested on 6–7B models (including Qwen-7B-Chat). The negative rejection testbed directly implements "insufficient context" conditions. Bilingual (English + Chinese). Moderate size but precisely designed.

The **Google Sufficient Context methodology** (Joren et al., ICLR 2025; arXiv 2411.06037) provides an autorater approach replicable with Qwen3 8B applied to HotpotQA, FreshQA, and MuSiQue. Its binary sufficient/insufficient labels, combined with correct/hallucinated/abstained error categories, directly support the proposed experimental design. Code available at github.com/hljoren/sufficientcontext.

### Tier 2 datasets provide scale and domain diversity

**RAGBench** (Friel et al., arXiv 2407.11005) offers **~100K examples** across five industry domains with TRACe evaluation labels covering relevance, utilization, adherence (hallucination detection), and completeness. Available on HuggingFace (`rungalileo/ragbench`). Contexts are manipulable for creating insufficient conditions. The scale and domain diversity make it excellent for demonstrating generalization.

**CRAG** (Yang et al., Meta; NeurIPS 2024 D&B; arXiv 2406.04744) provides **~4,400 QA pairs** with web search results and KG APIs across five domains. Its **4-way response classification** (perfect/acceptable/missing/incorrect) maps naturally to decision boundary thresholds. Temporal dynamism categories (real-time, fast-changing, slow-changing, stable) and entity popularity tiers (head/torso/tail) enable fine-grained analysis of when uncertainty is highest.

**HotpotQA** (Yang et al., EMNLP 2018; arXiv 1809.09600) provides **113K multi-hop questions** with sentence-level supporting fact annotations. These annotations enable precise context degradation—removing specific supporting sentences creates controlled insufficient-context conditions. The distractor setting (10 paragraphs including irrelevant ones) naturally tests context discrimination.

### Tool-use and abstention benchmarks enable the action-selection component

**MetaTool** (Huang et al., ICLR 2024; arXiv 2310.03128) is the most directly relevant tool-use benchmark, with **21,127 queries** testing two core decisions: tool usage awareness (should I use a tool at all?) and tool selection (which tool?). The binary tool-awareness labels map directly to the proceed/escalate branch of the decision boundary. Most LLMs struggle with this: ChatGPT achieves F1 >70% but others score as low as **11.53%**.

**BFCL** (Berkeley Function Calling Leaderboard; Patil et al.) provides deterministic ground-truth evaluation via AST matching. V2 contains **875 irrelevance detection entries** (model should not call any function) and relevance detection entries (model should recognize missing parameters and ask for clarification). V3 adds 200 missing-parameter entries specifically testing when essential information is absent. Together, these categories implement all three branches: correct call → proceed, irrelevant function → escalate, missing parameters → clarify.

**AbstentionBench** (Kirichenko et al., Meta; NeurIPS 2025; arXiv 2506.09038) provides **35,000+ queries** across 6 abstention scenarios. Its underspecification subsets (GSM8K-Abstain, GPQA-Abstain, MMLU-Abstain) are particularly relevant—they test whether models detect missing information. The benchmark uses Llama 3.1 8B Instruct as its evaluation judge, confirming compatibility with the 8B parameter class. Available on HuggingFace (`facebook/AbstentionBench`).

**T-Eval** (Chen et al., ACL 2024; arXiv 2312.14033) evaluates six sub-dimensions of tool utilization with human-verified golden annotations. Its decomposed evaluation isolates where models fail, useful for diagnosing whether comprehension probes improve specific sub-capabilities.

### Supplementary datasets round out the evaluation

**Natural Questions** (Kwiatkowski et al., TACL 2019) provides **307K real Google queries** with null annotations when Wikipedia doesn't contain the answer (~30% of examples). **FreshQA** (Vu et al., ACL 2024 Findings) includes false-premise questions testing rejection of flawed queries, though it is small (~600 questions) and lacks native retrieval context. **TruthfulQA** (Lin et al., ACL 2022; 817 questions) tests parametric truthfulness but lacks RAG context. **MS MARCO** (Nguyen et al., 2016) provides massive scale (1M queries) with "No Answer Present" labels but requires significant preprocessing. Additional recent benchmarks include GaRAGe (ACL 2025 Findings; 2,366 questions with grounding annotations), UAEval4RAG (ACL 2025; unanswerability in RAG), RefusalBench (2025; selective refusal with linguistic perturbations), mtRAG (TACL 2025; multi-turn conversational RAG), and CRUMQs (ACL 2025 Findings; controlled unanswerable multi-hop queries). API-Bank (Li et al., EMNLP 2023) provides 73 tools with ground-truth API call sequences. ToolBench (Qin et al., ICLR 2024 Spotlight) covers 16,464 real APIs but focuses on execution success rather than abstention decisions. AgentBench (Liu et al., ICLR 2024) tests 8 environments but requires heavy infrastructure.

### No single dataset covers the full experimental design

The critical finding is that **no "holy grail" dataset exists** combining QA + context sufficiency + three-way action selection. The recommended strategy is to construct a composite evaluation:

- **Comprehension probing**: MuSiQue-Full (answerable/unanswerable pairs with supporting fact annotations) + SQuAD 2.0 (answerable/unanswerable with `is_impossible` labels)
- **Context sufficiency**: Google Sufficient Context autorater applied to MuSiQue and HotpotQA + RGB negative rejection testbed
- **Action selection**: BFCL irrelevance/relevance detection (proceed/clarify/escalate labels constructible from existing categories) + MetaTool tool-awareness binary labels
- **Calibration benchmarking**: AbstentionBench underspecification subsets + TruthfulQA
- **Domain generalization**: RAGBench across five industry domains + CRAG across five domains with temporal variation

All datasets are text-based and compatible with logprob extraction from Qwen3 8B via Ollama's API. The Sufficient Context paper confirmed feasibility at the 7–8B parameter scale.

---

## Conclusion: positioning the research within a converging field

Three converging trends create the opening for this work. First, the RAG-paradox finding—that retrieval *suppresses* abstention—means deployed RAG systems need a pre-generation gate that current architectures lack. Second, the AbstentionBench result—that reasoning fine-tuning degrades abstention by 24%—shows the problem cannot be solved by scaling or training alone; external verification signals are needed. Third, the paraphrase and logprob literatures have independently established that neither signal alone is reliable for RLHF-tuned models, but consistency-based methods consistently outperform single-signal approaches.

The proposed approach is best positioned as a **lightweight, training-free verification layer** that complements rather than replaces existing UQ methods. It draws on semantic entropy's insight (measure meaning, not tokens), SPUQ's methodology (perturb inputs, measure consistency), the Sufficient Context paper's framing (gate on context adequacy), and SAGE-Agent's decision framework (proceed/clarify/escalate)—but combines them in a novel configuration targeting input comprehension rather than output confidence. The strongest methodological ancestors to cite are Kuhn et al. (2023) for semantic uncertainty foundations, Joren et al. (2025) for context sufficiency framing, Gao et al. (2024/SPUQ) for input perturbation methodology, Tian et al. (2023) for logprob unreliability motivation, and Kirichenko et al. (2025) for the abstention crisis that makes this work urgent. The Uncertainty Profiles paper (arXiv 2505.07309) formally acknowledges "Surface Form Uncertainty" as a distinct source—the proposed work provides the first practical method for measuring it.