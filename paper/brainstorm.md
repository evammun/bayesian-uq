# Brainstorm — Pre-Action UQ

**Last updated:** 2026-03-30

---

## 1. Core Idea

**One sentence:** Before a local LLM acts on query + RAG context, run paraphrased comprehension probes + logprob analysis to decide: proceed / clarify / escalate.

**Why novel:** Existing UQ = output-side ("is the answer right?"). Ours = input-side ("did the model understand?"). Nobody occupies this intersection — confirmed by lit review.

**Why it matters:**
- RAG paradox: bad context → *more* confident hallucination (Google Sufficient Context, ICLR 2025)
- AbstentionBench: reasoning training degrades abstention by ~24% (Meta, NeurIPS 2025)
- LLMs can't self-correct without external feedback (Huang & Chen, ICLR 2024)
- → External verification signals needed. That's us.

---

## 2. Three Uncertainty Types

### Query comprehension
- Does model understand what user is asking?
- Test: paraphrase user query N ways, same RAG context → measure interpretation stability via logprobs
- Maps directly to v2 paraphrase-aggregation framework

### Context sufficiency
- Does model recognise if RAG context contains needed info?
- Test: probe context comprehension; compare behaviour with sufficient vs insufficient context
- Key: RAG paradox means model won't naturally flag bad context — probe must catch it
- Connection to broken-premise work from v2 (same concept, different domain)

### Action selection
- Should model answer / call tool / clarify / escalate?
- Test: logprobs at action decision point across paraphrased inputs
- Two approaches: constrained MCQ ("should you: A) answer, B) search, C) clarify, D) escalate") or natural decision-point inspection
- Interesting Q: does action uncertainty *proxy* context sufficiency? If yes, simpler MCQ may subsume complex probes.

---

## 3. Probe Design Options (THE CRUX)

| Option | How | Pros | Cons |
|--------|-----|------|------|
| A: Paraphrase consistency | Paraphrase query, measure response stability | Simple, extends v2 directly | Consistency ≠ correctness; "just SPUQ for RAG" |
| B: Auto-generated MCQ | Teacher model generates comprehension MCQ | Directly tests comprehension | Extra inference call; circular if same model |
| C: Templated action MCQ | Fixed "should you: answer/search/clarify/escalate?" | Zero overhead, clean MCQ | Only covers action selection |
| D: Context-referencing probes | "Does context mention [entity]?" | Directly tests context processing | Requires entity extraction step |
| **E: Hybrid (likely best)** | A for query, D or with/without for context, C for action | Covers all three types | More complex evaluation |

**Key empirical question:** Does pure paraphrase consistency (Option A) already predict context sufficiency + action correctness? If yes, skip the fancy probes. **Pilot should test this.**

---

## 4. Dataset Strategy

### March 30 update: RGB dropped, QuALITY adopted

RGB dropped — all factoid lookup, no real comprehension, too small (500q), free-form answers = no clean logprob extraction. See `paper_running_notes.md` March 30 entry.

**Primary dataset: QuALITY** (Pang et al., NAACL 2022). ~6,000 MCQ over ~260 articles, ~5K-token passages, hard subset where speed-readers fail. CC BY 4.0. MCQ format = direct pipeline reuse.

**RQ-to-dataset mapping (updated):**

| RQ | Dataset | Why |
|---|---|---|
| **RQ2: Context sufficiency** (main) | **QuALITY** with constructed context conditions | Comprehension genuinely matters; MCQ format; we control the manipulation |
| **RQ1: Query comprehension** | **QuALITY** paraphrase consistency | Free — same experiment, different analysis cut |
| **RQ3: Action selection** | BFCL (deferred) | 875 irrelevance detection + missing params |

### 4.1 Context Conditions (constructed, not dataset-provided)

The experimental manipulation is *ours*, not the dataset's. For each QuALITY question:

| Condition | Construction | What it tests | Hypothesis |
|-----------|-------------|---------------|------------|
| **C1: Sufficient** | Correct article + question | Baseline — model has everything it needs | High confidence, high paraphrase consistency, correct answers |
| **C2: Insufficient** | Different article (same genre) + question | Model has *no* relevant information | Should show low confidence OR (if RAG paradox) high confidence + low consistency |
| **C3: Partial** | Truncated article (answer section removed) + question | Realistic RAG failure — *some* context but not enough | Most subtle case. Prediction: intermediate uncertainty, possibly fragile confidence |
| **C4: Counterfactual** | Article with key facts modified + question | The RAG paradox scenario — context looks right but is wrong | Dangerous if model is confident. Our signals should detect instability. |

**Design notes:**
- C2 (insufficient): swap article within same genre tag. QuALITY has genre info — use it.
- C3 (partial): need to identify which section of the passage contains the answer. Could use the gold answer + simple search, or manual annotation for pilot.
- C4 (counterfactual): hardest to construct automatically. Options: entity swap (replace key names/dates), negation insertion, or manual for pilot. Park for later — C1/C2/C3 are enough to start.
- **Start with C1 and C2 only.** Sufficient vs fully insufficient is the cleanest comparison and the easiest to construct. Add C3/C4 once we have baseline results.

**C5: Topically-relevant insufficient (April 2 idea)**

C2 tests "retrieval completely failed" (wrong article). But real RAG failures are subtler: the retrieved context is thematically related but doesn't contain the specific answer. The model's world knowledge fills the gap — confidently and incorrectly.

*Motivating example:* Article 61007 is a Garden of Eden retelling (man named Ha-Adamah, woman named Hawwah, forbidden fruit, naming animals). Q: "Why does the crew refer to Ha-Adamah as Adam?" Correct answer: Father Briton (the linguist) confirms Ha-Adamah is the Hebrew form of Adam — it's an etymological recognition. But Qwen 8B (CoT, sufficient context, 0.603 confidence, 6/10 agreement) chose: "The planet feels like Eden, so they begin to believe he IS Adam." The model latched onto the thematic reading from its Genesis training data instead of tracking the specific dialogue. Claude Opus, given the same question cold, cited the exact dialogue and got it right.

This matters because it mirrors the real failure mode: a legal assistant retrieves topically relevant docs about contract law, but the specific clause the user asked about isn't there. The model answers anyway from its legal training data. The answer sounds authoritative and is thematically consistent — but wrong.

*Design options:*
- Construct questions about familiar themes (religion, history, law) that are answerable from world knowledge but where the article's actual content diverges
- Check if existing QuALITY hard-wrong cases already correlate with "world-knowledge-plausible" wrong answers
- Could generate these automatically: take an article touching a well-known topic, ask questions the model's priors would answer differently than the text

### 4.2 Additional design axes
- **Easy vs hard questions:** QuALITY provides this split. Speed-reader accuracy as difficulty proxy. Key prediction: signals more discriminative on hard subset.
- **Think vs no-think:** Scaffolding absorption in comprehension setting. Does thinking mode suppress uncertainty the same way it did in MMLU?
- **With/without context comparison:** Run questions with no context at all → if logprob distribution doesn't change when context is added, model is answering from parametric knowledge, not context. Simple signal, possibly powerful. (From parking lot idea.)

---

## 5. Paper Sketch

**Target:** EMNLP 2026 (~June submission)

Intro → Related work → Method (probe design + logprob extraction + routing) → Setup (QuALITY, context conditions, Qwen 8B) → Results (do signals predict context sufficiency? does paraphrasing help? the 2D space? easy vs hard?) → Discussion

**Key results to aim for:**
- Comprehension probes predict context sufficiency (headline)
- Logprobs outperform self-assessment / verbalized confidence
- Paraphrasing adds value beyond single-probe (validates framework)
- 2D uncertainty space separates sufficient/insufficient conditions

---

## 6. Scope Boundaries

NOT doing: production system (paper = empirical validation), training anything (inference-time only), frontier models (Qwen 8B on laptop = the point), RAG retrieval (assume it happened), multi-turn, competing with semantic entropy (we're complementary — input-side vs output-side).

---

## 7. Misc Ideas / Parking Lot

- Action selection uncertainty as proxy for context sufficiency — test empirically
- Scaffolding absorption in RAG setting: does CoT suppress comprehension uncertainty the same way it suppressed MMLU uncertainty?
- Broken-premise taxonomy from v2 → types of context insufficiency? (entity missing, contradictory info, wrong domain, partial info)
- The with/without context comparison: if adding context doesn't change logprob distribution → model answering from parametric knowledge, not context. Simple signal, possibly powerful.
- Per-domain analysis: does UQ help more in some domains (legal, medical) than others?
- Qwen 3.5 vs Qwen 3: thinking mode implications for the pipeline
- Luigi's commercial framing: "verification layer as drop-in module" — paper validates, product packages

---

## 8. Timeline (rough)

- End March: lock probe design
- Early April: pilot on ~50-100 questions
- April: full experiments
- May: writing
- June: submit EMNLP
