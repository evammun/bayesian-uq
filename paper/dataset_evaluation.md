# Do These Datasets Actually Fit What We're Doing?

## An honest evaluation of MuSiQue, SQuAD 2.0, RGB, and Sufficient Context against the proposed research questions

---

## First: What are the research questions, and do they still hold?

The three uncertainty types we have been working with are:

1. **Query comprehension** — does the model understand what the user is asking?
2. **Context sufficiency** — does the retrieved context contain what is needed to answer?
3. **Action selection** — should the model answer, call a tool, ask for clarification, or escalate?

These were articulated during the initial brainstorm, and they tell a compelling story about a pre-action verification layer. But we need to pressure-test them before choosing datasets, because the dataset choice should follow from the RQs — not the other way around.

### The problem: these three are not equally developed

**Context sufficiency is the strongest RQ.** It is testable, it maps to a known failure mode (the RAG paradox — models hallucinate more confidently with insufficient context than without any context at all), and the existing pipeline (paraphrase-aggregated logprobs) has a clear mechanism for measuring it. The hypothesis is clean: if paraphrased versions of the same question produce inconsistent logprob distributions over answers given the same context, the context probably does not support a clear answer.

**Query comprehension is the most novel but hardest to operationalize.** What does it mean for a model to "understand" a query? In a real deployment, this would look like: a customer asks "can I return the blender?" and the model interprets this as a question about product specifications rather than return policy. But how do you test this systematically? You would need queries with annotated intents, ambiguous queries with multiple valid interpretations, and a way to measure whether paraphrase probes detect intent instability. None of the four Tier 1 datasets have this. They all assume the question is clear and well-formed — the uncertainty is in whether the answer is available, not whether the question was understood.

**Action selection is a routing problem that sits on top of the other two.** If comprehension uncertainty is high, ask for clarification. If context sufficiency uncertainty is high, retrieve more or escalate. If both are low, answer. This is a compelling system design, but it is a framework that depends on the other two signals working. It is not independently testable without a dataset that has ground-truth action labels (answer / search / clarify / escalate), and none of these four datasets have that.

### So what does this mean for dataset selection?

It means we should be honest that the four Tier 1 datasets, as they stand, primarily serve **RQ2 (context sufficiency)** and at best partially inform **RQ1 (query comprehension)**. They do not serve RQ3 at all.

This is not fatal — context sufficiency is arguably the most publishable and practically important of the three. But we should not pretend these datasets test the full three-part framework. The question is: do they test context sufficiency *well enough* and *in the right way* for what we are actually proposing?

---

## Dataset 1: MuSiQue (Trivedi et al., TACL 2022)

### What it is

MuSiQue is a multi-hop QA dataset built bottom-up by composing single-hop questions from existing datasets (SQuAD, Natural Questions, T-REx, etc.). MuSiQue-Ans has ~25K 2-4 hop questions. MuSiQue-Full adds unanswerable contrast pairs, where one supporting paragraph is swapped for an insufficient one. Each question comes with ~20 paragraphs (2-4 supporting + ~16 distractors), sub-question decompositions, and supporting paragraph annotations.

The construction is clever: for each answerable question, they create an unanswerable variant by replacing a critical supporting paragraph with one that is topically related but does not contain the needed information. This means the unanswerable questions are hard — the context looks plausible but has a gap.

### Why it looked promising

The answerable/unanswerable pairing is exactly the controlled "sufficient vs insufficient context" experimental design we need. The supporting paragraph annotations let you know exactly which piece of context matters. The dataset was used in the Google Sufficient Context paper, giving it pedigree.

### Why Eva's instinct is right — it is not a great fit

The core problem: **MuSiQue tests reasoning capability, not comprehension of input.**

Consider a typical MuSiQue question: "Who wrote the novel that the author of Armageddon in Retrospect is most famous for?" To answer this, the model needs to: (1) identify that Kurt Vonnegut wrote Armageddon in Retrospect, (2) identify that Vonnegut is most famous for Slaughterhouse-Five, (3) return "Kurt Vonnegut" as the author. The difficulty is in chaining these steps — decomposing the question into sub-queries and connecting information across paragraphs.

This is fundamentally different from what we are testing. In our RAG scenario, the question is: "Given this user query and this retrieved context, does the model understand what it received?" We are not asking the model to chain multi-hop reasoning — we are asking whether it grasped the situation well enough to act appropriately.

More specifically:

**The questions are artificial.** Nobody asks "who wrote the novel that the author of X is most famous for?" in a customer support setting. The multi-hop composition is designed to prevent reasoning shortcuts, which is great for evaluating multi-hop QA models but irrelevant for evaluating comprehension of realistic queries.

**The unanswerable condition tests a very specific failure mode.** A supporting paragraph is removed, creating a gap in the reasoning chain. But in real RAG, "insufficient context" looks different — it might be that the retrieved documents are topically adjacent but do not address the user's actual question, or that the context contains outdated information, or that it covers a related product but not the one asked about. MuSiQue's gaps are surgical excisions from reasoning chains, not the kind of context insufficiency that occurs in practice.

**The task format is extractive.** The answer is a short text span. Our pipeline extracts logprobs over answer tokens, which works for MCQ but would need adaptation for extractive QA. You would either need to reformulate as MCQ (losing the original task structure) or measure logprob entropy over the generation, which is a different signal.

**Paraphrase probing on multi-hop questions is confounded.** If you paraphrase "who wrote the novel that the author of Armageddon in Retrospect is most famous for?" you might change which reasoning path the model follows, introducing variance that is about the paraphrase's effect on reasoning rather than about comprehension uncertainty. The signal is muddied.

### Verdict: Weak fit

MuSiQue could serve as one data point in a larger evaluation (especially for showing that the uncertainty signal generalises to complex multi-hop settings), but it should not be a primary dataset. The gap between "multi-hop reasoning decomposition" and "did the model understand its input in a RAG deployment" is too wide.

---

## Dataset 2: SQuAD 2.0 (Rajpurkar et al., ACL 2018)

### What it is

~150K questions on Wikipedia paragraphs. SQuAD 1.1 had ~100K answerable questions; SQuAD 2.0 adds ~50K adversarially-crafted unanswerable questions. The unanswerable questions are written by crowdworkers who were shown the paragraph and asked to write questions that look like they should be answerable from the paragraph but are not — they reference entities in the paragraph but ask about something the paragraph does not cover. Each example: one paragraph, one question, either extractive answer or is_impossible = true.

### What is useful about it

The adversarial construction of unanswerable questions is directly relevant. A question like "Which laws faced significant opposition?" asked about a paragraph discussing legislation that had little opposition is exactly the kind of plausible-but-unanswerable query that a RAG system would face. The model has to recognise that the paragraph does not support the answer, not that the paragraph is irrelevant.

The is_impossible flag is a clean binary label that maps directly to the context sufficiency decision boundary: "Is the context sufficient to answer this question?" If we can show that our paraphrase-aggregated logprob signal predicts is_impossible better than baseline methods, that is a publishable result.

The scale (150K questions) means we can subsample meaningfully. The single-paragraph format keeps context short, which is practical for Qwen 8B with limited context window and keeps per-example runtime manageable.

### The problems

**It is 2018 and it shows.** SQuAD is the canonical reading comprehension benchmark, but it has been saturated by modern LLMs. BERT achieved near-human performance in 2018. An 8B parameter model in 2026 will likely find most of SQuAD 2.0 trivially easy, both for answering and for detecting unanswerable questions. If the model gets ~95% accuracy on answerability detection, there is not much room for our uncertainty signal to add value. We would need to verify this empirically before committing.

**Single-paragraph contexts are not realistic RAG.** In a real RAG deployment, the model receives multiple retrieved passages (often 3-10), some relevant, some not. SQuAD gives exactly one paragraph per question. This simplifies the context sufficiency problem substantially — there is no noise from irrelevant retrievals, no need to integrate across documents, no distractor passages.

**Extractive format, same problem as MuSiQue.** The answers are spans from the paragraph. Our logprob pipeline is built for MCQ-style extraction. We would need to reformulate, adding a layer of artificiality.

**No query ambiguity.** All questions are clear, grammatical, and well-formed. The "query comprehension" dimension is absent.

### Verdict: Moderate fit, but may be too easy

SQuAD 2.0 is a clean, well-understood benchmark for the context sufficiency RQ. It could serve as a baseline or sanity check — if the uncertainty signal cannot even predict is_impossible on SQuAD 2.0, something is wrong with the method. But it is unlikely to be the main result. Reviewers in 2026 will want something more realistic than SQuAD.

---

## Dataset 3: RGB — Retrieval-Augmented Generation Benchmark (Chen et al., AAAI 2024)

### What it is

RGB evaluates four fundamental RAG capabilities across separate testbeds:

1. **Noise robustness** — questions with varying ratios of noisy (irrelevant) documents mixed in with relevant ones (noise ratios: 0, 0.2, 0.4, 0.6, 0.8)
2. **Negative rejection** — all provided documents are irrelevant; the model should refuse to answer
3. **Information integration** — the answer spans multiple documents and must be synthesised
4. **Counterfactual robustness** — documents contain deliberate factual errors that the model should detect

Bilingual (English + Chinese). Tested on 7B-class models including Qwen-7B-Chat. Questions are based on recent news to avoid contamination from pretraining. Each testbed has its own evaluation metric (accuracy, rejection rate, error detection rate).

### Why this is the closest fit

The **negative rejection testbed** is almost exactly our experimental scenario. The model receives a query plus documents that look relevant but do not contain the answer. The correct behaviour is to refuse — to recognise that the context is insufficient. The key finding: LLMs struggle badly at this. ChatGPT's rejection rate was only around 45%, and smaller models scored even lower.

This is the failure mode our uncertainty signal is designed to detect. If paraphrased comprehension probes produce inconsistent logprob distributions when all documents are irrelevant, that inconsistency is the signal for "do not answer."

The **noise robustness testbed** provides a gradient — as irrelevant documents increase from 0% to 80% of context, how does the uncertainty signal change? This gives us a continuous relationship between context quality and uncertainty, not just a binary sufficient/insufficient split.

The inclusion of **Qwen-7B-Chat** in the original evaluation means direct comparability for our Qwen3 8B results.

### The problems

**Small scale.** RGB is a diagnostic benchmark, not a large evaluation suite. The total is modest (a few hundred per condition). This limits statistical power and may not support the kind of fine-grained analysis (AUROC curves, calibration plots) that our method produces best.

**Narrow question types.** The questions are based on recent news events, which means they are factoid questions about current affairs. This is one slice of the RAG use case space — no customer support queries, no product questions, no policy lookups.

**Evaluation is binary (rejected/not rejected).** RGB measures whether the model refused to answer, not the quality of the uncertainty signal. We would need to overlay our logprob analysis on top of RGB's format, measuring whether the uncertainty signal predicts when the model should reject, not just whether it does.

**The format requires adaptation.** RGB provides documents and expects free-form generation. We would need to either: (a) reformulate as MCQ for logprob extraction, or (b) adapt the pipeline to measure generation-level uncertainty (entropy over first tokens, consistency across paraphrased queries). Option (b) is more aligned with the real-world framing but is a pipeline extension.

### Verdict: Strong conceptual fit, needs augmentation for scale

RGB's negative rejection and noise robustness testbeds are the most directly relevant experimental conditions among all four datasets. The question is whether the scale is sufficient. RGB might be best used alongside a larger dataset — run RGB for the targeted diagnostic evaluation and a bigger dataset for the main results.

---

## Dataset 4: Sufficient Context (Joren et al., ICLR 2025)

### What it is

This is not a dataset per se — it is a methodology and a set of findings applied to existing datasets (HotpotQA, FreshQA, MuSiQue). The core contribution is a "sufficient context autorater" — an LLM-based classifier that labels each (question, context) pair as having sufficient or insufficient context to answer the question. The autorater achieves 93% accuracy.

The methodology is applied to stratify RAG performance: given the same model and the same questions, how does performance differ when context is sufficient vs insufficient? The answer: dramatically. Models hallucinate far more with insufficient context, and — the key paradox — RAG reduces abstention. Adding any context, even insufficient context, makes models more confident and less likely to say "I don't know."

They tested this on smaller open-source models including Llama 3.1 8B and Mistral 7B, confirming feasibility in the 8B parameter class. They also developed a selective generation method combining sufficient context signals with model confidence, improving correct-answer fraction by 2-10%.

### Why this matters enormously for us

The findings are the single strongest motivation for our research direction:

- RAG paradoxically reduces abstention — this is the problem we are trying to solve
- Smaller models (8B class) hallucinate frequently even with sufficient context — our target setting
- Combining sufficiency signals with confidence improves outcomes — our method provides a new kind of sufficiency signal (paraphrase-aggregated logprob consistency)
- The autorater can be replicated with Qwen3 8B to label any dataset

The selective generation method they propose is a linear model trained on (sufficient_context_score, model_confidence) to predict hallucinations. Our paraphrase-aggregated uncertainty is a third signal that could complement or improve on both. The paper itself suggests "a sufficiency check before generation" as a recommendation — we are building exactly that.

### The problems

**It is a methodology, not a dataset.** We still need to choose which QA dataset to apply it to, and that inherits whatever limitations the underlying dataset has. If we apply it to MuSiQue, we get MuSiQue's problems (multi-hop reasoning confound). If we apply it to HotpotQA, we get something more tractable but still academic QA.

**The autorater is a separate LLM call.** Running the sufficient context autorater on every example adds compute cost and introduces a dependency on another model's judgment. For our experimental design, we would need to either: (a) use their autorater to pre-label examples and treat those labels as ground truth, or (b) reimplement the autorater using Qwen3 8B itself, which introduces circularity.

**The paper's own findings suggest a ceiling for our approach.** They find that models answer 35-62% of questions correctly even with insufficient context, using parametric knowledge. This means our uncertainty signal cannot purely predict answerability from context sufficiency alone — the model might be right anyway using its own knowledge. Our signal would need to predict when the model is correctly using parametric knowledge vs confidently hallucinating, which is a harder problem than pure context sufficiency detection.

**It does not test comprehension probing.** The autorater classifies whether context is sufficient based on the information content of the context. It does not test whether the model understood the context — a distinction that is central to our novelty claim. The model might have sufficient context and still misinterpret it.

### Verdict: Essential as framing and methodology, not as a standalone dataset

The Sufficient Context paper should be a primary reference and methodological inspiration. Its findings motivate the research. Its autorater could provide labels for whatever primary dataset we choose. But it does not replace the need for a dataset that actually tests what our pipeline measures.

---

## The bigger picture: what is actually missing

Having gone through all four, a clear pattern emerges. These datasets were designed to test different things from what we are proposing:

| What we need to test | What these datasets test |
|---|---|
| Does the model understand the user's query? (RQ1) | Can the model answer a well-formed academic question? |
| Does the model recognise when context is insufficient? (RQ2) | Can the model abstain when the paragraph does not contain the answer? |
| Can uncertainty signals route to the right action? (RQ3) | (Not tested at all) |

The mismatch is most severe for RQ1 (query comprehension) and RQ3 (action selection). For RQ2 (context sufficiency), the datasets are partially usable but with significant caveats:

- The "insufficient context" conditions are constructed by academic researchers for academic benchmarks, not by real retrieval systems failing in the ways they actually fail
- The questions are clean, well-formed, and unambiguous — nothing like real user queries
- The format (extractive QA) does not match our pipeline (MCQ logprob extraction) without adaptation
- The scale is either too small (RGB) or the task is too easy for modern LLMs (SQuAD 2.0)

### What this suggests about the RQs

This exercise reveals that the three RQs, as currently framed, may be too ambitious for a single paper — especially given the dataset landscape. A more focused version might be:

**Primary RQ**: Can paraphrase-aggregated logprob signals predict whether a RAG-augmented local LLM will produce a correct response, and specifically, can they detect context insufficiency before the model generates?

This is essentially RQ2 with the practical framing intact. It is testable with modified versions of these datasets. It is the claim that connects most directly to the Sufficient Context paradox findings. And it is where the existing pipeline transfers most naturally.

RQ1 (query comprehension) and RQ3 (action selection) could be positioned as extensions or future work — or tested with smaller, purpose-built experiments rather than needing a full benchmark.

### If we proceed with this focus, the best dataset strategy is probably:

1. **RGB's negative rejection testbed** as the core diagnostic experiment — directly tests the "all context is irrelevant" condition, small enough for thorough per-example analysis
2. **Sufficient Context autorater applied to HotpotQA** (not MuSiQue) as the larger-scale evaluation — HotpotQA is 2-hop (simpler than MuSiQue's 2-4 hop), has supporting fact annotations for controlled context degradation, and the Sufficient Context paper already provides the methodology
3. **SQuAD 2.0** as a sanity check / baseline — clean binary labels, well-understood, establishes minimum viability
4. **The Sufficient Context paper's findings and autorater** as the motivating framework and labelling tool throughout

But honestly, we should also look beyond these four. RAGBench (100K examples across 5 industry domains with TRACe labels) and CRAG (4-way response classification, temporal dynamics, entity popularity) might actually be closer to the "real business RAG deployment" framing that makes this research distinctive. The Tier 1 designation in the literature review was based on having built-in sufficient/insufficient context pairing — but the trade-off is that they are all academic QA benchmarks, and the practical RAG framing is what makes this paper different from the dozen other UQ papers published this year.
