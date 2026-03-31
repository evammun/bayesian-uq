# Chain-of-Thought Prompting Strategy: Design Rationale

## Context

This document records the design decisions around prompting strategy for our Bayesian UQ framework experiments, specifically how we elicit answers from local LLMs (Qwen3 8B Q4) on MMLU-Redux multiple-choice questions. The key question: should we use direct prompting, chain-of-thought reasoning, or native thinking mode — and how should each be implemented?

## Background: Standard Evaluation Protocol

The standard evaluation protocol for MMLU, as implemented in EleutherAI's lm-evaluation-harness (the de facto standard for LLM benchmarking), uses **no system prompt** at all. The prompt format is simply the question text, the four choices formatted as A/B/C/D, and a minimal completion target. There is no roleplay framing ("You are a helpful assistant…"), no elaborate instructions, and no chain-of-thought elicitation. This is the format used to produce the published benchmark scores that papers report.

We initially used a system prompt ("You are a helpful assistant answering multiple-choice questions"), but removed it to align with standard evaluation practice and eliminate an uncontrolled variable. The "helpful assistant" framing has fallen out of favour in evaluation contexts because it adds prompt sensitivity without clear benefit — the MMLU-Pro paper (Wang et al., NeurIPS 2024) found that MMLU scores varied by 4–5% across different prompt styles, making prompt design a meaningful confound.

## The Case for Chain-of-Thought

Chain-of-thought prompting (Wei et al., 2022; Kojima et al., 2022) asks the model to produce intermediate reasoning steps before giving a final answer. For our UQ framework, CoT is interesting for two reasons:

1. **Accuracy improvement.** CoT consistently improves performance on reasoning-heavy tasks. The MMLU-Pro paper found that CoT improved performance on MMLU-Pro (the harder, 10-option variant) compared to direct answering — though notably, this effect was weaker on the original 4-option MMLU.

2. **Relevance to the UQ framework.** CoT gives the model more "space" to activate relevant knowledge before committing to an answer. This could increase answer diversity across paraphrases on questions where the model has partial knowledge, which is precisely the signal our Dirichlet posterior needs to detect uncertainty.

## Why Not Native Thinking Mode?

Qwen3 supports a native thinking mode (think=true) that enables internal chain-of-thought reasoning. We tested this and encountered two problems:

- **Unbounded runtime.** Think mode is open-ended — the model can reason for 30 seconds or 10 minutes, and we cannot control the duration. On certain questions (particularly formal logic and abstract algebra), the model entered extremely long reasoning chains that consumed 5+ minutes per query. With 100 queries per question and 5,330 questions, this made full experimental runs infeasible.

- **Generalisability.** Native thinking is a model-specific architecture feature unique to models like Qwen3, DeepSeek-R1, and similar reasoning-trained models. CoT prompting, by contrast, works on any instruction-following LLM, making findings more broadly applicable.

## The JSON Schema Ordering Insight

Our experiments enforce structured output via Ollama's JSON schema support. The model must return valid JSON matching a predefined schema, which eliminates parsing ambiguity. However, combining CoT with structured output requires care.

Research from the Instructor library benchmarks (2024) demonstrated that adding a `chain_of_thought` reasoning field to a JSON response schema improved accuracy by up to 60% on GSM8k. However, a critical finding emerged: **the order of fields in the JSON schema determines whether reasoning actually occurs**. When a JSON schema places the "answer" field before a "reasoning" field, models tend to commit to an answer first and then generate post-hoc rationalisation. When "reasoning" comes first, the model must generate its reasoning before the answer token exists.

This is not a prompting trick but a consequence of how autoregressive language models work: they generate tokens strictly left-to-right. When writing the reasoning field, the answer field does not yet exist — there are no future tokens to "read ahead" to. The model genuinely cannot see its own answer while producing its reasoning.

A caveat: the model may have already formed an internal "intention" in its hidden states before generating the reasoning tokens. The reasoning may therefore be unfaithful — a rationalisation of a pre-existing inclination rather than genuine deliberation. This phenomenon is documented (Bentham et al., 2024, "Chain-of-thought unfaithfulness as disguised accuracy"). However, even unfaithful CoT tends to improve accuracy in practice, because the generated reasoning tokens become part of the context for the final answer token, creating a real causal influence on the output.

For our UQ framework, faithfulness of reasoning is not a concern. We measure only the final answer choice, which feeds into the Dirichlet posterior. Whether the model reasoned genuinely or rationalised, the answer is what matters for our uncertainty quantification.

## Separate Findings on JSON Mode and Reasoning

A study on structured output formats ("Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models") found that strict JSON-mode enforcement can degrade reasoning performance compared to free-text generation. This is mitigated by including reasoning fields in the schema — the model retains its ability to reason when given explicit space to do so within the structured format.

Additionally, there is a practical concern about field naming. The Instructor library benchmarks found that schema field names significantly affect performance — renaming a field from "potential_final_choice" to "final_answer" improved accuracy from 4.5% to 95% in one test case. We use clear, conventional field names ("reasoning", "answer", "option_a" etc.) to avoid this pitfall.

## Our Three Prompting Conditions

Based on this analysis, we implement three prompting conditions as experimental variables:

### Condition 1: Direct (baseline)

No system message. Minimal user prompt. This matches the standard MMLU evaluation protocol as closely as possible within our structured output framework.

**User message:**
```
{question}

  A) {choice_a}
  B) {choice_b}
  C) {choice_c}
  D) {choice_d}

Respond with the letter of the correct answer.
```

**JSON schema:**
```json
{
  "type": "object",
  "properties": {
    "answer": {"type": "string", "enum": ["A", "B", "C", "D"]}
  },
  "required": ["answer"]
}
```

### Condition 2: Chain-of-Thought (CoT)

No system message. The user message adds a minimal reasoning instruction. The JSON schema includes a `reasoning` field **before** the `answer` field, forcing the model to generate reasoning before committing to an answer.

**User message:**
```
{question}

  A) {choice_a}
  B) {choice_b}
  C) {choice_c}
  D) {choice_d}

Consider each option, then give your answer.
```

**JSON schema:**
```json
{
  "type": "object",
  "properties": {
    "reasoning": {"type": "string"},
    "answer": {"type": "string", "enum": ["A", "B", "C", "D"]}
  },
  "required": ["reasoning", "answer"]
}
```

### Condition 3: Structured Chain-of-Thought (CoT-Structured)

No system message. The JSON schema forces the model to produce a brief evaluation of **each** option individually before giving a final answer. This is the most constrained condition, producing consistent-length reasoning and structured per-option evaluations.

**User message:**
```
{question}

  A) {choice_a}
  B) {choice_b}
  C) {choice_c}
  D) {choice_d}

Evaluate each option, then give your answer.
```

**JSON schema:**
```json
{
  "type": "object",
  "properties": {
    "option_a": {"type": "string"},
    "option_b": {"type": "string"},
    "option_c": {"type": "string"},
    "option_d": {"type": "string"},
    "answer": {"type": "string", "enum": ["A", "B", "C", "D"]}
  },
  "required": ["option_a", "option_b", "option_c", "option_d", "answer"]
}
```

This condition is particularly interesting for the UQ framework because the per-option evaluations can be analysed alongside the posterior. If the model writes ambivalent evaluations for multiple options, does that predict a split posterior across paraphrases? This is a secondary analysis question that falls out of the data at no additional cost.

## Experimental Design

Prompting mode is one of three binary/ternary experimental variables:

| Variable | Levels | Rationale |
|----------|--------|-----------|
| Prompt mode | direct / cot / cot_structured | Does reasoning improve accuracy and/or posterior quality? |
| Answer shuffling | on / off | Does randomising answer order integrate out position bias? |
| Paraphrasing | on / off | Does input diversification improve posterior calibration? |

All conditions use the same random seed, ensuring the same 100 (or 5,330) questions are sampled for each run, enabling paired comparisons.

## What We Deliberately Excluded

- **Prompt repetition.** A March 2025 paper (arXiv:2512.14982) found that repeating the entire prompt twice consistently improves performance on non-reasoning models by allowing all tokens to attend to all other tokens. This is an interesting technique but adds another variable to an already full design. Noted for future work.

- **Few-shot examples.** The standard MMLU evaluation uses 5-shot prompting (5 example question-answer pairs prepended to each question). We use zero-shot to keep the prompt clean and because our structured JSON output format is incompatible with the traditional few-shot format. This is a known trade-off — few-shot prompting typically adds 2–5% accuracy on MMLU.

- **System prompt / roleplay framing.** Removed to align with standard evaluation practice and reduce prompt sensitivity as a confound.

## References

- Wang, Y. et al. (2024). "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark." NeurIPS 2024. arXiv:2406.01574.
- Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.
- Kojima, T. et al. (2022). "Large Language Models are Zero-Shot Reasoners." NeurIPS 2022.
- Bentham, O., Stringham, N., & Marasovic, A. (2024). "Chain-of-thought unfaithfulness as disguised accuracy." Transactions on Machine Learning Research.
- Gema, A.P. et al. (2025). "Are We Done with MMLU?" NAACL 2025. arXiv:2406.04127.
- EleutherAI lm-evaluation-harness. https://github.com/EleutherAI/lm-evaluation-harness
- Instructor Library Benchmarks (2024). "Bad Schemas Could Break Your LLM Structured Outputs." https://python.useinstructor.com/blog/2024/09/26/bad-schemas-could-break-your-llm-structured-outputs/
- "Prompt Repetition Improves Non-Reasoning LLMs." arXiv:2512.14982 (March 2025).
