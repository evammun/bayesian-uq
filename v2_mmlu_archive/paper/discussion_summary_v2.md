# Logprob-Guided Adaptive Inference for LLMs: Discussion Summary

## 1. Logprob Access in Local Inference

When a language model generates text, it doesn't just output words — at each step, it computes a probability distribution over its entire vocabulary (typically 128K+ tokens). The token it outputs is sampled from this distribution. Log probabilities (logprobs) are the logarithm of these probabilities, and they tell you how confident the model was in each choice it made. A logprob of 0 means 100% confidence; a logprob of -2.3 means roughly 10% confidence.

Modern local inference backends — llama-server (llama.cpp), Ollama, and vLLM — all let you inspect these logprobs on generated output. You can ask for the top-N most probable tokens at each position, not just the one that was selected. This is what makes the ideas in this document possible: you can see not just what the model said, but what it *almost* said, and how close the decision was.

The key differences between backends:

- **llama-server** offers a unique `post_sampling_probs` toggle. When you apply grammar constraints (e.g., forcing the model to output valid JSON), the model's raw probability distribution gets masked — tokens that violate the grammar are zeroed out, and the remaining probabilities are renormalized. With this toggle, you can see the distribution both *before* and *after* this masking. This matters because it lets you detect when the grammar forced the model into an output it wasn't actually confident about.
- **vLLM** uniquely supports `prompt_logprobs` — log probabilities for each token in the *input* prompt, not just the output. This tells you how "surprising" each input token was to the model given the preceding context, which is essentially a perplexity measurement over your prompt.
- **Ollama** added logprobs in v0.12.11 (November 2025).
- **llama-server does not expose prompt logprobs** through its API, despite the underlying library supporting it. However, `llama-perplexity`, a separate binary that ships with llama.cpp, computes per-token logits over any input text and can be used as a workaround.

Any algorithm designed against the common denominator — output token logprobs with top-N alternatives — works across all these backends and most cloud APIs (OpenAI, Together, etc.). A thin adapter layer is all that's needed to normalize response formats.

For Qwen 3.5 specifically: thinking is enabled by default and cannot be soft-toggled mid-conversation (unlike Qwen 3). Disabling requires `--chat-template-kwargs '{"enable_thinking": false}'` at the server level or per-request.

---

## 2. What Existing Research Does (and Doesn't Do)

The ideas in this document sit at the intersection of three research directions. Each one solves part of the problem, but they've never been combined — and the combination is where the value lies.

### The self-consistency approach: sample the same question multiple times

The core observation here is simple: if you ask a model the same question multiple times with temperature > 0, it will sometimes give different answers. If most answers agree, you can be more confident the answer is correct. This is **Self-Consistency** (Wang et al., 2022), and it works remarkably well — +17.9% accuracy on the GSM8K math benchmark just by sampling 40 reasoning paths and taking the majority vote.

The interesting detail for our purposes: the authors also tried a smarter aggregation — weighting each answer by its sequence-level log probability instead of just counting votes. It didn't help. They concluded that model probabilities are poorly calibrated. This finding has been widely accepted, and it's why subsequent work (Adaptive-Consistency by Aggarwal et al., 2023; RASC by Wan et al., 2025) continued to use vote-counting and answer-level agreement as the signal, never revisiting token-level logprobs.

But "sequence-level log probability doesn't help" and "token-level logprobs don't help" are very different claims. Nobody tested the second one.

### The rephrasing approach: ask the same question differently

A separate line of work noticed that models are sensitive to how a question is phrased. The same model can answer correctly with one wording and incorrectly with another, even when the meaning is identical. **Rephrase and Respond** (Deng et al., 2024) showed that just asking the model to rephrase the question before answering achieves near-perfect accuracy on tasks GPT-4 previously found hard.

**"Just Rephrase It!"** (Beker et al., 2024) took this further by generating 10 rephrasings and using majority vote across them to estimate uncertainty. Crucially, the entire paper is framed as a workaround for closed-source models where logprobs are unavailable. The implicit assumption is: if you had logprobs, you wouldn't need rephrasing.

This assumption is wrong. Logprobs can be confidently wrong — a model can assign 95% probability to an incorrect answer because the specific phrasing happens to activate a misleading pattern in the weights. Rephrasing breaks that pattern. So what you actually want is both signals together: logprobs to detect *when* the model is uncertain, and rephrasing to test *whether* its confidence is robust or fragile. A model that's 95% confident on phrasing A but 40% confident on a different answer for phrasing B is telling you something that neither signal alone would reveal.

### The semantic entropy approach: uncertainty over meanings, not tokens

A third line of work tackles the problem that "Paris" and "The capital of France is Paris" are semantically identical but look completely different at the token level. **Semantic Entropy** (Farquhar et al., 2024, published in Nature) clusters multiple model generations by meaning before computing entropy. High semantic entropy means the model is producing diverse *meanings*, not just diverse *wordings* — a much stronger signal of genuine uncertainty. **Semantic Entropy Probes** (Kossen et al., 2024) later showed you can approximate this signal from a single generation's hidden states, avoiding the cost of multiple samples.

### What's missing

Nobody has combined these three ideas. Specifically:

1. **Using token-level logprobs** (not sequence-level) to detect uncertainty at specific decision points — which token is the model unsure about?
2. **Using that signal to trigger adaptive rephrasing** — only rephrase when the model is uncertain, and target the rephrasing to the specific source of uncertainty
3. **Aggregating across rephrasings using logprob-weighted voting** rather than raw majority vote
4. **Applying all of this to tool calling decisions**, where the consequences of a wrong choice are immediate and often irreversible

The pre/post-grammar logprob split (unique to llama-server) adds another dimension: you can detect when constrained decoding (like forcing valid tool call JSON) is pushing the model away from what it actually wants to generate.

---

## 3. Tool Calling: How It Works and Where It Breaks

### The mechanics

Tool calling in LLMs is not a special internal mechanism — it's just text generation with conventions. Here's what actually happens:

The chat template injects tool definitions into the prompt as structured text (names, descriptions, parameter schemas). The model then generates tokens like it always does. If it "decides" to call a tool, it outputs tokens that form structured JSON — something like `{"name": "get_weather", "arguments": {"location": "Paris"}}`. The serving infrastructure parses this output, and the *client* is responsible for actually executing the tool and feeding the result back.

After a tool call, the model doesn't "resume" — it's a completely new inference call. The framework constructs a fresh prompt containing the full conversation history plus the tool result, and the model does a new forward pass over everything. It could then call another tool, or respond. Each turn is independent.

For reasoning models like Qwen 3.5, the stream is: `<think>reasoning...</think>` followed by either a tool call or a direct response. In vLLM, both a reasoning parser and a tool call parser run simultaneously. Locally, you get logprobs on everything including the thinking tokens — cloud APIs like DeepSeek explicitly block logprobs during thinking mode.

### Where it breaks

Evaluations of tool-calling agents reveal consistent failure patterns. Docker's testing of local models found "eager invocation" (tools called for simple greetings) and wrong tool selection (searching when it should add to cart). Research shows tool selection accuracy drops below 50% with 100+ available tools. The current best fix is simply using a better model — prompt engineering and tool descriptions have negligible impact on overall tool correctness.

And with the non-zero temperatures that reasoning models require (Qwen 3.5 recommends 0.6), the framework isn't even picking the most probable tool — it's sampling from the distribution. The model might be 60% confident in `get_weather` and 35% confident in `web_search`, and sampling noise alone decides which one gets executed. Both cases are treated identically: execute blindly, feed the result back, continue.

---

## 4. The Proposed Framework

### Core insight

Every decision in an LLM agent — whether to call a tool, which tool to call, what arguments to pass, whether the result is sufficient — is just token prediction with inspectable logprobs. At each of these points, the model is telling you how confident it is. Nobody is listening.

### Two critical inspection points

The most impactful places to check logprobs in a tool-calling agent are:

1. **Should I call a tool at all?** At the token position where the model either starts generating `<tool_call>` or starts a direct text response, the logprobs reveal the model's confidence in this branching decision. If it's nearly 50/50, the model genuinely doesn't know whether it needs external information — and you could re-sample, rephrase the question, or fall back to a safer default before committing.

2. **Which tool?** The logprob on the tool name token tells you how confident the model is in its choice. If it's split between two tools, that's a detectable signal *before* you execute anything — before you make an API call, write to a database, or send a message that can't be unsent.

### Why nobody is doing this

Agent frameworks (LangChain, CrewAI, AutoGen) operate at the message level — you send messages, you get back tool calls or text. The token-level probabilities are completely abstracted away; there isn't even a hook to access them. The agent framework ecosystem grew up on cloud APIs where logprobs on tool calls were unavailable or limited, so nobody designed around the signal. Agent developers and inference/ML engineers are largely separate communities reading different papers. And "just use a better model" has been the easier fix — until you need reliability guarantees that no model alone can provide.

### Properties of the framework

- **Universally applicable**: The core signal (output logprobs with top-N alternatives) is available across llama-server, Ollama, vLLM, and cloud APIs. llama-server's pre/post-grammar split and vLLM's prompt logprobs add extra capabilities but aren't required.
- **Pre-execution**: Errors are caught before they propagate. A wrong tool call in step 1 of an agentic chain can waste every subsequent step — catching it at the logprob level prevents the entire cascade.
- **Cheap**: Inspecting logprobs that are already computed costs nothing. The model produces the full distribution over its vocabulary at every step anyway; you're just reading it.
- **Novel combination**: Token-level logprobs, adaptive rephrasing, grammar-constrained inspection, and tool calling confidence are all established ideas individually. Combining them into a unified framework — and applying it at every decision point in an agentic loop — is new.
