"""Quick validation of test run results."""
import json
from pathlib import Path

def validate(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    name = data["run_name"]
    model = data["config"]["model_name"]
    n = len(data["question_results"])
    completed = data.get("completed_at", "N/A")
    print(f"=== {name} ===")
    print(f"Model: {model}")
    print(f"Questions: {n}")
    print(f"Completed: {completed}")

    correct = sum(1 for q in data["question_results"] if q["correct"])
    print(f"Accuracy: {correct}/{n} ({100*correct/n:.1f}%)")

    has_time = sum(1 for q in data["question_results"] if q["query_log"][0].get("inference_time_s") is not None)
    has_ptok = sum(1 for q in data["question_results"] if q["query_log"][0].get("prompt_tokens") is not None)
    has_otok = sum(1 for q in data["question_results"] if q["query_log"][0].get("output_tokens") is not None)
    print(f"Timing fields: inference_time={has_time}, prompt_tokens={has_ptok}, output_tokens={has_otok}")

    times = [q["query_log"][0]["inference_time_s"] for q in data["question_results"] if q["query_log"][0].get("inference_time_s")]
    ptoks = [q["query_log"][0]["prompt_tokens"] for q in data["question_results"] if q["query_log"][0].get("prompt_tokens")]
    otoks = [q["query_log"][0]["output_tokens"] for q in data["question_results"] if q["query_log"][0].get("output_tokens")]
    if times:
        print(f"Inference time: mean={sum(times)/len(times):.2f}s, min={min(times):.2f}s, max={max(times):.2f}s")
    if ptoks:
        print(f"Prompt tokens: mean={sum(ptoks)/len(ptoks):.0f}, min={min(ptoks)}, max={max(ptoks)}")
    if otoks:
        print(f"Output tokens: mean={sum(otoks)/len(otoks):.0f}, min={min(otoks)}, max={max(otoks)}")

    valid_lp = 0
    for q in data["question_results"]:
        ql = q["query_log"][0]
        if ql.get("canonical_probs") and len(ql["canonical_probs"]) == 4 and abs(sum(ql["canonical_probs"]) - 1.0) < 0.01:
            valid_lp += 1
    print(f"Valid logprob distributions (4 options, sum~1): {valid_lp}/{n}")

    first_q = data["question_results"][0]["query_log"][0]
    print(f"Sample raw_response: {first_q['raw_response'][:50].encode('ascii', 'replace').decode()}")
    print(f"Sample display_answer: {first_q['display_answer']}")
    skipped = sum(1 for q in data["question_results"] if q.get("skipped"))
    print(f"Skipped: {skipped}")

    # Per-question detail for small sets
    if n <= 5:
        for i, qr in enumerate(data["question_results"]):
            ql = qr["query_log"][0]
            print(f"  Q{i}: {qr['question_id']} | correct={qr['correct']} | time={ql.get('inference_time_s','N/A')}s | ptok={ql.get('prompt_tokens','N/A')} | otok={ql.get('output_tokens','N/A')}")
            if ql.get("canonical_probs"):
                print(f"       probs={[round(p,3) for p in ql['canonical_probs']]} sum={sum(ql['canonical_probs']):.4f}")
    print()

def deep_validate(path):
    """Detailed analysis of result quality."""
    import statistics
    from collections import Counter

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    qr = data["question_results"]
    name = data["run_name"]
    print(f"=== DEEP ANALYSIS: {name} ===")

    confs = [max(q["mean_probs"]) for q in qr]
    print(f"Max-prob confidence: mean={statistics.mean(confs):.3f}, median={statistics.median(confs):.3f}, stdev={statistics.stdev(confs):.3f}")
    print(f"Very confident (>0.9): {sum(1 for c in confs if c > 0.9)}")
    print(f"Uncertain (<0.5): {sum(1 for c in confs if c < 0.5)}")

    ans = Counter(q["final_answer"] for q in qr)
    print(f"Answer distribution: {dict(sorted(ans.items()))}")

    c_conf = [max(q["mean_probs"]) for q in qr if q["correct"]]
    w_conf = [max(q["mean_probs"]) for q in qr if not q["correct"]]
    print(f"Correct confidence: mean={statistics.mean(c_conf):.3f} (n={len(c_conf)})")
    print(f"Wrong confidence: mean={statistics.mean(w_conf):.3f} (n={len(w_conf)})")

    indices = [0, 25, 50, 75, min(99, len(qr)-1)] if len(qr) >= 100 else list(range(len(qr)))
    for i in indices:
        q = qr[i]
        ql = q["query_log"][0]
        top = sorted(ql["display_letter_logprobs"].items(), key=lambda x: x[1], reverse=True)
        lp_str = ", ".join(f"{k}={v:.2f}" for k, v in top)
        print(f"\nQ{i}: {q['question_id']} | ans={ql['display_answer']} | correct={q['correct']}")
        print(f"  logprobs: {lp_str}")
        print(f"  probs: {[round(p,3) for p in q['mean_probs']]}")
        print(f"  time={ql['inference_time_s']:.2f}s, ptok={ql['prompt_tokens']}, otok={ql['output_tokens']}")
    print()


results_dir = Path("results")
gemma = list(results_dir.glob("test_gemma4_direct_noshuffle_*.json"))
qwen = list(results_dir.glob("test_qwen35_direct_noshuffle_*.json"))

if gemma:
    validate(sorted(gemma)[-1])
    deep_validate(sorted(gemma)[-1])
if qwen:
    validate(sorted(qwen)[-1])
